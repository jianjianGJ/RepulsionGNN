from typing import Optional, Callable
from torch_geometric.typing import Adj


import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import ModuleList, Linear, BatchNorm1d, ReLU
from torch_sparse import SparseTensor

from torch_geometric_autoscale.models import ScalableGNN

class BasicGNN(ScalableGNN):

    def __init__(self, num_nodes: int, in_channels: int, hidden_channels: int,
                 out_channels: int, num_layers: int, dropout: float = 0.0,
                 drop_input: bool = True, batch_norm: bool = False,
                 residual: bool = False, linear: bool = False, 
                 act: Optional[Callable] = ReLU(inplace=True),
                 pool_size: Optional[int] = None,
                 buffer_size: Optional[int] = None, device=None):
        super().__init__(num_nodes, hidden_channels, num_layers, pool_size,
                         buffer_size, device)
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        
        self.act = act
        self.dropout = dropout
        self.drop_input = drop_input
        self.batch_norm = batch_norm
        self.residual = residual
        
        self.linear = linear
        self.lins = ModuleList()
        self.convs = ModuleList()
        self.bns = ModuleList()
        for i in range(num_layers):
            bn = BatchNorm1d(hidden_channels)
            self.bns.append(bn)
        if linear:
            self.lins.append(Linear(in_channels, hidden_channels))
            self.lins.append(Linear(hidden_channels, out_channels))
    @property
    def reg_modules(self):
        if self.linear:
            return ModuleList(list(self.convs) + list(self.bns))
        else:
            return ModuleList(list(self.convs[:-1]) + list(self.bns))
    @property
    def nonreg_modules(self):
        return self.lins if self.linear else self.convs[-1:]
    
    def reset_parameters(self):
        super().reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj, label_p, cm, *args) -> Tensor:
        if self.drop_input:
            x = F.dropout(x, p=self.dropout, training=self.training)

        if self.linear:
            x = self.act(self.lins[0](x))
            x = F.dropout(x, p=self.dropout, training=self.training)
        for conv, bn, hist in zip(self.convs[:-1], self.bns, self.histories):
            h = conv(x, edge_index, label_p, cm)
            if self.batch_norm:
                h = bn(h)
            
            if self.residual and h.size(-1) == x.size(-1):
                h += x[:h.size(0)]
            x = self.act(h)
            #关键：PUSH AND PULL
            x = self.push_and_pull(hist, x, *args)
            x = F.dropout(x, p=self.dropout, training=self.training)
        h = self.convs[-1](x, edge_index, label_p, cm)
        if not self.linear:
            return h

        if self.batch_norm:
            h = self.bns[-1](h)
        if self.residual and h.size(-1) == x.size(-1):
            h += x[:h.size(0)]
        h = self.act(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.lins[1](h)
    @torch.no_grad()
    def forward_layer(self, layer, x, edge_index, label_p, cm, state):
        if layer == 0:
            if self.drop_input:
                x = F.dropout(x, p=self.dropout, training=self.training)
            if self.linear:
                x = self.act(self.lins[0](x))
                x = F.dropout(x, p=self.dropout, training=self.training)
        else:
            x = F.dropout(x, p=self.dropout, training=self.training)

        h = self.convs[layer](x, edge_index, label_p, cm)
        if layer < self.num_layers - 1 or self.linear:
            if self.batch_norm:
                h = self.bns[layer](h)
            if self.residual and h.size(-1) == x.size(-1):
                h += x[:h.size(0)]
            h = self.act(h)

        if self.linear:
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = self.lins[1](h)

        return h
    def __call__(
        self,
        x: Optional[Tensor] = None,
        edge_index: Optional[SparseTensor] = None,
        label_p = None,
        cm = None,
        batch_size: Optional[int] = None,
        n_id: Optional[Tensor] = None,
        offset: Optional[Tensor] = None,
        count: Optional[Tensor] = None,
        loader = None,
    ) -> Tensor:
        

        if loader is not None:
            return self.mini_inference(loader, cm)

        # We only perform asynchronous history transfer in case the following
        # conditions are met:
        #重要标志：self._async，是否使用异步传输
        self._async = (self.pool is not None and batch_size is not None
                       and n_id is not None and offset is not None
                       and count is not None)

        #在整个GNN模型开始前馈前开启所有异步pull
        if self._async:
            for hist in self.histories:
                #利用pool从CPU中读取嵌入至GPU
                self.pool.async_pull(hist.emb, None, None, n_id[batch_size:])
        #***************************前馈**************************************
        out = self.forward(x, edge_index, label_p, cm, batch_size, n_id, offset, count)

        #在整个GNN模型结束前馈后关闭所有的异步push
        if self._async:
            for hist in self.histories:
                #利用pool将GPU中嵌入储存至CPU
                self.pool.synchronize_push()

        self._async = False

        return out
    @torch.no_grad()
    def mini_inference(self, loader, cm):
        r"""一层前馈完前馈下一层"""


        loader = [sub_data + ({}, ) for sub_data in loader]

        # 第一层:
        for data, batch_size, n_id, offset, count, state in loader:
            x = data.x.to(self.device)
            adj_t = data.adj_t.to(self.device)
            label_p = data.label_p[:batch_size].to(self.device)###############################
            out = self.forward_layer(0, x, adj_t, label_p, cm, state)[:batch_size]###############################
            self.pool.async_push(out, offset, count, self.histories[0].emb)
        self.pool.synchronize_push()
        #中间层
        for i in range(1, len(self.histories)):
            # Pull the complete layer-wise history:
            for _, batch_size, n_id, offset, count, _ in loader:
                self.pool.async_pull(self.histories[i - 1].emb, offset, count,
                                     n_id[batch_size:])

            # Compute new output embeddings one-by-one and start pushing them
            # to the history.
            for batch, batch_size, n_id, offset, count, state in loader:
                adj_t = batch.adj_t.to(self.device)
                label_p = batch.label_p[:batch_size].to(self.device)###############################
                x = self.pool.synchronize_pull()[:n_id.numel()]
                out = self.forward_layer(i, x, adj_t, label_p, cm, state)[:batch_size]###############################
                self.pool.async_push(out, offset, count, self.histories[i].emb)
                self.pool.free_pull()
            self.pool.synchronize_push()

        # 从最后一层中获取嵌入:
        for _, batch_size, n_id, offset, count, _ in loader:
            self.pool.async_pull(self.histories[-1].emb, offset, count,
                                 n_id[batch_size:])

        # 最终嵌入写入一个输出嵌入矩阵：
        for batch, batch_size, n_id, offset, count, state in loader:
            adj_t = batch.adj_t.to(self.device)
            label_p = batch.label_p[:batch_size].to(self.device)###############################
            x = self.pool.synchronize_pull()[:n_id.numel()]
            out = self.forward_layer(self.num_layers - 1, x, adj_t, label_p, cm, state)[:batch_size]
            self.pool.async_push(out, offset, count, self._out)
            self.pool.free_pull()
        self.pool.synchronize_push()

        return self._out
    
    def get_rsl(self):
        rsl = 'beta for layers:'
        for i,conv in enumerate(self.convs):
            if conv.repulsion:
                rsl += f'{i}:{conv.rsl} '
        return rsl
    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, num_layers={self.num_layers})')