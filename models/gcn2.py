from typing import Optional,Tuple,Callable
from math import log
import torch
from torch import Tensor
import torch.nn.functional as F
from torch_sparse import SparseTensor,matmul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import  OptTensor
from torch.nn.parameter import Parameter
from torch_geometric.nn.inits import glorot
from torch_scatter import scatter_mean
from .basic_gnn import BasicGNN
from torch.nn import ReLU
class GCN2Conv(MessagePassing):
    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, channels: int, alpha: float, theta: float = None,
                 layer: int = None, shared_weights: bool = True,
                 n_class: int = None, 
                 momentum:float = 0.5, rsl:float = 0.0,
                 **kwargs):

        kwargs.setdefault('aggr', 'add')
        super(GCN2Conv, self).__init__(**kwargs)
        
        self.channels = channels
        self.alpha = alpha
        self.beta = 1.
        if theta is not None or layer is not None:
            assert theta is not None and layer is not None
            self.beta = log(theta / layer + 1)


        self.n_class = n_class
        self.register_buffer('centers', torch.zeros(n_class, channels))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        self.momentum = momentum
        self.rsl = rsl
        self.weight1 = Parameter(torch.Tensor(channels, channels))

        if shared_weights:
            self.register_parameter('weight2', None)
        else:
            self.weight2 = Parameter(torch.Tensor(channels, channels))

        self.reset_parameters()
        print(self.rsl)

    def reset_parameters(self):
        glorot(self.weight1)
        glorot(self.weight2)
        self._cached_edge_index = None
        self._cached_adj_t = None
    def forward(self, x, x_0, edge_index, label_p, cm,
                edge_weight: OptTensor = None) -> Tensor:
        """"""
        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)

########################################################################################
        if self.rsl>0.:
            x = x[:edge_index.size(0)]
            if self.training:
                if self.num_batches_tracked>0:
                    self.centers = (1.0-self.momentum)*self.centers +\
                                        self.momentum*scatter_mean(x,label_p,dim=0,
                                                                    dim_size=self.n_class).detach()
                    self.num_batches_tracked += 1
                else:
                    self.centers = scatter_mean(x,label_p,dim=0,dim_size=self.n_class).detach()
                    self.num_batches_tracked += 1
            c_repeat = self.centers.repeat(self.n_class,1,1)
            r = c_repeat.transpose(0,1) - c_repeat
            r_norm = r.norm(dim=-1, keepdim=True)
            r_norm[r_norm==0] = 1.
            r = r/r_norm
            #---------------------------------------------
            r = torch.matmul(cm,r).squeeze_()
            #---------------------------------------------
            r = r[label_p]
            a = out - x
            out = x + self.rsl*r + (1-self.rsl)*a
########################################################################################       
        
        out.mul_(1 - self.alpha)
        x_0 = self.alpha * x_0

        if self.weight2 is None:
            out = out+x_0
            out = torch.addmm(out, out, self.weight1, beta=1. - self.beta,
                              alpha=self.beta)
        else:
            out = torch.addmm(out, out, self.weight1, beta=1. - self.beta,
                              alpha=self.beta)
            out += torch.addmm(x_0, x_0, self.weight2, beta=1. - self.beta,
                               alpha=self.beta)

        return out

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, alpha={}, beta={})'.format(self.__class__.__name__,
                                                  self.channels, self.alpha,
                                                  self.beta)
class GCN2(BasicGNN):
    def __init__(self, num_nodes: int, in_channels, hidden_channels: int,
                 out_channels: int, num_layers: int, alpha: float,
                 theta: float, shared_weights: bool = True,
                 dropout: float = 0.0, drop_input: bool = True,
                 batch_norm: bool = False, residual: bool = False,
                 linear: bool = True, momentum = 0.5,
                 rsl=0.1,
                 act: Optional[Callable] = ReLU(inplace=True),
                 pool_size: Optional[int] = None,
                 buffer_size: Optional[int] = None, device=None):
        super().__init__(num_nodes, in_channels, hidden_channels,
                 out_channels, num_layers, dropout,
                 drop_input, batch_norm, residual, linear, 
                 act,pool_size,buffer_size, device)

        for i in range(num_layers):
            conv = GCN2Conv(hidden_channels, alpha,theta,i + 1,shared_weights,
                            out_channels, momentum,rsl)
            self.convs.append(conv)

    def forward(self, x: Tensor, adj_t: SparseTensor, label_p, cm, *args) -> Tensor:
        if self.drop_input:
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.lins[0](x).relu_()
        x_0 = x[:adj_t.size(0)]
        x = F.dropout(x, p=self.dropout, training=self.training)
        for conv, bn, hist in zip(self.convs[:-1], self.bns[:-1],
                                  self.histories):
            h = conv(x, x_0, adj_t, label_p, cm)
            if self.batch_norm:
                h = bn(h)
            if self.residual:
                h += x[:h.size(0)]
            x = h.relu_()
            x = self.push_and_pull(hist, x, *args)
            x = F.dropout(x, p=self.dropout, training=self.training)

        h = self.convs[-1](x, x_0, adj_t, label_p, cm)
        if self.batch_norm:
            h = self.bns[-1](h)
        if self.residual:
            h += x[:h.size(0)]
        x = h.relu_()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[1](x)
        return x

    @torch.no_grad()
    def forward_layer(self, layer, x, adj_t, label_p, cm, state):
        if layer == 0:
            if self.drop_input:
                x = F.dropout(x, p=self.dropout, training=self.training)

            x = x_0 = self.lins[0](x).relu_()
            state['x_0'] = x_0[:adj_t.size(0)]

        x = F.dropout(x, p=self.dropout, training=self.training)
        h = self.convs[layer](x, state['x_0'], adj_t, label_p, cm)
        if self.batch_norm:
            h = self.bns[layer](h)
        if self.residual and h.size(-1) == x.size(-1):
            h += x[:h.size(0)]
        x = h.relu_()

        if layer == self.num_layers - 1:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.lins[1](x)

        return x
    