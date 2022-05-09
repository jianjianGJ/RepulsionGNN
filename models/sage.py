from typing import Optional, Callable
import torch
from torch import Tensor
from torch.nn import ReLU,ModuleList, BatchNorm1d
from torch_sparse import SparseTensor
from torch_geometric.nn.conv import MessagePassing
from torch_sparse import  matmul
import torch.nn.functional as F

from torch_geometric.typing import Adj, Size

from torch.nn import Linear
from torch_scatter import scatter_mean
from torch_geometric.nn import PairNorm

class SAGEConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, 
                 n_class, rsl,
                 bias: bool = True, **kwargs):  # yapf: disable
        kwargs.setdefault('aggr', 'mean')
        super(SAGEConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.n_class = n_class
        self.rsl = rsl

        self.lin = Linear(in_channels, out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj, label_p, cm,
                size: Size = None) -> Tensor:
        # propagate_type: (x: OptPairTensor)
        x = self.lin(x)
        out = self.propagate(edge_index, x=x, size=size)
###############################################################################
        if self.rsl>0.:
            label, p = label_p
            centers = scatter_mean(x,label,dim=0,dim_size=self.n_class).detach()
            c_repeat = centers.repeat(self.n_class,1,1)
            r = c_repeat.transpose(0,1) - c_repeat
            r_norm = r.norm(dim=-1, keepdim=True)
            r_norm[r_norm==0] = 1.
            r = r/r_norm
            #---------------------------------------------
            r = torch.matmul(cm,r).squeeze_()
            #---------------------------------------------
            # r = r[label_p]
            r = torch.matmul(p,r)
            a = out - x
            out = x + self.rsl*r + (1-self.rsl)*a
###############################################################################    
        return out


    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
class SAGE(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, num_layers: int, dropout: float = 0.0,
                 batch_norm: bool = False, rsl=0.0, pairnorm=False):
        super().__init__()
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.pairnorm = pairnorm
        
        self.convs = ModuleList()
        self.bns = ModuleList()
        for i in range(num_layers):
            in_dim = out_dim = hidden_channels
            if i == 0:
                in_dim = in_channels
            if i == num_layers - 1:
                out_dim = out_channels
            conv = SAGEConv(in_dim, out_dim, out_channels, rsl)
            self.convs.append(conv)
        for i in range(num_layers):
            if pairnorm:######################################
                bn = PairNorm()
            else:
                bn = BatchNorm1d(hidden_channels)
            self.bns.append(bn)
    @property
    def reg_modules(self):
        return ModuleList(list(self.convs[:-1]) + list(self.bns))
    @property
    def nonreg_modules(self):
        return self.convs[-1:]
    def forward(self, x, edge_index, label_p, cm) -> Tensor:

        for conv, bn in zip(self.convs[:-1], self.bns):
            h = conv(x, edge_index, label_p, cm)
            if self.batch_norm or self.pairnorm:
                h = bn(h)
            x = h.relu_()
            x = F.dropout(x, p=self.dropout, training=self.training)
        h = self.convs[-1](x, edge_index, label_p, cm)
        return h
    @torch.no_grad()
    def get_embeddings(self, x, edge_index, label_p, cm) -> Tensor:
        self.eval()
        embeddings = []
        embeddings.append(x)
        for conv, bn in zip(self.convs[:-1], self.bns):
            h = conv(x, edge_index, label_p, cm)
            embeddings.append(h.detach())
            if self.batch_norm or self.pairnorm:
                h = bn(h)
            x = h.relu_()
            x = F.dropout(x, p=self.dropout, training=self.training)
        h = self.convs[-1](x, edge_index, label_p, cm)
        embeddings.append(h.detach())
        return embeddings
   