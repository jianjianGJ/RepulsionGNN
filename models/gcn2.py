from typing import Optional,Tuple
from math import log
import torch
from torch.nn import ModuleList, Linear, BatchNorm1d
from torch import Tensor
import torch.nn.functional as F
from torch_sparse import SparseTensor,matmul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import  OptTensor
from torch.nn.parameter import Parameter
from torch_geometric.nn.inits import glorot
from torch_scatter import scatter_mean
from torch_geometric.nn import PairNorm
class GCN2Conv(MessagePassing):
    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, channels: int, 
                 n_class, rsl,
                 alpha: float, theta: float = None,
                  layer: int = None, shared_weights: bool = True,
                  **kwargs):

        kwargs.setdefault('aggr', 'add')
        super(GCN2Conv, self).__init__(**kwargs)
        
        self.channels = channels
        
        self.n_class = n_class
        self.rsl = rsl
        
        self.alpha = alpha
        self.beta = 1.
        if theta is not None or layer is not None:
            assert theta is not None and layer is not None
            self.beta = log(theta / layer + 1)
        

        self.weight1 = Parameter(torch.Tensor(channels, channels))

        if shared_weights:
            self.register_parameter('weight2', None)
        else:
            self.weight2 = Parameter(torch.Tensor(channels, channels))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight1)
        glorot(self.weight2)
        self._cached_edge_index = None
        self._cached_adj_t = None
    def forward(self, x, x_0, edge_index, label_p, cm,
                edge_weight: OptTensor = None) -> Tensor:
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)
########################################################################################
        if self.rsl>0.:
            label, p = label_p
            centers = scatter_mean(x+0.1*x_0,label,dim=0,dim_size=self.n_class).detach()
            c_repeat = centers.repeat(self.n_class,1,1)
            r = c_repeat.transpose(0,1) - c_repeat
            r_norm = r.norm(dim=-1, keepdim=True)
            r_norm[r_norm==0] = 1.
            r = r/r_norm
            #---------------------------------------------
            r = torch.matmul(cm,r).squeeze_()
            #---------------------------------------------
            # r = r[label]
            r = torch.matmul(p,r)
            a = out - x
            out = x + self.rsl*r + (1-self.rsl)*a
########################################################################################  
        out.mul_(1 - self.alpha)
        x_0 = self.alpha * x_0

        if self.weight2 is None:
            out = out + x_0
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
class GCN2(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels: int,
                 out_channels: int, num_layers: int, alpha: float,
                 theta: float, shared_weights: bool = True,
                 dropout: float = 0.0, 
                 batch_norm: bool = False,rsl=0.0, pairnorm=False):
        super().__init__()
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.pairnorm = pairnorm
        
        
        self.lins = ModuleList()
        self.lins.append(Linear(in_channels, hidden_channels))
        self.lins.append(Linear(hidden_channels, out_channels))
        self.convs = ModuleList()
        for i in range(num_layers):
            conv = GCN2Conv(hidden_channels, out_channels,rsl, alpha,theta,i + 1,shared_weights)
            self.convs.append(conv)
        self.bns = ModuleList()
        for i in range(num_layers):
            if pairnorm:######################################
                bn = PairNorm()
            else:
                bn = BatchNorm1d(hidden_channels)
            self.bns.append(bn)
    @property
    def reg_modules(self):
        return ModuleList(list(self.convs) + list(self.bns))
    @property
    def nonreg_modules(self):
        return self.lins
    def forward(self, x, adj_t, label_p, cm) -> Tensor:
        # x = F.dropout(x, p=self.dropout, training=self.training)
        x = x_0 = self.lins[0](x).relu_()
        x = F.dropout(x, p=self.dropout, training=self.training)
        for conv, bn in zip(self.convs, self.bns):
            h = conv(x, x_0, adj_t, label_p, cm)
            if self.batch_norm or self.pairnorm:
                h = bn(h)
            x = h.relu_()
            x = F.dropout(x, p=self.dropout, training=self.training)

        out = self.lins[1](x)
        
        return out
    @torch.no_grad()
    def get_embeddings(self, x, adj_t, label_p, cm) -> Tensor:
        self.eval()
        embeddings = []
        x = x_0 = self.lins[0](x).relu_()
        embeddings.append(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        for conv, bn in zip(self.convs, self.bns):
            h = conv(x, x_0, adj_t, label_p, cm)
            embeddings.append(h.detach())
            if self.batch_norm or self.pairnorm:
                h = bn(h)
            x = h.relu_()
            x = F.dropout(x, p=self.dropout, training=self.training)
        out = self.lins[1](x)
        embeddings.append(out.detach())
        return embeddings
    
  