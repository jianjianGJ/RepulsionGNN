from typing import Optional, Callable
import torch
from torch import Tensor
from torch.nn import ReLU
from torch_sparse import SparseTensor
from torch.nn.parameter import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import  OptTensor
from torch_sparse import  matmul

from torch_scatter import scatter_mean

from .basic_gnn import BasicGNN

class GCNConv(MessagePassing):

    def __init__(self, in_channels: int, out_channels: int, 
                 n_class: int, momentum:float = 0.5, rsl:float = 0.,
                 bias: bool = True, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super(GCNConv, self).__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.n_class = n_class#################################################
        self.register_buffer('centers', torch.zeros(n_class, out_channels))
        self.register_buffer('num_batches_tracked', 
                             torch.tensor(0, dtype=torch.long))################
        self.momentum = momentum###############################################
        self.rsl = rsl
        
        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        print(self.rsl)
    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)

    def forward(self, x, edge_index, label_p, cm,
                edge_weight: OptTensor = None) -> Tensor:
        """"""
        x = x @ self.weight

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)
###############################################################################
        # pp = torch.rand(1)
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
###############################################################################
        if self.bias is not None:
            out += self.bias

        return out
    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
class GCN(BasicGNN):
    def __init__(self, num_nodes: int, in_channels: int, hidden_channels: int,
                 out_channels: int, num_layers: int, dropout: float = 0.0,
                 drop_input: bool = True, batch_norm: bool = False,
                 residual: bool = False, linear: bool = False, 
                 rsl=0.0, momentum=0.5,
                 act: Optional[Callable] = ReLU(inplace=True),
                 pool_size: Optional[int] = None,
                 buffer_size: Optional[int] = None, device=None):
        super().__init__(num_nodes, in_channels, hidden_channels,out_channels, 
                         num_layers, dropout, drop_input, batch_norm, residual, 
                         linear, act, pool_size, buffer_size)
        
        for i in range(num_layers):
            in_dim = out_dim = hidden_channels
            if i == 0 and not linear:
                in_dim = in_channels
            if i == num_layers - 1 and not linear:
                out_dim = out_channels
            conv = GCNConv(in_dim, out_dim, out_channels,
                           momentum=momentum, rsl=rsl)
            self.convs.append(conv)
        self.reset_parameters()
    