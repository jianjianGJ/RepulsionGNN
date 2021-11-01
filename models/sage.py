from typing import Optional, Callable
import torch
from torch import Tensor
from torch.nn import ReLU
from torch_sparse import SparseTensor
from torch_geometric.nn.conv import MessagePassing
from torch_sparse import  matmul

from torch_scatter import scatter_mean

from .basic_gnn import BasicGNN

from torch_geometric.typing import Adj, Size

from torch.nn import Linear

class SAGEConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, 
                 n_class: int = None, 
                 momentum: float = 0.5, rsl:float = 0.0,
                 bias: bool = True, **kwargs):  # yapf: disable
        kwargs.setdefault('aggr', 'mean')
        super(SAGEConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.n_class = n_class#################################################
        self.register_buffer('centers', torch.zeros(n_class, out_channels))
        self.register_buffer('num_batches_tracked', 
                             torch.tensor(0, dtype=torch.long))################
        self.momentum = momentum###############################################
        self.rsl = rsl
        

        self.lin = Linear(in_channels, out_channels, bias=bias)

        self.reset_parameters()
        print(self.rsl)

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj, label_p, cm,
                size: Size = None) -> Tensor:
        """"""

        # propagate_type: (x: OptPairTensor)
        x = self.lin(x)
        out = self.propagate(edge_index, x=x, size=size)
###############################################################################
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
        return out


    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
class SAGE(BasicGNN):
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
            conv = SAGEConv(in_dim, out_dim, n_class=out_channels, momentum=momentum,
                           rsl=rsl)
            self.convs.append(conv)
        self.reset_parameters()