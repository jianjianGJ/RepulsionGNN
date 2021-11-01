from typing import Union, Tuple, Optional, Callable
from torch_geometric.typing import Size, OptTensor
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import ModuleList, Linear, ELU
from torch_sparse import SparseTensor

from torch.nn.parameter import Parameter
from torch_geometric.nn.conv import MessagePassing

from torch_geometric.nn.inits import glorot, zeros
from torch_scatter import scatter_mean
from torch_geometric.utils import   softmax
from .basic_gnn import BasicGNN


class GATConv(MessagePassing):
    _alpha: OptTensor

    def __init__(self, in_channels: Union[int, Tuple[int, int]],out_channels: int,
                 heads: int = 1, concat: bool = True, bias: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0.0,
                 n_class:int = None, momentum:float = 0.5, rsl:float = 0.,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(GATConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_class = n_class
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        if concat:
            self.register_buffer('centers', torch.zeros(n_class, out_channels*heads))
        else:
            self.register_buffer('centers', torch.zeros(n_class, out_channels))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        self.momentum = momentum
        self.rsl = rsl

        if isinstance(in_channels, int):
            self.lin_l = Linear(in_channels, heads * out_channels, bias=False)
            self.lin_r = self.lin_l
        else:
            self.lin_l = Linear(in_channels[0], heads * out_channels, False)
            self.lin_r = Linear(in_channels[1], heads * out_channels, False)

        self.att_l = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_r = Parameter(torch.Tensor(1, heads, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None

        self.reset_parameters() 
        print(self.rsl)

    def reset_parameters(self):
        glorot(self.lin_l.weight)
        glorot(self.lin_r.weight)
        glorot(self.att_l)
        glorot(self.att_r)
        zeros(self.bias)

    def forward(self, x, edge_index, label_p, cm,
                size: Size = None, return_attention_weights=None):
        
        H, C = self.heads, self.out_channels

        x_l: OptTensor = None
        x_r: OptTensor = None
        alpha_l: OptTensor = None
        alpha_r: OptTensor = None
        if isinstance(x, Tensor):
            assert x.dim() == 2, 'Static graphs not supported in `GATConv`.'
            x_l = x_r = self.lin_l(x).view(-1, H, C)
            alpha_l = (x_l * self.att_l).sum(dim=-1)
            alpha_r = (x_r * self.att_r).sum(dim=-1)
        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2, 'Static graphs not supported in `GATConv`.'
            x_l = self.lin_l(x_l).view(-1, H, C)
            alpha_l = (x_l * self.att_l).sum(dim=-1)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)
                alpha_r = (x_r * self.att_r).sum(dim=-1)

        assert x_l is not None
        assert alpha_l is not None

        

        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)
        out = self.propagate(edge_index, x=(x_l, x_r),
                             alpha=(alpha_l, alpha_r), size=size)
#########################################################################################
        if self.rsl>0.:        
            x_l = x_l.view(-1, self.heads * self.out_channels)
            out = out.view(-1, self.heads * self.out_channels)        
            x = x_l[:edge_index.size(0)]
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
            out = out.view(-1, self.heads, self.out_channels)  
#########################################################################################

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)
    

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)
class GAT(BasicGNN):
    def __init__(self, num_nodes: int, in_channels, hidden_channels: int,
                 hidden_heads: int, out_channels: int, 
                 num_layers: int, dropout: float = 0.0,
                 rsl=0.0, momentum=0.5,
                 act: Optional[Callable] = ELU(inplace=True),
                 pool_size: Optional[int] = None,
                 buffer_size: Optional[int] = None, device=None):
        super().__init__(num_nodes, in_channels, hidden_channels*hidden_heads,out_channels, 
                         num_layers, dropout, False, False, False, 
                         False, act, pool_size, buffer_size)
        self.hidden_heads = hidden_heads
        
        for i in range(num_layers):
            in_dim = in_channels if i == 0 else hidden_channels * hidden_heads
            if i < num_layers - 1:
                conv = GATConv(in_dim, hidden_channels, hidden_heads, concat=True,
                               dropout=dropout, n_class=out_channels,
                               momentum=momentum, rsl=rsl)
            if i == num_layers - 1:
                conv = GATConv(hidden_channels * hidden_heads, out_channels,  1,
                               concat=False, dropout=dropout, n_class=out_channels,
                               momentum=momentum, rsl=rsl)
            self.convs.append(conv)

        self.convs.append(conv)
        self.reset_parameters()
    @property
    def reg_modules(self):
        return self.convs
    @property
    def nonreg_modules(self):
        return ModuleList()
    def forward(self, x: Tensor, adj_t: SparseTensor, label_p, cm, *args) -> Tensor:
        for conv, history in zip(self.convs[:-1], self.histories):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv((x, x[:adj_t.size(0)]), adj_t, label_p, cm)
            x = F.elu(x)
            x = self.push_and_pull(history, x, *args)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1]((x, x[:adj_t.size(0)]), adj_t, label_p, cm)
        return x

    @torch.no_grad()
    def forward_layer(self, layer, x, adj_t, label_p, cm, state):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[layer]((x, x[:adj_t.size(0)]), adj_t, label_p, cm)

        if layer < self.num_layers - 1:
            x =  F.elu(x)

        return x
    














