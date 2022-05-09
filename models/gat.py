from typing import Union, Tuple, Optional, Callable
from torch_geometric.typing import Size, OptTensor
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import ModuleList, Linear, ELU, BatchNorm1d
from torch_sparse import SparseTensor, matmul

from torch.nn.parameter import Parameter
from torch_geometric.nn.conv import MessagePassing

from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import   softmax
from torch_scatter import scatter_mean
from torch_geometric.nn import PairNorm

class GATConv(MessagePassing):
    _alpha: OptTensor

    def __init__(self, in_channels: Union[int, Tuple[int, int]],out_channels: int,
                 n_class, rsl, 
                 heads: int = 1, concat: bool = True, bias: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0.0,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(GATConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.n_class = n_class
        self.rsl = rsl

        self.lin_l = Linear(in_channels, heads * out_channels, bias=False)
        self.lin_r = self.lin_l

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
        assert x.dim() == 2, 'Static graphs not supported in `GATConv`.'
        x_l = x_r = self.lin_l(x).view(-1, H, C)
        alpha_l = (x_l * self.att_l).sum(dim=-1)
        alpha_r = (x_r * self.att_r).sum(dim=-1)

        assert x_l is not None
        assert alpha_l is not None

        

        out = self.propagate(edge_index, x=x_l,
                             alpha=(alpha_l, alpha_r), size=size)
        #########################################################################################
        if self.rsl>0.:        
            label, p = label_p
            x_l = x_l.view(-1, self.heads * self.out_channels)
            out = out.view(-1, self.heads * self.out_channels)        
            x = x_l
            centers = scatter_mean(x,label,dim=0,dim_size=self.n_class).detach()
            c_repeat = centers.repeat(self.n_class,1,1)
            r = c_repeat.transpose(0,1) - c_repeat
            r_norm = r.norm(dim=-1, keepdim=True)
            r_norm[r_norm==0] = 1.
            r = r/r_norm
            #---------------------------------------------
            r = torch.matmul(cm,r).squeeze_()
            #---------------------------------------------
            r = torch.matmul(p,r)
            # r = r[label_p]
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
class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels: int,
                 hidden_heads: int, out_channels: int, 
                 num_layers: int, dropout: float = 0.0,
                 batch_norm: bool = False,rsl=0.0, pairnorm=False):######################################
        super().__init__()
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.pairnorm = pairnorm######################################
        
        self.convs = ModuleList()
        self.bns = ModuleList()
        self.hidden_heads = hidden_heads
        
        for i in range(num_layers):
            in_dim = in_channels if i == 0 else hidden_channels * hidden_heads
            if i < num_layers - 1:
                conv = GATConv(in_dim, hidden_channels, out_channels, rsl, hidden_heads, concat=True,
                               dropout=dropout)
            if i == num_layers - 1:
                conv = GATConv(hidden_channels * hidden_heads, out_channels,  out_channels, rsl, 1,
                               concat=False, dropout=dropout)
            self.convs.append(conv)

        for i in range(num_layers):
            if pairnorm:######################################
                bn = PairNorm()
            else:
                bn = BatchNorm1d(hidden_channels * hidden_heads)
            self.bns.append(bn)
    @property
    def reg_modules(self):
        return ModuleList(list(self.convs) + list(self.bns))
    @property
    def nonreg_modules(self):
        return ModuleList()
    def forward(self, x, edge_index, label_p, cm) -> Tensor:

        for conv, bn in zip(self.convs[:-1], self.bns):
            h = conv(x, edge_index, label_p, cm)
            if self.batch_norm or self.pairnorm:######################################
                h = bn(h)
            x = F.elu(h)
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
            if self.batch_norm or self.pairnorm:######################################
                h = bn(h)
            x = F.elu(h)
            x = F.dropout(x, p=self.dropout, training=self.training)
        h = self.convs[-1](x, edge_index, label_p, cm)
        embeddings.append(h.detach())
        return embeddings
    
    















