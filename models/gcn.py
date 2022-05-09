from typing import Optional, Callable
import torch
from torch import Tensor
from torch.nn import ModuleList, BatchNorm1d
from torch_sparse import SparseTensor
from torch.nn.parameter import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import  OptTensor
from torch_sparse import  matmul
import torch.nn.functional as F
from torch_scatter import scatter_mean#########################################
from torch_geometric.nn import PairNorm
class GCNConv(MessagePassing):

    def __init__(self, in_channels: int, out_channels: int, 
                 n_class, rsl,#################################################
                 bias: bool = True, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super(GCNConv, self).__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.n_class = n_class
        self.rsl = rsl#########################################################
        
        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)

    def forward(self, x, edge_index, label_p, cm,##############################
                edge_weight: OptTensor = None) -> Tensor:
        """"""
        x = x @ self.weight
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)
        if self.bias is not None:
            out += self.bias
        #######################################################################
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
            r = torch.matmul(p,r)
            # r = r[label]
            a = out - x
            out = x + self.rsl*r + (1-self.rsl)*a
        #######################################################################

        return out
    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
class GCN(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, num_layers: int, dropout: float = 0.0,batch_norm: bool = False,
                 rsl=0.0, pairnorm=False):####################################################
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
            conv = GCNConv(in_dim, out_dim, out_channels, rsl)#################
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
    def forward(self, x, edge_index, label_p, cm) -> Tensor:###################

        for conv, bn in zip(self.convs[:-1], self.bns):
            h = conv(x, edge_index, label_p, cm)###############################
            if self.batch_norm or self.pairnorm:
                h = bn(h)
            x = h.relu_()
            x = F.dropout(x, p=self.dropout, training=self.training)
        h = self.convs[-1](x, edge_index, label_p, cm)#########################
        return h
    
    @torch.no_grad()
    def get_embeddings(self, x, edge_index, label_p, cm) -> Tensor:###################
        self.eval()
        embeddings = []
        embeddings.append(x)
        for conv, bn in zip(self.convs[:-1], self.bns):
            h = conv(x, edge_index, label_p, cm)###############################
            embeddings.append(h.detach())
            if self.batch_norm or self.pairnorm:
                h = bn(h)
            x = h.relu_()
            x = F.dropout(x, p=self.dropout, training=self.training)
        h = self.convs[-1](x, edge_index, label_p, cm)#########################
        embeddings.append(h.detach())
        return embeddings