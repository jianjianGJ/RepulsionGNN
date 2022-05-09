#%%
import os
from typing import Tuple
from torch_sparse import fill_diag, sum as sparsesum, mul
import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.datasets import (WikiCS, Coauthor, Amazon,
                                      Reddit2)
from ogb.nodeproppred import PygNodePropPredDataset
#%%
def index2mask(idx, size):
    mask = torch.zeros(size, dtype=torch.bool, device=idx.device)
    mask[idx] = True
    return mask
def gen_masks(y, train_per_class: int = 20, val_per_class: int = 30,
              num_splits: int = 20):
    num_classes = int(y.max()) + 1

    train_mask = torch.zeros(y.size(0), num_splits, dtype=torch.bool)
    val_mask = torch.zeros(y.size(0), num_splits, dtype=torch.bool)

    for c in range(num_classes):
        idx = (y == c).nonzero(as_tuple=False).view(-1)
        perm = torch.stack(
            [torch.randperm(idx.size(0)) for _ in range(num_splits)], dim=1)
        idx = idx[perm]

        train_idx = idx[:train_per_class]
        train_mask.scatter_(0, train_idx, True)
        val_idx = idx[train_per_class:train_per_class + val_per_class]
        val_mask.scatter_(0, val_idx, True)

    test_mask = ~(train_mask | val_mask)

    return train_mask, val_mask, test_mask
def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):

    fill_value = 2. if improved else 1.

    adj_t = edge_index
    if not adj_t.has_value():
        adj_t = adj_t.fill_value(1., dtype=dtype)
    if add_self_loops:
        adj_t = fill_diag(adj_t, fill_value)
    deg = sparsesum(adj_t, dim=1)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
    adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
    adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
    return adj_t
#%%

def get_wikics(root: str) -> Tuple[Data, int, int]:
    dataset = WikiCS(f'{root}/WIKICS', transform=T.ToSparseTensor())
    data = dataset[0]
    data.adj_t = data.adj_t.to_symmetric()
    data.val_mask = data.stopping_mask
    data.stopping_mask = None
    return data, dataset.num_features, dataset.num_classes


def get_coauthor(root: str, name: str) -> Tuple[Data, int, int]:
    dataset = Coauthor(f'{root}/Coauthor', name, transform=T.ToSparseTensor())
    data = dataset[0]
    torch.manual_seed(12345)
    data.train_mask, data.val_mask, data.test_mask = gen_masks(
        data.y, 20, 30, 10)
    return data, dataset.num_features, dataset.num_classes


def get_amazon(root: str, name: str) -> Tuple[Data, int, int]:
    dataset = Amazon(f'{root}/Amazon', name, transform=T.ToSparseTensor())
    data = dataset[0]
    torch.manual_seed(12345)
    data.train_mask, data.val_mask, data.test_mask = gen_masks(
        data.y, 20, 30, 10)
    return data, dataset.num_features, dataset.num_classes


def get_arxiv(root: str) -> Tuple[Data, int, int]:
    dataset = PygNodePropPredDataset('ogbn-arxiv', f'{root}/OGB',
                                     pre_transform=T.ToSparseTensor())
    data = dataset[0]
    data.adj_t = data.adj_t.to_symmetric()
    data.node_year = None
    data.y = data.y.view(-1)
    split_idx = dataset.get_idx_split()
    data.train_mask = index2mask(split_idx['train'], data.num_nodes)
    data.val_mask = index2mask(split_idx['valid'], data.num_nodes)
    data.test_mask = index2mask(split_idx['test'], data.num_nodes)
    return data, dataset.num_features, dataset.num_classes


def get_reddit(root: str) -> Tuple[Data, int, int]:
    dataset = Reddit2(f'{root}/Reddit2', pre_transform=T.ToSparseTensor())
    data = dataset[0]
    data.x = (data.x - data.x.mean(dim=0)) / data.x.std(dim=0)
    return data, dataset.num_features, dataset.num_classes

def get_data(root: str, name: str) -> Tuple[Data, int, int]:
    if name.lower() in ['coauthorcs', 'coauthorphysics']:
        return get_coauthor(root, name[8:])
    elif name.lower() in ['amazoncomputers', 'amazonphoto']:
        return get_amazon(root, name[6:])
    elif name.lower() == 'wikics':
        return get_wikics(root)
    elif name.lower() == 'reddit':
        return get_reddit(root)
    elif name.lower() in ['ogbn-arxiv', 'arxiv']:
        return get_arxiv(root)
    else:
        raise NotImplementedError

def load_data(data_name, root='/home/gj/SpyderWorkSpace/data/', 
              add_selfloop=True, norm=True, basemodel='GCN', silence=False):
    
    data, num_features, num_classes = get_data(root,data_name)
    #---------------------------------------------------------------------
    if os.path.exists(f'./datainfo-{basemodel}/{data_name}/cm.tensor'):
        cm = torch.load(f'./datainfo-{basemodel}/{data_name}/cm.tensor')
        cm = cm.reshape(num_classes,1,num_classes)
        data.cm = cm
    else:
        data.cm = None
        
    if os.path.exists(f'./datainfo-{basemodel}/{data_name}/label.tensor'):
        data.label_p = torch.load(f'./datainfo-{basemodel}/{data_name}/label.tensor')
        if not silence:
            print('label_p and cm are added!')
    else:
        data.label_p = None
    #---------------------------------------------------------------------

    if add_selfloop:
        # print('Adding self-loops...', flush=True)
        data.adj_t.set_value_(torch.ones(data.adj_t.nnz()),layout='coo')
        data.adj_t = data.adj_t.set_diag(torch.ones(data.adj_t.size(0))*1)
    if norm:
        # print('Normalizing(GCN) data...', flush=True)
        data.adj_t = gcn_norm(data.adj_t, add_self_loops=False)
    return data, num_features, num_classes
if __name__ == '__main__':
    data_name = 'amazoncomputers'
    data, d, c = load_data(data_name)
    print(d,c)
    print(data)
















