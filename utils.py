
import os
import math
import torch
import torch.nn.functional as F
from models import GCN, GAT, GCN2, SAGE
from torch_scatter import scatter_mean
import random
import numpy as np
#%% 日志格式设置，全局变量初始化
version = None
epsilon = 1 - math.log(2)
device = None
n_node_feats, n_classes = 0, 0
#%% 小技巧 函数
def seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def get_GNN(modelname):
    if modelname=='GCN':
        GNN = GCN
    elif modelname=='GAT':
        GNN = GAT
    elif modelname=='GCN2':
        GNN = GCN2
    elif modelname=='SAGE':
        GNN = SAGE
    return GNN
def custom_loss_function(x, labels):
    y = F.cross_entropy(x, labels, reduction="none")
    y = torch.log(epsilon + y) - math.log(epsilon)
    return torch.mean(y)
def get_version(args):
    layers = args.architecture['num_layers']
    version = f'{args.dataset}-{layers}-{args.rsl}-GCN'
    return version

def get_auxiliary(label_logits, n_classes):
    p_softmax = torch.softmax(label_logits, dim=-1)
    label = p_softmax.max(1)[1]
    
    cm = scatter_mean(p_softmax,label,dim=0,dim_size=n_classes)
    cm = cm-torch.diag(cm.diag())
    cm = cm.unsqueeze(1)
    return (label, p_softmax), cm
def exp_exists(path, version):
    exp_list_ = os.listdir(path)
    exp_list = [exp[:-7] for exp in exp_list_]
    if version in exp_list:
        return True
    else:
        return False

def compute_micro_f1(logits, y,mask = None) -> float:
    if mask is not None:
        logits, y = logits[mask], y[mask]

    if y.dim() == 1:
        return int(logits.argmax(dim=-1).eq(y).sum()) / y.size(0)
    else:
        y_pred = logits > 0
        y_true = y > 0.5

        tp = int((y_true & y_pred).sum())
        fp = int((~y_true & y_pred).sum())
        fn = int((y_true & ~y_pred).sum())

        try:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            return 2 * (precision * recall) / (precision + recall)
        except ZeroDivisionError:
            return 0.
