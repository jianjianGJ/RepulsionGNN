import argparse
import math
import time
import os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import sys
from models import GCN, GAT, GCN2, SAGE

from torch_geometric_autoscale import (SubgraphLoader, EvalSubgraphLoader,
                                       compute_micro_f1, dropout)
from parameters import set_args
from data_utils import load_data
from torch_scatter import scatter_mean
#python -u main.py --dataset=arxiv  --modelname=GCN --num-layers=3 --epochs=500 --n-runs=1 --rsl=0.2

version = None
epsilon = 1 - math.log(2)
device = None
n_node_feats, n_classes = 0, 0
#%% 小技巧 函数
def exp_exists(path, version):
    exp_list_ = os.listdir(path)
    exp_list = [exp[:-7] for exp in exp_list_]
    if version in exp_list:
        return True
    else:
        return False
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
def get_cm(label_logits, n_classes):
    p_softmax = torch.softmax(label_logits, dim=-1)
    label = p_softmax.argmax(dim=-1)
    cm = scatter_mean(p_softmax,label,dim=0,dim_size=n_classes)
    cm = cm-torch.diag(cm.diag())
    cm = cm.unsqueeze(1)
    return cm
#%% 训练 测试
def mini_train(model, loader, optimizer, max_steps, grad_norm=None,
               edge_dropout=0.0, cm=None):
    model.train()

    total_loss = total_examples = 0
    for i, (batch, batch_size, *args) in enumerate(loader):
        #获取batch，并将batch的数据放入device中
        x = batch.x.to(model.device)
        adj_t = batch.adj_t.to(model.device)
        y = batch.y[:batch_size].to(model.device)
        train_mask = batch.train_mask[:batch_size].to(model.device)
        label_p = batch.label_p[:batch_size].to(model.device)
        #如该batch内没有训练节点，则跳过
        if train_mask.sum() == 0:
            continue

        adj_t = dropout(adj_t, p=edge_dropout)

        #前馈及优化
        optimizer.zero_grad()
        out = model(x, adj_t, label_p, cm, batch_size, *args)
        loss = custom_loss_function(out[train_mask], y[train_mask])
        
        loss.backward()
        if grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
        optimizer.step()

        total_loss += float(loss) * int(train_mask.sum())
        total_examples += int(train_mask.sum())

        #避免模型权重更新过大，导致history不能近似真实嵌入，需要及时更新history
        if (i + 1) >= max_steps and (i + 1) < len(loader):
            break

    return total_loss / total_examples

#利用多态，model的前馈根据不同的输入执行不同的操作
@torch.no_grad()
def mini_test(model, loader, cm):
    model.eval()
    return model(loader=loader, cm=cm)

#%%
def run(args, data, ptr, n_running, cm):
    try:
        edge_dropout = args.edge_drop
    except: 
        edge_dropout = 0.0
    grad_norm = None if isinstance(args.grad_norm, str) else args.grad_norm
    #数据载入初始化
    train_loader = SubgraphLoader(data, ptr, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers,
                                  persistent_workers=args.num_workers > 0)
    eval_loader = EvalSubgraphLoader(data, ptr, batch_size=args.batch_size)


    t = time.perf_counter()
    print('Calculating buffer size...', end=' ', flush=True)
    # 为了高效传输，实际分配的buffer比需要的大
    buffer_size = max([n_id.numel() for _, _, n_id, _, _ in eval_loader])
    print(f'Done! [{time.perf_counter() - t:.2f}s] -> {buffer_size}')
    
    GNN = get_GNN(args.modelname)
    #初始化模型
    model = GNN(
        num_nodes=data.num_nodes,
        in_channels=n_node_feats,
        out_channels=n_classes,
        pool_size=args.pool_size,
        buffer_size=buffer_size,
        num_layers=args.num_layers,
        rsl=args.rsl,
        **args.architecture
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.005)#args.lr)
    t = time.perf_counter()
    print('Fill history...', end=' ', flush=True)
    #初始化“history embedding”
    mini_test(model, eval_loader, cm)
    print(f'Done! [{time.perf_counter() - t:.2f}s]')
    total_time = 0
    best_val_acc = 0
    # best_test_acc = 0
    best_out = None
    for epoch in tqdm(range(1, args.epochs + 1), desc=f'running {n_running}', ncols=80):
        tic = time.time()
        mini_train(model, train_loader, optimizer,
                          args.max_steps, grad_norm, edge_dropout, cm)
        toc = time.time()
        total_time += toc - tic
        # adjust_learning_rate(optimizer, args.lr, epoch)
        out = mini_test(model, eval_loader, cm)
        #更新cm
        if args.rsl>0.:
            cm = get_cm(out, n_classes).to(device)
        val_acc = compute_micro_f1(out, data.y, data.val_mask)
        # test_acc = compute_micro_f1(out, data.y, data.test_mask)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # best_test_acc = test_acc
            best_out = out.detach().argmax(dim=-1).cpu()
    per_epoch_time = total_time / args.epochs

    return best_out, per_epoch_time




def main():
    if not os.path.exists('./partitions/'):
        os.makedirs('./partitions/')
    if not os.path.exists('./result/'):
        os.makedirs('./result/')
    global device, n_node_feats, n_classes, epsilon

    argparser = argparse.ArgumentParser("Test",
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--gpu", type=int, default=0, help="GPU device ID.")
    argparser.add_argument("--n-runs", type=int, help="running times", default=2)
    argparser.add_argument("--epochs", type=int, help="number of epochs", default=500)

    argparser.add_argument("--dataset", type=str, default='arxiv')
    argparser.add_argument("--modelname", type=str, default='GCN')
    argparser.add_argument("--rsl", type=float, default=0.0)
    argparser.add_argument("--num-layers", type=int, help="number of layers", default=3)############2
    argparser.add_argument("--log-every", type=int, help="log every LOG_EVERY epochs", default=40)

    argparser.add_argument("--version", type=str, default='')
    args = argparser.parse_args()
    set_args(args, args.modelname, args.dataset)
    #%%
    result_path = f'./result/{args.modelname}/'
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    version = f'{args.dataset}-l{args.num_layers}-{args.rsl}'
    if exp_exists(result_path,version):
        print('Experiment has exists')
        return None
    #%%
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # load data
    if args.modelname=='SAGE':
        data, n_node_feats, n_classes, ptr = load_data(args.dataset, args.num_parts,add_selfloop=True, norm=False)
    else:
        data, n_node_feats, n_classes, ptr = load_data(args.dataset, args.num_parts)
    print((data.label_p==data.y).sum()/data.x.shape[0])
    cm = data.cm.to(device) if hasattr(data,'cm')  else None
    train_mask, val_mask, test_mask = data.train_mask, data.val_mask, data.test_mask
    # run
    best_outs = []
    test_accs = []
    per_epoch_times = []
    for n_running in range(1, args.n_runs + 1):
        if data.train_mask.dim()>1:
            data.train_mask = train_mask[:,n_running-1]
        if data.val_mask.dim()>1:
            data.val_mask = val_mask[:,n_running-1]
        if data.test_mask.dim()>1:
            data.test_mask = test_mask[:,n_running-1]
        best_out, per_epoch_time = run(args, data, ptr, n_running, cm)
        best_outs.append(best_out)
        per_epoch_times.append(per_epoch_time)
        acc=int(best_out[data.test_mask].eq(data.y[data.test_mask]).sum()) / data.y[data.test_mask].size(0)
        print(acc)
        test_accs.append(acc)
    results = torch.vstack(best_outs)
    torch.save(results,f'{result_path}{version}.tensor')
    with open(f'./result/{args.modelname}.txt','a') as f:
        f.write(f"{version}: {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}  {np.mean(per_epoch_times):.4f}\n")

if __name__ == "__main__":
    print(' '.join(sys.argv))
    main()


