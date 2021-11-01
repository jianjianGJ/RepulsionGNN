import math
import time
import os
from tqdm import tqdm
import torch
import torch.nn.functional as F
from models import GCN
from parameters import set_args
import argparse
from torch_geometric_autoscale import (SubgraphLoader, EvalSubgraphLoader,
                                       compute_micro_f1, dropout)
from data_utils import load_data
from torch_scatter import scatter_mean
#python ./prepare.py --dataset='arxiv' --epochs=2 --num-layers=3
#%% 日志格式设置，全局变量初始化
epsilon = 1 - math.log(2)
device = None
n_node_feats, n_classes = 0, 0
#%% 小技巧 函数
def custom_loss_function(x, labels):
    y = F.cross_entropy(x, labels, reduction="none")
    y = torch.log(epsilon + y) - math.log(epsilon)
    return torch.mean(y)
 


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
        label_p = None#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
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
#test时执行的新操作：将前馈后获得的嵌入存入CPU
# @torch.no_grad()
# def full_test(model, data,cm=None):
#     model.eval()
#     return model(data.x.to(model.device), data.label_p.to(model.device), 
#                  data.adj_t.to(model.device),cm=cm.to(model.device)).cpu()
@torch.no_grad()
def mini_test(model, loader, cm):
    model.eval()
    return model(loader=loader, cm=cm)

#%%
def run(args, data, ptr, cm):
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
    
    #初始化模型
    model = GCN(
        num_nodes=data.num_nodes,
        in_channels=n_node_feats,
        out_channels=n_classes,
        pool_size=args.pool_size,
        buffer_size=buffer_size,
        num_layers=args.num_layers,
        rsl=args.rsl,
        **args.architecture
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    t = time.perf_counter()
    print('Fill history...', end=' ', flush=True)
    #初始化“history embedding”
    mini_test(model, eval_loader, cm)
    print(f'Done! [{time.perf_counter() - t:.2f}s]')

    total_time = 0
    best_val_acc = test_acc = 0
    best_out = None#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    for epoch in tqdm(range(1, args.epochs + 1),  ncols=80):
        tic = time.time()
        mini_train(model, train_loader, optimizer,
                   args.max_steps, grad_norm, edge_dropout, cm)
        toc = time.time()
        total_time += toc - tic
        out = mini_test(model, eval_loader, cm)
        val_acc = compute_micro_f1(out, data.y, data.val_mask)
        tmp_test_acc = compute_micro_f1(out, data.y, data.test_mask)
    
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
            best_out = out.detach().cpu()#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    print('Saving best_out.tensor!')
    torch.save(best_out,'./partitions/'+args.dataset+f'-{args.num_parts}/best_out.tensor')#<<<<<<
    print('The prepare works are done!!')
    print(f'final test acc:{test_acc:.4f}')





#%%训练基础GCN模型，获取best_out
torch.manual_seed(0)


if __name__ == "__main__":
    if not os.path.exists('./partitions/'):
        os.makedirs('./partitions/')
    if not os.path.exists('./result/'):
        os.makedirs('./result/')
    argparser = argparse.ArgumentParser("Test",
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--epochs", type=int, help="number of epochs", default=500)
    argparser.add_argument("--dataset", type=str, default='arxiv')
    argparser.add_argument("--num-layers", type=int, help="number of layers", default=3)############2
    args = argparser.parse_args()
    args.gpu = 0
    args.rsl = 0.
    args.modelname = 'GCN'
    set_args(args, args.modelname, args.dataset)
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    
    # load data
    data, n_node_feats, n_classes, ptr = load_data(args.dataset, args.num_parts)
    if data.train_mask.dim()>1:
        data.train_mask = data.train_mask[:,0]
    if data.val_mask.dim()>1:
        data.val_mask = data.val_mask[:,0]
    if data.test_mask.dim()>1:
        data.test_mask = data.test_mask[:,0]
    # run
    #%% +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    partition_path = f'./partitions/{args.dataset}-{args.num_parts}/best_out.tensor'
    if not os.path.exists(partition_path):
        run(args, data, ptr, cm=None)
    #%% 获取保存预测标签
    label_logits = torch.load(f'./partitions/{args.dataset}-{args.num_parts}/best_out.tensor')
    p_softmax = torch.softmax(label_logits, dim=-1)
    prob, label = p_softmax.max(1)
    
    print((label==data.y).sum()/label.shape[0])
    label[data.train_mask] = data.y[data.train_mask]
    print((label==data.y).sum()/label.shape[0])
    print('Saving label.tensor!')
    torch.save(label,f'./partitions/{args.dataset}-{args.num_parts}/label.tensor')
    label = p_softmax.argmax(dim=-1)
    cm = scatter_mean(p_softmax,label,dim=0,dim_size=n_classes)
    cm = cm-torch.diag(cm.diag())
    cm = cm.unsqueeze(1)
    torch.save(cm,f'./partitions/{args.dataset}-{args.num_parts}/cm.tensor')
    print(cm.max())



























