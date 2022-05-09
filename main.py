import os
import sys
import math
import time
import argparse
import torch
import numpy as np
from tqdm import tqdm
from parameters import set_args
from data_utils import load_data
from utils import compute_micro_f1, get_GNN, custom_loss_function, get_version, get_auxiliary, seed
#%% 日志格式设置，全局变量初始化
version = None
epsilon = 1 - math.log(2)
device = None
n_node_feats, n_classes = 0, 0

#%% 
def train(model, adj_t, x, y, label_p, train_mask, optimizer, cm=None):
    model.train()
    optimizer.zero_grad()
    out = model(x, adj_t, label_p, cm)
    loss = custom_loss_function(out[train_mask], y[train_mask])
    loss.backward()
    optimizer.step()
    return loss.detach()

@torch.no_grad()
def inference(model, adj_t, x, label_p, cm):
    model.eval()
    out = model(x, adj_t, label_p, cm)
    return out

#%%
def run(args, adj_t, x, y, label_p, cm, train_mask, val_mask, n_running):


    seed(n_running)
    GNN = get_GNN(args.modelname)
    model = GNN(
        in_channels=n_node_feats,
        out_channels=n_classes,
        rsl = args.rsl,
        **args.architecture
    ).to(device)
    optimizer = torch.optim.Adam([
        dict(params=model.reg_modules.parameters(), weight_decay=args.reg_weight_decay),
        dict(params=model.nonreg_modules.parameters(), weight_decay=args.nonreg_weight_decay)
    ], lr=args.lr)
    
    total_time = 0
    best_val_acc = 0
    best_out = None
    for epoch in tqdm(range(1, args.epochs + 1), desc=f'running {n_running}', ncols=80):
        tic = time.time()
        train(model, adj_t, x, y, label_p, train_mask, optimizer, cm)
        toc = time.time()
        total_time += toc - tic
        out = inference(model, adj_t, x, label_p, cm)
        val_acc = compute_micro_f1(out, y, val_mask)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_out = out.detach().cpu()
    per_epoch_time = total_time / args.epochs
    return best_out, per_epoch_time




def main():
    
    global device, n_node_feats, n_classes, epsilon

    argparser = argparse.ArgumentParser("Test",
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--gpu", type=int, default=0, help="GPU device ID.")
    argparser.add_argument("--n-runs", type=int, help="running times", default=1)
    argparser.add_argument("--epochs", type=int, help="number of epochs", default=500)
    
    
    argparser.add_argument("--prepare", action='store_true', default=False)
    argparser.add_argument("--basemodel", type=str, default='GCN')
    
    argparser.add_argument("--dataset", type=str, default='arxiv')
    argparser.add_argument("--modelname", type=str, default='GCN')
    args = argparser.parse_args()
    if args.prepare:
        args.modelname = args.basemodel
    set_args(args)
    args.version = get_version(args)
    
    #%%
    if not os.path.exists(f'./datainfo-{args.basemodel}//'):
        os.makedirs(f'./datainfo-{args.basemodel}/')
    if not os.path.exists('./result/'):
        os.makedirs('./result/')
    #%%
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # load data
    if args.modelname=='SAGE':
        data, n_node_feats, n_classes = load_data(args.dataset,add_selfloop=True, norm=False, basemodel=args.basemodel)
    else:
        data, n_node_feats, n_classes = load_data(args.dataset, basemodel=args.basemodel)
    adj_t, x, y, label_p, cm, train_masks, val_masks, test_masks = \
        data.adj_t, data.x, data.y, data.label_p, data.cm, data.train_mask, data.val_mask, data.test_mask
    adj_t, x, y, train_masks, val_masks, test_masks = adj_t.to(device), x.to(device), y.to(device),  \
        train_masks.to(device), val_masks.to(device), test_masks.to(device)
    if args.rsl>0:
        label_p, cm = (label_p[0].to(device),label_p[1].to(device)), cm.to(device)
    best_outs = []
    test_accs = []
    per_epoch_times = []
    for n_running in range(1, args.n_runs + 1):
        if train_masks.dim()>1:
            train_mask = train_masks[:,(n_running-1)%train_masks.shape[1]]
            val_mask = val_masks[:,(n_running-1)%val_masks.shape[1]]
        if train_masks.dim()==1:
            train_mask = train_masks
            val_mask = val_masks
        if test_masks.dim()==1:
            test_mask = test_masks
        else:
            test_mask = test_masks[:,(n_running-1)%test_masks.shape[1]]
        best_out_logits, per_epoch_time = run(args, adj_t, x, y, label_p, cm, train_mask, val_mask, n_running)
        best_out = best_out_logits.argmax(dim=-1)
        if args.prepare:
            info_path = f'./datainfo-{args.basemodel}/{args.dataset}'
            if not os.path.exists(info_path):
                os.mkdir(info_path)
            label_p, cm = get_auxiliary(best_out_logits, n_classes)
            torch.save(label_p,f'{info_path}/label.tensor')
            torch.save(cm,f'{info_path}/cm.tensor')
            args.version += ' prepare '
        best_outs.append(best_out)
        per_epoch_times.append(per_epoch_time)
        acc=int(best_out[test_mask.cpu()].eq(y.cpu()[test_mask.cpu()]).sum()) / y.cpu()[test_mask.cpu()].size(0)
        print(acc)
        test_accs.append(acc)
    with open(f'./result/{args.modelname}.txt','a') as f:
        f.write(f"{args.version}: mean={np.mean(test_accs)*100:.2f} std={np.std(test_accs)*100:.2f}  t={np.mean(per_epoch_times):.4f} \n")

if __name__ == "__main__":
    print(' '.join(sys.argv))
    main()


