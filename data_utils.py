import os
import torch
from torch_geometric_autoscale import metis, permute, get_data
from torch_geometric.nn.conv.gcn_conv import gcn_norm
#%%
#../../hy-tmp/data
def load_data(data_name, num_parts, root='/home/gj/SpyderWorkSpace/data', add_selfloop=True, norm=True):
    data, num_features, num_classes = get_data(root,data_name)
    partition_path = f'./partitions/{data_name}-'+str(num_parts)
    if os.path.exists(partition_path):
        perm = torch.load(partition_path + '/perm.tensor')
        ptr = torch.load(partition_path + '/ptr.tensor')
    else:
        os.makedirs(partition_path)
        perm, ptr = metis(data.adj_t, num_parts=num_parts, log=True)
        torch.save(perm, partition_path+'/perm.tensor')
        torch.save(ptr, partition_path+'/ptr.tensor')
    #---------------------------------------------------------------------
    data = permute(data, perm, log=True)
    if os.path.exists(f'./partitions/{data_name}-{num_parts}/cm.tensor'):
        cm = torch.load(f'./partitions/{data_name}-{num_parts}/cm.tensor')
        cm = cm.reshape(num_classes,1,num_classes)
        data.cm = cm
    else:
        data.cm = None
        
    if os.path.exists(f'./partitions/{data_name}-{num_parts}/label.tensor'):
        data.label_p = torch.load(f'./partitions/{data_name}-{num_parts}/label.tensor')
    
        print('label_p and cm are added!')
    else:
        data.label_p = torch.ones(data.x.shape[0])
    #---------------------------------------------------------------------

    
    #添加自环
    if add_selfloop:
        print('Adding self-loops...', flush=True)
        data.adj_t = data.adj_t.set_diag()
    #规范化邻接矩阵（GCN方式）
    if norm:
        print('Normalizing data...', flush=True)
        data.adj_t = gcn_norm(data.adj_t, add_self_loops=False)
    return data, num_features, num_classes, ptr
#'coauthorcs', 'coauthorphysics', 'amazoncomputers', 'amazonphoto', 'wikics'
if __name__ == '__main__':
    data_name = 'wikics'
    data, d, c, ptr = load_data(data_name, 1)


















