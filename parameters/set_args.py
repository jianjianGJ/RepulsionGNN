from .gcn_args import gcn_parameters, rgcn_parameters
from .gcn2_args import gcn2_parameters, rgcn2_parameters
from .gat_args import gat_parameters, rgat_parameters
from .sage_args import sage_parameters, rsage_parameters
def set_args(args, model_name, data_name):
    if model_name=='GCN':
        if args.rsl>0.:
            parameters = rgcn_parameters
        else:
            parameters = gcn_parameters
    elif model_name=='GCN2':
        if args.rsl>0.:
            parameters = rgcn2_parameters
        else:
            parameters = gcn2_parameters
    elif model_name=='GAT':
        if args.rsl>0.:
            parameters = rgat_parameters
        else:
            parameters = gat_parameters
    elif model_name=='SAGE':
        if args.rsl>0.:
            parameters = rsage_parameters
        else:
            parameters = sage_parameters
    params = parameters[data_name]
    print("***********************************************")
    print(parameters['note']+data_name+'.')
    print("***********************************************")
    #args.num_layers = params['num_layers']
    args.architecture = params['architecture']
    args.num_parts = params['num_parts']
    args.batch_size = params['batch_size']
    args.max_steps = params['max_steps']
    args.pool_size = params['pool_size']
    args.num_workers = params['num_workers']
    args.lr = params['lr']
    args.reg_weight_decay = params['reg_weight_decay']
    args.nonreg_weight_decay = params['nonreg_weight_decay']
    args.grad_norm = params['grad_norm']
    args.edge_drop = params['edge_drop']


if __name__ == '__main__':
    class Args():
        pass
    args = Args()
    model_name = 'RGCN2'
    data_name = 'arxiv'
    set_args(args, model_name, data_name)
    print(args.architecture)
