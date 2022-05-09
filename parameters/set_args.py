from .gcn_args import gcn_parameters
from .gcn2_args import gcn2_parameters
from .gat_args import gat_parameters
from .sage_args import sage_parameters
def set_args(args, silence=False):
    model_name, data_name = args.modelname, args.dataset
    if model_name=='GCN':
        parameters = gcn_parameters
    elif model_name=='GCN2':
        parameters = gcn2_parameters
    elif model_name=='GAT':
        parameters = gat_parameters
    elif model_name=='SAGE':
        parameters = sage_parameters
    params = parameters[data_name]
    if not silence:
        print("***********************************************")
        print(parameters['note']+data_name+'.')
        print("***********************************************")
    
    args.rsl = params['rsl']
    args.architecture = params['architecture']
    args.lr = params['lr']
    args.reg_weight_decay = params['reg_weight_decay']
    args.nonreg_weight_decay = params['nonreg_weight_decay']
    
    #prepare--------------------------------------------------------
    if args.prepare:
        args.n_runs = 1
        args.rsl = 0.
