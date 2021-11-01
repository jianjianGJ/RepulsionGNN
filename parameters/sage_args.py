#'coauthorcs', 'coauthorphysics', 'amazoncomputers', 'amazonphoto', 'wikics'
sage_parameters = {
        'note':'Loading parameters for SAGE on dataset ',
        'arxiv':{
            'architecture':{
                'hidden_channels': 256,
                'dropout': 0.5,
                'drop_input': False,
                'batch_norm': True,
                'residual': False,},
            'num_parts': 80,
            'batch_size': 40,
            'max_steps': 2,
            'pool_size': 2,
            'num_workers': 0,
            'lr': 0.005,
            'reg_weight_decay': 0,
            'nonreg_weight_decay': 0,
            'grad_norm': None,
            'edge_drop':0.3},
        'reddit':{
            'architecture':{
                'hidden_channels': 128,
                'dropout': 0.5,
                'drop_input': False,
                'batch_norm': False,
                'residual': False,
                'linear': False},
            'num_parts': 200,
            'batch_size': 100,
            'max_steps': 2,
            'pool_size': 2,
            'num_workers': 0,
            'lr': 0.005,
            'reg_weight_decay': 0,
            'nonreg_weight_decay': 0,
            'grad_norm': None,
            'edge_drop':0.3},
        'coauthorcs':{
            'architecture':{
                'hidden_channels': 128,
                'dropout': 0.5,
                'drop_input': False,
                'batch_norm': False,
                'residual': False},
            'num_parts': 1,
            'batch_size': 1,
            'max_steps': 1,
            'pool_size': 1,
            'num_workers': 0,
            'lr': 0.005,
            'reg_weight_decay': 0,
            'nonreg_weight_decay': 0,
            'grad_norm': None,
            'edge_drop':0.3},
        'coauthorphysics':{
            'architecture':{
                'hidden_channels': 128,
                'dropout': 0.5,
                'drop_input': False,
                'batch_norm': False,
                'residual': False},
            'num_parts': 1,
            'batch_size': 1,
            'max_steps': 1,
            'pool_size': 1,
            'num_workers': 0,
            'lr': 0.005,
            'reg_weight_decay': 0,
            'nonreg_weight_decay': 0,
            'grad_norm': None,
            'edge_drop':0.3},
        'amazoncomputers':{
            'architecture':{
                'hidden_channels': 128,
                'dropout': 0.5,
                'drop_input': False,
                'batch_norm': False,
                'residual': False},
            'num_parts': 1,
            'batch_size': 1,
            'max_steps': 1,
            'pool_size': 1,
            'num_workers': 0,
            'lr': 0.005,
            'reg_weight_decay': 0,
            'nonreg_weight_decay': 0,
            'grad_norm': None,
            'edge_drop':0.3},
        'wikics':{
            'architecture':{
                'hidden_channels': 128,
                'dropout': 0.5,
                'drop_input': False,
                'batch_norm': False,
                'residual': False},
            'num_parts': 1,
            'batch_size': 1,
            'max_steps': 1,
            'pool_size': 1,
            'num_workers': 0,
            'lr': 0.005,
            'reg_weight_decay': 0,
            'nonreg_weight_decay': 0,
            'grad_norm': None,
            'edge_drop':0.3}
    }

rsage_parameters = {
        'note':'Loading parameters for RSAGE on dataset ',
        'arxiv':{
            'architecture':{
                'hidden_channels': 256,
                'dropout': 0.5,
                'drop_input': False,
                'batch_norm': True,
                'residual': False,},
            'num_parts': 80,
            'batch_size': 40,
            'max_steps': 2,
            'pool_size': 2,
            'num_workers': 0,
            'lr': 0.005,
            'reg_weight_decay': 0,
            'nonreg_weight_decay': 0,
            'grad_norm': None,
            'edge_drop':0.3},
        'reddit':{
            'architecture':{
                'hidden_channels': 128,
                'dropout': 0.5,
                'drop_input': False,
                'batch_norm': False,
                'residual': False,
                'linear': False},
            'num_parts': 200,
            'batch_size': 100,
            'max_steps': 2,
            'pool_size': 2,
            'num_workers': 0,
            'lr': 0.005,
            'reg_weight_decay': 0,
            'nonreg_weight_decay': 0,
            'grad_norm': None,
            'edge_drop':0.3},
        'coauthorcs':{
            'architecture':{
                'hidden_channels': 128,
                'dropout': 0.5,
                'drop_input': False,
                'batch_norm': False,
                'residual': False},
            'num_parts': 1,
            'batch_size': 1,
            'max_steps': 1,
            'pool_size': 1,
            'num_workers': 0,
            'lr': 0.005,
            'reg_weight_decay': 0,
            'nonreg_weight_decay': 0,
            'grad_norm': None,
            'edge_drop':0.3},
        'coauthorphysics':{
            'architecture':{
                'hidden_channels': 128,
                'dropout': 0.5,
                'drop_input': False,
                'batch_norm': False,
                'residual': False},
            'num_parts': 1,
            'batch_size': 1,
            'max_steps': 1,
            'pool_size': 1,
            'num_workers': 0,
            'lr': 0.005,
            'reg_weight_decay': 0,
            'nonreg_weight_decay': 0,
            'grad_norm': None,
            'edge_drop':0.3},
        'amazoncomputers':{
            'architecture':{
                'hidden_channels': 128,
                'dropout': 0.5,
                'drop_input': False,
                'batch_norm': False,
                'residual': False},
            'num_parts': 1,
            'batch_size': 1,
            'max_steps': 1,
            'pool_size': 1,
            'num_workers': 0,
            'lr': 0.005,
            'reg_weight_decay': 0,
            'nonreg_weight_decay': 0,
            'grad_norm': None,
            'edge_drop':0.3},

        'wikics':{
            'architecture':{
                'hidden_channels': 128,
                'dropout': 0.5,
                'drop_input': False,
                'batch_norm': False,
                'residual': False},
            'num_parts': 1,
            'batch_size': 1,
            'max_steps': 1,
            'pool_size': 1,
            'num_workers': 0,
            'lr': 0.005,
            'reg_weight_decay': 0,
            'nonreg_weight_decay': 0,
            'grad_norm': None,
            'edge_drop':0.3}
    }
































