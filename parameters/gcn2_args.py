#'coauthorcs', 'coauthorphysics', 'amazoncomputers', 'amazonphoto', 'wikics'
gcn2_parameters = {
        'note':'Loading parameters for GCN2 on dataset ',
        'arxiv':{
            'architecture':{
                'hidden_channels': 256,
                'dropout': 0.3,
                'drop_input': False,
                'batch_norm': True,
                'residual': False,
                'shared_weights': True,
                'alpha': 0.1,
                'theta': 0.5},
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
                'dropout': 0.3,
                'drop_input': True,
                'batch_norm': True,
                'residual': False,
                'shared_weights': True,
                'alpha': 0.1,
                'theta': 0.5},
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
                'dropout': 0.3,
                'drop_input': False,
                'batch_norm': False,
                'residual': False,
                'shared_weights': True,
                'alpha': 0.1,
                'theta': 0.5},
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
                'dropout': 0.3,
                'drop_input': False,
                'batch_norm': False,
                'residual': False,
                'shared_weights': True,
                'alpha': 0.1,
                'theta': 0.5},
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
                'dropout': 0.3,
                'drop_input': False,
                'batch_norm': False,
                'residual': False,
                'shared_weights': True,
                'alpha': 0.1,
                'theta': 0.5},
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
                'dropout': 0.3,
                'drop_input': False,
                'batch_norm': False,
                'residual': False,
                'shared_weights': True,
                'alpha': 0.1,
                'theta': 0.5},
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




rgcn2_parameters = {
        'note':'Loading parameters for RGCN2 on dataset ',
        'arxiv':{
            'architecture':{
                'hidden_channels': 256,
                'dropout': 0.3,
                'drop_input': False,
                'batch_norm': True,
                'residual': False,
                'shared_weights': True,
                'alpha': 0.1,
                'theta': 0.5},
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
                'dropout': 0.3,
                'drop_input': True,
                'batch_norm': True,
                'residual': False,
                'shared_weights': True,
                'alpha': 0.1,
                'theta': 0.5},
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
                'dropout': 0.3,
                'drop_input': False,
                'batch_norm': False,
                'residual': False,
                'shared_weights': True,
                'alpha': 0.1,
                'theta': 0.5},
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
                'dropout': 0.3,
                'drop_input': False,
                'batch_norm': False,
                'residual': False,
                'shared_weights': True,
                'alpha': 0.1,
                'theta': 0.5},
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
                'dropout': 0.3,
                'drop_input': False,
                'batch_norm': False,
                'residual': False,
                'shared_weights': True,
                'alpha': 0.1,
                'theta': 0.5},
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
                'dropout': 0.3,
                'drop_input': False,
                'batch_norm': False,
                'residual': False,
                'shared_weights': True,
                'alpha': 0.1,
                'theta': 0.5},
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





























