#'coauthorcs', 'coauthorphysics', 'amazoncomputers', 'amazonphoto', 'wikics'
gat_parameters = {
        'note':'Loading parameters for GAT on dataset ',
        'arxiv':{
            'architecture':{
                'hidden_channels': 64,
                'dropout': 0.5,
                'hidden_heads': 3},
            'num_parts': 80,
            'batch_size': 15,
            'max_steps': 4,
            'pool_size': 2,
            'num_workers': 0,
            'lr': 0.01,
            'reg_weight_decay': 0,
            'nonreg_weight_decay': 0,
            'grad_norm': None,
            'edge_drop':0.3},
        'reddit':{
            'architecture':{
                'hidden_channels': 64,
                'dropout': 0.5,
                'hidden_heads': 3},
            'num_parts': 200,
            'batch_size': 40,
            'max_steps': 4,
            'pool_size': 2,
            'num_workers': 0,
            'lr': 0.01,
            'reg_weight_decay': 0,
            'nonreg_weight_decay': 0,
            'grad_norm': None,
            'edge_drop':0.3},
        'coauthorcs':{
            'architecture':{
                'hidden_channels': 64,
                'dropout': 0.5,
                'hidden_heads': 3},
            'num_parts': 1,
            'batch_size': 1,
            'max_steps': 1,
            'pool_size': 1,
            'num_workers': 0,
            'lr': 0.01,
            'reg_weight_decay': 0,
            'nonreg_weight_decay': 0,
            'grad_norm': None,
            'edge_drop':0.3},
        'coauthorphysics':{
            'architecture':{
                'hidden_channels': 64,
                'dropout': 0.5,
                'hidden_heads': 3},
            'num_parts': 1,
            'batch_size': 1,
            'max_steps': 1,
            'pool_size': 1,
            'num_workers': 0,
            'lr': 0.01,
            'reg_weight_decay': 0,
            'nonreg_weight_decay': 0,
            'grad_norm': None,
            'edge_drop':0.3},
        'amazoncomputers':{
            'architecture':{
                'hidden_channels': 64,
                'dropout': 0.5,
                'hidden_heads': 3},
            'num_parts': 1,
            'batch_size': 1,
            'max_steps': 1,
            'pool_size': 1,
            'num_workers': 0,
            'lr': 0.01,
            'reg_weight_decay': 0,
            'nonreg_weight_decay': 0,
            'grad_norm': None,
            'edge_drop':0.3},
        'wikics':{
            'architecture':{
                'hidden_channels': 64,
                'dropout': 0.5,
                'hidden_heads': 3},
            'num_parts': 1,
            'batch_size': 1,
            'max_steps': 1,
            'pool_size': 1,
            'num_workers': 0,
            'lr': 0.01,
            'reg_weight_decay': 0,
            'nonreg_weight_decay': 0,
            'grad_norm': None,
            'edge_drop':0.3}
        }

rgat_parameters = {
        'note':'Loading parameters for RGAT on dataset ',
        'arxiv':{
            'architecture':{
                'hidden_channels': 64,
                'dropout': 0.5,
                'hidden_heads': 3},
            'num_parts': 80,
            'batch_size': 15,
            'max_steps': 4,
            'pool_size': 2,
            'num_workers': 0,
            'lr': 0.01,
            'reg_weight_decay': 0,
            'nonreg_weight_decay': 0,
            'grad_norm': None,
            'edge_drop':0.3},
        'reddit':{
            'architecture':{
                'hidden_channels': 64,
                'dropout': 0.5,
                'hidden_heads': 3},
            'num_parts': 200,
            'batch_size': 40,
            'max_steps': 4,
            'pool_size': 2,
            'num_workers': 0,
            'lr': 0.01,
            'reg_weight_decay': 0,
            'nonreg_weight_decay': 0,
            'grad_norm': None,
            'edge_drop':0.3},
        'coauthorcs':{
            'architecture':{
                'hidden_channels': 64,
                'dropout': 0.5,
                'hidden_heads': 3},
            'num_parts': 1,
            'batch_size': 1,
            'max_steps': 1,
            'pool_size': 1,
            'num_workers': 0,
            'lr': 0.01,
            'reg_weight_decay': 0,
            'nonreg_weight_decay': 0,
            'grad_norm': None,
            'edge_drop':0.3},
        'coauthorphysics':{
            'architecture':{
                'hidden_channels': 64,
                'dropout': 0.5,
                'hidden_heads': 3},
            'num_parts': 1,
            'batch_size': 1,
            'max_steps': 1,
            'pool_size': 1,
            'num_workers': 0,
            'lr': 0.01,
            'reg_weight_decay': 0,
            'nonreg_weight_decay': 0,
            'grad_norm': None,
            'edge_drop':0.3},
        'amazoncomputers':{
            'architecture':{
                'hidden_channels': 64,
                'dropout': 0.5,
                'hidden_heads': 3},
            'num_parts': 1,
            'batch_size': 1,
            'max_steps': 1,
            'pool_size': 1,
            'num_workers': 0,
            'lr': 0.01,
            'reg_weight_decay': 0,
            'nonreg_weight_decay': 0,
            'grad_norm': None,
            'edge_drop':0.3},
        'wikics':{
            'architecture':{
                'hidden_channels': 64,
                'dropout': 0.5,
                'hidden_heads': 3},
            'num_parts': 1,
            'batch_size': 1,
            'max_steps': 1,
            'pool_size': 1,
            'num_workers': 0,
            'lr': 0.01,
            'reg_weight_decay': 0,
            'nonreg_weight_decay': 0,
            'grad_norm': None,
            'edge_drop':0.3}
        }
































