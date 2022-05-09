#'coauthorcs', 'coauthorphysics', 'amazoncomputers', 'amazonphoto', 'wikics'
gcn2_parameters = {
        'note':'Loading parameters for GCN2 on dataset ',
        'arxiv':{
            'architecture':{
                'num_layers':3,
                'hidden_channels': 256,
                'dropout': 0.3,
                'batch_norm': True,
                'shared_weights': True,
                'alpha': 0.1,
                'theta': 0.5},
            'rsl':0.05,
            'lr': 0.01,
            'reg_weight_decay': 0.01,
            'nonreg_weight_decay': 0.},
        'reddit':{
            'architecture':{
                'num_layers':3,
                'hidden_channels': 128,
                'dropout': 0.3,
                'batch_norm': False,
                'shared_weights': True,
                'alpha': 0.1,
                'theta': 0.5},
            'rsl':0.05,
            'lr': 0.01,
            'reg_weight_decay': 0.01,
            'nonreg_weight_decay': 0.},
        'coauthorcs':{
            'architecture':{
                'num_layers':2,
                'hidden_channels': 128,
                'dropout': 0.3,
                'batch_norm': False,
                'shared_weights': True,
                'alpha': 0.1,
                'theta': 0.5},
            'rsl':0.05,
            'lr': 0.01,
            'reg_weight_decay': 0.01,
            'nonreg_weight_decay': 0.},
        'coauthorphysics':{
            'architecture':{
                'num_layers':2,
                'hidden_channels': 128,
                'dropout': 0.3,
                'batch_norm': False,
                'shared_weights': True,
                'alpha': 0.1,
                'theta': 0.5},
            'rsl':0.05,
            'lr': 0.01,
            'reg_weight_decay': 0.01,
            'nonreg_weight_decay': 0.},
        'amazoncomputers':{
            'architecture':{
                'num_layers':2,
                'hidden_channels': 128,
                'dropout': 0.3,
                'batch_norm': False,
                'shared_weights': True,
                'alpha': 0.1,
                'theta': 0.5},
            'rsl':0.05,
            'lr': 0.01,
            'reg_weight_decay': 0.01,
            'nonreg_weight_decay': 0.},
        
        'wikics':{
            'architecture':{
                'num_layers':2,
                'hidden_channels': 128,
                'dropout': 0.3,
                'batch_norm': False,
                'shared_weights': True,
                'alpha': 0.1,
                'theta': 0.5},


            'rsl':0.05,
            'lr': 0.01,
            'reg_weight_decay': 0.01,
            'nonreg_weight_decay': 0.}
        }





