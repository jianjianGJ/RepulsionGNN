#'coauthorcs', 'coauthorphysics', 'amazoncomputers', 'amazonphoto', 'wikics'
sage_parameters = {
        'note':'Loading parameters for SAGE on dataset ',
        'arxiv':{
            'architecture':{
                'num_layers':3,
                'hidden_channels': 256,
                'dropout': 0.5,
                'batch_norm': True},
            'rsl':0.1,
            'lr': 0.01,
            'reg_weight_decay': 0.0005,
            'nonreg_weight_decay': 0},
        'reddit':{
            'architecture':{
                'num_layers':3,
                'hidden_channels': 128,
                'dropout': 0.5,
                'batch_norm': False},
            'rsl':0.5,
            'lr': 0.01,
            'reg_weight_decay': 0.0005,
            'nonreg_weight_decay': 0},
        'coauthorcs':{
            'architecture':{
                'num_layers':2,
                'hidden_channels': 128,
                'dropout': 0.5,
                'batch_norm': False},
            'rsl':0.5,
            'lr': 0.01,
            'reg_weight_decay': 0.0005,
            'nonreg_weight_decay': 0},
        'coauthorphysics':{
            'architecture':{
                'num_layers':2,
                'hidden_channels': 128,
                'dropout': 0.5,
                'batch_norm': False},
            'rsl':0.5,
            'lr': 0.01,
            'reg_weight_decay': 0.0005,
            'nonreg_weight_decay': 0},
        'amazoncomputers':{
            'architecture':{
                'num_layers':2,
                'hidden_channels': 128,
                'dropout': 0.5,
                'batch_norm': False},
            'rsl':0.4,
            'lr': 0.01,
            'reg_weight_decay': 0.0005,
            'nonreg_weight_decay': 0},
        'wikics':{
            'architecture':{
                'num_layers':2,
                'hidden_channels': 128,
                'dropout': 0.5,
                'batch_norm': False},
            'rsl':0.5,
            'lr': 0.01,
            'reg_weight_decay': 0.0005,
            'nonreg_weight_decay': 0}
    }






























