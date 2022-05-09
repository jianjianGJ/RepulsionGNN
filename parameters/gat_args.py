#'coauthorcs', 'coauthorphysics', 'amazoncomputers', 'amazonphoto', 'wikics'
gat_parameters = {
        'note':'Loading parameters for GAT on dataset ',
        'arxiv':{
            'architecture':{
                'num_layers':3,
                'hidden_channels': 64,
                'dropout': 0.5,
                'hidden_heads': 3},
            'rsl':0.1,
            'lr': 0.01,
            'reg_weight_decay': 0.0005,
            'nonreg_weight_decay': 0},
        'reddit':{
            'architecture':{
                'num_layers':3,
                'hidden_channels': 64,
                'dropout': 0.5,
                'hidden_heads': 3},
            'rsl':0.1,
            'lr': 0.01,
            'reg_weight_decay': 0.0005,
            'nonreg_weight_decay': 0},
        'coauthorcs':{
            'architecture':{
                'num_layers':2,
                'hidden_channels': 64,
                'dropout': 0.5,
                'hidden_heads': 3},
            'rsl':0.5,
            'lr': 0.01,
            'reg_weight_decay': 0.0005,
            'nonreg_weight_decay': 0},
        'coauthorphysics':{
            'architecture':{
                'num_layers':2,
                'hidden_channels': 64,
                'dropout': 0.5,
                'hidden_heads': 3},
            'rsl':0.2,
            'lr': 0.01,
            'reg_weight_decay': 0.0005,
            'nonreg_weight_decay': 0},
        'amazoncomputers':{
            'architecture':{
                'num_layers':2,
                'hidden_channels': 64,
                'dropout': 0.5,
                'hidden_heads': 3},
            'rsl':0.4,
            'lr': 0.01,
            'reg_weight_decay': 0.0005,
            'nonreg_weight_decay': 0},
        'wikics':{
            'architecture':{
                'num_layers':2,
                'hidden_channels': 64,
                'dropout': 0.5,
                'hidden_heads': 3},
            'rsl':0.4,
            'lr': 0.01,
            'reg_weight_decay': 0.0005,
            'nonreg_weight_decay': 0}
        }

























