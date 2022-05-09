# RepulsionGNN

## Run
The root path in data_utils.py should be changed to your path.

1. run python -u main.py --dataset='wikics' --basemodel='GCN' --prepare --epochs=1000` to run the basic model. The basic prediction and Confusion Matrix can be obtained.

2. run `python -u main.py --dataset='wikics' --modelname='SAGE' --n-runs=30 --epochs=1000` to train the Repulsion-SAGE. The prediction results will be saved in `./result`. Relevant hyperparameters can be set in `./parameters/SAGE.py`.
## Requirement
* python 3.8
* pytorch 1.8
* ogb
* pytorch-geometric
