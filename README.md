# RepulsionGNN

## Run
The root path in data_utils.py should be changed to your path.

1. run `python ./prepare.py --dataset='arxiv' --epochs=2 --num-layers=3` to run the basic model. The basic prediction and Confusion Matrix can be obtained.

2. run `python -u main.py --dataset=arxiv  --modelname=GCN --num-layers=3 --epochs=500 --n-runs=1 --rsl=0.2` to train the Repulsion-GCN. The prediction results will be saved in ./result.
## Requirement
* python 3.8
* pytorch 1.8
* ogb
* pytorch-geometric
* pyg-autoscale
