playground/: temporary testing scripts
module/: algorithm
param/: default hyperparameter sets
utils/: small functions
dataset/: data

running scripts:

python train.py -lp XX<XX.yml in param/> -op <optional, e.g.: network.kwargs.nonlinear=ReLU,optimizer.name=Adam>
