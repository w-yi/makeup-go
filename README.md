playground/: temporary testing scripts

module/: algorithm

param/: default hyperparameter sets

utils/: small functions

dataset/: data

running scripts:

python train.py -lp XX<XX.yml in param/> -op XX.XX=XX<optional>

e.g. python train.py -lp naive -op network.kwargs.nonlinear=ReLU,optimizer.name=Adam
