from utils import registry
import torch.optim

registry.register('Optimizer', 'SGD')(torch.optim.SGD)
registry.register('Optimizer', 'Adam')(torch.optim.Adam)
registry.register('Optimizer', 'RMSprop')(torch.optim.RMSprop)
