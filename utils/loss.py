from utils import registry
import torch.nn


registry.register('Loss', 'CrossEntropy')(torch.nn.CrossEntropyLoss)
registry.register('Loss', 'MSE')(torch.nn.MSELoss)
