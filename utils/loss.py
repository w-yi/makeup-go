from utils import registry
import torch.nn


registry.register('Loss', 'CrossEntropy')(torch.nn.CrossEntropyLoss)
