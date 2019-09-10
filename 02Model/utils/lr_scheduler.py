from utils import registry
from torch.optim import lr_scheduler


def _no_change(_):
    return 1


@registry.register('LRScheduler', 'Constant')
class ConstantLR(lr_scheduler.LambdaLR):
    '''
    The learning rate remains unchanged during training.
    '''
    def __init__(self, optimizer):
        super().__init__(optimizer, _no_change)


registry.register('LRScheduler', 'Exponential')(lr_scheduler.ExponentialLR)
registry.register('LRScheduler', 'IterExponential')(lr_scheduler.ExponentialLR)
registry.register('LRScheduler', 'StepLR')(lr_scheduler.StepLR)
registry.register('LRScheduler', 'ReduceOnPlateau')(lr_scheduler.ReduceLROnPlateau)
registry.register('LRScheduler', 'MultiStep')(lr_scheduler.MultiStepLR)
