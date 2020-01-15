import torch
import math

class WarmStartCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    r"""Set the learning rate of each parameter group using a cosine annealing
    add warm start to CosineAnnealingLR
    """
    def __init__(self, optimizer,total_epochs , warm_epochs, eta_min=0, last_epoch=-1):
        self.T_max = total_epochs
        self.T_warm = warm_epochs
        self.eta_min = eta_min

        super(WarmStartCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.T_warm:
            return [base_lr * ((self.last_epoch + 1) / self.T_warm) for base_lr in self.base_lrs]
        else:
            return [self.eta_min + (base_lr - self.eta_min) *
                    (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
                    for base_lr in self.base_lrs]
