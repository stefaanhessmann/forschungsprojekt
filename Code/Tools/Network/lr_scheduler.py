from torch.optim.lr_scheduler import _LRScheduler

class AbcExponentialLR(_LRScheduler):

    def __init__(self, optimizer, b, c, last_epoch=-1):
        self.b = b
        self.c = c
        super(AbcExponentialLR, self).__init__(optimizer, last_epoch)


    def get_lr(self):
        return [base_lr * self.b ** (self.last_epoch/self.c)
                for base_lr in self.base_lrs]
