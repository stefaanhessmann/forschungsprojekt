from torch.optim.lr_scheduler import _LRScheduler

class AbcExponentialLR(_LRScheduler):

    def __init__(self, optimizer, b, c, last_epoch=-1):
        #self.a = a
        self.b = b
        self.c = c
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * self.b ** (0.6*self.last_epoch/self.c)
                for base_lr in self.base_lrs]
