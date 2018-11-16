import abc
from abc import abstractmethod


class Abstract_Optim_Wrapper(object, metaclass=abc.ABCMeta):
    """An optimizer wrapper class for learning rate scheduling"""
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def step(self):
        """Step by the inner optimizer"""
        self.optimizer.step()

    def zero_grad(self):
        """Zero out the gradients by the inner optimizer"""
        self.optimizer.zero_grad()

    def _decrease_learning_rate(self, dec_rate):
        #cur_lr_list = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * (1 - dec_rate)
            #cur_lr_list.append(param_group['lr'])
        # cur_lr = ' '.join(map(lambda v: str(v), cur_lr_list))
        # print('Current learning rate:', cur_lr)

    def _increase_learning_rate(self, inc_rate):
        #cur_lr_list = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * (1 + inc_rate)
            #cur_lr_list.append(param_group['lr'])
        # cur_lr = ' '.join(map(lambda v: str(v), cur_lr_list))
        # print('Current learning rate:', cur_lr)

    @abstractmethod
    def update_lr_per_step(self):
        """Learning rate scheduling per step """
        pass

    @abstractmethod
    def update_lr_per_epoch(self):
        """Learning rate scheduling per epoch """
        pass