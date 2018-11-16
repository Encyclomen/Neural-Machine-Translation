from Optim.abstract_optim_wrapper import Abstract_Optim_Wrapper


class RNNSearch_Optim_Wrapper(Abstract_Optim_Wrapper):
    def __init__(self, optimizer, dec_rate):
        super().__init__(optimizer)
        self.dec_rate = dec_rate

    def update_lr_per_step(self):
        pass

    def update_lr_per_epoch(self):
        self._decrease_learning_rate(self.dec_rate)
