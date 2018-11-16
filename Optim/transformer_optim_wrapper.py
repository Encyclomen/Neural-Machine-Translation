import numpy as np

from Optim.abstract_optim_wrapper import Abstract_Optim_Wrapper


class Transformer_Optim_Wrapper(Abstract_Optim_Wrapper):
    def __init__(self, optimizer, d_model, n_warmup_step):
        super().__init__(optimizer)
        self.d_model = d_model
        self.n_warmup_step = n_warmup_step
        self.current_step = 0

    def update_lr_per_step(self):
        self.current_step += 1
        new_lr = np.power(self.d_model, -0.5) * np.min([
            np.power(self.current_step, -0.5),
            np.power(self.n_warmup_step, -1.5) * self.current_step
        ])

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

    def update_lr_per_epoch(self):
        pass
