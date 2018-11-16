from Optim.abstract_optim_wrapper import Abstract_Optim_Wrapper


class Default_Optim_Wrapper(Abstract_Optim_Wrapper):
    """A simple wrapper class for learning rate scheduling"""
    def __init__(self, optimizer):
        super().__init__(optimizer)

    def update_lr_per_step(self):
        pass

    def update_lr_per_epoch(self):
        pass