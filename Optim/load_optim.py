from torch import optim


def load_optimizer(model, optim_class, **kwargs):
    param_list = model.parameters()
    optimizer = optim_class(param_list, **kwargs)

    return optimizer


def load_optim_wrapper(optimizer, optim_wrapper_class, **kwargs):
    optim_wrapper = optim_wrapper_class(optimizer, **kwargs)

    return optim_wrapper
