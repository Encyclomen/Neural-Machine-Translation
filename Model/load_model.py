import torch


def load_model(model_class, *args, cuda=False, if_load_state_dict=False, saved_state_dict=None, if_init=True):
    model = model_class(*args)
    if cuda:
        model.cuda()
    if if_load_state_dict:
        state_dict = torch.load(saved_state_dict)
        model.load_state_dict(state_dict)
    elif if_init:  # initialize the model's parameters
        model.param_init()

    return model
