def load_loss_function(loss_class, *args, **kwargs):
    loss_function = loss_class(*args, **kwargs)

    return loss_function
