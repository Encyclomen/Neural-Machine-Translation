import os
import time
import torch


def save_checkpoint_model(model, checkpoint_dir, epoch, info='checkpoint_model'):
    # date = time.strftime('%m-%d_%H-%M', time.localtime(time.time()))
    # checkpoint = 'model_%s__info_%s__epoch_%d.pt' % (info, epoch)
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, info).replace('\\', '/'))


def save_min_loss_model(model, checkpoint_dir, batch_idx, epoch, min_loss, info='min_loss_model'):
    # date = time.strftime('%m-%d_%H-%M', time.localtime(time.time()))
    # checkpoint = 'model_%s__info_%s__loss_%f__batch_idx_%d__epoch_%d.pt' % (info, min_loss, batch_idx, epoch)
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, info).replace('\\', '/'))


def save_max_bleu_model(model, saved_model_dir, batch_idx, epoch, max_bleu, info='max_bleu_model'):
    # date = time.strftime('%m-%d_%H-%M', time.localtime(time.time()))
    # saved_model = 'model_%s__info_%s__bleu_%f__batch_idx_%d__epoch_%d.pt' % (info, max_bleu, batch_idx, epoch)
    torch.save(model.state_dict(), os.path.join(saved_model_dir, info).replace('\\', '/'))