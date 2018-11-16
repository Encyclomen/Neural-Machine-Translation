import numpy as np
import torch


def position_encoding_init(n_position, d_pos_vec, cuda=False):
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / d_pos_vec) for j in range(d_pos_vec)]
        if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1
    position_enc = torch.from_numpy(position_enc).type(torch.FloatTensor)
    if cuda is True:
        position_enc = position_enc.cuda()

    return position_enc


def get_attn_padding_mask(q_seq_lens, k_seq_lens, cuda=False):
    '''
    assert np.size(q_seq_lens, 0) == np.size(k_seq_lens, 0)
    batch_size = np.size(q_seq_lens, 0)
    max_q_seq_len = q_seq_lens.max()
    max_k_seq_len = k_seq_lens.max()
    pad_attn_mask = torch.zeros(batch_size, max_q_seq_len, max_k_seq_len, dtype=torch.uint8)  # construct mask
    for i in range(batch_size):
        pad_attn_mask[i, :, 0:k_seq_lens[i]] = 1
    '''
    assert q_seq_lens.size(0) == k_seq_lens.size(0)
    batch_size = q_seq_lens.size(0)
    max_q_seq_len = int(q_seq_lens.max())
    max_k_seq_len = int(k_seq_lens.max())
    # construct mask
    pad_attn_mask = torch.zeros(batch_size, max_q_seq_len, max_k_seq_len, dtype=torch.uint8)
    for i in range(batch_size):
        pad_attn_mask[i, :, 0:k_seq_lens[i]] = 1
    if cuda is True:
        pad_attn_mask=pad_attn_mask.cuda()

    return pad_attn_mask


def get_attn_subsequent_mask(seq_lens, cuda=False):
    batch_size = seq_lens.size(0)
    max_seq_len = int(seq_lens.max())
    subsequent_mask = np.tril(np.ones((batch_size, max_seq_len, max_seq_len)), k=0).astype('uint8')
    subsequent_mask = torch.from_numpy(subsequent_mask)
    if cuda is True:
        subsequent_mask = subsequent_mask.cuda()

    return subsequent_mask