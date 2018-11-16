from torch import optim


def load_optimizer(model, optim_type='Adam', **kwargs):
    print(type(model))
    param_list = model.parameters()
    optimizer_types = {
        'Adam': lambda: optim.Adam(param_list, **kwargs),
        'RMSprop': lambda: optim.RMSprop(param_list, **kwargs),
    }
    assert optim_type in optimizer_types.keys(), "\033[0;31mInvalid Optimizer Type!\033[0m"
    optimizer = optimizer_types.get(optim_type)()

    return optimizer


def batch_str2idx_with_flag(batch, vocab, unk=None, pad=None, sos=None, eos=None, reverse=False):
    """
    Convert words in a batch into corresponding indices in the assigned vocabulary and append necessary flags.
    :param batch: the batch containing sentences
    :param vocab: the assigned vocabulary
    :param unk: the index of <unk> in the assigned vocabulary
    :param pad: the index of <pad> in the assigned vocabulary
    :param sos: the index of <sos> in the assigned vocabulary
    :param eos: the index of <eos> in the assigned vocabulary
    :param reverse: if reverse == True, the sentences in a batch will be converted reversely
    :return: padded_output: the converted batch filled with corresponding word indices after padded
    """
    max_len = max(len(x) for x in batch)  # 获得一个batch中最长句子的长度
    padded = []
    for x in batch:
        # 为batch中每个句子添加'<sos>'和'<eos>'
        if reverse:
            padded.append(
                ([] if eos is None else [eos]) +
                list(x[::-1]) +
                ([] if sos is None else [sos]))
        else:
            padded.append(
                ([] if sos is None else [sos]) +
                list(x) +
                ([] if eos is None else [eos]))
        padded[-1] = padded[-1] + [pad] * max(0, max_len - len(x))  # 在较短的句子后面添加'<pad>'
        padded[-1] = list(map(lambda v: vocab.word2index(v) if v in vocab.get_w2i_vocab() else vocab.word2index(unk), padded[-1]))  # 将该句中每个word转化成字典中对应index

    return padded


def sort_batch(batch):
    """
    Sort the sentences in a batch according to sentence length in descending order
    :param batch: batch containing sentences of random lengths
    :return: sorted_batch: the batch sorted according to the lengths of sentences in former batch
    """
    batch = zip(*batch)
    sorted_batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)  # 得到以batch中sentence length降序排列的sorted_batch
    sorted_batch = list(zip(*sorted_batch))

    return sorted_batch