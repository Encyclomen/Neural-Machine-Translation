import torch


def to_Tensor(*sequences, tensor_type=torch.FloatTensor, cuda=False):
    tensor_sequence = [torch.Tensor(sequence).type(tensor_type) for sequence in sequences]
    # ------------------------------------------------------------------------------------------------------------------
    if cuda is True:
        tensor_sequence = [tensor.cuda() for tensor in tensor_sequence]

    if len(tensor_sequence) == 1:
        return tensor_sequence[0]
    return tensor_sequence


def dict_value_to_Tensor(*dicts, tensor_type=torch.FloatTensor, cuda=False):
    for dict_ in dicts:
        for key, value in dict_.items():
            dict_[key] = torch.Tensor(value).type(tensor_type)
    # ------------------------------------------------------------------------------------------------------------------
    if cuda is True:
        for dict_ in dicts:
            for key, value in dict_.items():
                dict_[key] = value.cuda()

    if len(dicts) == 1:
        return dicts[0]
    return dicts


def parallel_corpus_collate_fn(x):
    """ This function is merely used for the generation of parallel dataset batch iterators """
    x = list(zip(*x))

    return x


def get_batch_mask(batch, vocab, pad=None):
    """
    :param batch: the batch containing sentences (Tensor)
    :param vocab: the assigned vocabulary
    :return: mask: the mask matrix composed of 0s and 1s that is used to retrieve valid sentence lengths in a batch
    """
    mask = batch.ne(vocab.word2index(pad))  # batch中每一项与vocab中pad值进行比较，生成mask

    return mask


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


def batch_idx2str(batch, vocab):
    """
    Convert word indices in a batch into corresponding words in the assigned vocabulary
    :param batch: the batch containing word indices
    :param vocab: the assigned vocabulary
    :return: output: the converted batch filled with corresponding words
    """
    output = []
    for x in batch:
        output.append(list(map(lambda v: vocab.index2word(int(v)), x)))
    return output


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
