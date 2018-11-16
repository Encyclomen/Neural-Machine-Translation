import json
import _pickle as pickle


def invert_vocab(vocab):
    v = {}
    for k, idx in vocab.items():
        v[idx] = k

    return v


def load_vocab_file(file, file_type='pkl'):
    if file_type is 'pkl':
        f = open(file, 'rb')
        vocab = pickle.load(f)
    elif file_type is 'json':
        f = open(file, 'r', encoding='UTF-8')
        vocab = json.load(f)
    f.close()

    return vocab


class Vocabulary():
    def __init__(self, vocab_file, i2w=True, file_type='pkl'):
        if i2w is True:
            self.i2w_vocab = load_vocab_file(vocab_file, file_type=file_type)
            self.w2i_vocab = invert_vocab(self.i2w_vocab)  # {sentence: index} -> {index: sentence}
        else:
            self.w2i_vocab = load_vocab_file(vocab_file, file_type=file_type)
            self.i2w_vocab = invert_vocab(self.w2i_vocab)  # {sentence: index} -> {index: sentence}

    def get_vocab_len(self):
        # get the size of vocabulary
        assert len(self.i2w_vocab) == len(self.w2i_vocab)

        return len(self.i2w_vocab)

    def get_i2w_vocab(self):
        return self.i2w_vocab

    def get_w2i_vocab(self):
        return self.w2i_vocab

    def index2word(self, i):
        return self.i2w_vocab[i]

    def word2index(self, w):
        return self.w2i_vocab[w]