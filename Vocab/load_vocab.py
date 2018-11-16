

def load_vocab(vocabulary_class, vocab_file, i2w=True, file_type='pkl'):
    vocab = vocabulary_class(vocab_file, i2w, file_type)

    return vocab
