import itertools

import torch.utils.data


class Parallel_Corpus_Dataset(torch.utils.data.Dataset):
    def __init__(self, p_src, p_trg, src_max_len=None, trg_max_len=None):
        """
        :param p_src: path of the file containing source language corpus
        :param p_trg: paths of the files containing target language corpus, type of str or list
        :param src_max_len: max length of source language sentences. default: None
        :param trg_max_len: max length of target language sentences, default: None
        """
        # 将源语料，目标语料文件名存入p_list，p_list[p_src, p_trg1, p_trg2...]
        p_trg = p_trg.split()  # 将trg_corpus字符串中包含的所有语料的文件名转换成列表, 为了处理有多组target language corpus的情况
        p_list = [p_src]
        if isinstance(p_trg, str):
            p_list.append(p_trg)
        else:  # 处理拥有多组target language语料的情况，此时type(p_trg) == list
            p_list.extend(p_trg)

        # 加载source language和target language的语料集
        corpus_set = []
        for p in p_list:
            with open(p, 'r', encoding='UTF-8-sig') as f:
                corpus_set.append(f.readlines())  # 加载语料同时去除掉utf-8默认行首的'\ufeffi'
        assert len(corpus_set[0]) == len(corpus_set[1])

        # 将源语料与目标语料中不超过设定的最大长度的语句对应成sentence pair，存入data，用来后续生成batch
        self.data = []
        for line in itertools.zip_longest(*corpus_set):
            line = list(line)  # 将zip后每一行对应的中英文句子tuple转化为list,以后续进行修改
            line = list(map(lambda v: v.lower().strip(), line))  # 字母小写化
            if not any(line):
                continue
            line = list(map(lambda v: v.split(), line))
            # 如果源语言或目标语言句子超过设定的最大长度，就舍弃该句
            if (src_max_len and len(line[0]) > src_max_len) \
                    or (trg_max_len and len(line[1]) > trg_max_len):
                continue
            self.data.append(line)
        self.length = len(self.data)

    def __len__(self):

        return self.length

    def __getitem__(self, index):

        return self.data[index]
