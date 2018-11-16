import argparse
import json
from collections import Counter

def parseargs():
    parser = argparse.ArgumentParser(description='Training Attention-based Neural Machine Translation Model')

    corpus_root = '../Corpus/'  # corpus和vocab的加载路径
    parser.add_argument('--corpus_path', type=str, default=corpus_root+'chinese.corpus', help='')
    # parser.add_argument('--corpus_path', type=str, default=corpus_root+'english.corpus', help='')
    parser.add_argument('--limit', type=int, default=30000, help='')
    parser.add_argument('--output', type=str, default=corpus_root+'src_chinese_vocab.json', help='')
    # parser.add_argument('--output', type=str, default=corpus_root+'trg_english_vocab.json', help='')

    args = parser.parse_args()
    return args

def create_dictionary(corpus, lim=0):
    global_counter = Counter()
    fd = open(corpus, 'r', encoding='UTF-8')

    for line in fd:
        if '\xef\xbb\xbf' in line:# 针对带有BOM的UTF-8用replace替换掉'\xef\xbb\xbf'
            line = line.replace('\xef\xbb\xbf', '')
        words = line.encode('utf-8').decode('utf-8-sig').strip().split()

        global_counter.update(words)

    combined_counter = global_counter

    if lim <= 2:  # lim<=2代表不对词典大小进行限制
        lim = len(combined_counter) + 2

    vocab_count = combined_counter.most_common(lim - 2)
    total_counts = sum(combined_counter.values())
    print(100.0 * sum([count for _, count in vocab_count]) / total_counts)

    vocab = {"<unk>": 0, "<pad>": 1, "<sos>": 2, "<eos>": 3}

    for i, (word, count) in enumerate(vocab_count):
        vocab[word] = i + 4

    return vocab


if __name__ == "__main__":
    args = parseargs()
    vocab = create_dictionary(args.corpus_path, args.limit)

    temp=[]
    for key in vocab.keys():
        temp.append(key)
    print(temp)
    fd = open(args.output, 'w')
    fd.write(json.dumps(vocab))
    fd.close()
