import copy
import os
import argparse
import sys

import torch
from torch.multiprocessing import Pool
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
sys.path.append(".")
sys.path.append("./Dataset")
sys.path.append("./Model")
sys.path.append("./Optim")
sys.path.append("./Util")
sys.path.append("./Vocabulary")
from Model.RNNSearch.constants import *
from Dataset.load_dataset import load_dataset, load_dataloader
from Dataset.parallel_corpus_dataset import Parallel_Corpus_Dataset
from Model.load_model import load_model
from Model.RNNSearch.rnnsearch import RNNSearch
from Optim.load_optim import load_optimizer, load_optim_wrapper
from Optim.rnnsearch_optim_wrapper import RNNSearch_Optim_Wrapper
from Vocab.load_vocab import load_vocab
from Vocab.vocabulary import Vocabulary
from Util.utils import to_Tensor, get_batch_mask, batch_str2idx_with_flag, \
                        batch_idx2str, sort_batch, parallel_corpus_collate_fn
from Util.save_model import save_checkpoint_model, save_min_loss_model, save_max_bleu_model


def parse_args():
    root = '.'  # 工程根目录
    corpus_dir = 'Corpus'  # 语料目录
    saved_model_dir = 'Output/Saved_Models'  # 模型保存目录
    checkpoint_dir = 'Output/Checkpoints'  # 模型检查点保存目录
    translation_output_dir = 'Output/Translation'  # 翻译输出目录
    src_vocab_json = 'src_chinese_vocab.json'
    trg_vocab_json = 'trg_english_vocab.json'
    train_src_corpus = 'chinese.corpus'
    train_trg_corpus = 'english.corpus'
    vldt_src_corpus = 'test.src'
    vldt_trg_corpus_list = ['test.ref0', 'test.ref1', 'test.ref2', 'test.ref3']
    saved_model = 'RNNSearch_max_bleu_model'
    checkpoint = 'RNNSearch_min_loss_model'
    test_src_corpus = 'nist02.src'
    test_trg_corpus_list = ['nist02.tok.ref0', 'nist02.tok.ref1', 'nist02.tok.ref2', 'nist02.tok.ref3']
    translation_output = 'nist02_translation_output.txt'
# all options-----------------------------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='Training RNNSearch Model')
    # necessary directories
    parser.add_argument('--root', type=str, default=root,help="the root directory of this project")
    parser.add_argument('--corpus_dir', type=str, default=corpus_dir, help='the directory of corpus')
    parser.add_argument('--saved_model_dir', type=str, default=saved_model_dir, help='the directory of saved model')
    parser.add_argument('--checkpoint_dir', type=str, default=checkpoint_dir, help='the directory of saved checkpoints')
    parser.add_argument('--translation_output_dir', type=str, default=translation_output_dir, help='the directory in which output of translation will be stored')
    # corpus, vocab, batch and dataset parameters
    parser.add_argument('--train_src_corpus', type=str, default=train_src_corpus, help='the source language corpus for training')
    parser.add_argument('--train_trg_corpus', type=str, default= train_trg_corpus, help='the target language corpus for training')
    parser.add_argument('--train_batch_size', type=int, default=80, help='the batch size used in training')
    parser.add_argument('--src_max_len', type=int, default=50, help='the max length of valid sentence in source language corpus')
    parser.add_argument('--trg_max_len', type=int, default=50, help='the max length of valid sentence in target language corpus')
    # vocab parameters
    parser.add_argument('--src_vocab_json', type=str, default=src_vocab_json, help='the source language vocabulary')
    parser.add_argument('--trg_vocab_json', type=str, default=trg_vocab_json, help='the target language vocabulary')
    # validation parameters
    parser.add_argument('--vldt_src_corpus', type=str, default=vldt_src_corpus, help='the source language corpus for validating')
    parser.add_argument('--vldt_trg_corpus_list', type=str, default=vldt_trg_corpus_list, nargs='+', help='the target language corpus for validating')
    parser.add_argument('--vldt_freq', type=int, default=1500, help='the frequency for validation')
    # optimization parameters
    parser.add_argument('--interval', type=int, default=1, help='the batch interval for parameter updating')
    parser.add_argument('--LR', type=float, default=0.0005, help="the learning rate of model's parameters")
    parser.add_argument('--weight_decay_l2', type=float, default=0, help='the weight decay coefficient of L2 normalization')
    parser.add_argument('--max_norm', type=float, default=1.0, help='the max norm of model parameters')
    parser.add_argument('--dec_rate', type=float, default=0.5, help='the decreasing rate of LR per epoch')
    # model hyperparameters that can be manually set in command line
    parser.add_argument('--saved_model', type=str, default=saved_model, help='the saved model')
    parser.add_argument('--d_src_emb', type=int, default=620, help='the number of features in source word embedding')
    parser.add_argument('--d_trg_emb', type=int, default=620, help='the number of features in target word embedding')
    parser.add_argument('--enc_nhid', type=int, default=1000, help='the number of features in source side hidden states')
    parser.add_argument('--dec_nhid', type=int, default=1000, help='the number of features in target side hidden states')
    parser.add_argument('--dec_natt', type=int, default=1000, help='the number of features in target side attention layer')
    parser.add_argument('--nreadout', type=int, default=620, help='the number of maxout layer')
    parser.add_argument('--src_emb_dropout', type=float, default=0.3, help='the dropout rate for source word embedding')
    parser.add_argument('--trg_emb_dropout', type=float, default=0.3, help='the dropout rate for target word embedding')
    parser.add_argument('--enc_hid_dropout', type=float, default=0.3, help='the dropout rate for encoder hidden state')
    parser.add_argument('--readout_dropout', type=float, default=0.3, help='the dropout rate for readout layer')
    # checkpoint parameters
    parser.add_argument('--if_load_checkpoint', type=bool, default=False, help='if the model loads existing checkpoint')
    parser.add_argument('--checkpoint', type=str, default=checkpoint, help='the checkpoint file')
    # evaluation parameters
    parser.add_argument('--test_src_corpus', type=str, default=test_src_corpus, help='the source language corpus for testing')
    parser.add_argument('--test_trg_corpus_list', type=str, default=test_trg_corpus_list, help='the target language corpus for testing')
    parser.add_argument('--beam_size', type=int, default=10, help='size of beam for beamsearch')
    parser.add_argument('--translation_output', type=str, default=translation_output, help='the translation output')
    # Misc.
    parser.add_argument('--mode', type=str, default='translate', help='the working mode of the model')
    parser.add_argument('--cuda', type=bool, default=False, help='if cuda() is implemented')
    parser.add_argument('--epoch', type=int, default=0, help='the initial epoch to train')
    parser.add_argument('--nepoch', type=int, default=5, help='the number of epoch to train')
    parser.add_argument('--info', type=str, default='', help='info of the model')
    parser.add_argument('--seed', type=int, default=123, help='random number seed')
# ----------------------------------------------------------------------------------------------------------------------
    opt = parser.parse_args()
    opt.corpus_dir = os.path.join(opt.root, opt.corpus_dir).replace('\\', '/')
    opt.saved_model_dir = os.path.join(opt.root, opt.saved_model_dir).replace('\\', '/')
    opt.checkpoint_dir = os.path.join(opt.root, opt.checkpoint_dir).replace('\\', '/')
    opt.translation_output_dir = os.path.join(opt.root, opt.translation_output_dir).replace('\\', '/')
    opt.src_vocab_json = os.path.join(opt.corpus_dir, opt.src_vocab_json).replace('\\', '/')
    opt.trg_vocab_json = os.path.join(opt.corpus_dir, opt.trg_vocab_json).replace('\\', '/')
    opt.train_src_corpus = os.path.join(opt.corpus_dir, opt.train_src_corpus).replace('\\', '/')
    opt.train_trg_corpus = os.path.join(opt.corpus_dir, opt.train_trg_corpus).replace('\\', '/')
    opt.vldt_src_corpus = os.path.join(opt.corpus_dir, opt.vldt_src_corpus).replace('\\', '/')
    opt.vldt_trg_corpus_list = ' '.join(
        map(lambda trg_corpus: os.path.join(opt.corpus_dir, trg_corpus).replace('\\', '/'), opt.vldt_trg_corpus_list)
    )
    opt.saved_model = os.path.join(opt.saved_model_dir, opt.saved_model).replace('\\', '/')
    opt.checkpoint = os.path.join(opt.checkpoint_dir, opt.checkpoint).replace('\\', '/')
    opt.test_src_corpus = os.path.join(opt.corpus_dir, opt.test_src_corpus).replace('\\', '/')
    opt.test_trg_corpus_list = ' '.join(
        map(lambda trg_corpus: os.path.join(opt.corpus_dir, trg_corpus).replace('\\', '/'), opt.test_trg_corpus_list)
    )
    opt.translation_output = os.path.join(opt.translation_output_dir, opt.translation_output).replace('\\', '/')

    return opt


def my_callback(result):
    global max_bleu

    bleu = result[0]
    if bleu > max_bleu:
        max_bleu = bleu
        print('!!!!! max bleu: %f' % (max_bleu * 100))
        batch_idx = result[1]
        cur_epoch = result[2]
        save_max_bleu_model(model, opt.saved_model_dir, batch_idx, cur_epoch, max_bleu*100, info='RNNSearch_max_bleu_model')


def train(model, src_vocab, trg_vocab, optim_wrapper, train_iter, vldt_iter):
    global opt, min_loss, max_bleu
    subprocess_pool = Pool(2)

    # start training
    model.train()
    print('!!!train', id(model))
    for epoch in range(opt.epoch, opt.nepoch):
        cur_epoch = epoch + 1
        total_loss = 0
        print('############### epoch = %d ###############\n' % cur_epoch)
        for batch_idx, batch in enumerate(train_iter, start=1):
            sorted_batch = sort_batch(batch)
            src_raw = sorted_batch[0]
            trg_raw = sorted_batch[1]
            # 获得以word indices表示的源句子和目标语句
            src = batch_str2idx_with_flag(src_raw, src_vocab, unk=UNK, pad=PAD, sos=SOS, eos=EOS)
            f_trg = batch_str2idx_with_flag(trg_raw, trg_vocab, unk=UNK, pad=PAD, sos=SOS, eos=EOS)
            src, f_trg = to_Tensor(src, f_trg, tensor_type=torch.LongTensor, cuda=opt.cuda)
            src_mask = get_batch_mask(src, src_vocab, PAD)
            f_trg_mask = get_batch_mask(f_trg, trg_vocab, PAD)
            '''
            # b_trg = batch_str2idx_with_flag(trg_raw, trg_vocab, unk=UNK, pad=PAD, sos=SOS, eos=EOS, reverse=True)  # 目标端反向的句子batch，暂时不用
            # src, f_trg, b_trg = to_Tensor(src, f_trg, b_trg, tensor_type=torch.LongTensor, cuda=opt.cuda)
            # b_trg_mask = get_batch_mask(b_trg, trg_vocab, PAD)
            '''
            loss = model(src, src_mask, f_trg, f_trg_mask)  # TODO
            total_loss = total_loss + float(loss)
            loss.backward()
            if batch_idx % opt.interval == 0:
                total_loss = total_loss / opt.interval
                if total_loss < min_loss:
                    print('& epoch = %d batch_idx = %d min_loss = %f &\n' % (cur_epoch, batch_idx/opt.interval, total_loss))
                    min_loss = total_loss
                    save_min_loss_model(model, opt.checkpoint_dir, batch_idx/opt.interval, cur_epoch, min_loss,info='RNNSearch_min_loss_model')
                else:
                    print('- batch_idx = %d, loss = %f -\n' % (batch_idx/opt.interval, total_loss))
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.max_norm, norm_type=2)  # 参数更新前执行梯度裁剪,默认取L2范数
                optim_wrapper.step()
                optim_wrapper.zero_grad()
                total_loss = 0
                optim_wrapper.update_lr_per_step()
                '''
              # 开启额外cpu进程测试开发集bleu时调用下面语句
              # 从第4轮训练开始，每隔opt.vldt_freq个batch，另开子进程测试一次bleu
              if cur_epoch >= 4 and (batch_idx * opt.interval) % opt.vldt_freq == 0:
                  cpu_model = copy.deepcopy(model).cpu()
                  subprocess_pool.apply_async(evaluate, args=(opt, cpu_model, src_vocab, trg_vocab, vldt_iter, batch_idx, cur_epoch), callback=my_callback)
              '''
        optim_wrapper.zero_grad()
        optim_wrapper.update_lr_per_epoch()
        save_checkpoint_model(model, opt.checkpoint_dir, cur_epoch, info='RNNSearch_checkpoint_model')
        print('$ min_loss: %f, max_bleu: %f $\n' % (min_loss, max_bleu))
    # 关闭进程池等待开发集bleu测试完成
    subprocess_pool.close()
    subprocess_pool.join()


def evaluate(opt, model, src_vocab, trg_vocab, corpus_iter, batch_idx, cur_epoch):
    try:
        model.eval()
        print('!!!eval', id(model))
        hyp_list = []
        ref_list = []

        print('sub: ', os.getpid())
        print('num: ', batch_idx)
        for idx, batch in enumerate(corpus_iter, start=1):
            print(idx)
            batch = list(batch)
            src_raw = batch[0]
            trg_raw = batch[1:]
            ref = list(map(lambda x: x[0], trg_raw))
            ref_list.append(ref)
            src = batch_str2idx_with_flag(src_raw, src_vocab, unk=UNK, pad=PAD, sos=SOS, eos=EOS)
            src = to_Tensor(src, tensor_type=torch.LongTensor, cuda=opt.cuda)
            src_mask = get_batch_mask(src, src_vocab, PAD)
            with torch.no_grad():
                sentences_output, scores_output = model.beamsearch(src, src_mask, opt.beam_size, normalize=True)
                best_sentence, best_score = sentences_output[0], scores_output[0]
                best_sentence = batch_idx2str([best_sentence], trg_vocab)
                hyp_list.append(best_sentence[0])

        bleu = corpus_bleu(ref_list, hyp_list, smoothing_function=SmoothingFunction().method1)

        return bleu, batch_idx, cur_epoch
    except Exception as ex:
        msg = "subprcess wrong: %s" % ex
        print(msg)


def translate(model, src_vocab, trg_vocab, corpus_iter, translation_output=None):
    global opt

    model.eval()
    hyp_list = []

    for idx, batch in enumerate(corpus_iter, start=1):
        print(idx)
        batch=list(batch)
        src_raw = batch[0]
        src = batch_str2idx_with_flag(src_raw, src_vocab, unk=UNK, pad=PAD, sos=SOS, eos=EOS)
        src = to_Tensor(src, tensor_type=torch.LongTensor, cuda=opt.cuda)
        src_mask = get_batch_mask(src, src_vocab, PAD)
        with torch.no_grad():
            sentences_output, scores_output = model.beamsearch(src, src_mask, opt.beam_size, normalize=True)
            best_sentence, best_score = sentences_output[0], scores_output[0]
            best_sentence = batch_idx2str([best_sentence], trg_vocab)
            hyp_list.append(best_sentence[0])

    with open(translation_output, 'w') as f:
        for sentence in hyp_list:
            sentence = ' '.join(sentence)
            f.write(sentence + '\n')


if __name__ == "__main__":
    min_loss = float("inf")
    max_bleu = 0

    print("*******************Start RNNSearch*******************", os.getpid())
    # parse arguments that may need to be manually set in command line
    opt = parse_args()
    # set the random seed manually
    torch.manual_seed(opt.seed)
    opt.cuda = opt.cuda and torch.cuda.is_available()
    if opt.cuda:
        torch.cuda.manual_seed(opt.seed)
    # load vocabularies for source and target language
    src_vocab = load_vocab(Vocabulary, opt.src_vocab_json, i2w=False, file_type='json')
    trg_vocab = load_vocab(Vocabulary, opt.trg_vocab_json, i2w=False, file_type='json')
    # some model's hyperparameters that can not be manually set in command line
    enc_ntok = src_vocab.get_vocab_len()  # the size of source language vocabulary, taking sos,eos,pad,unk into account
    dec_ntok = trg_vocab.get_vocab_len()  # the size of target language vocabulary, taking sos,eos,pad,unk into account
    if opt.mode == 'train':
        # load the assigned model
        if opt.if_load_checkpoint:
            model = load_model(RNNSearch,
                               opt.d_src_emb, opt.enc_nhid, opt.src_emb_dropout, opt.enc_hid_dropout,
                               opt.d_trg_emb, opt.dec_nhid, opt.dec_natt, opt.nreadout,
                               opt.readout_dropout, opt.trg_emb_dropout,
                               opt.enc_ntok, dec_ntok,
                               cuda=opt.cuda, if_load_state_dict=True, saved_state_dict=opt.checkpoint)
        else:
            model = load_model(RNNSearch,
                               opt.d_src_emb, opt.enc_nhid, opt.src_emb_dropout, opt.enc_hid_dropout,
                               opt.d_trg_emb, opt.dec_nhid, opt.dec_natt, opt.nreadout,
                               opt.readout_dropout, opt.trg_emb_dropout,
                               enc_ntok, dec_ntok,
                               cuda=opt.cuda)
        # load the optimizer and optimizer wrapper of the training
        optimizer = load_optimizer(model, torch.optim.Adam,
                                   lr=opt.LR, weight_decay=opt.weight_decay_l2)
        optim_wrapper = load_optim_wrapper(optimizer, RNNSearch_Optim_Wrapper, dec_rate=opt.dec_rate)
        train_dataset = load_dataset(Parallel_Corpus_Dataset, opt.train_src_corpus, opt.train_trg_corpus,
                                     opt.src_max_len, opt.trg_max_len)
        train_dataloader = load_dataloader(train_dataset, torch.utils.data.DataLoader, batch_size=opt.train_batch_size,
                                           shuffle=True, collate_fn=parallel_corpus_collate_fn)
        vldt_dataset = load_dataset(Parallel_Corpus_Dataset, opt.vldt_src_corpus, opt.vldt_trg_corpus_list)
        vldt_dataloader = load_dataloader(vldt_dataset, torch.utils.data.DataLoader, batch_size=1,
                                           shuffle=False, collate_fn=parallel_corpus_collate_fn)
        train(model, src_vocab, trg_vocab, optim_wrapper, train_dataloader, vldt_dataloader)  # start training
    else:
        # load the assigned model
        model = load_model(RNNSearch,
                           opt.d_src_emb, opt.enc_nhid, opt.src_emb_dropout, opt.enc_hid_dropout,
                           opt.d_trg_emb, opt.dec_nhid, opt.dec_natt, opt.nreadout,
                           opt.readout_dropout, opt.trg_emb_dropout,
                           enc_ntok, dec_ntok,
                           cuda=opt.cuda, if_load_state_dict=True, saved_state_dict=opt.saved_model)
        test_dataset = load_dataset(Parallel_Corpus_Dataset, opt.test_src_corpus, opt.test_trg_corpus_list)
        test_dataloader = load_dataloader(test_dataset, torch.utils.data.DataLoader, batch_size=1,
                                          shuffle=False, collate_fn=parallel_corpus_collate_fn)
        translate(model, src_vocab, trg_vocab, test_dataloader,
                  translation_output=opt.translation_output)  # execute translation
