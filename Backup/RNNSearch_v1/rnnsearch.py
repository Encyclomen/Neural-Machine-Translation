import torch
import torch.nn as nn
import torch.nn.functional as F

from Model.RNNSearch import constants
from Beam.beam import Beam

from Model.RNNSearch.decoder import Decoder
from Model.RNNSearch.encoder import Encoder


class RNNSearch(nn.Module):
    """The whole RNNSearch model structure"""
    def __init__(self, d_src_emb, enc_nhid, src_emb_dropout, enc_hid_dropout, d_trg_emb, dec_nhid, dec_natt,
                 nreadout, readout_dropout, trg_emb_dropout, enc_ntok, dec_ntok):
        super(RNNSearch, self).__init__()
        # model's hyperparameters that can be manually set in command line
        self.d_src_emb = d_src_emb
        self.enc_nhid = enc_nhid
        self.src_emb_dropout = src_emb_dropout
        self.enc_hid_dropout = enc_hid_dropout
        self.d_trg_emb = d_trg_emb
        self.dec_nhid = dec_nhid
        self.dec_natt = dec_natt
        self.nreadout = nreadout
        self.readout_dropout = readout_dropout
        self.trg_emb_dropout = trg_emb_dropout
        # model's hyperparameters that can not be manually set in command line
        self.enc_pad = constants.enc_pad
        self.dec_sos = constants.dec_sos
        self.dec_eos = constants.dec_eos
        self.dec_pad = constants.dec_pad
        self.enc_ntok = enc_ntok
        self.dec_ntok = dec_ntok
        # model's submodules
        self.encoder = Encoder(d_src_emb, enc_nhid, enc_ntok, self.enc_pad, src_emb_dropout, enc_hid_dropout)
        self.dec_emb = nn.Embedding(dec_ntok, d_trg_emb, padding_idx=self.dec_pad)
        self.decoder = Decoder(d_trg_emb, dec_nhid, 2 * enc_nhid, dec_natt, nreadout, readout_dropout)
        self.affine = nn.Linear(nreadout, dec_ntok)
        self.init_affine = nn.Linear(2 * enc_nhid, dec_nhid)
        self.dec_emb_dp = nn.Dropout(trg_emb_dropout)

    def param_init(self):
        for param in self.parameters():
            param.data.uniform_(-0.1, 0.1)

    def forward(self, src, src_mask, f_trg, f_trg_mask, b_trg=None, b_trg_mask=None):
        """
        :param src: batch of source language sentences
        :param src_mask: mask matrix composed of 0s and 1s that is used to retrieve valid sentence lengths in a source sentence batch
        :param f_trg:  batch of forward target language sentences
        :param f_trg_mask: mask matrix composed of 0s and 1s that is used to retrieve valid sentence lengths in a forward target sentence batch
        :param b_trg: batch of backward target language sentences, NOT YET USED right now
        :param b_trg_mask: mask matrix composed of 0s and 1s that is used to retrieve valid sentence lengths in a backward target sentence batch, NOT YET USED right now
        :return: loss: the loss
        """
        enc_hiddens = self.encoder(src, src_mask)
        enc_hiddens = enc_hiddens.contiguous()

        # decoder隐层初始化
        sum_enc_hiddens = enc_hiddens.sum(1)
        src_seq_len = src_mask.long().sum(1).unsqueeze(-1).expand_as(sum_enc_hiddens)
        avg_enc_hidden = sum_enc_hiddens / src_seq_len.float()  # 计算encoder隐层的平均值
        init_dec_hidden = torch.tanh(self.init_affine(avg_enc_hidden))  # 将encoder隐层平均值经线性变换并tanh激活后用于decoder隐层初始化
        # 按time step前向传播，累计loss
        dec_hidden = init_dec_hidden
        loss = 0
        for i in range(f_trg.size(1) - 1):
            dec_output, dec_hidden = self.decoder(self.dec_emb_dp(self.dec_emb(f_trg[:, i])), dec_hidden, src_mask, enc_hiddens)
            y_prob = self.affine(dec_output)  # 对dec_output做线性变换，得到该位置输出单词的概率分布y_prob
            loss += F.cross_entropy(y_prob, f_trg[:, i + 1], reduce=True, ignore_index=self.dec_pad)
        # loss = loss.sum() / f_trg_mask[:, 1:].sum().float()
        loss = loss / (f_trg.size(1)-1)

        return loss

    def beamsearch(self, src, src_mask, beam_size=10, normalize=False, max_len=None, min_len=None):
        max_len = src.size(1) * 3 if max_len is None else max_len
        min_len = src.size(1) / 2 if min_len is None else min_len

        enc_hiddens = self.encoder(src, src_mask)
        enc_hiddens = enc_hiddens.contiguous()

        sum_enc_hiddens = enc_hiddens.sum(1)
        src_seq_len = src_mask.long().sum(1).unsqueeze(-1).expand_as(sum_enc_hiddens)
        avg_enc_hidden = sum_enc_hiddens / src_seq_len.float()
        init_dec_hidden = torch.tanh(self.init_affine(avg_enc_hidden))

        dec_hidden = init_dec_hidden
        prev_beam = Beam(beam_size)
        prev_beam.candidates = [[self.dec_sos]]  # 存储每步搜索的候选词
        prev_beam.scores = [0]
        f_done = (lambda x: x[-1] == self.dec_eos)

        valid_size = beam_size

        sentence_list, score_list = [], []
        for k in range(max_len):
            candidates = prev_beam.candidates
            input = src.new(list(map(lambda cand: cand[-1], candidates)))
            input = self.dec_emb_dp(self.dec_emb(input))  # 对输入到decoder中的embeddings执行dropout操作
            output, dec_hidden = self.decoder(input, dec_hidden, src_mask, enc_hiddens)
            log_prob = F.log_softmax(self.affine(output), dim=1)
            if k < min_len:
                log_prob[:, self.dec_eos] = -float('inf')  # 使句子不会在小于min_len的情况下结束
            if k == max_len - 1:  # 如果达到最大长度，就让除了eos之外的词生成概率为0
                eos_prob = log_prob[:, self.dec_eos].clone()
                log_prob[:, :] = -float('inf')
                log_prob[:, self.dec_eos] = eos_prob
            next_beam = Beam(valid_size)
            sentence_done_list, score_done_list, remain_list = next_beam.step(-log_prob, prev_beam, f_done)  # 根据当前目标单词概率分布，以及之前的beam，进行下一步搜索
            sentence_list.extend(sentence_done_list)  # 将翻译完成的句子加入hyp_list
            score_list.extend(score_done_list)
            valid_size = valid_size - len(sentence_done_list)  # 更新下一个beam的valid_size

            if valid_size == 0:  # beam search结束
                break

            beam_remain_idx = src.new(remain_list)
            enc_hiddens = enc_hiddens.index_select(0, beam_remain_idx)  # 根据上步beam search结果更新对应源语句的encoder hidden states
            src_mask = src_mask.index_select(0, beam_remain_idx)  # 根据上步beam search结果更新对应源语句的mask
            dec_hidden = dec_hidden.index_select(0, beam_remain_idx)  # 根据上步beam search结果更新对应源语句的上一个decoder hidden state
            prev_beam = next_beam  # 更新beam

        sentence_list = [sentence[1: sentence.index(self.dec_eos)] if self.dec_eos in sentence else sentence[1:] for sentence in sentence_list]
        if normalize:  # 如指定normalize为True，则对beam search得到的每句话对应score进行标准化
            for k, (hyp, score) in enumerate(zip(sentence_list, score_list)):
                if len(hyp) > 0:
                    score_list[k] = score_list[k] / len(hyp)
        score = dec_hidden.new(score_list)
        sort_score, sort_idx = torch.sort(score)
        sentences_output, scores_output = [], []
        for idx in sort_idx.tolist():
            sentences_output.append(sentence_list[idx])
            scores_output.append(score[idx])
        
        return sentences_output, scores_output
