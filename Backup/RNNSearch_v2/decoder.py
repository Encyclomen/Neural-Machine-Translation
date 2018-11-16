import torch
import torch.nn as nn
import torch.nn.functional as F

from Model.RNNSearch.attention import Attention


class Decoder(nn.Module):
    """Decoder that decodes information from both encoder and attention mechanism"""
    def __init__(self, ninp, dec_nhid, enc_ncontext, natt, dec_nout, dec_out_dropout):
        """
        :param ninp: the number of features in input target word embeddings
        :param dec_nhid: the number of features in decoder hidden states, the same as that of the hidden states proposal
        :param enc_ncontext: the number of features in context vectors
        :param natt: the number of features in intermediate alignment vectors of attention layer
        :param dec_noutput: the number of features in decoder's computed output
        :param dec_out_dropout: the dropout rate of decoder's computed output
        """
        super(Decoder, self).__init__()
        self.gru1 = nn.GRUCell(ninp, dec_nhid)
        self.gru2 = nn.GRUCell(enc_ncontext, dec_nhid)
        self.enc_attn = Attention(dec_nhid, enc_ncontext, natt)
        self.e2o = nn.Linear(ninp, dec_nout)
        self.h2o = nn.Linear(dec_nhid, dec_nout)
        self.c2o = nn.Linear(enc_ncontext, dec_nout)
        self.readout_dp = nn.Dropout(dec_out_dropout)

    def forward(self, prev_trg_emb, prev_dec_hidden, enc_mask, enc_hiddens):
        """
        :param prev_trg_emb: the word embedding of previous target word y(i-1)
        :param prev_dec_hidden: the previous hidden state in decoder (i.e. s(i-1))
        :param enc_mask: the mask matrix composed of 0s and 1s that is used to retrieve valid sentence lengths in a batch
        :param enc_hiddens: the batch of encoder hidden states
        :return: dec_output: the batch of decoder's outputs at current timestamp i for all sentences
                  dec_hidden: the batch of decoder hidden states s(i) at current timestamp i for all sentences
        """
        dec_hidden_proposal = self.gru1(prev_trg_emb, prev_dec_hidden)  # 将s(i-1)先与y(i-1)融合，得到s'(i-1)作为decoder在i-1时刻的hidden state proposal
                                                          # (hidden state proposal) s'(i)的维度与(hidden state) s(i)相同
        attn_enc = self.enc_attn(dec_hidden_proposal, enc_mask, enc_hiddens)  # 获得Attention层输出的加权后的context vector c(i)
        dec_hidden = self.gru2(attn_enc, dec_hidden_proposal)  # 将c(i)和s'(i-1)再度融合得到当前decoder hidden state s(i)
        dec_output = torch.tanh(self.e2o(prev_trg_emb) + self.h2o(dec_hidden) + self.c2o(attn_enc))  # 输入y(i-1), c(i)与s'(i-1)融合后的结果, c(i),得到dec_output
        dec_output = self.readout_dp(dec_output)  # 对dec_output执行dropout操作
        return dec_output, dec_hidden
