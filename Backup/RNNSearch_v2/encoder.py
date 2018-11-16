import torch
import torch.nn as nn
import torch.nn.functional as F

from Model.RNNSearch.MyGRU import MyGRU


class Encoder(nn.Module):
    """Encoder that encodes the input sequence with Bi-GRU"""
    def __init__(self, ninp, enc_nhid, ntok, padding_idx, emb_dropout, hid_dropout):
        """
        :param ninp: the number of features in input source word embeddings
        :param enc_nhid: the number of features in encoder hidden states
        :param ntok: the size of vocabulary
        :param padding_idx: the index used to represent position to be padded
        :param emb_dropout: the dropout rate of input embeddings
        :param hid_dropout: the dropout rate of computed hidden states
        """
        super(Encoder, self).__init__()
        self.nhid = enc_nhid
        self.emb = nn.Embedding(ntok, ninp, padding_idx=padding_idx)
        self.bi_gru = MyGRU(ninp, enc_nhid, batch_first=True, bidirectional=True)
        #self.bi_gru = nn.GRU(ninp, enc_nhid, 1, batch_first=True, bidirectional=True)
        self.enc_emb_dp = nn.Dropout(emb_dropout)
        self.enc_hid_dp = nn.Dropout(hid_dropout)

    def init_hidden(self, batch_size):
        """
        :param batch_size: the number of sentences contained in a batch
        :return h0: the initialized bidirectional encoder hidden states, not yet cancatenated
        """
        weight = next(self.parameters()).data
        h0 = weight.new(2, batch_size, self.nhid).zero_()  # 初始化batch_size个双向encoder hidden states

        return h0

    def forward(self, input, mask):
        """
        :param input: batch of word indices sequences, whose shape is (batch_size, seq_len)
        :param mask: mask matrix composed of 0s and 1s that is used to retrieve valid length of every sentence in a batch
        :return enc_hiddens:  batch of all encoder hidden states
                last_enc_hidden :batch of the last concatenated encoder's bidirectional hidden states.
        """
        hidden = self.init_hidden(input.size(0))  # 告诉hidden state输入批量的batch_size,并返回初始化了的hidden state
        input_emb = self.enc_emb_dp(self.emb(input))  # 接收input的index序列，查找lookup table获得相应embedding，并进行对embedding的dropout操作
        enc_hiddens = self.bi_gru(input_emb, hidden, mask)
        enc_hiddens = self.enc_hid_dp(enc_hiddens)  # 对于获得的隐层序列执行drop_out操作

        return enc_hiddens