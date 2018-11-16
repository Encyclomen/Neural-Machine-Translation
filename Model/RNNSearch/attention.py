import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """Attention Mechanism"""
    def __init__(self, dec_nhid, enc_nhid, natt):
        """
        :param dec_nhid: the dimension of unidirectional decoder hidden states
        :param enc_nhid: the number of features in encoder's bidirectional hidden states
        :param natt: the number of features in intermediate alignment vectors
        """
        super(Attention, self).__init__()
        self.s2s = nn.Linear(enc_nhid, natt)
        self.h2s = nn.Linear(dec_nhid, natt)
        self.a2o = nn.Linear(natt, 1)

    def forward(self, dec_hidden, mask, enc_hiddens):
        """
        :param dec_hidden: the previous hidden state in decoder (i.e. s(i-1))
        :param mask: mask matrix composed of 0s and 1s that is used to retrieve valid sentence lengths in a batch
        :param enc_hiddens: batch of encoder hidden states
        :return: context: the context vector computed from the weighted sum of encoder hidden states
        """
        shape = enc_hiddens.size()  # context.size() == [batch_size, longest sentence length in a batch, nenc_hid]
        attn_h = self.s2s(enc_hiddens.view(-1, shape[2]))  # 先用enc_hiddens计算attn_h=U.*h
        attn_h = attn_h.view(shape[0], shape[1], -1)  # 复原attn_h的批量表示，使attn_h.size() == [shape[0], shape[1], natt]
        attn_h += self.h2s(dec_hidden).unsqueeze(1).expand_as(attn_h)  # 计算U.*h + W.*s(i−1)
        logit = self.a2o(torch.tanh(attn_h)).view(shape[0], shape[1])  # 将tanh(U.*h + W.*s(i−1))输入感知层，
                                                                   # 得到当前隐层s(i)对于所有h(j)的打分，
                                                                   # 即e(i,1),e(i,2), ..., e(i,n)
        if mask.any():
            logit.data.masked_fill_(1 - mask, -float('inf'))  # 将batch中句子的无效的padded element得分值置为负无穷，
                                                              # 用以后续进行softmax归一化。
        softmax = F.softmax(logit, dim=1)  # 进行softmax归一化，得到注意力权重
        enc_context = torch.bmm(softmax.unsqueeze(1), enc_hiddens).squeeze(1)  # 利用上面算出的权重计算encoder hidden states的
                                                                               # 加权和，作为Attention层获得的context vector
        return enc_context