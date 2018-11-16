import math

import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F


class Multi_Head_Self_Attention(nn.Module):
    def __init__(self, d_model, d_q, d_k, d_v, h, attn_dropout):
        super(Multi_Head_Self_Attention, self).__init__()
        self.d_model = d_model
        self.d_q = d_q
        self.d_k = d_k
        self.d_v = d_v
        self.h = h
        # model's submodules
        self.inp2q = Parameter(torch.Tensor(h, d_model, d_q), requires_grad=True)
        self.inp2k = Parameter(torch.Tensor(h, d_model, d_k), requires_grad=True)
        self.inp2v = Parameter(torch.Tensor(h, d_model, d_v), requires_grad=True)
        self.attn_dp = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(d_model, d_model)

    def multi_head_scaled_dot_product_attention(self, multi_head_q, multi_head_k, attn_mask=None):
        e = multi_head_q.matmul(multi_head_k.permute(0, 1, 3, 2)) / math.sqrt(
            self.d_model)  # e.size()==(batch_size, h, seq_len, seq_len)
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).expand(-1, self.h, -1, -1)
            e = e.masked_fill_(1 - attn_mask, float('-inf'))
            multi_head_attn = F.softmax(e, 3)
        else:
            multi_head_attn = F.softmax(e, 3)
        multi_head_attn_out = self.attn_dp(multi_head_attn)  # dropout on attention

        return multi_head_attn_out

    def forward(self, multi_head_q, multi_head_k, multi_head_v, attn_mask=None):
        # multi_head_$W$.size()==(batch_size, h, seq_len, d$), where $ -> {q, k, v}-------------------------------------
        multi_head_qWq = multi_head_q.matmul(self.inp2q)
        multi_head_kWk = multi_head_k.matmul(self.inp2k)
        multi_head_vWv = multi_head_v.matmul(self.inp2v)
        # --------------------------------------------------------------------------------------------------------------
        multi_head_attn = self.multi_head_scaled_dot_product_attention(multi_head_qWq, multi_head_kWk,
                                                                       attn_mask=attn_mask)
        multi_head_context = multi_head_attn.matmul(multi_head_vWv)
        # concat multi-head vectors back to the original shape, multi_head_concat.size()==(batch_size, seq_len, h*dv)
        multi_head_concat = torch.cat(multi_head_context.split(1, dim=1), dim=-1).squeeze(1)
        multi_head_attn_output = self.proj(multi_head_concat)

        return multi_head_attn_output


class Layer_Normalization(nn.Module):
    def __init__(self, d_hid, epsilon=1e-8):
        super(Layer_Normalization, self).__init__()
        self.d_hid = d_hid
        self.epsilon = epsilon
        # model's trainable parameters
        self.gain = Parameter(torch.ones(d_hid), requires_grad=True)  # TODO
        self.offset = Parameter(torch.zeros(d_hid), requires_grad=True)  # TODO

    def forward(self, input):
        mean, std_deviation = input.mean(-1, keepdim=True), input.std(-1, keepdim=True, unbiased=False)
        normalized = (input - mean) / (std_deviation + self.epsilon)
        output = self.gain * normalized + self.offset

        return output


class Positionwise_Feedforward(nn.Module):
    def __init__(self, d_model, d_inner_hid, dropout=0.1):
        super(Positionwise_Feedforward, self).__init__()
        self.conv1 = nn.Conv1d(d_model, d_inner_hid, kernel_size=1)
        self.conv2 = nn.Conv1d(d_inner_hid, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, input):
        hidden = self.relu(self.conv1(input.transpose(1, 2)))
        output = self.dropout(self.conv2(hidden).transpose(2, 1))

        return output