import torch
from torch import nn
import torch.nn.functional as F


class MyGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(MyGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        # GRUCell learnable parameters
        self.linear_ir = nn.Linear(input_size, hidden_size, bias=bias)
        self.linear_hr = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.linear_iz = nn.Linear(input_size, hidden_size, bias=bias)
        self.linear_hz = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.linear_in = nn.Linear(input_size, hidden_size, bias=bias)
        self.linear_hn = nn.Linear(hidden_size, hidden_size, bias=bias)

    def forward(self, input, prev_hidden):
        r = F.relu(self.linear_ir(input) + self.linear_hr(prev_hidden))
        z = F.relu(self.linear_iz(input) + self.linear_hz(prev_hidden))
        n = F.tanh(self.linear_in(input) + r * self.linear_hn(prev_hidden))
        hidden = (1 - z) * prev_hidden + z * n

        return hidden


class MyGRU(nn.Module):
    def __init__(self, input_size, hidden_size,
                num_layers=1, bias=True, batch_first=False,
                dropout=0, bidirectional=True):
        super(MyGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size=hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        # GRU learnable parameters
        self.linear_ir = nn.Linear(input_size, hidden_size, bias=bias)
        self.linear_hr = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.linear_iz = nn.Linear(input_size, hidden_size, bias=bias)
        self.linear_hz = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.linear_in = nn.Linear(input_size, hidden_size, bias=bias)
        self.linear_hn = nn.Linear(hidden_size, hidden_size, bias=bias)

    def forward(self, input, hidden, mask):
        if self.batch_first:
            input = input.transpose(0, 1)
            mask = mask.transpose(0, 1)
        mask = mask.unsqueeze(-1)
        max_seq_len = int(input.size(0))
        batch_size = int(input.size(1))
        all_hiddens = []
        f_hidden = hidden[0]
        for i in range(max_seq_len):
            f_prev_hidden = f_hidden
            r = F.relu(self.linear_ir(input[i]) + self.linear_hr(f_prev_hidden))
            z = F.relu(self.linear_iz(input[i]) + self.linear_hz(f_prev_hidden))
            n = F.tanh(self.linear_in(input[i]) + r * self.linear_hn(f_prev_hidden))
            f_hidden = (1 - z) * f_prev_hidden + z * n
            f_hidden.masked_fill_(1 - mask[i], 0)
            all_hiddens.append(f_hidden)
        if self.bidirectional:
            b_hidden = hidden[1]
            for i in range(max_seq_len-1, -1, -1):
                b_prev_hidden = b_hidden
                r = F.relu(self.linear_ir(input[i]) + self.linear_hr(b_prev_hidden))
                z = F.relu(self.linear_iz(input[i]) + self.linear_hz(b_prev_hidden))
                n = F.tanh(self.linear_in(input[i]) + r * self.linear_hn(b_prev_hidden))
                b_hidden = (1 - z) * b_prev_hidden + z * n
                b_hidden.masked_fill_(1 - mask[i], 0)
                all_hiddens[i] = torch.cat((all_hiddens[i], f_hidden), dim=-1)
            all_hiddens = [hidden.unsqueeze(0) for hidden in all_hiddens]
            all_hiddens = torch.cat(all_hiddens, dim=0)
        if self.batch_first:
            all_hiddens = all_hiddens.transpose(1, 0)

        return all_hiddens
