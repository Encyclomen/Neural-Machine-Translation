from torch import nn

from Model.Transformer import constants
from Model.Transformer.sublayer import Multi_Head_Self_Attention, Layer_Normalization, Positionwise_Feedforward
from Model.Transformer.utils import position_encoding_init, get_attn_padding_mask


class Encoder_Layer(nn.Module):
    def __init__(self, d_model, d_q, d_k, d_v, h, d_inner_hid, dropout):
        super(Encoder_Layer, self).__init__()
        self.h = h
        # model's submodules
        self.multi_head_attn = Multi_Head_Self_Attention(d_model, d_q, d_k, d_v, h, dropout)
        self.residual_dp1 = nn.Dropout(dropout)
        self.layer_norm1 = Layer_Normalization(d_model)
        self.pos_ffn = Positionwise_Feedforward(d_model, d_inner_hid, dropout=dropout)
        self.residual_dp2 = nn.Dropout(dropout)
        self.layer_norm2 = Layer_Normalization(d_model)

    def forward(self, input, src_seq_lens):
        # Multi-head Attention Sublayer
        multi_head_input = input.unsqueeze(1).expand(-1, self.h, -1, -1)
        multi_head_q, multi_head_k, multi_head_v = multi_head_input, multi_head_input, multi_head_input
        enc_slf_attn_mask = get_attn_padding_mask(src_seq_lens, src_seq_lens, cuda=input.is_cuda)
        multi_head_attn_output = self.multi_head_attn(multi_head_q, multi_head_k, multi_head_v, enc_slf_attn_mask)
        multi_head_attn_output = self.residual_dp1(multi_head_attn_output)  # dropout before residual connection
        residual_output1 = self.layer_norm1(input + multi_head_attn_output)  # Add(residual connection) & Norm
        # Position-wise Feedforward Sublayer
        feedforwad_output = self.pos_ffn(residual_output1)
        feedforwad_output = self.residual_dp2(feedforwad_output)  # dropout before residual connection
        output = self.layer_norm2(residual_output1 + feedforwad_output)  # Add(residual connection) & Norm

        return output


class Encoder(nn.Module):
    def __init__(self, d_model, d_q, d_k, d_v, h, N, d_inner_hid, dropout, enc_ntok):
        super(Encoder, self).__init__()
        # model's submodules
        self.src_emb = nn.Embedding(enc_ntok, d_model, padding_idx=constants.enc_pad)
        self.emb_dp = nn.Dropout(dropout)
        self.layer_stack = nn.ModuleList([
            Encoder_Layer(d_model, d_q, d_k, d_v, h, d_inner_hid, dropout)
            for _ in range(N)])

    def forward(self, src, src_seq_lens):
        enc_emb = self.src_emb(src)
        enc_pos_emb = position_encoding_init(enc_emb.size(1), enc_emb.size(2), cuda=enc_emb.is_cuda)
        enc_input = self.emb_dp(enc_emb + enc_pos_emb)  # add positional embeddings to input embeddings
        enc_output = enc_input
        for enc_layer in self.layer_stack:
            enc_output = enc_layer(enc_output, src_seq_lens)

        return enc_output