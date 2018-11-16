from torch import nn

from Model.Transformer import constants
from Model.Transformer.sublayer import Multi_Head_Self_Attention, Layer_Normalization, Positionwise_Feedforward
from Model.Transformer.utils import position_encoding_init, get_attn_padding_mask, get_attn_subsequent_mask


class Decoder_Layer(nn.Module):
    def __init__(self, d_model, d_q, d_k, d_v, h, d_inner_hid, dropout):
        super(Decoder_Layer, self).__init__()
        self.h = h
        # model's submodules
        self.masked_multi_head_attn = Multi_Head_Self_Attention(d_model, d_q, d_k, d_v, h, dropout)
        self.residual_dp1 = nn.Dropout(dropout)
        self.layer_norm1 = Layer_Normalization(d_model)
        self.multi_head_attn = Multi_Head_Self_Attention(d_model, d_q, d_k, d_v, h, dropout)
        self.residual_dp2 = nn.Dropout(dropout)
        self.layer_norm2 = Layer_Normalization(d_model)
        self.pos_ffn = Positionwise_Feedforward(d_model, d_inner_hid, dropout=dropout)
        self.residual_dp3 = nn.Dropout(dropout)
        self.layer_norm3 = Layer_Normalization(d_model)

    def forward(self, input, trg_seq_lens, enc_output, src_seq_lens):
        # Masked Multi-head Attention Sublayer
        masked_multi_head_input = input.unsqueeze(1).expand(-1, self.h, -1, -1)
        multi_head_q, multi_head_k, multi_head_v = masked_multi_head_input, masked_multi_head_input, masked_multi_head_input
        dec_slf_attn_mask = get_attn_subsequent_mask(trg_seq_lens, cuda=input.is_cuda)
        masked_multi_head_attn_output = self.masked_multi_head_attn(multi_head_q, multi_head_k, multi_head_v, dec_slf_attn_mask)
        masked_multi_head_attn_output = self.residual_dp1(masked_multi_head_attn_output)  # dropout before residual connection
        residual_output1 = self.layer_norm1(input + masked_multi_head_attn_output)  # Add(residual connection) & Norm
        # Multi-head Attention Sublayer
        multi_head_q = residual_output1.unsqueeze(1).expand(-1, self.h, -1, -1)
        multi_head_input = enc_output.unsqueeze(1).expand(-1, self.h, -1, -1)
        multi_head_k, multi_head_v = multi_head_input, multi_head_input
        dec_enc_attn_mask = get_attn_padding_mask(trg_seq_lens, src_seq_lens, cuda=residual_output1.is_cuda)
        multi_head_attn_output = self.multi_head_attn(multi_head_q, multi_head_k, multi_head_v, dec_enc_attn_mask)
        multi_head_attn_output = self.residual_dp2(multi_head_attn_output)  # dropout before residual connection
        residual_output2 = self.layer_norm2(residual_output1 + multi_head_attn_output)  # Add(residual connection) & Norm
        # Position-wise Feedforward Sublayer
        feedforwad_output = self.pos_ffn(residual_output2)
        feedforwad_output = self.residual_dp3(feedforwad_output)  # dropout before residual connection
        output = self.layer_norm3(residual_output2 + feedforwad_output)  # Add(residual connection) & Norm

        return output


class Decoder(nn.Module):
    def __init__(self, d_model, d_q, d_k, d_v, h, N, d_inner_hid, dropout, dec_ntok):
        super(Decoder, self).__init__()
        # model's submodules
        self.dec_emb = nn.Embedding(dec_ntok, d_model, padding_idx=constants.dec_pad)
        self.emb_dp = nn.Dropout(dropout)
        self.layer_stack = nn.ModuleList([
            Decoder_Layer(d_model, d_q, d_k, d_v, h, d_inner_hid, dropout)
            for _ in range(N)])

    def forward(self, trg, trg_seq_lens, enc_output, src_seq_lens):
        dec_emb = self.dec_emb(trg)
        dec_pos_emb = position_encoding_init(dec_emb.size(1), dec_emb.size(2), cuda=dec_emb.is_cuda)
        dec_input = self.emb_dp(dec_emb + dec_pos_emb)
        dec_output = dec_input
        for dec_layer in self.layer_stack:
            dec_output = dec_layer(dec_output, trg_seq_lens, enc_output, src_seq_lens)

        return dec_output