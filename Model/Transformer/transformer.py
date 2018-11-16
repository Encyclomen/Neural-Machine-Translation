import numpy as np
import torch
import torch.nn as nn

from Model.Transformer import constants
from Model.Transformer.decoder import Decoder
from Model.Transformer.encoder import Encoder
from Model.Transformer.sublayer import Multi_Head_Self_Attention, Layer_Normalization
from Model.Transformer.utils import position_encoding_init
from Util.process_data import to_Tensor


class Transformer(nn.Module):
    def __init__(self, d_model, d_q, d_k, d_v, h, enc_N, dec_N, d_inner_hid, dropout, enc_ntok, dec_ntok):
        super(Transformer, self).__init__()
        # model's hyperparameters that can be manually set in command line
        self.d_model = d_model
        self.dq, self.dk, self.dv = d_q, d_k, d_v
        self.h = h
        self.enc_N, self.dec_N = enc_N, dec_N
        self.d_inner_hid = d_inner_hid
        self.dropout = dropout
        # model's hyperparameters that can not be manually set in command line
        self.enc_pad = constants.enc_pad
        self.dec_sos = constants.dec_sos
        self.dec_eos = constants.dec_eos
        self.dec_pad = constants.dec_pad
        self.enc_ntok, self.dec_ntok = enc_ntok, dec_ntok
        # model's submodules
        self.encoder = Encoder(d_model, d_q ,d_k, d_v, h, enc_N, d_inner_hid,
                               dropout, enc_ntok)
        self.decoder = Decoder(d_model, d_q ,d_k, d_v, h, dec_N, d_inner_hid,
                               dropout, dec_ntok)
        self.affine = nn.Linear(d_model, dec_ntok)

    '''
        def __randomly_init_param_list(self):
        """ Avoid updating the position encoding """

        enc_layer_norm1_gain_ids = set(map(id, (layer.layer_norm1.gain for layer in self.encoder.layer_stack)))
        enc_layer_norm1_offset_ids = set(map(id, (layer.layer_norm1.offset for layer in self.encoder.layer_stack)))
        enc_layer_norm2_gain_ids = set(map(id, (layer.layer_norm2.gain for layer in self.encoder.layer_stack)))
        enc_layer_norm2_offset_ids = set(map(id, (layer.layer_norm2.offset for layer in self.encoder.layer_stack)))
        dec_layer_norm1_gain_ids = set(map(id, (layer.layer_norm1.gain for layer in self.decoder.layer_stack)))
        dec_layer_norm1_offset_ids = set(map(id, (layer.layer_norm1.offset for layer in self.decoder.layer_stack)))
        dec_layer_norm2_gain_ids = set(map(id, (layer.layer_norm2.gain for layer in self.decoder.layer_stack)))
        dec_layer_norm2_offset_ids = set(map(id, (layer.layer_norm2.offset for layer in self.decoder.layer_stack)))
        dec_layer_norm3_gain_ids = set(map(id, (layer.layer_norm3.gain for layer in self.decoder.layer_stack)))
        dec_layer_norm3_offset_ids = set(map(id, (layer.layer_norm3.offset for layer in self.decoder.layer_stack)))
        predefined_param_ids = enc_layer_norm1_gain_ids | enc_layer_norm1_offset_ids | \
                               enc_layer_norm2_gain_ids | enc_layer_norm2_offset_ids | \
                               dec_layer_norm1_gain_ids | dec_layer_norm1_offset_ids | \
                               dec_layer_norm2_gain_ids | dec_layer_norm2_offset_ids | \
                               dec_layer_norm3_gain_ids | dec_layer_norm3_offset_ids

        return [param for param in self.parameters() if id(param) not in predefined_param_ids]
    '''
    def param_init(self):
        # for param in self.__randomly_init_param_list():
        # param.data.normal_(0, self.d_model ** -0.5)
        for param in self.parameters():
            param.data.normal_(0, self.d_model ** -0.5)
        for param in self.parameters():
            if isinstance(param, torch.nn.Conv1d):
                param.weight.data.xavier_normal_()
            elif isinstance(param, Multi_Head_Self_Attention):
                param.inp2q.data.xavier_normal_()
                param.inp2k.data.xavier_normal_()
                param.inp2v.data.xavier_normal_()
            elif isinstance(param, Layer_Normalization):
                param.gain.data.fill_(1)
                param.offset.data.zero_()

    def forward(self, src, src_mask, trg, trg_mask):
        src_seq_lens = src_mask.sum(1)
        trg_seq_lens = trg_mask.sum(1)
        # --------------------------------------------------------------------------------------------------------------
        enc_output = self.encoder(src, src_seq_lens)
        dec_output = self.decoder(trg, trg_seq_lens, enc_output, src_seq_lens)
        y_prob = self.affine(dec_output)

        return y_prob

    def greedysearch(self, src, src_mask, max_len=None, min_len=None, cuda=False):
        max_len = src.size(1) * 3 if max_len is None else max_len
        min_len = src.size(1) / 2 if min_len is None else min_len
        src_seq_lens = src_mask.sum(1)
        enc_emb = self.encoder.src_emb(src)
        enc_pos_emb = position_encoding_init(enc_emb.size(1), enc_emb.size(2), cuda=enc_emb.is_cuda)
        enc_input = enc_emb + enc_pos_emb # add positional embeddings to input embeddings
        enc_output = enc_input
        for enc_layer in self.encoder.layer_stack:
            enc_output = enc_layer(enc_output, src_seq_lens)
        sentence = [self.dec_sos]
        output = to_Tensor([sentence], tensor_type=torch.LongTensor, cuda=cuda)
        for k in range(max_len):
            dec_emb = self.decoder.dec_emb(output)
            dec_pos_emb = position_encoding_init(dec_emb.size(1), dec_emb.size(2), cuda=dec_emb.is_cuda)
            dec_input = dec_emb + dec_pos_emb
            dec_layer_output = dec_input
            for dec_layer in self.decoder.layer_stack:
                multi_head_input = dec_layer_output.unsqueeze(1).expand(-1, self.h, -1, -1)
                multi_head_q, multi_head_k, multi_head_v = multi_head_input, multi_head_input, multi_head_input
                self_attn_output = dec_layer.masked_multi_head_attn(multi_head_q, multi_head_k, multi_head_v)
                residual_output1 = dec_layer.layer_norm1(dec_layer_output + self_attn_output)  # Add(residual connection) & Norm

                multi_head_q = residual_output1.unsqueeze(1).expand(-1, self.h, -1, -1)
                multi_head_input = enc_output.unsqueeze(1).expand(-1, self.h, -1, -1)
                multi_head_k, multi_head_v = multi_head_input, multi_head_input
                dec_enc_attn_output = dec_layer.multi_head_attn(multi_head_q, multi_head_k, multi_head_v)
                residual_output2 = dec_layer.layer_norm2(residual_output1 + dec_enc_attn_output)  # Add(residual connection) & Norm
                # Position-wise Feedforward Sublayer
                feedforwad_output = dec_layer.pos_ffn(residual_output2)
                dec_layer_output = dec_layer.layer_norm3(residual_output2 + feedforwad_output)  # Add(residual connection) & Norm
            dec_output = dec_layer_output
            y_prob = self.affine(dec_output)
            cur_word_idx = int(y_prob[0][k].argmax())
            #if cur_word_idx is self.dec_eos:
                #break
            sentence.append(cur_word_idx)
            output = to_Tensor([sentence], tensor_type=torch.LongTensor, cuda=cuda)

        return sentence
