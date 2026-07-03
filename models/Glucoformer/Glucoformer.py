import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from models.Glucoformer.cross_encoder import Encoder
from models.Glucoformer.cross_decoder import Decoder
from models.Glucoformer.attn import FullAttention, MSPatchLinearAttention, AttentionLayer, TwoStageAttentionLayer
from models.Glucoformer.cross_embed import *

from math import ceil

class Glucoformer(nn.Module):
    def __init__(self, data_dim, in_len, out_len, seg_len, output_size, win_size = 2,
                factor=4, d_model=512, d_ff = 1024, n_heads=8, e_layers=3, 
                dropout=0.0, output_attention=False, baseline = False, device=torch.device('cuda:0'),
                use_sse = True, use_etsa = True, use_decoder_self_attn = False):
        super(Glucoformer, self).__init__()
        self.data_dim = data_dim
        self.in_len = in_len
        self.out_len = out_len
        self.seg_len = seg_len
        self.merge_win = win_size
        self.output_attention = output_attention
        self.baseline = baseline
        self.device = device

        # The padding operation to handle invisible sgemnet length
        self.pad_in_len = ceil(1.0 * in_len / seg_len) * seg_len
        self.pad_out_len = ceil(1.0 * out_len / seg_len) * seg_len
        self.in_len_add = self.pad_in_len - self.in_len

        if use_sse:
            # [Glucoformer2 Upgrade] embedding now includes segment statistics before TSA.
            self.enc_value_embedding = SSE_embedding(seg_len, d_model)
        else:
            self.enc_value_embedding = DSW_embedding(seg_len, d_model)
        self.enc_pos_embedding = nn.Parameter(torch.randn(1, data_dim, (self.pad_in_len // seg_len), d_model))
        self.pre_norm = nn.LayerNorm(d_model)

        attn1 = FullAttention(scale=True, attention_dropout=dropout)
        attn2 = MSPatchLinearAttention(attention_dropout=dropout)

        # Encoder
        self.encoder = Encoder(attn2, attn1, e_layers, win_size, d_model, n_heads, d_ff, block_depth = 1, \
                                    dropout = dropout,in_seg_num = (self.pad_in_len // seg_len), factor = factor, use_etsa = use_etsa)
        
        # Decoder
        self.dec_pos_embedding = nn.Parameter(torch.randn(1, data_dim, (self.pad_out_len // seg_len), d_model))
        self.decoder = Decoder(attn2, attn1, attn1, seg_len, e_layers + 1, d_model, n_heads, d_ff, dropout, \
                                    out_seg_num = (self.pad_out_len // seg_len), factor = factor, use_decoder_self_attn = use_decoder_self_attn)
        self.linear_pred = nn.Linear(data_dim, output_size)
        
    def forward(self, x_seq, encoder_input_mark, decoder_input ,decoder_input_mark):
        batch_size = x_seq.shape[0]

        if (self.in_len_add != 0):
            x_seq = torch.cat((x_seq[:, :1, :].expand(-1, self.in_len_add, -1), x_seq), dim = 1)
        x_seq = self.enc_value_embedding(x_seq, None)
        x_seq = x_seq + self.enc_pos_embedding
        x_seq = self.pre_norm(x_seq)
        
        enc_out = self.encoder(x_seq)

        dec_in = repeat(self.dec_pos_embedding, 'b ts_d l d -> (repeat b) ts_d l d', repeat = batch_size)
        predict_y, attn_weights = self.decoder(dec_in, enc_out)
        predict_y = self.linear_pred(predict_y)

        if self.output_attention:
            return predict_y[:, :self.out_len, :], attn_weights
        else:
            return predict_y[:, :self.out_len, :]