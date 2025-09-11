import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

import math

class DSW_embedding(nn.Module):
    def __init__(self, seg_len, d_model):
        super(DSW_embedding, self).__init__()
        self.seg_len = seg_len

        self.linear = nn.Linear(seg_len, d_model)

    def forward(self, x, x_mark):
        if x_mark is not None:
            x_seq = torch.cat([x_mark, x], -1)
            batch, ts_len, ts_dim = x_seq.shape
            x_segment = rearrange(x_seq, 'b (seg_num seg_len) d -> (b d seg_num) seg_len', seg_len = self.seg_len)
            x_embed = self.linear(x_segment)
            output = rearrange(x_embed, '(b d seg_num) d_model -> b d seg_num d_model', b = batch, d = ts_dim)
        else:
            batch, ts_len, ts_dim = x.shape
            x_segment = rearrange(x, 'b (seg_num seg_len) d -> (b d seg_num) seg_len', seg_len = self.seg_len)
            x_embed = self.linear(x_segment)
            output = rearrange(x_embed, '(b d seg_num) d_model -> b d seg_num d_model', b = batch, d = ts_dim)

        return output


class DSW_embedding2(nn.Module):
    def __init__(self, seg_len, d_model):
        super(DSW_embedding2, self).__init__()
        self.seg_len = seg_len

        self.linear = nn.Linear(seg_len, d_model)

    def forward(self, x, x_mark):
        batch1, ts_len1, ts_dim1 = x.shape
        batch2, ts_len2, ts_dim2 = x_mark.shape
        x_mark = rearrange(x_mark, 'b (seg_num seg_len) d -> (b d seg_num) seg_len', seg_len = self.seg_len)
        x_segment = rearrange(x, 'b (seg_num seg_len) d -> (b d seg_num) seg_len', seg_len = self.seg_len)
        x_embed = self.linear(x_segment)
        time_embed = self.linear(x_mark)
        x_embed = rearrange(x_embed, '(b d seg_num) d_model -> b d seg_num d_model', b = batch1, d = ts_dim1)
        time_embed = rearrange(time_embed, '(b d seg_num) d_model -> b d seg_num d_model', b = batch2, d = ts_dim2)
        output = torch.cat([time_embed, x_embed], 1)
        
        return output