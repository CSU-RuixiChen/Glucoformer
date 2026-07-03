import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

class SSE_embedding(nn.Module):
    def __init__(self, seg_len, d_model):
        super(SSE_embedding, self).__init__()
        self.seg_len = seg_len

        # [Glucoformer Upgrade] segment embedding now mixes raw values with segment statistics.
        self.value_projection = nn.Linear(seg_len, d_model)
        self.stat_projection = nn.Linear(2, d_model)
        self.fusion_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, x, x_mark):
        if x_mark is not None:
            x_seq = torch.cat([x_mark, x], -1)
            batch, ts_len, ts_dim = x_seq.shape
            x_segment = rearrange(x_seq, 'b (seg_num seg_len) d -> (b d seg_num) seg_len', seg_len = self.seg_len)
        else:
            batch, ts_len, ts_dim = x.shape
            x_segment = rearrange(x, 'b (seg_num seg_len) d -> (b d seg_num) seg_len', seg_len = self.seg_len)

        segment_value = self.value_projection(x_segment)
        segment_mean = x_segment.mean(dim=-1, keepdim=True)
        segment_std = x_segment.std(dim=-1, keepdim=True, unbiased=False)
        segment_stats = torch.cat([segment_mean, segment_std], dim=-1)
        segment_stats = self.stat_projection(segment_stats)

        gate = self.fusion_gate(torch.cat([segment_value, segment_stats], dim=-1))
        x_embed = self.output_norm(segment_value + gate * segment_stats)
        output = rearrange(x_embed, '(b d seg_num) d_model -> b d seg_num d_model', b = batch, d = ts_dim)

        return output


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