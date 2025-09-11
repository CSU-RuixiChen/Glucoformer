import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from Glucoformer.Glucoformer_models.attn import FullAttention, AttentionLayer, TwoStageAttentionLayer
from math import ceil

# class ConvLayer(nn.Module):
#     def __init__(self, d_model):
#         super(ConvLayer, self).__init__()
#         padding = 1 if torch.__version__>='1.5.0' else 2
#         self.conv0 = nn.Conv1d(in_channels=d_model,
#                                   out_channels=d_model,
#                                   kernel_size=3,
#                                   padding=padding)
#         self.conv1 = nn.Conv1d(d_model, d_model, kernel_size=3, dilation=1, padding=1)
#         self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, dilation=2, padding=2)
#         self.conv3 = nn.Conv1d(d_model, d_model, kernel_size=3, dilation=3, padding=3)

#         self.norm1 = nn.BatchNorm1d(d_model)
#         # self.activation = nn.ELU()
#         self.activation = nn.ReLU()
#         self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
#         self.norm2 = torch.nn.LayerNorm(d_model)
#         self.SegMerging = SegMerging(d_model)

#     def forward(self, x):
#         batch_size, ts_d, seg_num, d_model = x.shape
#         x = rearrange(x, 'b ts_d seg_num d_model -> (b ts_d) seg_num d_model')
        
#         x = self.activation(self.conv1(x.permute(0, 2, 1)))
#         x = self.activation(self.conv2(x))
#         x = self.activation(self.conv3(x))

#         # x = self.activation(self.conv0(x.permute(0, 2, 1)))
        
#         x = self.norm1(x)

#         x = self.maxPool(x).permute(0, 2, 1)
#         x = self.norm2(x)
#         x = rearrange(x, '(b ts_d) seg_num d_model -> b ts_d seg_num d_model', b = batch_size)

#         # x = rearrange(x, '(b ts_d) d_model seg_num -> b ts_d seg_num d_model', b = batch_size)
#         # x = self.SegMerging(x)
    
#         return x


class SegMerging(nn.Module):
    '''
    Segment Merging Layer.
    The adjacent `win_size' segments in each dimension will be merged into one segment to
    get representation of a coarser scale
    we set win_size = 2 in our paper
    '''
    def __init__(self, d_model, win_size=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.d_model = d_model
        self.win_size = win_size
        self.norm = norm_layer(win_size * d_model)
        self.linear_trans = nn.Linear(win_size * d_model, d_model)
        

    def forward(self, x):
        """
        x: B, ts_d, L, d_model
        """
        batch_size, ts_d, seg_num, d_model = x.shape
        pad_num = seg_num % self.win_size
        if pad_num != 0: 
            pad_num = self.win_size - pad_num
            x = torch.cat((x, x[:, :, -pad_num:, :]), dim = -2)

        seg_to_merge = []
        for i in range(self.win_size):
            seg_to_merge.append(x[:, :, i::self.win_size, :])
        x = torch.cat(seg_to_merge, -1)  # [B, ts_d, seg_num/win_size, win_size*d_model]

        x = self.norm(x)
        x = self.linear_trans(x)


        return x


class scale_block(nn.Module):
    '''
    We can use one segment merging layer followed by multiple TSA layers in each scale
    the parameter `depth' determines the number of TSA layers used in each scale
    We set depth = 1 in the paper
    '''
    def __init__(self, attn1, attn2, win_size, d_model, n_heads, d_ff, depth, dropout, \
                    seg_num = 10, factor=10):
        super(scale_block, self).__init__()

        if (win_size > 1):
            self.merge_layer = SegMerging(d_model, win_size, nn.LayerNorm)
        else:
            self.merge_layer = None
        
        self.encode_layers = nn.ModuleList()
        
        for i in range(depth):
            self.encode_layers.append(TwoStageAttentionLayer(attn1, attn2, seg_num, factor, d_model, n_heads, \
                                                        d_ff, dropout))

        # if (win_size > 1):
        #     self.conv = ConvLayer(d_model)
        # else:
        #     self.conv = None   
    
    def forward(self, x):
        _, ts_dim, _, _ = x.shape

        if self.merge_layer is not None:
            x = self.merge_layer(x)
        
        for layer in self.encode_layers:
            x = layer(x)
        
        # if self.conv is not None:
        #     x = self.conv(x)
        
        return x

class Encoder(nn.Module):
    '''
    The Encoder of Crossformer.
    '''
    def __init__(self, attn1, attn2, e_blocks, win_size, d_model, n_heads, d_ff, block_depth, dropout,
                in_seg_num = 10, factor=10):
        super(Encoder, self).__init__()
        self.encode_blocks = nn.ModuleList()

        self.encode_blocks.append(scale_block(attn1, attn2, 1, d_model, n_heads, d_ff, block_depth, dropout,\
                                            in_seg_num, factor))
        for i in range(1, e_blocks):
            self.encode_blocks.append(scale_block(attn1, attn2, win_size, d_model, n_heads, d_ff, block_depth, dropout,\
                                            ceil(in_seg_num/win_size**(i)), factor))

    def forward(self, x):
        encode_x = []
        encode_x.append(x)
        
        for block in self.encode_blocks:
            x = block(x)
            encode_x.append(x)

        return encode_x