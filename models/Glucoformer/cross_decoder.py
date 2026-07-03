import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from models.Glucoformer.attn import FullAttention, AttentionLayer, TwoStageAttentionLayer, ETSA_Layer

class DecoderLayer(nn.Module):
    '''
    The decoder layer of Crossformer, each layer will make a prediction at its scale
    '''
    def __init__(self, attn1, attn2, attn3, seg_len, d_model, n_heads, d_ff=None, dropout=0.1, out_seg_num = 10, factor = 10, use_decoder_self_attn = False):
        super(DecoderLayer, self).__init__()

        self.use_decoder_self_attn = use_decoder_self_attn
        if self.use_decoder_self_attn:
            self.self_attention = ETSA_Layer(attn1, attn2, out_seg_num, factor, d_model, n_heads, \
                                    d_ff, dropout)
        else:
            dummy = ETSA_Layer(attn1, attn2, out_seg_num, factor, d_model, n_heads, \
                                d_ff, dropout)
            del dummy
        
        # self.self_attention = ETSA_Layer(attn1, attn2, out_seg_num, factor, d_model, n_heads, \
        #                             d_ff, dropout)
                
        self.cross_attention = AttentionLayer(attn3, d_model, n_heads, dropout = dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_model),
                                nn.GELU(),
                                nn.Linear(d_model, d_model))
        self.linear_pred = nn.Linear(d_model, seg_len)


    def forward(self, x, cross):
        '''
        x: the output of last decoder layer
        cross: the output of the corresponding encoder layer
        '''

        batch = x.shape[0]
        if self.use_decoder_self_attn:
            x = self.self_attention(x)
        x = rearrange(x, 'b ts_d out_seg_num d_model -> (b ts_d) out_seg_num d_model')
        
        cross = rearrange(cross, 'b ts_d in_seg_num d_model -> (b ts_d) in_seg_num d_model')
        tmp, attn = self.cross_attention(
            x, cross, cross,
        )
        x = x + self.dropout(tmp)
        y = x = self.norm1(x)
        y = self.MLP1(y)
        dec_output = self.norm2(x+y)
        
        dec_output = rearrange(dec_output, '(b ts_d) seg_dec_num d_model -> b ts_d seg_dec_num d_model', b = batch)
        layer_predict = self.linear_pred(dec_output)
        layer_predict = rearrange(layer_predict, 'b out_d seg_num seg_len -> b (out_d seg_num) seg_len')

        return dec_output, layer_predict, attn

class Decoder(nn.Module):
    '''
    The decoder of Crossformer, making the final prediction by adding up predictions at each scale
    '''
    def __init__(self, attn1, attn2, attn3, seg_len, d_layers, d_model, n_heads, d_ff, dropout,\
                output_attention=False, out_seg_num = 10, factor=10, use_decoder_self_attn = False):
        super(Decoder, self).__init__()

        self.output_attention = output_attention
        self.decode_layers = nn.ModuleList()
        for i in range(d_layers):
            self.decode_layers.append(DecoderLayer(attn1, attn2, attn3, seg_len, d_model, n_heads, d_ff, dropout, \
                                        out_seg_num, factor, use_decoder_self_attn))

    def forward(self, x, cross):
        final_predict = None
        i = 0
        attn_list = []
        ts_d = x.shape[1]
        for layer in self.decode_layers:
            cross_enc = cross[i]
            x, layer_predict, attn = layer(x,  cross_enc)
            if final_predict is None:
                final_predict = layer_predict
            else:
                final_predict = final_predict + layer_predict
            i += 1
            attn_list.append(attn)
            
        # for idx, a in enumerate(attn_list):
        #     if a is not None:
        #         print(f"Decoder Layer {idx} cross-attention shape: {a.shape}")
                
        final_predict = rearrange(final_predict, 'b (out_d seg_num) seg_len -> b (seg_num seg_len) out_d', out_d = ts_d)

        return final_predict, attn_list

