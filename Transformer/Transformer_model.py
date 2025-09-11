import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, device, max_len=5000):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0).transpose(1, 0).to(device)
 
    def forward(self, x):
        return self.pe[:x.size(0), :]
 
class Transformer_Model(nn.Module):
    def __init__(self, input_size, pred_length, d_model, device, nhead, 
                 num_encoder_layers, num_decoder_layers, dim_feedforward, output_size, dropout=0.1):
        super(Transformer_Model, self).__init__()
        self.pred_length = pred_length
 
        self.value_encoding = nn.Linear(input_size, d_model)
 
        self.positional_encoding = PositionalEncoding(d_model, device)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=dim_feedforward,
                                          dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(d_model, output_size)
 
    def forward(self, src,  encoder_input_mark,  tgt, decoder_input_mark, tgt_mask=None):
        if tgt_mask is None:
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1))
 
        src = self.value_encoding(src) + self.positional_encoding(src)
        tgt = self.value_encoding(tgt) + self.positional_encoding(tgt)
 
        output = self.transformer(src, tgt, tgt_mask=tgt_mask)

        output = self.fc_out(output)

 
        return output[:,-self.pred_length:,:]