import torch
import torch.nn as nn

class LSTM_Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, forecast_window, output_size, dropout=0.2):
        super(LSTM_Model, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.forecast_window = forecast_window


        # LSTM 层
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=dropout
        )  # 在 LSTM 中启用 dropout（仅在 num_layers > 1 时生效）

        # 全连接层
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, encoder_input_mark, decoder_input ,decoder_input_mark):

        # LSTM 前向传播
        out, _ = self.lstm(x)

        out = self.fc(out[:, -self.forecast_window:, :])  # 全连接层

        return out
    







