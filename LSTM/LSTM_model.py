import torch.nn as nn

class LSTM_Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, forecast_window, output_size, dropout=0.2):
        super(LSTM_Model, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.forecast_window = forecast_window

        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=dropout
        )


        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, encoder_input_mark, decoder_input ,decoder_input_mark):

        out, _ = self.lstm(x)

        out = self.fc(out[:, -self.forecast_window:, :])

        return out
    







