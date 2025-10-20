import torch.nn as nn

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, forecast_window, output_size):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.forecast_window = forecast_window
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, encoder_input_mark, decoder_input ,decoder_input_mark):
        
        out, _ = self.gru(x)

        out = self.fc(out[:, -self.forecast_window:, :])
        return out