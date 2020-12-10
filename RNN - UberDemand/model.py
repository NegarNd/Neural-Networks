import torch.nn as nn


class Predictor(nn.Module):
    CELL_MAP = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}

    def __init__(self, cell_type, hidden_size=32, dropout=0.0, num_layers=1):
        super().__init__()
        self.is_lstm = cell_type.lower() == 'lstm'
        self.cell = self.CELL_MAP[cell_type.lower()](input_size=18,
                                                     hidden_size=hidden_size,
                                                     num_layers=num_layers, bias=True,
                                                     batch_first=True, dropout=dropout,
                                                     bidirectional=False)
        self.fc = nn.Linear(hidden_size, 4)

    def forward(self, x):
        if self.is_lstm:
            h = self.cell(x)[1][0]
        else:
            h = self.cell(x)[1]
        return self.fc(h.squeeze(0))