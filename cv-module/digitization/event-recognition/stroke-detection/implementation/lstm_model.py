import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout_rate=0.1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        lstm_out, _ = self.lstm(x)  # lstm_out: (batch_size, seq_len, hidden_size)
        # Take the output from the last time step
        lstm_out = lstm_out[:, -1, :]  # shape: (batch_size, hidden_size)
        lstm_out = self.dropout(lstm_out)
        output = self.fc(lstm_out)
        return output
