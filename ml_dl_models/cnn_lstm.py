import math
import numpy as np
import torch
import torch.nn as nn

# -------------------------
# CNN-LSTM model
# -------------------------
class CNNLSTMModel(nn.Module):
    def __init__(self, input_dim, cnn_channels=32, lstm_hidden=128, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv1d(input_dim, cnn_channels, kernel_size=kernel_size, padding=1)
        self.ln = nn.LayerNorm(cnn_channels)
        self.lstm = nn.LSTM(cnn_channels, lstm_hidden, batch_first=True)
        self.fc = nn.Linear(lstm_hidden, 1)
        nn.init.xavier_uniform_(self.fc.weight); nn.init.zeros_(self.fc.bias)
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = torch.nan_to_num(x)
        x_t = x.permute(0,2,1)  # -> (batch, input_dim, seq_len)
        h = self.conv(x_t)      # (batch, channels, seq_len)
        h = h.permute(0,2,1)    # (batch, seq_len, channels)
        h = self.ln(h)
        out, _ = self.lstm(h)
        last = out[:, -1, :]
        return self.fc(last).squeeze(-1)
