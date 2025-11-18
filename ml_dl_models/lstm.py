import math
import numpy as np
import torch
import torch.nn as nn

# -------------------------
# LSTM model
# -------------------------
class LSTMPolicy(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=1, num_layers=1, dropout=0.1):
        super(LSTMPolicy, self).__init__()
        
        # LSTM core
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Fully-connected head
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # Initialize weights
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        """
        x: Tensor of shape (batch, input_dim)
        If used as sequence: (batch, seq_len, input_dim)
        Here, we just wrap it for a single timestep per batch.
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, seq_len=1, input_dim)
        
        out, (h, c) = self.lstm(x)   # h: (num_layers, batch, hidden_dim)
        out = self.fc(h[-1])         # last layer’s hidden state
        return out
