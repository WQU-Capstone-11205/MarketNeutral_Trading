import math
import numpy as np
import torch
import torch.nn as nn

# -------------------------
# Transformer model
# -------------------------
class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, hidden_dim=128):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dim_feedforward=hidden_dim)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)
        nn.init.xavier_uniform_(self.fc.weight); nn.init.zeros_(self.fc.bias)
    def forward(self, x):
        x = torch.nan_to_num(x)
        h = self.input_proj(x)
        h = self.transformer(h)
        last = h[:, -1, :]
        return self.fc(last).squeeze(-1)
