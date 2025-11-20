import torch
import torch.nn as nn
import torch.optim as optim

class CNNLSTMModel(nn.Module):
    # Add z_dim and seq_len to init
    def __init__(self, input_dim, cnn_channels=32, hidden_dim=128, kernel_size=3, z_dim=16, seq_len=50):
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.cnn_channels = cnn_channels
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim

        # CNN layer to process the sequence
        self.conv = nn.Conv1d(input_dim, cnn_channels, kernel_size=kernel_size, padding=kernel_size // 2) # Added padding
        self.relu = nn.ReLU() # Added ReLU
        self.ln_cnn = nn.LayerNorm(cnn_channels) # Layer Norm after CNN

        # LSTM layer to process the output of CNN
        # Input to LSTM will be (batch, seq_len, cnn_channels)
        self.lstm = nn.LSTM(cnn_channels, hidden_dim, batch_first=True)
        self.ln_lstm = nn.LayerNorm(hidden_dim) # Layer Norm after LSTM

        # Fully connected layer(s) to combine LSTM output and latent variable
        # Input to FC will be (hidden_dim + z_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim + z_dim, hidden_dim // 2), # Combine LSTM output and z
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1) # Output action
        )

        # Initialize weights (optional but good practice)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    # Modify forward to accept sequence input and latent variable
    def forward(self, x_seq, z):
        # x_seq: (batch, seq_len, input_dim)
        # z: (batch, z_dim)

        # Ensure x_seq is float and handle potential NaNs
        x_seq = torch.nan_to_num(x_seq.float())

        # Permute for Conv1d: (batch, input_dim, seq_len)
        x_conv_in = x_seq.permute(0, 2, 1)

        # Pass through CNN
        h_conv = self.conv(x_conv_in) # (batch, cnn_channels, seq_len)
        h_conv = self.relu(h_conv)
        h_conv = self.ln_cnn(h_conv.permute(0, 2, 1)).permute(0, 2, 1) # Layer norm expects (batch, seq_len, channels) then permute back

        # Permute back for LSTM: (batch, seq_len, cnn_channels)
        h_lstm_in = h_conv.permute(0, 2, 1)

        # Pass through LSTM
        out_lstm, _ = self.lstm(h_lstm_in) # out_lstm: (batch, seq_len, hidden_dim)
        out_lstm = self.ln_lstm(out_lstm)

        # Get the output of the last time step from LSTM
        last_lstm_out = out_lstm[:, -1, :] # (batch, hidden_dim)

        # Concatenate LSTM output with latent variable z
        combined_features = torch.cat([last_lstm_out, z], dim=-1) # (batch, hidden_dim + z_dim)

        # Pass through FC layers to get action
        action = self.fc(combined_features) # (batch, 1) or (batch,) depending on FC output

        return action.squeeze(-1) # Ensure output is (batch,) or (batch, 1)

def build_optimizer(model, optimizer_name="Adam", lr=1e-3, weight_decay=0.0):
    """
    Factory function to build an optimizer based on a string name.
    """
    if optimizer_name.lower() == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name.lower() == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name.lower() == "sgd":
        return optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
