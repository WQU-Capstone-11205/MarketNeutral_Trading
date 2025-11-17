import torch
import torch.nn as nn

# -------------------------
# VAE latent encoder
# -------------------------
class VAEEncoder(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128, z_dim=16, seq_len=50):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim

        # GRU encoder
        self.gru_enc = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, z_dim)
        self.fc_logvar = nn.Linear(hidden_dim, z_dim)

        # GRU decoder
        self.fc_z = nn.Linear(z_dim, hidden_dim)
        self.gru_dec = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        _, h = self.gru_enc(x)
        h = h[-1]  # last hidden state
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        # changes to make it deterministic
        # eps = torch.randn_like(std) # commented for deterministic
        # the following 4 lines added to make deterministic
        seed = 42
        g = torch.Generator()
        g.manual_seed(seed)
        eps = torch.randn(mu.shape, generator=g, device=mu.device, dtype=mu.dtype)
        return mu + eps * std

    def decode(self, z, seq_len=None):
        seq_len = seq_len or self.seq_len
        h0 = self.fc_z(z).unsqueeze(0)
        dec_input = torch.zeros(z.size(0), seq_len, self.hidden_dim, device=z.device)
        out, _ = self.gru_dec(dec_input, h0)
        x_hat = self.fc_out(out)
        return x_hat

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z, seq_len=x.size(1))
        return x_hat, mu, logvar, z

def vae_loss(x, x_hat, mu, logvar, kl_weight=0.01):
    recon = nn.MSELoss()(x_hat, x)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + kl_weight * kl, recon.item(), kl.item()
