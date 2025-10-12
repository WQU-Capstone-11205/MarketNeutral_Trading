def train_loop_cnn_lstm(
    stream,
    num_epochs=50,
    save_dir="checkpoints_cnnlstm",
    state_window=50,
    seq_len_for_vae=50,
    total_steps=10000,
    bocpd_hazard=300.0,
    device='cpu'
):
    if isinstance(stream, pd.Series):
        data = stream.values
    else:
        data = np.asarray(stream)

    bocpd = BOCPD(ConstantHazard(bocpd_hazard), StudentT(mu=0, kappa=1, alpha=1, beta=1))
    input_dim = 2
    z_dim = 16

    encoder = VAEEncoder(input_dim=input_dim, hidden_dim=128, z_dim=z_dim, seq_len=seq_len_for_vae).to(device)
    policy = CNNLSTMPolicy(input_dim=z_dim + 1, hidden_dim=128).to(device)

    opt_vae = torch.optim.Adam(encoder.parameters(), lr=1e-3)
    opt_policy = torch.optim.Adam(policy.parameters(), lr=1e-4)
    rms = RunningMeanStd()

    for epoch in range(num_epochs):
        total_recon, total_kl, total_policy = 0, 0, 0
        rms.update(data[:state_window])
        idx = state_window
        state_returns = list(data[:state_window])
        T = min(total_steps, len(data) - state_window - 1)

        for step in range(T):
            cur_ret = data[idx]
            rms.update([cur_ret])
            norm_ret = (cur_ret - rms.mean) / (math.sqrt(rms.var) + 1e-8)
            change_prob = bocpd.update(norm_ret)

            # ---- Encode latent z ----
            seq_start = max(0, idx - seq_len_for_vae + 1)
            seq_rets = data[seq_start: idx + 1]
            if len(seq_rets) < seq_len_for_vae:
                seq_rets = np.concatenate([np.zeros(seq_len_for_vae - len(seq_rets)), seq_rets])

            seq_inp = np.stack([
                (seq_rets - rms.mean) / (math.sqrt(rms.var) + 1e-8),
                np.ones_like(seq_rets) * change_prob
            ], axis=-1)[None, ...]
            seq_inp_t = torch.tensor(seq_inp, dtype=torch.float32).to(device)
            x_hat, mu, logvar, z_t = encoder(seq_inp_t)
            kl_w = min(1.0, epoch / 100)
            loss_vae, recon, kl = vae_loss(seq_inp_t, x_hat, mu, logvar, kl_weight=kl_w)
            opt_vae.zero_grad(); loss_vae.backward(); opt_vae.step()

            # ---- Policy forward ----
            state_norm = (np.array(state_returns[-state_window:]) - rms.mean) / (math.sqrt(rms.var) + 1e-8)
            state_t = torch.tensor(state_norm, dtype=torch.float32)[None, :, None].to(device)

            # combine latent z as extra channel for CNN-LSTM
            z_rep = z_t.unsqueeze(1).repeat(1, state_window, 1)
            inp_t = torch.cat([state_t, z_rep], dim=-1)
            action_t = torch.tanh(policy(inp_t))
            next_ret = data[idx + 1]
            pnl = action_t * next_ret
            loss_policy = -pnl.mean()

            opt_policy.zero_grad(); loss_policy.backward(); opt_policy.step()

            total_policy += loss_policy.item()
            total_recon += recon
            total_kl += kl
            state_returns.append(next_ret)
            idx += 1

        print(f"[CNN-LSTM] Epoch {epoch:03d} | recon={total_recon/T:.4f} | kl={total_kl/T:.4f} | policy={total_policy/T:.4f}")
