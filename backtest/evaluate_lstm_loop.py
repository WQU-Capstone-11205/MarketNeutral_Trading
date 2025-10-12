import numpy as np
import torch
import math
import matplotlib.pyplot as plt
import pandas as pd
import torch.optim as optim

@torch.no_grad()
def evaluate_lstm_loop(
    stream,
    model_path,
    state_window=50,
    seq_len_for_vae=50,
    bocpd_hazard_default=300.0, # Use a default in case it's not in cfg
    device='cpu',
    plot=True
):
    """
    Evaluate a trained LSTM policy and encoder with VAE + BOCPD context.
    Automatically loads saved models using load_all_models().

    Args:
        stream: pd.Series or np.ndarray of returns
        model_path: path to the saved models directory (where save_all_models() stored them)
        state_window: number of past returns for state
        seq_len_for_vae: lookback length for encoder input
        bocpd_hazard_default: default hazard rate for BOCPD if not found in loaded config
        device: 'cpu' or 'cuda'
        plot: whether to show cumulative return plot

    Returns:
        dict with Sharpe ratio, total return, cumulative returns, actions, and pnl
    """
    input_dim = 2
    z_dim = 16
    state_dim = state_window

    # ---- Load models ----
    from util.models_io import load_models  # adjust path if needed
    from ml_dl_models.rnn_vae import VAEEncoder
    from ml_dl_models.lstm import LSTMPolicy

    encoder = VAEEncoder(input_dim=input_dim, hidden_dim=128, z_dim=z_dim, seq_len=seq_len_for_vae).to(device)
    policy_lstm = LSTMPolicy(input_dim=state_dim + z_dim, hidden_dim=128).to(device)
    opt_vae = optim.Adam(encoder.parameters(), lr=1e-3)
    opt_policy = optim.Adam(policy_lstm.parameters(), lr=1e-4)

    bocpd_cfg, meta = load_models(model_path, policy_lstm, encoder,
                    opt_policy, opt_vae,
                    device, step=None)

    # ---- Setup BOCPD ----
    # Extract the hazard rate from the loaded dictionary
    bocpd_hazard = bocpd_cfg.get("bocpd_hazard", bocpd_hazard_default)
    bocpd = BOCPD(ConstantHazard(bocpd_hazard), StudentT(mu=0, kappa=1, alpha=1, beta=1))

    encoder.eval()
    policy_lstm.eval()
    bocpd.reset_params()


    # ---- Prepare data ----
    if isinstance(stream, pd.Series):
        data = stream.values
    else:
        data = np.asarray(stream)

    # ---- Initialize helpers ----
    rms = RunningMeanStd()
    rms.update(data[:state_window])

    idx = state_window
    state_returns = list(data[:state_window])
    prev_action = torch.zeros(1, 1, device=device)

    actions = [0]*state_window
    pnls = [0]*state_window
    cp_flag_list = [0]*state_window
    rt_mle = [0]*state_window

    # ---- Main evaluation loop ----
    while idx < len(data) - 1:
        cur_ret = data[idx]
        rms.update([cur_ret])
        norm_ret = (cur_ret - rms.mean) / (math.sqrt(rms.var) + 1e-8)

        # --- BOCPD ---
        change_prob = bocpd.update(norm_ret)
        rt_mle.append(bocpd.rt)
        cp_flag = 1 if rt_mle[idx] < rt_mle[idx-1] else 0
        cp_flag_list.append(cp_flag)
        
        # --- Build sequence input for encoder ---
        seq_start = max(0, idx - seq_len_for_vae + 1)
        seq_rets = data[seq_start: idx + 1]
        if len(seq_rets) < seq_len_for_vae:
            seq_rets = np.concatenate([np.zeros(seq_len_for_vae - len(seq_rets)), seq_rets])

        seq_inp = np.stack([
            (seq_rets - rms.mean) / (math.sqrt(rms.var) + 1e-8),
            np.ones_like(seq_rets) * change_prob
        ], axis=-1)[None, ...]
        seq_inp_t = torch.tensor(seq_inp, dtype=torch.float32).to(device)

        # --- Encode latent z_t ---
        _, _, _, z_t = encoder(seq_inp_t)

        # --- Prepare state + latent input for policy LSTM ---
        state_arr = np.array(state_returns[-state_window:])
        state_norm = (state_arr - rms.mean) / (math.sqrt(rms.var) + 1e-8)
        state_t = torch.tensor(state_norm.astype(np.float32))[None, :].to(device)

        inp_t = torch.cat([state_t, z_t.detach()], dim=-1)
        action_t = torch.tanh(policy_lstm(inp_t))
        actions.append(action_t.item())

        # --- Compute PnL ---
        next_ret = data[idx + 1]
        pnl = action_t.item() * next_ret
        pnls.append(pnl)

        # Slide
        prev_action = action_t.detach()
        state_returns.append(next_ret)
        idx += 1

    # ---- Compute performance metrics ----
    pnls = np.array(pnls)
    cumulative_returns = np.cumsum(pnls)
    mean_ret = np.mean(pnls)
    std_ret = np.std(pnls) + 1e-8
    sharpe = (mean_ret / std_ret) * np.sqrt(252)  # annualized Sharpe (daily freq)

    metrics = {
        "sharpe_ratio": sharpe,
        "total_return": cumulative_returns[-1],
        "cumulative_returns": cumulative_returns,
        "pnl_series": pnls,
        "actions": np.array(actions),
        "change_points": np.array(cp_flag_list)
    }

    # ---- Optional visualization ----
    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(cumulative_returns, label="LSTM Policy Cumulative Return", linewidth=2)
        plt.title(f"Cumulative Return | Sharpe={sharpe:.3f}")
        plt.xlabel("Time Steps")
        plt.ylabel("Cumulative PnL")
        plt.grid(True)
        plt.legend()
        plt.show()

    return metrics
