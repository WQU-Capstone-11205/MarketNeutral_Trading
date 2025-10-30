import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import torch.optim as optim
import os, json
import torch
import torch.nn as nn
from tqdm import trange
from util.running_mean_std import RunningMeanStd
from structural_break.bocpd import BOCPD
from structural_break.hazard import ConstantHazard
from structural_break.distribution import StudentT
from ml_dl_models.rnn_vae import VAEEncoder, vae_loss
from ml_dl_models.lstm import LSTMPolicy
from ml_dl_models.actor_critic import Critic
from util.weighted_replay_buffer import WeightedReplayBuffer
from util.models_io import load_models

@torch.no_grad()
def evaluate_loop_lstm(
          stream,
          bocpd_params, 
          vae_params, 
          lstm_params, 
          joint_params, 
          save_dir="checkpoints_lstm",
          total_steps = 100000, 
          device="cpu"
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
    # ---- Prepare data ----
    if isinstance(stream, pd.Series):
        data = stream.values
        date = stream.index
    else:
        data = np.asarray(stream)

    state_window = joint_params['state_window']
    seq_len_for_vae = vae_params['vae_seq_len']
    state_dim = state_window

    vae_encoder = VAEEncoder(
                          input_dim=vae_params['input_dim'],
                          hidden_dim=vae_params['hidden_dim'],
                          z_dim=vae_params['latent_dim'],
                          seq_len=seq_len_for_vae
                          ).to(device)

    policy_lstm = LSTMPolicy(
                          input_dim=state_dim + lstm_params['z_dim'], 
                          hidden_dim=lstm_params['hidden_dim']
                          ).to(device)

    vae_opt = optim.Adam(vae_encoder.parameters(), lr=vae_params['lr'])
    lstm_opt = optim.Adam(policy_lstm.parameters(), lr=lstm_params['lr'])
    
    bocpd_cfg, meta = load_models(save_dir, policy_lstm, vae_encoder,
                              lstm_opt, vae_opt,  device, step=None)


    # action noise base sigma
    base_action_sigma = joint_params['base_action_sigma']
    # walkthrough
    T = min(total_steps, len(data) - 1)

    # ---- Setup BOCPD ----
    # Extract the hazard rate from the loaded dictionary
    bocpd_hazard_default=bocpd_params['hazard']
    bocpd_hazard = bocpd_cfg.get("bocpd_hazard", bocpd_hazard_default)
    bocpd = BOCPD(
                ConstantHazard(bocpd_hazard),
                StudentT(
                    mu=bocpd_params['mu'],
                    kappa=bocpd_params['kappa'],
                    alpha=bocpd_params['alpha'],
                    beta=bocpd_params['beta']
                    )
                )

    vae_encoder.eval()
    policy_lstm.eval()
    bocpd.reset_params()


    # ---- Initialize helpers ----
    rms = RunningMeanStd()
    rms.update(data[:state_window])

    state_returns = [0.0]*(state_window-1)
    state_returns.append(data[0])
    vae_state_diff = np.array([0.0]*seq_len_for_vae)
    last_action = 0.0

    # action noise base sigma
    all_recons = []
    pnls = []
    capital = 1.0

    # ---- Main evaluation loop ----
    for step in trange(T):
        cur_ret = data[step]
        rms.update([cur_ret])

        norm_ret = (cur_ret - rms.mean) / (math.sqrt(rms.var) + 1e-8)

        # --- BOCPD ---
        change_prob, _ = bocpd.update(norm_ret)
        
        # build encoder input sequence (seq_len_for_vae)
        seq_start = max(0, step - seq_len_for_vae + 1)
        seq_rets = data[seq_start: step + 1]
        if step == 0:
            cur_dif = data[step]
        else:
            cur_dif = data[step] - data[step-1]
        vae_state_diff = np.append(vae_state_diff, cur_dif)
        seq_diff = vae_state_diff[-seq_len_for_vae:]
        # seq_diff = seq_diff / (math.sqrt(rms.var) + 1e-8)
        # pad if needed
        if len(seq_rets) < seq_len_for_vae:
            pad = np.zeros(seq_len_for_vae - len(seq_rets))
            seq_rets = np.concatenate([pad, seq_rets])
        # form encoder input: (seq_len, input_dim) where input_dim = [norm_ret, change_prob]
        seq_inp = np.stack([ (seq_rets - rms.mean) / (math.sqrt(rms.var)+1e-8), seq_diff,
                              np.ones_like(seq_rets) * change_prob ], axis=-1)[None, ...]  # batch=1
        seq_inp_t = torch.tensor(seq_inp, dtype=torch.float32).to(device)

        with torch.no_grad():
            x_hat, mu, logvar, z_t = vae_encoder(seq_inp_t)
            recon_np = x_hat.detach().cpu().numpy()[0, :, 0]  # seq_len values
            # take last timestep reconstruction (corresponds to current idx)
            recon_last_norm = recon_np[-1]
            recon_last_denorm = recon_last_norm * math.sqrt(rms.var) + rms.mean
            all_recons.append(recon_last_denorm)

        # --- Prepare state + latent input for policy LSTM ---
        state_arr = np.array(state_returns[-state_window:])
        state_norm = (state_arr - rms.mean) / (math.sqrt(rms.var) + 1e-8)
        state_t = torch.tensor(state_norm.astype(np.float32))[None, :].to(device)

        inp_t = torch.cat([state_t, mu.detach()], dim=-1)
        action_t = torch.tanh(policy_lstm(inp_t))
    
        # --- PnL computation (with gradient flow) ---
        next_ret = data[step + 1]
        next_ret_t = torch.tensor([next_ret - cur_ret], dtype=torch.float32, device=device)
        pnl_t = action_t * next_ret_t
        tc = 0 # transaction_cost * torch.abs(action_t - prev_action) # disabled for now
        pnl_net_t = pnl_t - tc  # subtract cost # maximize pnl
        loss_policy = -pnl_net_t.mean()  # maximize pnl
        pnl_scalar = float(pnl_net_t.detach().cpu().numpy().squeeze())
        pnls.append(pnl_scalar)
        prev_action = action_t.detach().clone()  # store detached value

        state_returns.append(next_ret)

    change_probs, rt_mle, cp_flags = bocpd.results
    all_recons.append(all_recons[-1])
    print("\nEvaluation complete.")
    metrics = { 
        'change_probs' : np.array(change_probs), 
        'rt_mle' : np.array(rt_mle), 
        'cp_flags' : np.array(cp_flags), 
        'recons' : np.array(all_recons), 
        'pnl' : np.array(pnls) 
    }
    return metrics
