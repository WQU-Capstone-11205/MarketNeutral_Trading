#-----------------------------------------------------------------
# evaluate_loop_cnn_lstm.py
#-----------------------------------------------------------------
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import torch.optim as optim
import random
import os, json
import torch
import torch.nn as nn
from tqdm import trange
from util.running_mean_std import RunningMeanStd
from structural_break.bocpd import BOCPD
from structural_break.hazard import ConstantHazard
from structural_break.distribution import StudentT
from ml_dl_models.rnn_vae import VAEEncoder, vae_loss
from ml_dl_models.cnn_lstm import CNNLSTMModel, build_optimizer
from ml_dl_models.actor_critic import Critic
from util.weighted_replay_buffer import WeightedReplayBuffer
from util.models_io import load_models
from util.seed_random import seed_random

@torch.no_grad()
def evaluate_loop_cnnlstm(
          stream,
          bocpd_params,
          vae_params,
          cnnlstm_params,
          joint_params,
          load_dir="checkpoints_cnnlstm",
          total_steps = 100000,
          device="cpu",
          stop_loss_threshold=-0.02, #same stop-loss threshold as training (e.g., −2%)
          stop_loss_penalty=0.001,    # optional penalty for hitting stop-loss
          seed: int = 42
    ):
    """
    Evaluate a trained LSTM policy and encoder with VAE + BOCPD context.
    Automatically loads saved models using load_all_models().

    Args:
        stream: pd.Series or np.ndarray of returns
        bocpd_params, vae_params, cnnlstm_params, joint_params: dict with parameters
        load_dir: path to the saved models directory (where save_all_models() stored them)
        device: 'cpu' or 'cuda'
        stop_loss_threshold, stop_loss_penalty: used for early stopping
        seed: for random seeding

    Returns:
        dict with change probabilities, most likely run lengths estimates, changepoint flags, 
        reconstructed outputs of VAE, portfolio returns, actions, and portfolio returns series
    """
    # ---- Prepare data ----
    if isinstance(stream, pd.Series):
        data = stream.values
        dates = stream.index
    else:
        data = np.asarray(stream)
        dates = None

    seed_random()
    state_window = joint_params['state_window']
    seq_len_for_vae = vae_params['vae_seq_len']
    tc = joint_params.get('transaction_cost', 0.0)

    vae_encoder = VAEEncoder(
                          input_dim=vae_params['input_dim'],
                          hidden_dim=vae_params['hidden_dim'],
                          z_dim=vae_params['latent_dim'],
                          seq_len=seq_len_for_vae
                          ).to(device)

    vae_opt = optim.Adam(vae_encoder.parameters(), lr=vae_params['lr'])

    policy_cnnlstm = CNNLSTMModel(
                      input_dim=cnnlstm_params['input_dim'],
                      cnn_channels=cnnlstm_params['cnn_channels'],
                      hidden_dim=cnnlstm_params['hidden_dim'],
                      kernel_size=cnnlstm_params['kernel_size'],
                      z_dim=cnnlstm_params['z_dim'],
                      seq_len=seq_len_for_vae
                    ).to(device)

    opt_policy = build_optimizer(
                    policy_cnnlstm,
                    optimizer_name=cnnlstm_params['optimizer'],
                    lr=cnnlstm_params['lr'],
                    weight_decay=cnnlstm_params['weight_decay']
                )

    bocpd_cfg, meta = load_models(load_dir, policy_cnnlstm, vae_encoder,
                              opt_policy, vae_opt,  device, step=None)

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
    policy_cnnlstm.eval()
    bocpd.reset_params()

    # ---- Initialize helpers ----
    rms = RunningMeanStd()
    rms_stats = np.load(os.path.join(load_dir, "rms_stats.npz"))
    # --- Assign back to rms object ---
    rms.mean = rms_stats["mean"]
    rms.var = rms_stats["var"]

    state_returns = [0.0]*(state_window-1)
    state_returns.append(data[0])
    prev_action = 0.0

    # action noise base sigma
    actions = []
    all_recons = []
    portfolio_returns = []
    cp_probs = []
    capital = 1.0
    transaction_cost = joint_params.get('transaction_cost', 0.0)
    eps = 1e-8
    cumulative_pnl = 0.0          # track total PnL
    stop_loss_count = 0

    global_mean = 0.0
    global_var = 0.0
    recon_probs = []
    errors_ch0 = []
    errors_ch1 = []

    # ---- Main evaluation loop ----
    for step in trange(T):
        # reseed at each step to ensure deterministic behavior
        step_seed = seed + step + 2000
        if device.startswith("cuda") and torch.cuda.is_available():
            gen = torch.Generator(device='cuda')
        else:
            gen = torch.Generator(device='cpu')
        gen.manual_seed(step_seed)

        np.random.seed(step_seed)
        random.seed(step_seed)
        torch.manual_seed(step_seed)

        cur_ret = data[step]
        rms.update([cur_ret])
        global_mean = rms.mean
        global_var = rms.var
        norm_ret = (cur_ret - rms.mean) / (math.sqrt(rms.var) + 1e-8)

        # --- BOCPD ---
        change_prob, _ = bocpd.update(norm_ret)
        cp_probs.append(change_prob)
        # build encoder input sequence (seq_len_for_vae)
        seq_start = max(0, step - seq_len_for_vae + 1)
        seq_rets = data[seq_start: step + 1]
        cps_seq = cp_probs[seq_start: step + 1]
        # pad if needed
        if len(seq_rets) < seq_len_for_vae:
            pad = np.zeros(seq_len_for_vae - len(seq_rets))
            seq_rets = np.concatenate([pad, seq_rets])
            cps_pad = np.zeros(seq_len_for_vae - len(cps_seq))
            cps_seq = np.concatenate([cps_pad, cps_seq])

        # form encoder input: (seq_len, input_dim) where input_dim = [norm_ret, change_prob]
        seq_inp = np.stack([ (seq_rets - rms.mean) / (math.sqrt(rms.var)+1e-8),
                              cps_seq ], axis=-1)[None, ...]  # batch=1
        seq_inp_t = torch.tensor(seq_inp, dtype=torch.float32).to(device)

        with torch.no_grad():
            x_hat, mu, logvar, z_t = vae_encoder(seq_inp_t)
            recon_np = x_hat.detach().cpu().numpy()[0, :, 0]  # seq_len values
            recon_cp = x_hat.detach().cpu().numpy()[0, :, 1]
            # take last timestep reconstruction (corresponds to current idx)
            recon_last_norm = recon_np[-1]
            recon_last_denorm = recon_last_norm * math.sqrt(rms.var) + rms.mean
            all_recons.append(recon_last_denorm)
            recon_cp_last_norm = recon_cp[-1]
            recon_probs.append(recon_cp_last_norm)

            inp_ch0 = cur_ret #denorm_spread(inp_norm, global_mean, global_var)
            recon_ch0 = recon_last_denorm #denorm_spread(recon_norm, global_mean, global_var)
            inp_ch1 = change_prob #cp_probs[i]   # cp_probs typically already 0..1
            recon_ch1 = recon_cp_last_norm #recon_probs[i]
            errors_ch0.append(np.mean((inp_ch0 - recon_ch0)**2))
            errors_ch1.append(np.mean((inp_ch1 - recon_ch1)**2))

        # --- Policy (CNNLSTM) ---
        # Pass the sequence input and the latent variable to the CNNLSTMModel
        z_t = mu;
        action_t = torch.tanh(policy_cnnlstm(seq_inp_t, z_t.detach()))  # [-1, 1]
        action_t = torch.clamp(action_t, -1.0, 1.0)

        reward = data[step + 1] - cur_ret
        # --- PnL computation (with gradient flow) ---
        reward_t = torch.tensor([reward], dtype=torch.float32, device=device)
        pnl_t = action_t * reward_t
        # normalize reward by volatility and include transaction costs
        eps = 1e-8
        pnl_t_norm = pnl_t / (math.sqrt(rms.var) + eps)          # delta spread × position
        trans_cost = float(tc * torch.abs(action_t - prev_action).item())
        pnl_net_t = pnl_t_norm - trans_cost  # subtract cost # maximize pnl

        # entropy-like penalty, This discourages the LSTM from pushing
        # outputs to extremes (−1 or +1) unless strongly justified
        entropy_reg = - (action_t * torch.log(torch.abs(action_t) + 1e-8)).mean()
        loss_policy = -pnl_net_t.mean() - 1e-3 * entropy_reg

        pnl_scalar = float(pnl_net_t.detach().cpu().numpy().squeeze())
        cumulative_pnl  += pnl_scalar

        # STOP-LOSS CHECK
        # stop_triggered = False
        if cumulative_pnl <= stop_loss_threshold:
            pnl_scalar -= abs(stop_loss_penalty)   # penalize hitting stop-loss
            action_t = torch.zeros_like(action_t) # force close position
            stop_triggered = True
            stop_loss_count += 1
            cumulative_pnl = 0.0

        portfolio_returns.append(pnl_scalar)
        prev_action = float(action_t.detach().cpu().numpy().squeeze())
        actions.append(prev_action)

    change_probs, rt_mle, cp_flags = bocpd.results
    all_recons.append(all_recons[-1])
    rmse_ch0 = float(np.sqrt(np.mean(errors_ch0)))
    rmse_ch1 = float(np.sqrt(np.mean(errors_ch1)))
    print(f"\nRMSE channel0 (spread, denorm): {rmse_ch0:.6f}")
    print(f"RMSE channel1 (cp prob):        {rmse_ch1:.6f}")

    print("\nEvaluation complete.")
    metrics = {
        'change_probs' : np.array(change_probs),
        'rt_mle' : np.array(rt_mle),
        'cp_flags' : np.array(cp_flags),
        'recons' : np.array(all_recons),
        'portfolio_returns' : np.array(portfolio_returns),
        'actions' : np.array(actions),
        'rets': pd.Series(portfolio_returns, index= dates[:len(dates)-1]) if dates is not None else pd.Series(portfolio_returns)
    }
    return metrics
