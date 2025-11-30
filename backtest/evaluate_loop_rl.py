# ============================================================
#  EVALUATION LOOP
# ============================================================
import os, json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import trange
import torch.optim as optim
import math
import random

from util.running_mean_std import RunningMeanStd
from util.seed_random import seed_random
from structural_break.bocpd import BOCPD
from structural_break.hazard import ConstantHazard
from structural_break.distribution import StudentT
from ml_dl_models.rnn_vae import VAEEncoder, vae_loss
from ml_dl_models.actor_critic import Actor
from ml_dl_models.actor_critic import Critic
from util.weighted_replay_buffer import WeightedReplayBuffer
from util.models_io import load_RLmodels

def evaluate_loop_rl(
          stream,
          bocpd_params,
          vae_params,
          rl_params,
          joint_params,
          total_steps = 100000,
          load_dir="checkpoints",
          device="cpu",
          exploration=False,
          stop_loss_threshold=-0.02, #same stop-loss threshold as training (e.g., −2%)
          stop_loss_penalty=0.001,    # optional penalty for hitting stop-loss
          seed: int = 42
):
    """
    Evaluate a trained RL policy and encoder with VAE + BOCPD context.
    Automatically loads saved models using load_all_models().

    Args:
        stream: pd.Series or np.ndarray of returns
        bocpd_params, vae_params, rl_params, joint_params: dict with parameters
        total_steps: maximum time steps limit 
        load_dir: path to the saved models directory (where save_all_models() stored them)
        device: 'cpu' or 'cuda'
        exploration: used if stochasticity is required for the evaluation loop
        stop_loss_threshold, stop_loss_penalty: used for early stopping
        seed: for random seeding

    Returns:
        dict with change probabilities, most likely run lengths estimates, changepoint flags, 
        reconstructed outputs of VAE, portfolio returns, actions, and portfolio returns series
    """
    if isinstance(stream, pd.Series):
        data = stream.values  # just the spread values
        dates = stream.index    # keep dates for later if you want plotting
    else:
        data = np.asarray(stream)
        dates = None
    seed_random(seed, device=device)
    state_window = joint_params['state_window']
    seq_len_for_vae = vae_params['vae_seq_len']
    tc = joint_params.get('transaction_cost', 0.0)
    state_dim = state_window

    vae_encoder = VAEEncoder(
                          input_dim=vae_params['input_dim'],
                          hidden_dim=vae_params['hidden_dim'],
                          z_dim=vae_params['latent_dim'],
                          seq_len=seq_len_for_vae
                          ).to(device)

    actor = Actor(
                state_dim = state_dim,
                z_dim=rl_params['state_dim'],
                hidden_dim=rl_params['hidden_dim'],
                action_dim=rl_params['action_dim']
                ).to(device)

    critic = Critic(
              state_dim=state_dim,
              z_dim=rl_params['state_dim'],
              hidden_dim=rl_params['hidden_dim']
              ).to(device)

    vae_opt = torch.optim.Adam(vae_encoder.parameters(), lr=vae_params['lr'])
    actor_opt = torch.optim.Adam(actor.parameters(), lr=rl_params['lr'])
    critic_opt = torch.optim.Adam(critic.parameters(), lr=rl_params['lr'])
    bocpd_cfg, meta = load_RLmodels(load_dir, actor, critic, vae_encoder,
                                      actor_opt, critic_opt, vae_opt,
                                      device)

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
    actor.eval()
    bocpd.reset_params()

    rms = RunningMeanStd()
    rms_stats = np.load(os.path.join(load_dir, "rms_stats.npz"))
    # --- Assign back to rms object ---
    rms.mean = rms_stats["mean"]
    rms.var = rms_stats["var"]

    # action noise base sigma
    base_action_sigma = joint_params['base_action_sigma']
    # walkthrough
    T = min(total_steps, len(data) - 1)

    # initialize state: last `state_window` returns
    state_returns = [0.0]*(state_window-1)
    state_returns.append(data[0])
    vae_state_diff = np.array([0.0]*seq_len_for_vae)
    last_action = 0.0

    # action noise base sigma
    all_recons = []
    rewards = []
    actions = []
    pnl = []
    portfolio_returns = []
    equity_curve = []
    capital = 1.0
    cp_probs = []
    eps = 1e-8
    cumulative_pnl = 0.0          # track total PnL
    stop_loss_count = 0
    
    global_mean = 0.0
    global_var = 0.0
    recon_probs = []
    errors_ch0 = []
    errors_ch1 = []

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
              
        # BOCPD update (normalized observation)
        norm_ret = float((cur_ret - rms.mean) / (math.sqrt(rms.var) + 1e-8))
        change_prob, _ = bocpd.update(norm_ret)  # float in [0,1]
        cp_probs.append(change_prob)
        
        # build VAE input window
        seq_start = max(0, step - seq_len_for_vae + 1)
        seq_rets = data[seq_start: step + 1]
        cps_seq = cp_probs[seq_start: step + 1] ####
        if step == 0:
            cur_dif = data[step]
        else:
            cur_dif = data[step] - data[step-1]
        vae_state_diff = np.append(vae_state_diff, cur_dif)
        
        # pad if needed
        if len(seq_rets) < seq_len_for_vae:
            pad = np.zeros(seq_len_for_vae - len(seq_rets))
            seq_rets = np.concatenate([pad, seq_rets])
            cps_pad = np.zeros(seq_len_for_vae - len(cps_seq)) ####
            cps_seq = np.concatenate([cps_pad, cps_seq]) ####
        
        # form encoder input: (seq_len, input_dim) where input_dim = [norm_ret, change_prob]
        seq_inp = np.stack([ (seq_rets - rms.mean) / (math.sqrt(rms.var)+1e-8),
                              cps_seq ], axis=-1)[None, ...]  # batch=1
        seq_inp_t = torch.tensor(seq_inp, dtype=torch.float32).to(device)

        # VAE forward (no-grad)
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

            inp_ch0 = cur_ret
            recon_ch0 = recon_last_denorm
            inp_ch1 = change_prob
            recon_ch1 = recon_cp_last_norm
            # accumulate recon errors
            errors_ch0.append(np.mean((inp_ch0 - recon_ch0)**2))
            errors_ch1.append(np.mean((inp_ch1 - recon_ch1)**2))

        # ---- BOCPD-based gating ----
        noise_scale = 0.05 * (1.0 + 5.0 * change_prob)  # increase noise if break detected
        # ---- Policy action ----
        # state vector for policy: flatten last `state_window` normalized returns
        state_arr = np.array(state_returns[-state_window:])
        state_norm = (state_arr - rms.mean) / (math.sqrt(rms.var) + 1e-8)

        state_t = torch.tensor(state_norm.astype(np.float32))[None, :].to(device)

        # actor forward
        with torch.no_grad():
            z_t_det = mu # z_t
            action_mean = actor(state_t, z_t_det).cpu().numpy().squeeze()

        # exploration vs deterministic gating by cp-prob
        if exploration:
            # live adaptive (adds regime-scaled noise)
            noise_sigma = base_action_sigma * (1.0 + 5.0 * change_prob) # alpha = 5.0
            action = action_mean + np.random.normal(scale=noise_sigma, size=action_mean.shape)
        else:
            action = action_mean * (1.0 - 0.5 * change_prob) # 0.5 * change_prob
        action = np.clip(action, -1.0, 1.0)
        actions.append(action)
        next_ret = data[step + 1]

        eps = 1e-8
        # reward with transaction cost
        raw_reward = float(action * (next_ret - cur_ret))
        reward = raw_reward / (math.sqrt(rms.var) + eps)
        trans_cost = tc * float(np.abs(action - last_action).sum())
        reward = reward - trans_cost
        last_action = action.copy()
        cumulative_pnl += reward                     # track cumulative profit/loss

        # ---------------------------
        # Stop-loss check
        # ---------------------------
        stop_triggered = False
        if cumulative_pnl <= stop_loss_threshold:
            reward -= abs(stop_loss_penalty)          # penalize
            action = 0.0                              # force flat
            stop_triggered = True
            stop_loss_count += 1
            cumulative_pnl = 0.0
            done_flag = True                          # end evaluation early (optional)

        portfolio_returns.append(reward)
        if step == 0:
            equity_curve.append(1+ reward)
            pnl.append(reward)
        else:
            equity_curve.append(equity_curve[-1]*(1+ reward))
            pnl.append(equity_curve[-1] - 1)
        rewards.append(reward)
        state_returns.append(data[step+1])

    if stop_loss_count > 0:
        print(f"Stop-loss triggered for {stop_loss_count} PnLs")

    change_probs, rt_mle, cp_flags = bocpd.results
    all_recons.append(all_recons[-1])
    rmse_ch0 = float(np.sqrt(np.mean(errors_ch0)))
    rmse_ch1 = float(np.sqrt(np.mean(errors_ch1)))
    print(f"\nRMSE channel0 (spread, denorm): {rmse_ch0:.6f}")
    print(f"RMSE channel1 (cp prob):        {rmse_ch1:.6f}")

    print("Evaluation complete.")
    metrics = {
                'change_probs' : np.array(change_probs),
                'rt_mle' : np.array(rt_mle),
                'cp_flags' : np.array(cp_flags),
                'recons' : np.array(all_recons),
                'actions' : np.array(actions),
                'portfolio_returns' : np.array(portfolio_returns),
                'equity_curve' : np.array(equity_curve),
                'pnl' : np.array(pnl),
                'rets' : pd.Series(portfolio_returns, index= dates[:len(dates)-1])
    }
    return metrics
