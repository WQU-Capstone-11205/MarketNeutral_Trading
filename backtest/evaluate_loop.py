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
from util.running_mean_std import RunningMeanStd
from structural_break.bocpd import BOCPD
from structural_break.hazard import ConstantHazard
from structural_break.distribution import StudentT
from ml_dl_models.rnn_vae import VAEEncoder, vae_loss
from ml_dl_models.actor_critic import Actor
from ml_dl_models.actor_critic import Critic
from util.weighted_replay_buffer import WeightedReplayBuffer
from util.file_operations import load_models

def evaluate_loop(data, last_step, seq_len = 50, total_steps = 100000, load_dir="checkpoints", device="cpu", exploration=False):
    state_window = 50
    seq_len_for_vae = 50
    input_dim = 2  # [return, bocpd_prob] per timestep into encoder
    z_dim = 16
    state_dim = state_window

    vae_encoder = VAEEncoder(input_dim=input_dim, hidden_dim=128, z_dim=z_dim, seq_len=seq_len_for_vae).to(device)
    actor = Actor(state_dim = state_dim, z_dim=z_dim, action_dim=1).to(device)
    critic = Critic(state_dim=state_dim, z_dim=z_dim).to(device)
    vae_opt = torch.optim.Adam(vae_encoder.parameters(), lr=1e-3)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=1e-3)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=1e-3)
    bocpd_cfg, meta = load_models(load_dir, actor, critic, vae_encoder,
                                      actor_opt, critic_opt, vae_opt,
                                      device, step=(last_step-1))
    
    # Extract the hazard rate from the loaded dictionary
    bocpd_hazard = bocpd_cfg.get("bocpd_hazard", bocpd_hazard_default)
    bocpd = BOCPD(ConstantHazard(bocpd_hazard), StudentT(mu=0, kappa=1, alpha=1, beta=1))
    vae_encoder.eval()
    actor.eval()
    bocpd.reset_params()

    rms = RunningMeanStd()
    # action noise base sigma
    base_action_sigma = 0.05
    # walkthrough
    T = min(total_steps, len(data) - state_window - 1)
    rms.update(data[:state_window])

    # initialize state: last `state_window` returns
    idx = 0
    state_returns = list(data[idx: idx + state_window])
    idx += state_window
    last_action = 0.0

    # action noise base sigma
    change_probs, rewards = [], []
    all_recons = [0]*state_window
    actions = [0]*state_window

    # for i in range(len(data_n) - seq_len):
    for step in trange(T):
        cur_ret = data[idx]
        rms.update([cur_ret])
        # BOCPD expects scalar observation -> use normalized return
        norm_ret = float((cur_ret - rms.mean) / (math.sqrt(rms.var) + 1e-8))
        change_prob = bocpd.update(norm_ret)  # float in [0,1]
        change_probs.append(change_prob)

        # build encoder input sequence (seq_len_for_vae)
        seq_start = max(0, idx - seq_len_for_vae + 1)
        seq_rets = data[seq_start: idx + 1]
        # pad if needed
        if len(seq_rets) < seq_len_for_vae:
            pad = np.zeros(seq_len_for_vae - len(seq_rets))
            seq_rets = np.concatenate([pad, seq_rets])
        # form encoder input: (seq_len, input_dim) where input_dim = [norm_ret, change_prob]
        seq_inp = np.stack([ (seq_rets - rms.mean) / (math.sqrt(rms.var)+1e-8),
                              np.ones_like(seq_rets) * change_prob ], axis=-1)[None, ...]  # batch=1
        seq_inp_t = torch.tensor(seq_inp, dtype=torch.float32).to(device)

        with torch.no_grad():
            x_hat, mu, logvar, z_t = vae_encoder(seq_inp_t)
            recon_np = x_hat.detach().cpu().numpy()[0, :, 0]  # seq_len values
            # take last timestep reconstruction (corresponds to current idx)
            recon_last_norm = recon_np[-1]
            recon_last_denorm = recon_last_norm * math.sqrt(rms.var) + rms.mean
            all_recons.append(recon_last_denorm)

            # ---- BOCPD-based gating ----
            noise_scale = 0.05 * (1.0 + 5.0 * change_prob)  # increase noise if break detected
            # ---- Policy action ----
            # state vector for policy: flatten last `state_window` normalized returns
            state_arr = np.array(state_returns[-state_window:])
            state_norm = (state_arr - rms.mean) / (math.sqrt(rms.var) + 1e-8)

            state_t = torch.tensor(state_norm.astype(np.float32))[None, :].to(device)

            z_t_det = z_t # mu
            action_mean = actor(state_t, z_t_det).cpu().numpy().squeeze()
            if exploration:
                # live adaptive (adds regime-scaled noise)
                noise_sigma = base_action_sigma * (1.0 + 5.0 * change_prob) # alpha = 5.0
                action = action_mean + np.random.normal(scale=noise_sigma, size=action_mean.shape)
            else:
                # stabilized adaptive (no noise, but can still scale amplitude)
                action = action_mean * (1.0 - 0.5 * change_prob)
            action = np.clip(action, -1.0, 1.0)
            actions.append(action.item())
            reward = -nn.MSELoss()(x_hat, seq_inp_t).item()
            rewards.append(reward)
            #all_recons.append(torch.mean((x_hat - seq_inp_t) ** 2).item())

        idx += 1    

    actions.append(0)
    all_recons.append(all_recons[-1])
    # ---- Final metrics ----
    print(f"Average recon error: {np.mean(all_recons):.6f}")
    print(f"Average change prob: {np.mean(change_probs):.6f}")
    # pnl = np.cumsum(np.array(actions) * np.array(rewards))
    # plt.plot(pnl, label="RL PnL")
    # plt.plot(np.cumsum(np.array(rewards)), label="Buy&Hold")
    # plt.legend(); 
    # plt.title("PnL vs Buy&Hold")
    # plt.show()
    print("Evaluation complete.")
    return actions, all_recons
