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
from util.models_io import load_RLmodels

def evaluate_loop_rl(
          data, 
          bocpd_params, 
          vae_params, 
          rl_params, 
          joint_params, 
          seq_len = 50, 
          total_steps = 100000, 
          load_dir="checkpoints", 
          device="cpu", 
          exploration=False
):
    state_window = joint_params['state_window']
    seq_len_for_vae = vae_params['vae_seq_len']
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
    capital = 1.0

    # for i in range(len(data_n) - seq_len):
    for step in trange(T):
        cur_ret = data[step]
        rms.update([cur_ret])
        # BOCPD expects scalar observation -> use normalized return
        norm_ret = float((cur_ret - rms.mean) / (math.sqrt(rms.var) + 1e-8))
        change_prob, _ = bocpd.update(norm_ret)  # float in [0,1]

        # build encoder input sequence (seq_len_for_vae)
        seq_start = max(0, step - seq_len_for_vae + 1)
        seq_rets = data[seq_start: step + 1]
        if step == 0:
            cur_dif = data[step]
        else:
            cur_dif = data[step] - data[step-1]
        vae_state_diff = np.append(vae_state_diff, cur_dif)
        seq_diff = vae_state_diff[-seq_len_for_vae:]
        seq_diff = seq_diff / (math.sqrt(rms.var) + 1e-8)
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

        # ---- BOCPD-based gating ----
        noise_scale = 0.05 * (1.0 + 5.0 * change_prob)  # increase noise if break detected
        # ---- Policy action ----
        # state vector for policy: flatten last `state_window` normalized returns
        state_arr = np.array(state_returns[-state_window:])
        state_norm = (state_arr - rms.mean) / (math.sqrt(rms.var) + 1e-8)

        state_t = torch.tensor(state_norm.astype(np.float32))[None, :].to(device)

        with torch.no_grad():
            z_t_det = mu # z_t
            action_mean = actor(state_t, z_t_det).cpu().numpy().squeeze()
        if exploration:
            # live adaptive (adds regime-scaled noise)
            noise_sigma = base_action_sigma * (1.0 + 5.0 * change_prob) # alpha = 5.0
            action = action_mean + np.random.normal(scale=noise_sigma, size=action_mean.shape)
        else:
            # stabilized adaptive (no noise, but can still scale amplitude)
            action = action_mean * (1.0 - 0.5 * change_prob)
        action = np.clip(action, -1.0, 1.0)
        #position = action * (1 - change_prob) / (math.sqrt(rms.var) + 1e-8)
        actions.append(action)
        next_ret = data[step + 1]
        reward = action * (next_ret - cur_ret)
        #reward = float(action * next_ret)
        rewards.append(reward)
        capital *= (1 + reward)
        pnl.append(capital)
        #all_recons.append(torch.mean((x_hat - seq_inp_t) ** 2).item())

        state_returns.append(data[step+1])
    
    change_probs, rt_mle, cp_flags = bocpd.results
    all_recons.append(all_recons[-1])
    print("Evaluation complete.")
    metrics = { 
                'change_probs' : change_probs, 
                'rt_mle' : rt_mle, 
                'cp_flags' : cp_flags, 
                'recons' : all_recons, 
                'pnl' : pnl 
    }
    return metrics
    return actions, all_recons, rewards
