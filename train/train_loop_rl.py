# train_loop.py
import torch
import torch.nn as nn
import numpy as np
import os, json
import matplotlib.pyplot as plt
import math
import random
from collections import deque, namedtuple
import torch.optim as optim
from tqdm import trange
import pandas as pd
from util.running_mean_std import RunningMeanStd
from structural_break.bocpd import BOCPD
from structural_break.hazard import ConstantHazard
from structural_break.distribution import StudentT
from ml_dl_models.rnn_vae import VAEEncoder, vae_loss
from ml_dl_models.actor_critic import Actor
from ml_dl_models.actor_critic import Critic
from util.weighted_replay_buffer import WeightedReplayBuffer
from util.eval_strategy import evaluate_strategy
from util.models_io import save_RLmodels

def train_loop_rl(
    stream,
    bocpd_params=None,
    vae_params=None,
    rl_params=None,
    joint_params=None,
    num_epochs=25,
    save_dir="checkpoints",
    total_steps=10000,
    device='cpu'
):
    state_window=joint_params['state_window']
    seq_len_for_vae=vae_params['vae_seq_len']
    bocpd_hazard=bocpd_params['hazard']
    if isinstance(stream, pd.Series):
        data = stream.values  # just the spread values
        dates = stream.index    # keep dates for later if you want plotting
    else:
        data = np.asarray(stream)
        dates = None

    bocpd = BOCPD(
                  ConstantHazard(bocpd_hazard),
                  StudentT(
                            mu=bocpd_params['mu'],
                            kappa=bocpd_params['kappa'],
                            alpha=bocpd_params['alpha'],
                            beta=bocpd_params['beta']
                    )
            )

    state_dim = state_window  # using flattened returns as state; in practice use richer features

    actor = Actor(
                state_dim=state_window, 
                z_dim=rl_params['state_dim'], 
                hidden_dim=rl_params['hidden_dim'], 
                action_dim=rl_params['action_dim']
                ).to(device)

    critic = Critic(
                state_dim=state_window, 
                z_dim=rl_params['state_dim'], 
                hidden_dim=rl_params['hidden_dim']
                ).to(device)

    encoder = VAEEncoder(
                input_dim=vae_params['input_dim'], 
                hidden_dim=vae_params['hidden_dim'], 
                z_dim=vae_params['latent_dim'], 
                seq_len=seq_len_for_vae
                ).to(device)

    actor_opt = optim.Adam(actor.parameters(), lr=rl_params['lr'])
    critic_opt = optim.Adam(critic.parameters(), lr=rl_params['lr'])
    opt_vae = optim.Adam(encoder.parameters(), lr=vae_params['lr'])
    buffer = WeightedReplayBuffer(capacity=30000)
    best_val_sharpe = -np.inf

    # training loop
    for epoch in range(num_epochs):
        rms = RunningMeanStd()
        # action noise base sigma
        base_action_sigma = joint_params['base_action_sigma']
        # walkthrough
        T = min(total_steps, len(data) - 1)

        # initialize state: last `state_window` returns
        state_returns = [0.0]*(state_window-1)
        state_returns.append(data[0])
        last_action = 0.0
        out_recon = []
        actions_pnl = []
        total_recon, total_kl, total_policy_loss = 0, 0, 0

        for step in trange(T):
            cur_ret = data[step]
            rms.update([cur_ret])
            # BOCPD expects scalar observation -> use normalized return
            norm_ret = float((cur_ret - rms.mean) / (math.sqrt(rms.var) + 1e-8))
            change_prob, cp_flag = bocpd.update(norm_ret)  # float in [0,1]

            # build encoder input sequence (seq_len_for_vae)
            seq_start = max(0, step - seq_len_for_vae + 1)
            seq_rets = data[seq_start: step + 1]
            # pad if needed
            if len(seq_rets) < seq_len_for_vae:
                pad = np.zeros(seq_len_for_vae - len(seq_rets))
                seq_rets = np.concatenate([pad, seq_rets])
            # form encoder input: (seq_len, input_dim) where input_dim = [norm_ret, change_prob]
            seq_inp = np.stack([ (seq_rets - rms.mean) / (math.sqrt(rms.var)+1e-8),
                                  np.ones_like(seq_rets) * change_prob ], axis=-1)[None, ...]  # batch=1
            seq_inp_t = torch.tensor(seq_inp, dtype=torch.float32).to(device)
            # # --- VAE encoder ---
            x_hat, mu, logvar, z_t = encoder(seq_inp_t)
            loss_vae, recon_loss, kl_loss = vae_loss(seq_inp_t, x_hat, mu, logvar, kl_weight=vae_params['kl_wt'])
            opt_vae.zero_grad(); loss_vae.backward(); opt_vae.step()

            # keep denormalized reconstruction for plotting if desired
            # assume first channel is the "return" we are reconstructing
            with torch.no_grad():
                recon_np = x_hat.detach().cpu().numpy()[0, :, 0]  # seq_len values
                # take last timestep reconstruction (corresponds to current idx)
                recon_last_norm = recon_np[-1]
                recon_last_denorm = recon_last_norm * math.sqrt(rms.var) + rms.mean
                out_recon.append(recon_last_denorm)

            # state vector for policy: flatten last `state_window` normalized returns
            state_arr = np.array(state_returns[-state_window:])
            state_norm = (state_arr - rms.mean) / (math.sqrt(rms.var) + 1e-8)

            state_t = torch.tensor(state_norm.astype(np.float32))[None, :].to(device)
            with torch.no_grad():
                z_t_det = z_t
                action_mean = actor(state_t, z_t_det).cpu().numpy().squeeze()
            # exploration scale increases with change_prob
            noise_sigma = base_action_sigma * (1.0 + 5.0 * change_prob)  # alpha=5 scaling
            action = action_mean + np.random.normal(scale=noise_sigma, size=action_mean.shape)
            action = np.clip(action, -1.0, 1.0)
            actions_pnl.append(action)
            # interpret action: e.g., fraction of capital to long (positive) or short (negative)
            # reward: simple PnL = action * next_return
            next_ret = data[step + 1]
            reward = float(action * next_ret)

            # store transition in buffer with initial weight 1.0
            buffer.push(
                state_norm.astype(np.float32),
                action.astype(np.float32),
                reward,
                None, False,
                1.0,
                seq_inp.squeeze(0).astype(np.float32)  # shape (seq_len, input_dim)
            )

            # if change_prob large, upweight recent transitions
            if cp_flag == 1:
                buffer.upweight_recent(window=200, multiplier=joint_params['wt_multplier'])

            # periodic updates
            if buffer.size() >= joint_params['buffer_size_updates'] and step % 8 == 0:
                batch = buffer.sample(joint_params['sample_batch_size'])

                # prepare tensors for actor/critic
                states = torch.tensor(np.stack([b.state for b in batch]), dtype=torch.float32).to(device)
                actions = torch.tensor(np.stack([b.action for b in batch]), dtype=torch.float32).to(device)
                rewards = torch.tensor(np.stack([b.reward for b in batch]), dtype=torch.float32).unsqueeze(-1).to(device)

                # critic update
                values = critic(states, torch.zeros(states.size(0), rl_params['state_dim']).to(device))  # critic input placeholder z
                targets = rewards
                critic_loss = nn.MSELoss()(values, targets)
                critic_opt.zero_grad(); critic_loss.backward(); critic_opt.step()

                # actor update
                with torch.no_grad():
                    adv = (rewards - values).detach()
                pred_actions = actor(states, torch.zeros(states.size(0), rl_params['state_dim']).to(device))
                actor_loss = - (pred_actions * adv).mean()
                actor_opt.zero_grad(); actor_loss.backward(); actor_opt.step()
                total_policy_loss += actor_loss.item()

            # update state
            state_returns.append(next_ret)

            total_recon += recon_loss
            total_kl += kl_loss

        if (epoch + 1) == num_epochs:
            print()
        avg_recon = total_recon / len(data)
        avg_kl = total_kl / len(data)
        avg_policy = total_policy_loss / len(data)
        print(f"Epoch {epoch:03d} | recon loss = {avg_recon:.4f} | kl loss = {avg_kl:.4f} | policy loss = {avg_policy:.4f}")

        # ============================================================
        #   Save models for best Sharpe ratio
        # ============================================================
        val_metrics = evaluate_strategy(actions_pnl)
        val_sharpe = val_metrics["sharpe_ratio"]

        print(f"Val Sharpe={val_sharpe:.3f}")

        # --- save best checkpoint ---
        if val_sharpe > best_val_sharpe:
            best_val_sharpe = val_sharpe
            meta = {"epoch": epoch, "recon loss": (avg_recon), "kl loss": (avg_kl), "policy loss" : (avg_policy)}
            bocpd_cfg = {"bocpd_hazard": bocpd_hazard}
            save_RLmodels(save_dir, actor, critic, encoder,
                            actor_opt, critic_opt, opt_vae,
                            bocpd_cfg, meta)
            print(f"Saved best models at epoch {epoch:03d} (Sharpe={val_sharpe:.3f})")

    np.savez(os.path.join(save_dir, "rms_stats.npz"), mean=rms.mean, var=rms.var)
    print("Training complete.")

# # Optional: quick test
# if __name__ == "__main__":
#     dummy_series = pd.Series(np.random.randn(2000),
#                              index=pd.date_range("2020-01-01", periods=2000))
#     actor, critic, encoder, bocpd = train_loop_rl(dummy_series)
#     print("train_loop ran successfully!")
