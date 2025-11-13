import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os, json
from tqdm import trange
import math
import random
from collections import deque, namedtuple
import pandas as pd
from util.running_mean_std import RunningMeanStd
from structural_break.bocpd import BOCPD
from structural_break.hazard import ConstantHazard
from structural_break.distribution import StudentT
from ml_dl_models.rnn_vae import VAEEncoder, vae_loss
from ml_dl_models.lstm import LSTMPolicy
from util.weighted_replay_buffer import WeightedReplayBuffer
from util.eval_strategy import evaluate_strategy
from util.models_io import save_models
from util.seed_random import seed_random


def train_loop_lstm(
    stream,
    bocpd_params=None,
    vae_params=None,
    lstm_params=None,
    joint_params=None,
    num_epochs = 10,
    save_dir="checkpoints_lstm",
    total_steps=10000,
    device='cpu',
    stop_loss_threshold=-0.02,
    stop_loss_penalty=0.001
):
    seed_random()
    gamma = lstm_params.get("gamma", 0.99) # disabled for now # 0.99,                # discount factor (same as RL)
    replay_alpha_cp = 0.6      # weight mix: alpha*cp + (1-alpha)*|reward|
    state_window=joint_params['state_window']
    seq_len_for_vae=vae_params['vae_seq_len']
    bocpd_hazard=bocpd_params['hazard']
    # ----- Data setup -----
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

    encoder = VAEEncoder(
                input_dim=vae_params['input_dim'],
                hidden_dim=vae_params['hidden_dim'],
                z_dim=vae_params['latent_dim'],
                seq_len=seq_len_for_vae
                ).to(device)

    opt_vae = optim.Adam(encoder.parameters(), lr=vae_params['lr'])

    policy_lstm = LSTMPolicy(
                input_dim = joint_params['state_window'] + lstm_params['z_dim'],
                hidden_dim = lstm_params['hidden_dim']
                ).to(device)

    opt_policy = optim.Adam(policy_lstm.parameters(), lr=lstm_params['lr'])
    best_val_sharpe = -np.inf
    buffer = WeightedReplayBuffer(capacity=20000)
    rms = RunningMeanStd()

    # ----- Training loop -----
    for epoch in range(num_epochs):
        T = min(total_steps, len(data) - 1)
        # initialize state: last `state_window` returns
        state_returns = [0.0]*(state_window-1)
        state_returns.append(data[0])
        last_action = 0.0
        out_recon = []
        portfolio_returns = []
        total_recon, total_kl, total_policy_loss = 0.0, 0.0, 0.0
        cp_probs = []
        cumulative_pnl = 0.0
        stop_loss_count = 0

        for step in trange(T):
            cur_ret = data[step]
            rms.update([cur_ret])
            norm_ret = float((cur_ret - rms.mean) / (math.sqrt(rms.var) + 1e-8))

            # --- BOCPD change-point probability ---
            change_prob, cp_flag = bocpd.update(norm_ret)

            # --- Encoder (VAE) ---
            seq_start = max(0, step - seq_len_for_vae + 1)
            seq_rets = data[seq_start: step + 1]
            cp_probs.append(change_prob)
            cps_seq = cp_probs[seq_start: step + 1]

            if len(seq_rets) < seq_len_for_vae:
                pad = np.zeros(seq_len_for_vae - len(seq_rets))
                seq_rets = np.concatenate([pad, seq_rets])
                cps_pad = np.zeros(seq_len_for_vae - len(cps_seq))
                cps_seq = np.concatenate([cps_pad, cps_seq])

            seq_inp = np.stack([
                (seq_rets - rms.mean) / (math.sqrt(rms.var) + 1e-8),
                cps_seq
            ], axis=-1)[None, ...]
            seq_inp_t = torch.tensor(seq_inp, dtype=torch.float32).to(device)
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

            # --- Policy (LSTM)) ---
            state_arr = np.array(state_returns[-state_window:])
            state_norm = (state_arr - rms.mean) / (math.sqrt(rms.var) + 1e-8)
            state_t = torch.tensor(state_norm.astype(np.float32))[None, :].to(device)

            inp_t = torch.cat([state_t, mu.detach()], dim=-1)
            action_t = torch.tanh(policy_lstm(inp_t))  # [-1, 1]
    
            # The following two lines are for training loop only
            action_t = action_t + torch.randn_like(action_t) * 0.01
            action_t = torch.clamp(action_t, -1.0, 1.0)

            next_ret = data[step + 1]
            # --- PnL computation (with gradient flow) ---
            next_ret_t = torch.tensor([next_ret - cur_ret], dtype=torch.float32, device=device)

            pnl_t = action_t * next_ret_t
            # normalize reward by volatility and include transaction costs
            eps = 1e-8
            pnl_t_norm = pnl_t / (math.sqrt(rms.var) + eps)

            tc = joint_params.get('transaction_cost', 0.0)
            trans_cost = tc * float(np.abs((action_t.detach().cpu().numpy().squeeze()) - last_action).sum())  # sum if vector action
            pnl_net_t = pnl_t_norm - trans_cost  # subtract cost # maximize pnl

            # entropy-like penalty, This discourages the LSTM from pushing 
            # outputs to extremes (−1 or +1) unless strongly justified
            entropy_reg = - (action_t * torch.log(torch.abs(action_t) + 1e-8)).mean()
            loss_policy = -pnl_net_t.mean() - 1e-3 * entropy_reg

            #loss_policy = -pnl_net_t.mean()  # maximize pnl
            opt_policy.zero_grad()
            loss_policy.backward()
            torch.nn.utils.clip_grad_norm_(policy_lstm.parameters(), max_norm=1.0)
            opt_policy.step()

            pnl_scalar = float(pnl_net_t.detach().cpu().numpy().squeeze())
            total_policy_loss += float(loss_policy.detach().cpu().numpy())
            cumulative_pnl += pnl_scalar

            # STOP-LOSS CHECK
            stop_triggered = False
            if cumulative_pnl <= stop_loss_threshold:
                pnl_scalar -= abs(stop_loss_penalty)   # penalize hitting stop-loss
                action_t = torch.zeros_like(action_t) # force close position
                stop_triggered = True
                stop_loss_count += 1
                cumulative_pnl = 0.0

            # transaction cost: proportional to change in action magnitude
            portfolio_returns.append(pnl_scalar)
            last_action = float(action_t.detach().cpu().numpy().squeeze())

            # compute weight: mix BOCPD surprise and reward magnitude
            w_cp = float(change_prob)
            w_ret = abs(pnl_scalar)
            weight = float(replay_alpha_cp * w_cp + (1.0 - replay_alpha_cp) * w_ret + 1e-8)
            # --- Store transition in buffer (for future stability) ---
            buffer.push(
                state_norm.astype(np.float32),
                float(action_t.detach().cpu().numpy().squeeze()),
                float(pnl_scalar),
                None,
                False,
                weight,
                None
            )

            # Upweight near detected changes
            if cp_flag == 1:
                buffer.upweight_recent(window=200, multiplier=joint_params['wt_multplier'])

            # periodic updates
            if ((buffer.size() >= joint_params['buffer_size_updates']) and (step % 8 == 0)):
                batch = buffer.sample(joint_params['sample_batch_size'])

                # prepare tensors
                states = torch.tensor(np.stack([b.state for b in batch]), dtype=torch.float32, device=device)
                actions = torch.tensor(np.stack([b.action for b in batch]), dtype=torch.float32, device=device).unsqueeze(-1)
                rewards = torch.tensor(np.stack([b.reward for b in batch]), dtype=torch.float32, device=device).unsqueeze(-1)
                sample_weights = [b.weight for b in batch]
                sample_weights_t = torch.tensor(sample_weights, dtype=torch.float32, device=device).unsqueeze(-1)

                # compute predicted actions and weighted loss (policy)
                with torch.no_grad():
                    z_placeholder = torch.zeros(states.size(0), lstm_params['z_dim'], device=device)  # if you want to include z, adapt
                policy_actions = policy_lstm(torch.cat([states, z_placeholder], dim=-1))
                pred_actions = torch.tanh(policy_actions)

                # policy loss: -pred_actions * reward (we want actions that produce positive reward)
                per_sample_loss = - (pred_actions * rewards)  # (N,1)
                weighted_loss = (per_sample_loss * sample_weights_t).mean()
                opt_policy.zero_grad()
                weighted_loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_lstm.parameters(), max_norm=1.0)
                opt_policy.step()
                policy_loss = per_sample_loss.mean()
                total_policy_loss += float(policy_loss.detach().cpu().item())

            # Move window
            state_returns.append(next_ret)
            total_recon += float(recon_loss)
            total_kl += float(kl_loss)

        if stop_loss_count > 0:
            print(f"Stop-loss triggered for {stop_loss_count} PnLs")

        avg_recon = total_recon / float(T)
        avg_kl = total_kl / float(T)
        avg_policy = total_policy_loss / float(T)
        print(f"Epoch {epoch:03d} | recon={avg_recon:.4f} | kl={avg_kl:.4f} | policy={avg_policy:.4f}")

        # ============================================================
        #   Save models for best Sharpe ratio
        # ============================================================
        val_metrics = evaluate_strategy(portfolio_returns)
        val_sharpe = val_metrics["sharpe_ratio"]

        print(f"Sharpe = {val_sharpe:.3f}")

        # --- save best checkpoint ---
        if val_sharpe > best_val_sharpe:
            best_val_sharpe = val_sharpe
            meta = {"epoch": epoch, "recon loss": float(avg_recon), "kl loss": float(avg_kl), "policy loss" : float(avg_policy)}
            bocpd_cfg = {"bocpd_hazard": bocpd_hazard}
            save_models(
                        save_dir,
                        policy_lstm,
                        encoder,
                        opt_policy,
                        opt_vae,
                        bocpd_cfg,
                        meta
                    )
            print(f"Saved best models at epoch {epoch:03d} (Sharpe={val_sharpe:.3f})")

    np.savez(os.path.join(save_dir, "rms_stats.npz"), mean=rms.mean, var=rms.var)
    print("LSTM policy training complete.")

# # Optional: quick test
# if __name__ == "__main__":
#     # dummy_series = pd.Series(np.random.randn(2000),
#     #                          index=pd.date_range("2020-01-01", periods=2000))
#     train_loop_lstm(train_spread, num_epochs = 1)
#     print("train_loop (lstm) ran successfully!")
