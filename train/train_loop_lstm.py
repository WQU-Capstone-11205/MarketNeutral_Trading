import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
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


def train_loop_lstm(
    stream,
    bocpd_params=None,
    vae_params=None,
    lstm_params=None,
    joint_params=None,
    num_epochs = 10,
    save_dir="checkpoints_lstm",
    total_steps=10000,
    device='cpu'
):
    gamma = 1 # disabled for now # 0.99,                # discount factor (same as RL)
    transaction_cost = 0 # disabled for now #0.0005,   # optional, small transaction cost per trade
    replay_alpha_cp = 0.6      # weight mix: alpha*cp + (1-alpha)*|reward|
    state_window=joint_params['state_window']
    seq_len_for_vae=vae_params['vae_seq_len']
    bocpd_hazard=bocpd_params['hazard']
    # ----- Data setup -----
    if isinstance(stream, pd.Series):
        data = stream.values
    else:
        data = np.asarray(stream)

    bocpd = BOCPD(
                  ConstantHazard(bocpd_hazard),
                  StudentT(
                            mu=bocpd_params['mu'],
                            kappa=bocpd_params['kappa'],
                            alpha=bocpd_params['alpha'],
                            beta=bocpd_params['beta']
                    )
            )
    
    state_dim = state_window

    encoder = VAEEncoder(
                input_dim=vae_params['input_dim'], 
                hidden_dim=vae_params['hidden_dim'], 
                z_dim=vae_params['latent_dim'], 
                seq_len=seq_len_for_vae
                ).to(device)

    opt_vae = optim.Adam(encoder.parameters(), lr=vae_params['lr'])
        
    policy_lstm = LSTMPolicy(
                input_dim=state_dim + lstm_params['z_dim'], 
                hidden_dim = lstm_params['hidden_dim']
                ).to(device)
    
    opt_policy = optim.Adam(policy_lstm.parameters(), lr=lstm_params['lr'])
    best_val_sharpe = -np.inf
    
    # ----- Training loop -----
    for epoch in range(num_epochs):
        buffer = WeightedReplayBuffer(capacity=30000)
        rms = RunningMeanStd()
        T = min(total_steps, len(data) - 1)

        total_pnl = 0.0
        discounted_pnl = 0.0
        prev_action = 0.0  # for transaction cost calc

        # initialize state: last `state_window` returns
        state_returns = [0.0]*(state_window-1)
        state_returns.append(data[0])
        vae_state_diff = np.array([0.0]*seq_len_for_vae)
        last_action = 0.0
        out_recon = []
        actions_pnl = []
        total_recon, total_kl, total_policy_loss = 0.0, 0.0, 0.0

        for step in trange(T):
            cur_ret = data[step]
            rms.update([cur_ret])
            norm_ret = float((cur_ret - rms.mean) / (math.sqrt(rms.var) + 1e-8))

            # --- BOCPD change-point probability ---
            change_prob, cp_flag = bocpd.update(norm_ret)

            # --- Encoder (VAE) ---
            seq_start = max(0, step - seq_len_for_vae + 1)
            seq_rets = data[seq_start: step + 1]
            if step == 0:
                cur_dif = data[step]
            else:
                cur_dif = data[step] - data[step-1]
            vae_state_diff = np.append(vae_state_diff, cur_dif)
            seq_diff = vae_state_diff[-seq_len_for_vae:]
            if len(seq_rets) < seq_len_for_vae:
                pad = np.zeros(seq_len_for_vae - len(seq_rets))
                seq_rets = np.concatenate([pad, seq_rets])

            seq_inp = np.stack([
                (seq_rets - rms.mean) / (math.sqrt(rms.var) + 1e-8), seq_diff,
                np.ones_like(seq_rets) * change_prob
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

            # --- Policy (Actor) ---
            state_arr = np.array(state_returns[-state_window:])
            state_norm = (state_arr - rms.mean) / (math.sqrt(rms.var) + 1e-8)
            state_t = torch.tensor(state_norm.astype(np.float32))[None, :].to(device)

            inp_t = torch.cat([state_t, mu.detach()], dim=-1)
            action_t = torch.tanh(policy_lstm(inp_t))  # [-1, 1]
    
            next_ret = data[step + 1]

            # --- PnL computation (with gradient flow) ---
            next_ret_t = torch.tensor([next_ret - cur_ret], dtype=torch.float32, device=device)

            pnl_t = action_t * next_ret_t
            tc = 0 # transaction_cost * torch.abs(action_t - prev_action) # disabled for now
            pnl_net_t = pnl_t - tc  # subtract cost # maximize pnl
            
            loss_policy = -pnl_net_t.mean()  # maximize pnl

            opt_policy.zero_grad()
            loss_policy.backward()
            opt_policy.step()

            pnl_scalar = float(pnl_net_t.detach().cpu().numpy().squeeze())
        
            total_policy_loss += float(loss_policy.detach().cpu().numpy())
            total_pnl += pnl_scalar
            actions_pnl.append(pnl_scalar)
            discounted_pnl += (gamma ** step) * pnl_scalar

            prev_action = action_t.detach().clone()  # store detached value

            # compute weight: mix BOCPD surprise and reward magnitude
            w_cp = float(change_prob)
            w_ret = abs(pnl_scalar)
            weight = float(replay_alpha_cp * w_cp + (1.0 - replay_alpha_cp) * w_ret + 1e-8)
            # --- Store transition in buffer (for future stability) ---
            buffer.push(state_norm.astype(np.float32),
                        action_t.detach().cpu().numpy().squeeze().astype(np.float32),
                        float(pnl_scalar), #.item(),
                        None, False, weight,
                        inp_t.squeeze(0).cpu().numpy().astype(np.float32))

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
                policy_loss = per_sample_loss.mean()
                opt_policy.zero_grad()
                weighted_loss.backward()
                opt_policy.step()

                total_policy_loss += policy_loss #float(weighted_loss.detach().cpu().numpy())
            
            # Move window
            state_returns.append(next_ret)
            total_recon += recon_loss
            total_kl += kl_loss

        avg_recon = total_recon / len(data)
        avg_kl = total_kl / len(data)
        avg_policy = total_policy_loss / len(data)
        print(f"Epoch {epoch:03d} | recon={avg_recon:.4f} | kl={avg_kl:.4f} | policy={avg_policy:.4f}")

        # ============================================================
        #   Save models for best Sharpe ratio
        # ============================================================
        val_metrics = evaluate_strategy(actions_pnl)
        val_sharpe = val_metrics["sharpe_ratio"]

        print(f"Sharpe = {val_sharpe:.3f}")

        # --- save best checkpoint ---
        if val_sharpe > best_val_sharpe:
            best_val_sharpe = val_sharpe
            meta = {"epoch": epoch, "recon loss": (total_recon/len(data)), "kl loss": (total_kl/len(data))}
            bocpd_cfg = {"bocpd_hazard": bocpd_hazard}
            save_models(save_dir, policy_lstm, encoder, opt_policy, opt_vae, bocpd_cfg, meta)
            print(f"Saved best models at epoch {epoch:03d} (Sharpe={val_sharpe:.3f})")

    print("LSTM policy training complete.")

# # Optional: quick test
# if __name__ == "__main__":
#     # dummy_series = pd.Series(np.random.randn(2000),
#     #                          index=pd.date_range("2020-01-01", periods=2000))
#     train_loop_lstm(train_spread, num_epochs = 1)
#     print("train_loop (lstm) ran successfully!")
