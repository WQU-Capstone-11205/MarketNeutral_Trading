# train_loop.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os, json
import matplotlib.pyplot as plt
import math
import random
from collections import deque, namedtuple
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
from util.seed_random import seed_random

def train_loop_rl(
    stream,
    bocpd_params=None,
    vae_params=None,
    rl_params=None,
    joint_params=None,
    num_epochs=10,
    save_dir="checkpoints",
    total_steps=10000,
    device='cpu',
    stop_loss_threshold=-0.02,
    stop_loss_penalty=0.001,
    seed: int = 42
):
    """
    Training loop for RL policy with VAE + BOCPD context.
    Args:
        stream: pd.Series or np.ndarray of returns
        bocpd_params, vae_params, rl_params, joint_params: dict with parameters
        num_epochs: models to be trained for number of epochs
        save_dir: path to the saved models directory (where save_all_models() stored them)
        total_steps: maximum limit for Time steps
        device: 'cpu' or 'cuda'
        stop_loss_threshold, stop_loss_penalty: used for early stopping
        seed: for random seeding

    Exit:
        Save models for the best score.
        Early stopping when the scores don't change beyond a threshold
        Displays stop-loss triggered for number of PnLs
    """
    seed_random(seed, device=device)
    
    state_window=joint_params['state_window']
    seq_len_for_vae=vae_params['vae_seq_len']
    bocpd_hazard=bocpd_params['hazard']
   
    if isinstance(stream, pd.Series):
        data = stream.values  # just the spread values
        dates = stream.index    # keep dates for later if you want plotting
    else:
        data = np.asarray(stream)
        dates = None

    # Initialize all models
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
    gamma = rl_params.get("gamma", 0.99)
    buffer = WeightedReplayBuffer(capacity=20000)
    rms = RunningMeanStd()
    # EARLY STOPPING PARAMETERS
    patience = rl_params.get("patience", 5)
    min_delta = rl_params.get("min_delta", 1e-4)
    es_counter = 0
    best_val_sharpe = -np.inf
    stopped_early = False

    # CHANGED: reward-shaping hyperparams (tunable via rl_params or joint_params)
    cp_weight = rl_params.get("cp_weight", 1.0)           # multiplies reward when change_prob high
    var_penalty = rl_params.get("var_penalty", 0.25)      # penalty * recent variance
    dd_penalty = rl_params.get("dd_penalty", 2.0)         # penalty multiplier for drawdown exceed
    dd_thr = rl_params.get("dd_threshold", 0.10)         # acceptable drawdown before penalty
    var_window = rl_params.get("var_window", 20)         # window for recent var
    tc_scale = joint_params.get("tc_scale", 0.3)         # scale for transaction cost (reduce penalty) # CHANGED
    exploration_alpha = joint_params.get("exploration_alpha", 10.0)  # CHANGED: was 5.0 before

    # training loop
    for epoch in range(num_epochs):
        # reseed per-epoch so runs are reproducible and deterministic across epochs
        # we use deterministic offset seeds so all randomness is identical for same seed value
        epoch_seed = seed + epoch
        seed_random(epoch_seed, device=device)
        
        # action noise base sigma
        base_action_sigma = joint_params['base_action_sigma']
        # walkthrough
        T = min(total_steps, len(data) - 1)

        # initialize state: last `state_window` returns
        state_returns = [0.0]*(state_window-1)
        state_returns.append(data[0])
        vae_state_diff = np.array([0.0]*seq_len_for_vae)
        last_action = 0.0
        out_recon = []
        portfolio_returns = []
        total_recon, total_kl, total_policy_loss = 0, 0, 0
        cp_probs = []
        cumulative_pnl = 0.0
        peak_pnl = 0.0   # CHANGED: track peak for drawdown computation
        stop_loss_count = 0

        for step in trange(T):
            # reseed per-step for any sampling/noise used during step
            step_seed = epoch_seed + step + 1000
            # PyTorch generator for randn-like draws (per-device)
            if device.startswith("cuda") and torch.cuda.is_available():
                gen = torch.Generator(device='cuda')
            else:
                gen = torch.Generator(device='cpu')
            gen.manual_seed(step_seed)

            # keep numpy and python random deterministic for any sampling inside this step (e.g., buffer pushes)
            np.random.seed(step_seed)
            random.seed(step_seed)
            torch.manual_seed(step_seed)  # ensures CPU-side rng deterministic for code that uses torch.randn()
            
            cur_ret = data[step]
            rms.update([cur_ret])
            # BOCPD expects scalar observation -> use normalized return
            norm_ret = float((cur_ret - rms.mean) / (math.sqrt(rms.var) + 1e-8))
            change_prob, cp_flag = bocpd.update(norm_ret)  # float in [0,1]

            # build encoder input sequence (seq_len_for_vae)
            seq_start = max(0, step - seq_len_for_vae + 1)
            seq_rets = data[seq_start: step + 1]
            cp_probs.append(change_prob)
            cps_seq = cp_probs[seq_start: step + 1]
            if step == 0:
                cur_dif = data[step]
            else:
                cur_dif = data[step] - data[step-1]
            vae_state_diff = np.append(vae_state_diff, cur_dif)
            # pad if needed
            if len(seq_rets) < seq_len_for_vae:
                pad = np.zeros(seq_len_for_vae - len(seq_rets))
                seq_rets = np.concatenate([pad, seq_rets])
                cps_pad = np.zeros(seq_len_for_vae - len(cps_seq))
                cps_seq = np.concatenate([cps_pad, cps_seq]) 
            # form encoder input: (seq_len, seq_diff, input_dim) where input_dim = [norm_ret, change_prob]
            seq_inp = np.stack([ (seq_rets - rms.mean) / (math.sqrt(rms.var)+1e-8),
                                  cps_seq ], axis=-1)[None, ...]  # batch=1 # (1, seq_len_for_vae, 2)
            seq_inp_t = torch.tensor(seq_inp, dtype=torch.float32).to(device)
            # # --- VAE encoder ---
            x_hat, mu, logvar, z_t = encoder(seq_inp_t)
            loss_vae, recon_loss, kl_loss = vae_loss(seq_inp_t, x_hat, mu, logvar, kl_weight=vae_params['kl_wt'])
            opt_vae.zero_grad(); loss_vae.backward(); opt_vae.step()

            # keep denormalized reconstruction for plotting if desired
            # assume first channel is the "return" we are reconstructing
            with torch.no_grad():
                recon_np = x_hat.detach().cpu().numpy()[0, :, 0]  # seq_len values
                # take last timestep reconstruction (corresponds to current step)
                recon_last_norm = recon_np[-1]
                recon_last_denorm = recon_last_norm * math.sqrt(rms.var) + rms.mean
                out_recon.append(recon_last_denorm)

            # state vector for policy: flatten last `state_window` normalized returns
            state_arr = np.array(state_returns[-state_window:])
            state_norm = (state_arr - rms.mean) / (math.sqrt(rms.var) + 1e-8)

            state_t = torch.tensor(state_norm.astype(np.float32))[None, :].to(device)
            with torch.no_grad():
                z_t_det = mu.detach() # z_t
                action_mean = actor(state_t, z_t_det).cpu().numpy().squeeze()
            # exploration scale increases with change_prob
            noise_sigma = base_action_sigma * (1.0 + exploration_alpha * change_prob)  # alpha=5 scaling
            action = action_mean + np.random.normal(scale=noise_sigma, size=action_mean.shape)
            action = np.clip(action, -1.0, 1.0)
            next_ret = data[step + 1]
            next_state_arr = np.array(state_returns[-(state_window-1):] + [next_ret])
            next_state_norm = (next_state_arr - rms.mean) / (math.sqrt(rms.var) + 1e-8)

            # normalize reward by volatility and include transaction costs
            eps = 1e-8
            raw_reward = float(action * (next_ret - cur_ret))          # delta spread × position
            # base normalized reward
            base_reward = raw_reward / (math.sqrt(rms.var) + eps)

            # CHANGED: compute recent rolling variance for variance penalty
            if len(portfolio_returns) >= 2:
                recent_window = max(1, min(var_window, len(portfolio_returns)))
                rolling_var = float(np.var(portfolio_returns[-recent_window:]))
            else:
                rolling_var = 1e-8  # fallback

            # CHANGED: apply cp-weighting, variance penalty, drawdown penalty, and scale transaction cost
            # cp amplification
            reward = base_reward * (1.0 + cp_weight * change_prob)  # CHANGED: amplify reward when CP high

            # drawdown bookkeeping BEFORE applying stop-loss
            # cumulative_pnl currently tracks normalized rewards sum
            # update peak for drawdown calc
            if cumulative_pnl > peak_pnl:
                peak_pnl = cumulative_pnl

            # compute drawdown as fraction of peak (safe denom)
            if peak_pnl > 1e-8:
                cur_dd = (peak_pnl - cumulative_pnl) / (abs(peak_pnl) + 1e-8)
            else:
                cur_dd = 0.0

            # variance penalty (reduces reward when recent variance high)
            reward = reward - (var_penalty * rolling_var)

            # drawdown penalty (only applied when exceeding threshold)
            if cur_dd > dd_thr:
                reward = reward - dd_penalty * (cur_dd - dd_thr)

            cumulative_pnl += reward       # track cumulative profit/loss

            # STOP-LOSS CHECK
            stop_triggered = False
            if cumulative_pnl <= stop_loss_threshold:
                reward -= abs(stop_loss_penalty)   # penalize hitting stop-loss
                action = 0.0                         # force close position
                stop_triggered = True
                stop_loss_count += 1
                cumulative_pnl = 0.0
                done_flag = True                   # optionally end episode early

            # transaction cost: proportional to change in action magnitude
            tc = joint_params.get('transaction_cost', 0.0)
            trans_cost = tc * float(np.abs(action - last_action).sum())  # sum if vector action
            # CHANGED: scale down effective tc to avoid over-penalizing turnover
            reward = reward - (tc_scale * trans_cost)
            
            portfolio_returns.append(reward)

            # store transition in buffer with initial weight 1.0
            buffer.push(
                state_norm.astype(np.float32),
                np.array([action]).astype(np.float32), # Ensure action is a numpy array
                reward,
                next_state_norm.astype(np.float32),
                False,
                1.0,
                None
            )

            last_action = action
            
            # if change_prob large, upweight recent transitions
            if cp_flag == 1:
                buffer.upweight_recent(window=200, multiplier=joint_params['wt_multplier'])

            # periodic updates
            if buffer.size() >= joint_params['buffer_size_updates'] and step % 8 == 0:
                # reseed numpy/random before sampling to ensure consistent sampled indices
                np.random.seed(step_seed + 12345)
                random.seed(step_seed + 12345)

                batch = buffer.sample(joint_params['sample_batch_size'])

                # prepare tensors for actor/critic
                states = torch.tensor(np.stack([b.state for b in batch]), dtype=torch.float32).to(device)
                actions = torch.tensor(np.stack([b.action for b in batch]), dtype=torch.float32).to(device)
                rewards = torch.tensor(np.stack([b.reward for b in batch]), dtype=torch.float32).unsqueeze(-1).to(device)
                next_states = torch.tensor(np.stack([b.next_state for b in batch]), dtype=torch.float32).to(device)
                
                # critic update
                values = critic(states, torch.zeros(states.size(0), rl_params['state_dim']).to(device))  # critic input placeholder z

                with torch.no_grad():
                    next_values = critic(next_states, torch.zeros(next_states.size(0), rl_params['state_dim']).to(device))
                    targets = rewards + (gamma * next_values)
                critic_loss = nn.MSELoss()(values, targets)
                critic_opt.zero_grad();
                critic_loss.backward();
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                critic_opt.step()

                # actor update: advantage-based
                with torch.no_grad():
                    adv = (targets - values).detach()  # shape (batch, 1)
                # actor update
                pred_actions = actor(states, torch.zeros(states.size(0), rl_params['state_dim']).to(device))
                actor_loss = - (pred_actions * adv).mean()

                # CHANGED: optionally add small L2 regularization on policy parameters to avoid collapse (if desired)
                if rl_params.get('actor_l2', 0.0) > 0.0:
                    l2_reg = 0.0
                    for p in actor.parameters():
                        l2_reg += (p**2).sum()
                    actor_loss = actor_loss + rl_params['actor_l2'] * l2_reg
                    
                actor_opt.zero_grad();
                actor_loss.backward();
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                actor_opt.step()
                total_policy_loss += actor_loss.item()

            # update state
            state_returns.append(next_ret)

            total_recon += recon_loss
            total_kl += kl_loss

        if stop_loss_count > 0:
            print(f"Stop-loss triggered for {stop_loss_count} PnLs")

        if (epoch + 1) == num_epochs:
            print()
        avg_recon = total_recon / T
        avg_kl = total_kl / T
        avg_policy = total_policy_loss / T
        print(f"Epoch {epoch:03d} | recon loss = {avg_recon:.4f} | kl loss = {avg_kl:.4f} | policy loss = {avg_policy:.4f}")

        # ============================================================
        #   Save models for best Sharpe ratio
        # ============================================================
        val_metrics = evaluate_strategy(portfolio_returns)
        val_sharpe = val_metrics["sharpe_ratio"]

        print(f"Sharpe = {val_sharpe:.3f}")

        # --- save best checkpoint ---
        if val_sharpe > best_val_sharpe + min_delta:
            best_val_sharpe = val_sharpe
            es_counter = 0  # reset patience counter
            meta = {"epoch": epoch, "recon loss": (avg_recon), "kl loss": (avg_kl), "policy loss" : (avg_policy)}
            bocpd_cfg = {"bocpd_hazard": bocpd_hazard}
            save_RLmodels(save_dir, actor, critic, encoder,
                            actor_opt, critic_opt, opt_vae,
                            bocpd_cfg, meta)
            print(f"Saved best models at epoch {epoch:03d} (Sharpe={val_sharpe:.3f})")
        else:
            es_counter += 1
            print(f"No improvement. Early stopping patience counter = {es_counter}/{patience}")
            if es_counter >= patience:
                print(f"EARLY STOPPING TRIGGERED at epoch {epoch}")
                stopped_early = True
                break

    np.savez(os.path.join(save_dir, "rms_stats.npz"), mean=rms.mean, var=rms.var)
    if stopped_early:
        print("Training stopped early due to no improvement in Sharpe.")
    else:
        print("Training completed all epochs.")
    print("RL policy training complete.")

# # Optional: quick test
# if __name__ == "__main__":
#     dummy_series = pd.Series(np.random.randn(2000),
#                              index=pd.date_range("2020-01-01", periods=2000))
#     actor, critic, encoder, bocpd = train_loop_rl(dummy_series)
#     print("train_loop ran successfully!")
