import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import trange
import math
import random
import os, json
from collections import deque, namedtuple
import pandas as pd
from util.running_mean_std import RunningMeanStd
from structural_break.bocpd import BOCPD
from structural_break.hazard import ConstantHazard
from structural_break.distribution import StudentT
from ml_dl_models.rnn_vae import VAEEncoder, vae_loss
from ml_dl_models.transformer import TransformerModel
from util.weighted_replay_buffer import WeightedReplayBuffer
from util.eval_strategy import evaluate_strategy
from util.models_io import save_models
from util.seed_random import seed_random

def train_loop_trafo(
    stream,
    bocpd_params=None,
    vae_params=None,
    trafo_params=None,
    joint_params=None,
    num_epochs = 50,
    save_dir="checkpoints_trafo",
    total_steps=10000,
    device='cpu',
    stop_loss_threshold=-0.02,
    stop_loss_penalty=0.001,
    seed: int = 42
):
    """
    Training loop for Transformer policy with VAE + BOCPD context.
    
    Args:
        stream: pd.Series or np.ndarray of returns
        bocpd_params, vae_params, trafo_params, joint_params: dict with parameters
        save_dir: path to the save models directory
        total_steps: maximum time steps limit 
        device: 'cpu' or 'cuda'
        stop_loss_threshold, stop_loss_penalty: used for early stopping
        seed: for random seeding

    Exit:
        Save models for the best score.
        Early stopping when the scores don't change beyond a threshold
        Displays stop-loss triggered for number of PnLs
    """
    # seed once before model construction and data setup
    seed_random(seed, device=device)

    state_window=joint_params['state_window']
    seq_len_for_vae=vae_params['vae_seq_len']
    bocpd_hazard=bocpd_params['hazard']
    replay_alpha_cp = 0.6
    tc = joint_params.get('transaction_cost', 0.0)

    # ----- Data setup -----
    if isinstance(stream, pd.Series):
        data = stream.values
        dates = stream.index
    else:
        data = np.asarray(stream)
        dates = None

    # BOCPD, encoder, transformer initialization
    bocpd = BOCPD(
                  ConstantHazard(bocpd_hazard),
                  StudentT(
                            mu=bocpd_params['mu'],
                            kappa=bocpd_params['kappa'],
                            alpha=bocpd_params['alpha'],
                            beta=bocpd_params['beta']
                    )
            )

    # IMPORTANT: model parameter initialization must be deterministic too (we already seeded above).
    encoder = VAEEncoder(
                input_dim=vae_params['input_dim'],
                hidden_dim=vae_params['hidden_dim'],
                z_dim=vae_params['latent_dim'],
                seq_len=seq_len_for_vae
                ).to(device)

    opt_vae = optim.Adam(encoder.parameters(), lr=vae_params['lr'])

    transformer = TransformerModel(
                      input_dim=(2 * seq_len_for_vae) + trafo_params['z_dim'],
                      d_model=trafo_params['d_model'], 
                      nhead=trafo_params['nhead'], 
                      num_layers=trafo_params['num_layers'], 
                      hidden_dim=(trafo_params['hidden_dim'] * trafo_params['d_model'])
                    ).to(device)

    opt_policy = optim.Adam(transformer.parameters(), lr=trafo_params['lr'])

    gamma = trafo_params.get("gamma", 0.99)
    buffer = WeightedReplayBuffer(capacity=20000)
    rms = RunningMeanStd()
    # EARLY STOPPING PARAMETERS
    patience = trafo_params.get("patience", 5)
    min_delta = trafo_params.get("min_delta", 1e-4)
    es_counter = 0
    best_val_sharpe = -np.inf
    stopped_early = False
    exploration_alpha = joint_params.get("exploration_alpha", 10.0)  # CHANGED: was 5.0 before
    
    # ----- Training loop -----
    for epoch in range(num_epochs):
        # reseed per-epoch so runs are reproducible and deterministic across epochs
        # we use deterministic offset seeds so all randomness is identical for same seed value
        epoch_seed = seed + epoch
        seed_random(epoch_seed, device=device)

        base_action_sigma = joint_params['base_action_sigma']
        T = min(total_steps, len(data) - 1)

        # initialize state
        state_returns = [0.0]*(state_window-1)
        state_returns.append(data[0])
        vae_state_diff = np.array([0.0]*seq_len_for_vae)
        prev_action = 0.0
        out_recon = []
        portfolio_returns = []
        total_recon, total_kl, total_policy_loss = 0, 0, 0
        cp_probs = []
        cumulative_pnl = 0.0
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
            norm_ret = float((cur_ret - rms.mean) / (math.sqrt(rms.var) + 1e-8))

            # --- BOCPD change-point probability ---
            change_prob, cp_flag = bocpd.update(norm_ret)

            # --- Encoder (VAE) and Policy (Transformer) Input ---
            seq_start = max(0, step - seq_len_for_vae + 1)
            seq_rets = data[seq_start: step + 1]
            cp_probs.append(change_prob)
            cps_seq = cp_probs[seq_start: step + 1]
            if len(seq_rets) < seq_len_for_vae:
                pad = np.zeros(seq_len_for_vae - len(seq_rets))
                seq_rets = np.concatenate([pad, seq_rets])
                cps_pad = np.zeros(seq_len_for_vae - len(cps_seq))
                cps_seq = np.concatenate([cps_pad, cps_seq])

            seq_inp = np.stack([ (seq_rets - rms.mean) / (math.sqrt(rms.var)+1e-8),
                                  cps_seq ], axis=-1)[None, ...]
            seq_inp_t = torch.tensor(seq_inp, dtype=torch.float32).to(device)

            # --- VAE encoder ---
            # If the VAE uses torch.randn inside, the manual seed above ensures deterministic draws.
            x_hat, mu, logvar, z_t = encoder(seq_inp_t)
            loss_vae, recon_loss, kl_loss = vae_loss(seq_inp_t, x_hat, mu, logvar, kl_weight=vae_params['kl_wt'])
            opt_vae.zero_grad(); loss_vae.backward(); opt_vae.step()

            # --- Policy (Transformer) ---
            z_t = z_t.detach().squeeze(0)
            z_expand = z_t.unsqueeze(0).unsqueeze(1).repeat(1, seq_len_for_vae, 1)
            full_input = torch.cat([seq_inp_t, z_expand], dim=-1)
            action_t = torch.tanh(transformer(full_input))

            # exploration scale increases with change_prob
            noise_sigma = base_action_sigma * (1.0 + exploration_alpha * change_prob)

            # deterministic noise draw using generator
            # note: torch.randn_like accepts generator=<gen>
            # noise = torch.randn_like(action_t, generator=gen) * noise_sigma
            noise = torch.randn(action_t.shape, generator=gen, device=action_t.device, dtype=action_t.dtype)
            action_t = action_t + noise
            action_t = torch.clamp(action_t, -1.0, 1.0)

            reward = data[step + 1] - cur_ret
            reward_t = torch.tensor([reward], dtype=torch.float32, device=device)
            pnl_t = action_t * reward_t

            eps = 1e-8
            pnl_t_norm = pnl_t / (math.sqrt(rms.var) + eps)
            trans_cost = float(tc * torch.abs(action_t - prev_action).item())
            pnl_net_t = pnl_t_norm - trans_cost

            entropy_reg = - (action_t * torch.log(torch.abs(action_t) + 1e-8)).mean()
            loss_policy = -pnl_net_t.mean() - 1e-3 * entropy_reg

            opt_policy.zero_grad()
            loss_policy.backward()
            torch.nn.utils.clip_grad_norm_(transformer.parameters(), max_norm=1.0)
            opt_policy.step()

            pnl_scalar = float(pnl_net_t.detach().cpu().numpy().squeeze())
            total_policy_loss += float(loss_policy.detach().cpu().numpy())
            total_recon += recon_loss
            total_kl += kl_loss
            cumulative_pnl  += pnl_scalar

            # STOP-LOSS CHECK
            stop_triggered = False
            if cumulative_pnl <= stop_loss_threshold:
                pnl_scalar -= abs(stop_loss_penalty)
                action_t = torch.zeros_like(action_t)
                stop_triggered = True
                stop_loss_count += 1
                cumulative_pnl = 0.0

            portfolio_returns.append(pnl_scalar)
            prev_action = float(action_t.detach().cpu().numpy().squeeze())

            # compute weight: mix BOCPD surprise and reward magnitude
            w_cp = float(change_prob)
            w_ret = abs(pnl_scalar)
            weight = float(replay_alpha_cp * w_cp + (1.0 - replay_alpha_cp) * w_ret + 1e-8)

            # --- Store transition in buffer ---
            buffer.push(seq_inp.squeeze(0).astype(np.float32),
                        float(action_t.detach().cpu().numpy().squeeze()),
                        float(pnl_scalar),
                        None,
                        False,
                        weight,
                        None
                    )

            if cp_flag == 1:
                buffer.upweight_recent(window=200, multiplier=joint_params['wt_multplier'])

            # periodic updates (make sampling deterministic by reseeding right before sample)
            if ((buffer.size() >= joint_params['buffer_size_updates']) and (step % 8 == 0)):
                # reseed numpy/random before sampling to ensure consistent sampled indices
                np.random.seed(step_seed + 12345)
                random.seed(step_seed + 12345)

                batch = buffer.sample(joint_params['sample_batch_size'])

                # prepare tensors
                states = torch.tensor(np.stack([b.state for b in batch]), dtype=torch.float32, device=device)
                actions = torch.tensor(np.stack([b.action for b in batch]), dtype=torch.float32, device=device).unsqueeze(-1)
                rewards = torch.tensor(np.stack([b.reward for b in batch]), dtype=torch.float32, device=device).unsqueeze(-1)
                sample_weights = [b.weight for b in batch]
                sample_weights_t = torch.tensor(sample_weights, dtype=torch.float32, device=device).unsqueeze(-1)

                # compute predicted actions and weighted loss (policy)
                with torch.no_grad():
                    z_placeholder = torch.zeros(states.size(0), trafo_params['z_dim'], device=device)
                z_placeholder_expand = z_placeholder.unsqueeze(1).repeat(1, seq_len_for_vae, 1)
                states_full_input = torch.cat([states, 8 * z_placeholder_expand], dim=-1)
                pred_actions = torch.tanh(transformer(states_full_input))

                per_sample_loss = - (pred_actions * rewards)
                # weighted_loss = (per_sample_loss * sample_weights_t).mean()
                weighted_loss = (per_sample_loss).mean()

                # optionally add small L2 regularization on policy parameters to avoid collapse
                if trafo_params.get('trafo_l2', 0.0) > 0.0:
                    l2_reg = 0.0
                    for p in transformer.parameters():
                        l2_reg += (p**2).sum()
                    weighted_loss = weighted_loss + trafo_params['trafo_l2'] * l2_reg

                opt_policy.zero_grad()
                weighted_loss.backward()
                torch.nn.utils.clip_grad_norm_(transformer.parameters(), max_norm=1.0)
                opt_policy.step()
                policy_loss = per_sample_loss.mean()
                total_policy_loss += float(policy_loss.detach().cpu().item())

        if stop_loss_count > 0:
            print(f"Stop-loss triggered for {stop_loss_count} PnLs")
        
        avg_recon = total_recon / T
        avg_kl = total_kl / T
        avg_policy = total_policy_loss / T
        print(f"Epoch {epoch:03d} | recon={avg_recon:.4f} | kl={avg_kl:.4f} | policy={avg_policy:.4f}")

        val_metrics = evaluate_strategy(portfolio_returns[seq_len_for_vae:])
        val_sharpe = val_metrics["sharpe_ratio"]
        print(f"Train Epoch Sharpe={val_sharpe:.3f}")

        if val_sharpe > best_val_sharpe + min_delta:
            best_val_sharpe = val_sharpe
            es_counter = 0  # reset patience counter
            meta = {"epoch": epoch, "recon loss": avg_recon, "kl loss": avg_kl, "train_sharpe": val_sharpe}
            bocpd_cfg = {"bocpd_hazard": bocpd_hazard}
            save_models(save_dir, transformer, encoder, opt_policy, opt_vae, bocpd_cfg, meta)
            print(f"Saved best models at epoch {epoch:03d} (Train Sharpe={val_sharpe:.3f})")
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
    print("Transformer policy training complete.")

# # Optional: quick test
# if __name__ == "__main__":
#     # dummy_series = pd.Series(np.random.randn(2000),
#     #                          index=pd.date_range("2020-01-01", periods=2000))
#     train_loop_transformer(train_spread, num_epochs = 1)
#     print("train_loop (Transformer) ran successfully!")
