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
from ml_dl_models.cnn_lstm import CNNLSTMModel
from util.weighted_replay_buffer import WeightedReplayBuffer
from util.eval_strategy import evaluate_strategy
from util.models_io import save_models


def train_loop_cnnlstm(
    stream,
    num_epochs = 50,
    save_dir="checkpoints_cnnlstm",
    state_window=50, # This will now be used to define the input_dim for the CNN-LSTM if needed separately, or removed. Let's keep it for now but the sequence length is seq_len_for_vae
    seq_len_for_vae=50,
    total_steps=10000,
    bocpd_hazard=300.0,
    gamma=0.99,                # discount factor (same as RL)
    transaction_cost=0.0005,   # optional, small transaction cost per trade
    device='cpu'
):
    # ----- Data setup -----
    if isinstance(stream, pd.Series):
        data = stream.values
    else:
        data = np.asarray(stream)

    bocpd = BOCPD(ConstantHazard(bocpd_hazard), StudentT(mu=0, kappa=1, alpha=1, beta=1))
    input_dim_vae = 2 # [normalized_return, change_prob] for VAE sequence input
    z_dim = 16
    # state_dim = state_window # This might not be needed as a separate input dimension if we use the sequence

    encoder = VAEEncoder(input_dim=input_dim_vae, hidden_dim=128, z_dim=z_dim, seq_len=seq_len_for_vae).to(device)
    # Modify CNNLSTMModel to take seq_inp_t and z_t
    policy_cnnlstm = CNNLSTMModel(input_dim=input_dim_vae, hidden_dim=128, z_dim=z_dim, seq_len=seq_len_for_vae).to(device)


    opt_vae = optim.Adam(encoder.parameters(), lr=1e-3)
    opt_policy = optim.Adam(policy_cnnlstm.parameters(), lr=1e-4)
    best_val_sharpe = -np.inf
    buffer = WeightedReplayBuffer(capacity=30000)

    # ----- Training loop -----
    for epoch in range(num_epochs):
        rms = RunningMeanStd()
        T = min(total_steps, len(data) - seq_len_for_vae - 1) # Adjust T based on seq_len_for_vae

        rms.update(data[:seq_len_for_vae]) # Initialize RMS with first sequence
        idx = 0 # Start index for the current return
        # state_returns = list(data[idx: idx + state_window]) # This state representation might change
        # idx += state_window # Adjust starting index

        total_recon, total_kl, total_policy_loss = 0, 0, 0
        total_pnl = 0.0
        discounted_pnl = 0.0
        rt_mle = [0]*seq_len_for_vae # Adjust initial rt_mle length
        cp_flag_list = [0]*seq_len_for_vae # Adjust initial cp_flag_list length
        actions_pnl = [0]*seq_len_for_vae # Adjust initial actions_pnl length
        prev_action = 0.0  # for transaction cost calc

        # Start processing after the initial sequence for VAE
        idx = seq_len_for_vae

        for step in trange(T):
            cur_ret = data[idx]
            rms.update([cur_ret])
            norm_ret = float((cur_ret - rms.mean) / (math.sqrt(rms.var) + 1e-8))

            # --- BOCPD change-point probability ---
            change_prob = bocpd.update(norm_ret)
            rt_mle.append(bocpd.rt)
            cp_flag = 1 if rt_mle[idx] < rt_mle[idx-1] else 0
            cp_flag_list.append(cp_flag)

            # --- Encoder (VAE) and Policy (CNNLSTM) Input ---
            # Use the sequence ending at the current index (idx)
            seq_start = max(0, idx - seq_len_for_vae + 1)
            seq_rets = data[seq_start: idx + 1] # length seq_len_for_vae
            # No padding needed here if T is adjusted correctly

            # form encoder/policy input: (seq_len, input_dim_vae) where input_dim_vae = [norm_ret, change_prob]
            # We need the historical change probabilities for the sequence input to the VAE/Policy
            # This requires storing past change probabilities or recomputing them for the sequence.
            # Let's simplify for now and use the current change_prob for the whole sequence input to VAE/Policy,
            # but ideally, we'd use the change probability at each timestep in the sequence.
            # For now, we'll use the current change_prob broadcasted.
            # A more accurate approach would require a BOCPD history.

            # Simplified seq_inp using current change_prob (less accurate but matches original VAE input approach)
            seq_norm = (seq_rets - rms.mean) / (math.sqrt(rms.var) + 1e-8)
            seq_inp = np.stack([
                seq_norm,
                np.ones_like(seq_norm) * change_prob # Using current change_prob for all steps in sequence
            ], axis=-1)[None, ...] # (1, seq_len_for_vae, input_dim_vae)

            seq_inp_t = torch.tensor(seq_inp, dtype=torch.float32).to(device)

            # --- Encoder (VAE) ---
            x_hat, mu, logvar, z_t = encoder(seq_inp_t)
            kl_w = min(1.0, epoch / 100)
            loss_vae, recon_loss, kl_loss = vae_loss(seq_inp_t, x_hat, mu, logvar, kl_weight=kl_w)
            opt_vae.zero_grad(); loss_vae.backward(); opt_vae.step()

            # --- Policy (CNNLSTM) ---
            # Pass the sequence input and the latent variable to the CNNLSTMModel
            action_t = torch.tanh(policy_cnnlstm(seq_inp_t, z_t.detach()))  # [-1, 1]


            next_ret = data[idx + 1] # This is the return *after* the current index (idx)

            # --- PnL computation (with gradient flow) ---
            next_ret_t = torch.tensor([next_ret], dtype=torch.float32, device=device)
            pnl = action_t * next_ret_t
            tc = transaction_cost * torch.abs(action_t - prev_action)
            pnl = pnl - tc  # subtract cost

            loss_policy = -pnl.mean()  # maximize pnl

            opt_policy.zero_grad()
            loss_policy.backward()
            opt_policy.step()

            total_policy_loss += loss_policy.item()
            total_pnl += pnl.item()
            actions_pnl.append(pnl.item())
            discounted_pnl += (gamma ** step) * pnl.item()

            prev_action = action_t.detach()  # store detached value

            total_recon += recon_loss
            total_kl += kl_loss

            # --- Store transition in buffer (for future stability) ---
            # Store the sequence input used for VAE/Policy
            buffer.push(seq_inp.squeeze(0).astype(np.float32), # Store sequence as state
                        action_t.detach().cpu().numpy().astype(np.float32),
                        pnl, #.item(), # Store PnL as reward
                        None, False, 1.0,
                        seq_inp.squeeze(0).astype(np.float32) # Store sequence again for consistency? Or maybe next state? Let's store seq_inp.
                       )


            # Upweight near detected changes
            if cp_flag == 1:
                buffer.upweight_recent(window=200, multiplier=1.8)

            # Move window - idx already points to the current return, so just increment
            idx += 1

        avg_recon = total_recon / T # Use T as the number of steps
        avg_kl = total_kl / T
        avg_policy = total_policy_loss / T
        print(f"Epoch {epoch:03d} | recon={avg_recon:.4f} | kl={avg_kl:.4f} | policy={avg_policy:.4f}")

        # ============================================================
        #   Save models for best Sharpe ratio
        # ============================================================
        # Evaluate strategy on the PnL generated during training for this epoch
        val_metrics = evaluate_strategy(actions_pnl[seq_len_for_vae:]) # Evaluate PnL after the initial sequence
        val_sharpe = val_metrics["sharpe_ratio"]

        print(f"Train Epoch Sharpe={val_sharpe:.3f}") # Renamed to avoid confusion with actual validation set

        # --- save best checkpoint ---
        if val_sharpe > best_val_sharpe:
            best_val_sharpe = val_sharpe
            meta = {"epoch": epoch, "recon loss": avg_recon, "kl loss": avg_kl, "train_sharpe": val_sharpe}
            bocpd_cfg = {"bocpd_hazard": bocpd_hazard}
            save_models(save_dir, policy_cnnlstm, encoder, opt_policy, opt_vae, bocpd_cfg, meta)
            print(f"Saved best models at epoch {epoch:03d} (Train Sharpe={val_sharpe:.3f})")

    print("CNN-LSTM policy training complete.")

# # Optional: quick test
# if __name__ == "__main__":
#     # dummy_series = pd.Series(np.random.randn(2000),
#     #                          index=pd.date_range("2020-01-01", periods=2000))
#     train_loop_cnnlstm(train_spread, num_epochs = 1)
#     print("train_loop (CNN-LSTM) ran successfully!")
