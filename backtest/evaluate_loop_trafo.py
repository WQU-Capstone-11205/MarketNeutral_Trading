#--------------------------------------------------------------------------
#     evaluate_loop_trafo.py
#--------------------------------------------------------------------------
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
from ml_dl_models.transformer import TransformerModel
from ml_dl_models.actor_critic import Critic
from util.weighted_replay_buffer import WeightedReplayBuffer
from util.models_io import load_models
from util.seed_random import seed_random

@torch.no_grad()
def evaluate_loop_trafo(
          stream,
          bocpd_params,
          vae_params,
          trafo_params,
          joint_params,
          load_dir="checkpoints_trafo",
          total_steps = 100000,
          device="cpu",
          stop_loss_threshold=-0.02,
          stop_loss_penalty=0.001,
          seed: int = 42
    ):
    """
    Evaluate a trained Transformer policy with VAE + BOCPD context.
    Automatically loads saved models using load_all_models().

    Args:
        stream: pd.Series or np.ndarray of returns
        bocpd_params, vae_params, trafo_params, joint_params: dict with parameters
        load_dir: path to the saved models directory (where save_all_models() stored them)
        total_steps: maximum time steps limit 
        device: 'cpu' or 'cuda'
        exploration: used if stochasticity is required for the evaluation loop
        stop_loss_threshold, stop_loss_penalty: used for early stopping
        seed: for random seeding

    Returns:
        dict with change probabilities, most likely run lengths estimates, changepoint flags, 
        reconstructed outputs of VAE, portfolio returns, actions, and portfolio returns series
    """
    # prepare and seed
    if isinstance(stream, pd.Series):
        data = stream.values
        dates = stream.index
    else:
        data = np.asarray(stream)
        dates = None

    # seed deterministically for evaluation
    seed_random(seed, device=device)

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
    
    transformer = TransformerModel(
                      input_dim=(2 * seq_len_for_vae) + trafo_params['z_dim'],
                      d_model=trafo_params['d_model'], 
                      nhead=trafo_params['nhead'], 
                      num_layers=trafo_params['num_layers'], 
                      hidden_dim=(trafo_params['hidden_dim'] * trafo_params['d_model'])
                    ).to(device)

    opt_policy = optim.Adam(transformer.parameters(), lr=trafo_params['lr'])

    bocpd_cfg, meta = load_models(load_dir, transformer, vae_encoder,
                              opt_policy, vae_opt,  device, step=None)

    T = min(total_steps, len(data) - 1)

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
    transformer.eval()
    bocpd.reset_params()

    rms = RunningMeanStd()
    rms_stats = np.load(os.path.join(load_dir, "rms_stats.npz"))
    rms.mean = rms_stats["mean"]
    rms.var = rms_stats["var"]

    state_returns = [0.0]*(state_window-1)
    state_returns.append(data[0])
    prev_action = 0.0

    all_recons = []
    portfolio_returns = []
    cp_probs = []
    actions = []
    capital = 1.0
    transaction_cost = joint_params.get('transaction_cost', 0.0)
    eps = 1e-8
    cumulative_pnl = 0.0
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
        norm_ret = (cur_ret - rms.mean) / (math.sqrt(rms.var) + 1e-8)

        change_prob, _ = bocpd.update(norm_ret)
        cp_probs.append(change_prob)

        seq_start = max(0, step - seq_len_for_vae + 1)
        seq_rets = data[seq_start: step + 1]
        cps_seq = cp_probs[seq_start: step + 1]
        if len(seq_rets) < seq_len_for_vae:
            pad = np.zeros(seq_len_for_vae - len(seq_rets))
            seq_rets = np.concatenate([pad, seq_rets])
            cps_pad = np.zeros(seq_len_for_vae - len(cps_seq))
            cps_seq = np.concatenate([cps_pad, cps_seq])

        seq_inp = np.stack([ (seq_rets - rms.mean) / (math.sqrt(rms.var)+1e-8),
                              cps_seq ], axis=-1)[None, ...]
        seq_inp_t = torch.tensor(seq_inp, dtype=torch.float32).to(device)

        with torch.no_grad():
            x_hat, mu, logvar, z_t = vae_encoder(seq_inp_t)
            recon_np = x_hat.detach().cpu().numpy()[0, :, 0]
            recon_cp = x_hat.detach().cpu().numpy()[0, :, 1]
            recon_last_norm = recon_np[-1]
            recon_last_denorm = recon_last_norm * math.sqrt(rms.var) + rms.mean
            all_recons.append(recon_last_denorm)
            recon_cp_last_norm = recon_cp[-1]
            recon_probs.append(recon_cp_last_norm)

            inp_ch0 = cur_ret
            recon_ch0 = recon_last_denorm
            inp_ch1 = change_prob
            recon_ch1 = recon_cp_last_norm
            errors_ch0.append(np.mean((inp_ch0 - recon_ch0)**2))
            errors_ch1.append(np.mean((inp_ch1 - recon_ch1)**2))

        z_t = mu.detach().squeeze(0)
        z_expand = z_t.unsqueeze(0).unsqueeze(1).repeat(1, seq_len_for_vae, 1)
        full_input = torch.cat([seq_inp_t, z_expand], dim=-1)
        action_t = torch.tanh(transformer(full_input))
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

        pnl_scalar = float(pnl_net_t.detach().cpu().numpy().squeeze())
        cumulative_pnl  += pnl_scalar

        if cumulative_pnl <= stop_loss_threshold:
            pnl_scalar -= abs(stop_loss_penalty)
            action_t = torch.zeros_like(action_t)
            stop_triggered = True
            stop_loss_count += 1
            cumulative_pnl = 0.0

        portfolio_returns.append(pnl_scalar)
        prev_action = float(action_t.detach().cpu().numpy().squeeze())
        actions.append(prev_action)

    if stop_loss_count > 0:
        print(f"Stop-loss triggered for {stop_loss_count} PnLs")

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
        'actions' : np.array(actions),
        'portfolio_returns' : np.array(portfolio_returns),
        'rets': pd.Series(portfolio_returns, index= dates[:len(dates)-1]) if dates is not None else pd.Series(portfolio_returns)
    }
    return metrics
