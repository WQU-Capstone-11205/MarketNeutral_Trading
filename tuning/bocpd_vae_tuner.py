import torch
from torch import nn, optim
from typing import Dict, Any, List, Callable
import math
import numpy as np
import random
from typing import Dict, List, Any
from itertools import product

from util.running_mean_std import RunningMeanStd
from metrics.stats import evaluate_composite_score
from util.seed_random import seed_random
from ml_dl_models.actor_critic import Actor
from ml_dl_models.actor_critic import Critic
from util.weighted_replay_buffer import WeightedReplayBuffer
from structural_break.bocpd import BOCPD
from structural_break.hazard import ConstantHazard
from structural_break.distribution import StudentT
from ml_dl_models.rnn_vae import VAEEncoder, vae_loss

# ------------------------------------------------------------
# Hyperparameter tuning class
# ------------------------------------------------------------
class BOCPD_VAE_Tuner:
    default_bocpd_space = {
        "hazard": [50, 100, 250],
        "mu": [0, 1, 2],
        "kappa": [0.1, 1.0, 10.0],
        "alpha": [0.1, 1.0, 10.0],
        "beta": [0.1, 1.0, 10.0]
    }

    #This is a bit concise form to get the best score
    default_vae_space = {
        "input_dim": [2],
        "latent_dim": [16],
        "hidden_dim": [32, 64, 128, 256],
        "lr": [1e-2, 1e-3, 1e-4],
        "vae_seq_len": [1, 5, 25],
        "kl_wt": [0.0005, 0.0001, 0.005, 0.001, 0.01]
    }

    best_bocpd_params = {
        "hazard": 50,
        "mu": 0,
        "kappa": 0.1,
        "alpha": 1.0,
        "beta": 0.1
    }

    best_vae_params = {
        "input_dim": 2,
        "latent_dim": 16,
        "hidden_dim": 128,
        "lr": 1e-3,
        "vae_seq_len": 1,
        "kl_wt": 0.001
    }

    def __init__(self, custom_bocpd_space: Dict[str, List[Any]]=None,
                 custom_vae_space: Dict[str, List[Any]]=None):
        """
        Args:
            bocpd_space: dict of BOCPD hyperparameter ranges
            vae_space: dict of VAE hyperparameter ranges
        """
        self.bocpd_space = BOCPD_VAE_Tuner.default_bocpd_space.copy()
        if custom_bocpd_space:
            self.bocpd_space.update(custom_bocpd_space)
        self.vae_space = BOCPD_VAE_Tuner.default_vae_space.copy()
        if custom_vae_space:
            self.vae_space.update(custom_vae_space)
        self.best_bocpd_params = BOCPD_VAE_Tuner.best_bocpd_params.copy()
        self.best_vae_params = BOCPD_VAE_Tuner.best_vae_params.copy()

    def tune_bocpd(self, data):
        """
        Auto tune BOCPD model hyperparameters
        Input:
            data: Training data spread
 
        Exit:
            Save and return the best hyperparameters
            Return the best score
            best_change_probs: Best Change probabilities from BOCPD
            best_cp_flags: Best Change point flags from BOCPD
            best_rts: Best most likely run length estimates
        """
        seed_random()
        best_score, best_params = -np.inf, None
        best_change_probs, best_cp_flags, best_rts = None, None, None
        for params in self._grid(self.bocpd_space):
            model = BOCPD(ConstantHazard(params['hazard']), StudentT(params['mu'], params['kappa'], params['alpha'], params['beta']))
            rms = RunningMeanStd()
            for i in range(len(data)):
                cur_ret = data.iloc[i] # Use iloc for pandas Series
                rms.update([cur_ret])
                # BOCPD expects scalar observation -> use normalized return
                norm_ret = float((cur_ret - rms.mean) / (math.sqrt(rms.var) + 1e-8))
                model.update(norm_ret)

            change_probs, rt_mle, cp_flags = model.results
            change_probs = np.array(change_probs)
            score = -np.sum(change_probs > 0.5)
            print(f'bocpd score = {round(score, 3)} :: params = {params}')
            if score > best_score:
                best_score, best_params = score, params
                best_change_probs, best_cp_flags, best_rts = change_probs, cp_flags, rt_mle

        self.best_bocpd_params = best_params
        return best_params, best_score, best_change_probs, best_cp_flags, best_rts

    def tune_vae(self, data, cp_probs):
        """
        Auto tune VAE model hyperparameters
        Input:
            data: Training data spread
            cp_probs: Change probabilities from BOCPD model
 
        Exit:
            Save and return the best hyperparameters
            Return the best score
            best_z_ts: Best latent variables from VAE
        """
        seed_random()
        best_score, best_params = -np.inf, None
        best_z_ts = None
        for params in self._grid(self.vae_space):
            device = 'cpu'
            seq_len_vae = params['vae_seq_len']
            encoder = VAEEncoder(input_dim=params['input_dim'], hidden_dim=params['hidden_dim'], z_dim=params['latent_dim'], seq_len=seq_len_vae).to(device)
            opt_vae = optim.Adam(encoder.parameters(), lr=params['lr'])
            rms = RunningMeanStd()
            total_recon_loss = 0.0
            total_kl_loss = 0.0
            total_vae_loss = 0.0
            state_returns = np.array([0.0]*seq_len_vae)
            z_ts = []
            for i in range(len(data)):
                seq_ret = data.iloc[i] # Use iloc for pandas Series
                rms.update([seq_ret])
                change_prob = cp_probs[i] # cp_probs is now a 1D array
                # build encoder input sequence (seq_len_for_vae)
                seq_start = max(0, i - seq_len_vae + 1)
                seq_rets = data[seq_start: i + 1]
                cps_seq = cp_probs[seq_start: i + 1] ####
                if i == 0:
                    cur_ret = data.iloc[i]
                else:
                    cur_ret = data.iloc[i] - data.iloc[i-1]
                state_returns = np.append(state_returns, cur_ret)
                # pad if needed
                if len(seq_rets) < seq_len_vae:
                    pad = np.zeros(seq_len_vae - len(seq_rets))
                    seq_rets = np.concatenate([pad, seq_rets])
                    cps_pad = np.zeros(seq_len_vae - len(cps_seq))
                    cps_seq = np.concatenate([cps_pad, cps_seq])
                # form encoder input: (seq_len, input_dim) where input_dim = [norm_ret, change_prob]
                seq_inp = np.stack([ (seq_rets - rms.mean) / (math.sqrt(rms.var)+1e-8),
                                      cps_seq ], axis=-1)[None, ...]  # batch=1
                seq_inp_t = torch.tensor(seq_inp, dtype=torch.float32).to(device)

                # # --- VAE encoder ---
                x_hat, mu, logvar, z_t = encoder(seq_inp_t)
                z_ts.append(mu)
                loss_vae, recon_loss, kl_loss = vae_loss(seq_inp_t, x_hat, mu, logvar, kl_weight=params['kl_wt'])
                opt_vae.zero_grad(); loss_vae.backward(); opt_vae.step()
                total_recon_loss += recon_loss
                total_kl_loss += kl_loss #.item()
                total_vae_loss += loss_vae.item()

            score = -total_vae_loss # VAE score
            print(f'vae score = {round(score,4)} :: params = {params}')
            if score > best_score:
                best_score, best_params = score, params
                best_z_ts = z_ts

        self.best_vae_params = best_params
        return best_params, best_score, best_z_ts

    def _grid(self, space_dict):
        keys = list(space_dict.keys())
        values = list(space_dict.values())
        for combo in product(*values):
            yield dict(zip(keys, combo))
