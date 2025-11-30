#------------------------------------------------------------------------------------
# bocpd_vae_cnnlstm_tuner.py
#------------------------------------------------------------------------------------
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
from ml_dl_models.cnn_lstm import CNNLSTMModel, build_optimizer
from util.weighted_replay_buffer import WeightedReplayBuffer
from tuning.bocpd_vae_tuner import BOCPD_VAE_Tuner
from util.seed_random import seed_random

class BOCPD_VAE_CNNLSTM_Tuner(BOCPD_VAE_Tuner):
    #CNN LSTM hyperparameter ranges
    default_cnnlstm_space = {
        "input_dim": [2],
        "cnn_channels": [32, 48],
        "kernel_size": [3, 5],
        "z_dim": [16], #[16, 24, 32]
        "hidden_dim": [64, 128, 192],
        "optimizer": ["Adam", "AdamW"],
        "lr": [1e-4, 5e-4, 1e-5],
        "weight_decay": [0, 1e-5],
        "grad_clip": [0.5, 1.0],
        "cnnlstm_l2": [1e-5, 1e-4, 1e-3, 1e-2], 
        "gamma": [0.95, 0.99, 1.0]
    }

    #Joint hyperparameter ranges
    default_joint_space = {
        "state_window": [25, 50],
        "base_action_sigma": [0.01, 0.1],
        "wt_multplier": [1.5, 1.8],
        "buffer_size_updates": [16, 64, 128, 256],
        "sample_batch_size": [8, 16, 64, 128],
        "transaction_cost": [0.001, 0.01, 0.1, 0.2],
        "exploration_alpha": [5.0, 6.5, 10.0]
    }

    #Best CNN-LSTM hyperparameters configured
    best_cnnlstm_params = {
        "input_dim" : 2,
        "cnn_channels": 32,
        "kernel_size": 3,
        "z_dim": 16,
        "hidden_dim": 128,
        "optimizer": "AdamW",
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "grad_clip": 0.5,
        "cnnlstm_l2": 1e-3,
        "gamma": 0.99
    }

    #Best Joint hyperparameters configured
    best_joint_params = {
        "state_window": 25,
        "base_action_sigma": 0.01,
        "wt_multplier": 1.5,
        "buffer_size_updates": 256,
        "sample_batch_size": 64,
        "transaction_cost": 0.001,
        "exploration_alpha": 6.5
    }

    def __init__(self, custom_bocpd_space: Dict[str, List[Any]]=None,
                 custom_vae_space: Dict[str, List[Any]]=None,
                 custom_cnnlstm_space: Dict[str, List[Any]]=None,
                 custom_joint_space: Dict[str, List[Any]]=None):
        """
        Args:
            custom_bocpd_space: dict of BOCPD, StudentT hyperparameter ranges
            custom_vae_space: dict of VAE hyperparameter ranges
            custom_cnnlstm_space: dict of CNN-LSTM hyperparameter ranges
            custom_joint_space: dict for joint tuning of BOCPD, VAE, and CNN-LSTM
        """
        super().__init__(custom_bocpd_space, custom_vae_space)
        self.cnnlstm_space = BOCPD_VAE_CNNLSTM_Tuner.default_cnnlstm_space.copy()
        if custom_cnnlstm_space:
            self.cnnlstm_space.update(custom_cnnlstm_space)
        self.joint_space = BOCPD_VAE_CNNLSTM_Tuner.default_joint_space.copy()
        if custom_joint_space:
            self.joint_space.update(custom_joint_space)
        self.best_cnnlstm_params = BOCPD_VAE_CNNLSTM_Tuner.best_cnnlstm_params.copy()
        self.best_joint_params = BOCPD_VAE_CNNLSTM_Tuner.best_joint_params.copy()

    def tune_cnnlstm(self, data, change_probs, cpflags, mus):
        """
        Auto tune CNN-LSTM hybrid model hyperparameters
        Input:
            data: Training data spread
            change_probs: Change probabilities from BOCPD
            cpflags: Change point flags from BOCPD
            mus: Latent variables from VAE

        Exit:
            Save and return the best hyperparameters
            Return the best score
        """
        # initialize random seed
        seed_random()
        stop_loss_threshold=-0.02 # Hardcoded for now (maybe tunable like hyperparams)
        stop_loss_penalty=0.001 # Hardcoded for now

        transaction_cost = self.best_joint_params['transaction_cost']  # optional, small transaction cost per trade
        replay_alpha_cp = 0.6      # weight mix: alpha*cp + (1-alpha)*|reward|
        base_action_sigma = self.best_joint_params['base_action_sigma']
        state_window = self.best_joint_params['state_window']
        seq_len_for_vae = self.best_vae_params['vae_seq_len']
        device = 'cpu'
        best_score, best_params = -np.inf, None

        for params in self._grid(self.cnnlstm_space):
            gamma = params['gamma'] # to disable set to 1.0 (discount factor) (same as RL)
            state_returns = [0.0]*(state_window-1)
            state_returns.append(data.iloc[0])
            # Modify CNNLSTMModel to take seq_inp_t and z_t
            policy_cnnlstm = CNNLSTMModel(
                              input_dim=params['input_dim'],
                              cnn_channels=params['cnn_channels'],
                              hidden_dim=params['hidden_dim'],
                              kernel_size=params['kernel_size'],
                              z_dim=params['z_dim'],
                              seq_len=seq_len_for_vae
                            ).to(device)

            opt_policy = build_optimizer(
                            policy_cnnlstm,
                            optimizer_name=params['optimizer'],
                            lr=params['lr'],
                            weight_decay=params['weight_decay']
                        )

            rms = RunningMeanStd()
            buffer = WeightedReplayBuffer(capacity=30000)
            prev_action = torch.tensor(0.0, dtype=torch.float32, device=device)  # tensor on device
            total_policy_loss = 0.0
            discounted_pnl = 0.0
            cumulative_pnl = 0.0
            stop_loss_count = 0
            pnls = []

            for i in range(len(data) - 1):
                rms.update([data.iloc[i]]) # Use iloc for pandas Series
                # --- Policy (CNN-LSTM) ---
                seq_start = max(0, i - seq_len_for_vae + 1)
                seq_rets = data[seq_start: i + 1]
                cps_seq = change_probs[seq_start: i + 1]
                # pad if needed
                if len(seq_rets) < seq_len_for_vae:
                    pad = np.zeros(seq_len_for_vae - len(seq_rets))
                    seq_rets = np.concatenate([pad, seq_rets])
                    cps_pad = np.zeros(seq_len_for_vae - len(cps_seq))
                    cps_seq = np.concatenate([cps_pad, cps_seq])
                # form encoder input: (seq_len, input_dim) where input_dim = [norm_ret, change_prob]
                seq_inp = np.stack([ (seq_rets - rms.mean) / (math.sqrt(rms.var)+1e-8),
                                      cps_seq ], axis=-1)[None, ...]  # batch=1 # (1, seq_len_for_vae, 2)
                seq_inp_t = torch.tensor(seq_inp, dtype=torch.float32).to(device)

                with torch.no_grad():
                    z_t = mus[i]
                action_t = torch.tanh(policy_cnnlstm(seq_inp_t, z_t.detach()))  # [-1, 1] #z_t.detach()
                action_t = torch.clamp(action_t, -1.0, 1.0)

                reward = data[i + 1] - data[i]
                # --- PnL computation (with gradient flow) ---
                reward_t = torch.tensor([reward], dtype=torch.float32, device=device)
                pnl_t = action_t * reward_t
                # normalize reward by volatility and include transaction costs
                eps = 1e-8
                pnl_t_norm = pnl_t / (math.sqrt(rms.var) + eps)
                #tc = transaction_cost * torch.abs(action_t - prev_action)
                #trans_cost = transaction_cost * float(np.abs((action_t.detach().cpu().numpy().squeeze()) - prev_action).sum())  # sum if vector action
                trans_cost = float(transaction_cost * torch.abs(action_t - prev_action).item())
                pnl_net_t = pnl_t_norm - trans_cost  # subtract cost # maximize pnl

                # entropy-like penalty, This discourages the LSTM from pushing
                # outputs to extremes (−1 or +1) unless strongly justified
                entropy_reg = - (action_t * torch.log(torch.abs(action_t) + 1e-8)).mean()
                loss_policy = -pnl_net_t.mean() - 1e-3 * entropy_reg

                opt_policy.zero_grad()
                loss_policy.backward()
                torch.nn.utils.clip_grad_norm_(policy_cnnlstm.parameters(), params['grad_clip'])
                opt_policy.step()

                pnl_scalar = float(pnl_net_t.detach().cpu().numpy().squeeze())
                total_policy_loss += float(loss_policy.detach().cpu().numpy())
                cumulative_pnl  += pnl_scalar

                # STOP-LOSS CHECK
                stop_triggered = False
                if cumulative_pnl <= stop_loss_threshold:
                    pnl_scalar -= abs(stop_loss_penalty)   # penalize hitting stop-loss
                    action_t = torch.zeros_like(action_t) # force close position
                    stop_triggered = True
                    stop_loss_count += 1
                    cumulative_pnl = 0.0

                pnls.append(pnl_scalar)
                prev_action = float(action_t.detach().cpu().numpy().squeeze())

                # compute weight: mix BOCPD surprise and reward magnitude
                w_cp = float(change_probs[i])
                w_ret = abs(pnl_scalar)
                weight = float(replay_alpha_cp * w_cp + (1.0 - replay_alpha_cp) * w_ret + 1e-8)
                # --- Store transition in buffer (for future stability) ---
                buffer.push(seq_inp.squeeze(0).astype(np.float32),
                            float(action_t.detach().cpu().numpy().squeeze()),
                            float(pnl_scalar), #.item(),
                            None,
                            False,
                            weight,
                            None
                        )

                # Upweight near detected changes
                if cpflags[i] == 1:
                    buffer.upweight_recent(window=200, multiplier=self.best_joint_params['wt_multplier'])

                # periodic updates
                if ((buffer.size() >= self.best_joint_params['buffer_size_updates']) and (i % 8 == 0)):
                    batch = buffer.sample(self.best_joint_params['sample_batch_size'])

                    # prepare tensors
                    states = torch.tensor(np.stack([b.state for b in batch]), dtype=torch.float32, device=device)
                    actions = torch.tensor(np.stack([b.action for b in batch]), dtype=torch.float32, device=device).unsqueeze(-1)
                    rewards = torch.tensor(np.stack([b.reward for b in batch]), dtype=torch.float32, device=device).unsqueeze(-1)
                    sample_weights = [b.weight for b in batch]
                    sample_weights_t = torch.tensor(sample_weights, dtype=torch.float32, device=device).unsqueeze(-1)

                    # compute predicted actions and weighted loss (policy)
                    with torch.no_grad():
                        z_placeholder = torch.zeros(states.size(0), params['z_dim'], device=device)  # if you want to include z, adapt
                    #policy_actions = policy_cnnlstm(torch.cat([states, z_placeholder], dim=-1))
                    policy_actions = policy_cnnlstm(states, z_placeholder)
                    pred_actions = torch.tanh(policy_actions )

                    # policy loss: -pred_actions * reward (we want actions that produce positive reward)
                    per_sample_loss = - (pred_actions * rewards)  # (N,1)
                    weighted_loss = (per_sample_loss * sample_weights_t).mean()

                    opt_policy.zero_grad()
                    weighted_loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy_cnnlstm.parameters(), params['grad_clip'])
                    opt_policy.step()
                    policy_loss = per_sample_loss.mean()
                    total_policy_loss += float(policy_loss.detach().cpu().item())

                # Move window
                state_returns.append(data[i + 1])

            avg_policy = total_policy_loss / len(data)

            score = evaluate_composite_score(pnls, cost_per_trade=0.001)
            print(f'CNN-LSTM score = {round(score,4)} :: params = {params}')
            if score > best_score:
                best_score, best_params = score, params

        self.best_cnnlstm_params = best_params
        return best_params, best_score

    def joint_tuning(self, data, best_lstm_params, change_probs, cpflags, mus):
        """
        Auto tune joint hyperparameters of the CNN-LSTM hybrid model
        Input:
            data: Training data spread
            best_lstm_params: best cnn-lstm parameters
            change_probs: Change probabilities from BOCPD
            cpflags: Change point flags from BOCPD
            mus: Latent variables from VAE

        Exit:
            Save and return the best joint hyperparameters
            Return the best score
        """
        # initialize random seed
        seed_random()
        stop_loss_threshold=-0.02 # Hardcoded for now (maybe tunable like hyperparams)
        stop_loss_penalty=0.001 # Hardcoded for now
        gamma = self.best_cnnlstm_params['gamma'] # discount factor (same as RL)
        seq_len_for_vae = self.best_vae_params['vae_seq_len']
        replay_alpha_cp = 0.6      # weight mix: alpha*cp + (1-alpha)*|reward|

        device = 'cpu'
        best_score, best_params = -np.inf, None
        for params in self._grid(self.joint_space):
            if params['sample_batch_size'] >= params['buffer_size_updates']:
                continue
            base_action_sigma = params['base_action_sigma']
            state_window = params['state_window']
            transaction_cost = params['transaction_cost'] # optional, small transaction cost per trade
            state_returns = [0.0]*(state_window-1)
            state_returns.append(data.iloc[0])
            rms = RunningMeanStd()
            buffer = WeightedReplayBuffer(capacity=30000)
            # Modify CNNLSTMModel to take seq_inp_t and z_t
            policy_cnnlstm = CNNLSTMModel(
                              input_dim=self.best_cnnlstm_params['input_dim'],
                              cnn_channels=self.best_cnnlstm_params['cnn_channels'],
                              hidden_dim=self.best_cnnlstm_params['hidden_dim'],
                              kernel_size=self.best_cnnlstm_params['kernel_size'],
                              z_dim=self.best_cnnlstm_params['z_dim'],
                              seq_len=seq_len_for_vae
                            ).to(device)

            opt_policy = build_optimizer(
                            policy_cnnlstm,
                            optimizer_name=self.best_cnnlstm_params['optimizer'],
                            lr=self.best_cnnlstm_params['lr'],
                            weight_decay=self.best_cnnlstm_params['weight_decay']
                        )

            prev_action = torch.tensor(0.0, dtype=torch.float32, device=device)  # tensor on device

            total_policy_loss = 0.0
            cumulative_pnl = 0.0
            stop_loss_count = 0
            pnls = []

            for i in range(len(data) - 1):
                rms.update([data.iloc[i]]) # Use iloc for pandas Series

                seq_start = max(0, i - seq_len_for_vae + 1)
                seq_rets = data[seq_start: i + 1]
                cps_seq = change_probs[seq_start: i + 1]
                # pad if needed
                if len(seq_rets) < seq_len_for_vae:
                    pad = np.zeros(seq_len_for_vae - len(seq_rets))
                    seq_rets = np.concatenate([pad, seq_rets])
                    cps_pad = np.zeros(seq_len_for_vae - len(cps_seq))
                    cps_seq = np.concatenate([cps_pad, cps_seq])
                # form encoder input: (seq_len, input_dim) where input_dim = [norm_ret, change_prob]
                seq_inp = np.stack([ (seq_rets - rms.mean) / (math.sqrt(rms.var)+1e-8),
                                      cps_seq ], axis=-1)[None, ...]  # batch=1 # (1, seq_len_for_vae, 2)
                seq_inp_t = torch.tensor(seq_inp, dtype=torch.float32).to(device)
                with torch.no_grad():
                    z_t = mus[i]
                action_t = torch.tanh(policy_cnnlstm(seq_inp_t, z_t.detach()))  # [-1, 1] #z_t.detach()
                action_t = torch.clamp(action_t, -1.0, 1.0)

                reward = data[i + 1] - data[i]
                # --- PnL computation (with gradient flow) ---
                reward_t = torch.tensor([reward], dtype=torch.float32, device=device)
                pnl_t = action_t * reward_t
                # normalize reward by volatility and include transaction costs
                eps = 1e-8
                pnl_t_norm = pnl_t / (math.sqrt(rms.var) + eps)
                #trans_cost = transaction_cost * float(np.abs((action_t.detach().cpu().numpy().squeeze()) - prev_action).sum())  # sum if vector action
                trans_cost = float(transaction_cost * torch.abs(action_t - prev_action).item())
                pnl_net_t = pnl_t_norm - trans_cost  # subtract cost # maximize pnl
                # entropy-like penalty, This discourages the LSTM from pushing
                # outputs to extremes (−1 or +1) unless strongly justified
                entropy_reg = - (action_t * torch.log(torch.abs(action_t) + 1e-8)).mean()
                loss_policy = -pnl_net_t.mean() - 1e-3 * entropy_reg

                opt_policy.zero_grad()
                loss_policy.backward()
                torch.nn.utils.clip_grad_norm_(policy_cnnlstm.parameters(), self.best_cnnlstm_params['grad_clip'])
                opt_policy.step()

                pnl_scalar = float(pnl_net_t.detach().cpu().numpy().squeeze())

                total_policy_loss += float(loss_policy.detach().cpu().numpy())
                cumulative_pnl  += pnl_scalar

                # STOP-LOSS CHECK
                stop_triggered = False
                if cumulative_pnl <= stop_loss_threshold:
                    pnl_scalar -= abs(stop_loss_penalty)   # penalize hitting stop-loss
                    action_t = torch.zeros_like(action_t) # force close position
                    stop_triggered = True
                    stop_loss_count += 1
                    cumulative_pnl = 0.0

                pnls.append(pnl_scalar)
                prev_action = float(action_t.detach().cpu().numpy().squeeze())

                # compute weight: mix BOCPD surprise and reward magnitude
                w_cp = float(change_probs[i])
                w_ret = abs(pnl_scalar)
                weight = float(replay_alpha_cp * w_cp + (1.0 - replay_alpha_cp) * w_ret + 1e-8)
                # --- Store transition in buffer (for future stability) ---
                buffer.push(seq_inp.squeeze(0).astype(np.float32),
                            float(action_t.detach().cpu().numpy().squeeze()),
                            float(pnl_scalar), #.item(),
                            None,
                            False,
                            weight,
                            None
                          )

                # if change_prob large, upweight recent transitions
                if cpflags[i] == 1:
                    buffer.upweight_recent(window=200, multiplier=params['wt_multplier'])

                # periodic updates
                if ((buffer.size() >= params['buffer_size_updates']) and (i % 8 == 0)):
                    batch = buffer.sample(params['sample_batch_size'])

                    # prepare tensors
                    states = torch.tensor(np.stack([b.state for b in batch]), dtype=torch.float32, device=device)
                    actions = torch.tensor(np.stack([b.action for b in batch]), dtype=torch.float32, device=device).unsqueeze(-1)
                    rewards = torch.tensor(np.stack([b.reward for b in batch]), dtype=torch.float32, device=device).unsqueeze(-1)
                    sample_weights = [b.weight for b in batch]
                    sample_weights_t = torch.tensor(sample_weights, dtype=torch.float32, device=device).unsqueeze(-1)

                    # compute predicted actions and weighted loss (policy)
                    with torch.no_grad():
                        z_placeholder = torch.zeros(states.size(0), self.best_cnnlstm_params['z_dim'], device=device)  # if you want to include z, adapt
                    policy_actions = policy_cnnlstm(states, z_placeholder)
                    #policy_actions  = policy_cnnlstm(torch.cat([states, z_placeholder], dim=-1))
                    pred_actions = torch.tanh(policy_actions )

                    # policy loss: -pred_actions * reward (we want actions that produce positive reward)
                    per_sample_loss = - (pred_actions * rewards)  # (N,1)
                    weighted_loss = (per_sample_loss * sample_weights_t).mean()

                    opt_policy.zero_grad()
                    weighted_loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy_cnnlstm.parameters(), self.best_cnnlstm_params['grad_clip'])
                    opt_policy.step()

                    policy_loss = per_sample_loss.mean()
                    total_policy_loss += float(policy_loss.detach().cpu().item())

                state_returns.append(data[i + 1])

            score = evaluate_composite_score(pnls, cost_per_trade=0.001)
            print(f'Joint Tuning score = {round(score,4)} :: params = {params}')
            if score > best_score:
                best_score, best_params = score, params

        self.best_joint_params = best_params
        return best_params, best_score

    def tune(self, data):
        """
        Auto tune hyperparameters main function for CNN-LSTM hybrid model
        Input:
            data: Training data spread
        
        Exit:
            Display the best hyperparameters
            Display the best score
        """
        best_bocpd_params, best_bocpd_score, cps, cpflags, runtime_len = self.tune_bocpd(data)
        print(f"Best BOCPD parameters: {best_bocpd_params}")
        print(f"Best BOCPD score: {round(best_bocpd_score,3)}")

        best_vae_params, best_vae_core, z_t = self.tune_vae(data, cps)
        print(f"Best vae parameters: {best_vae_params}")
        print(f"Best vae score: {round(best_vae_core,4)}")

        best_cnnlstm_params, best_cnnlstm_score = self.tune_cnnlstm(data, cps, cpflags, z_t)
        print(f"Best CNN-LSTM parameters: {best_cnnlstm_params}")
        print(f"Best CNN-LSTM score: {round(best_cnnlstm_score,4)}")

        best_joint_tuning_params, best_joint_tuning_score = self.joint_tuning(data, best_cnnlstm_params, cps, cpflags, z_t)
        print(f"Best Joint Tuning parameters: {best_joint_tuning_params}")
        print(f"Best Joint Tuning score: {round(best_joint_tuning_score,4)}")


    @property
    def best_params(self):
        return self.best_bocpd_params, self.best_vae_params, self.best_cnnlstm_params, self.best_joint_params
