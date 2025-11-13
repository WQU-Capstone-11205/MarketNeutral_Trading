#------------------------------------------------------------------------------------
# bocpd_vae_lstm_tuner.py
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
from util.metrics import evaluate_composite_score
from ml_dl_models.lstm import LSTMPolicy
from util.weighted_replay_buffer import WeightedReplayBuffer
from tuning.bocpd_vae_tuner import BOCPD_VAE_Tuner
from util.seed_random import seed_random

class BOCPD_VAE_LSTM_Tuner(BOCPD_VAE_Tuner):
    default_lstm_space = {
        "z_dim": [16],
        "hidden_dim": [64, 128, 256, 512],
        "lr": [1e-4, 5e-5, 1e-5, 5e-6]
    }

    default_joint_space = {
        "state_window": [25, 50],
        "base_action_sigma": [0.01, 0.1],
        "wt_multplier": [1.5, 1.8],
        "buffer_size_updates": [16, 64, 128, 256],
        "sample_batch_size": [8, 16, 64, 128]
    }

    best_lstm_params = {
        "z_dim": 16,
        "hidden_dim": 128,
        "lr": 5e-6,
        "gamma": 0.99
    }

    best_joint_params = {
        "state_window": 25,
        "base_action_sigma": 0.1,
        "wt_multplier": 1.5,
        "buffer_size_updates": 64,
        "sample_batch_size": 8,
        "transaction_cost": 0.001
    }

    def __init__(self, custom_bocpd_space: Dict[str, List[Any]]=None,
                 custom_vae_space: Dict[str, List[Any]]=None,
                 custom_lstm_space: Dict[str, List[Any]]=None,
                 custom_joint_space: Dict[str, List[Any]]=None):
        """
        Args:
            custom_lstm_space: dict of LSTM hyperparameter ranges
            custom_joint_space: dict for joint tuning of BOCPD, VAE, and LSTM
        """
        super().__init__(custom_bocpd_space, custom_vae_space)
        self.lstm_space = BOCPD_VAE_LSTM_Tuner.default_lstm_space.copy()
        if custom_lstm_space:
            self.lstm_space.update(custom_lstm_space)
        self.joint_space = BOCPD_VAE_LSTM_Tuner.default_joint_space.copy()
        if custom_joint_space:
            self.joint_space.update(custom_joint_space)
        self.best_lstm_params = BOCPD_VAE_LSTM_Tuner.best_lstm_params.copy()
        self.best_joint_params = BOCPD_VAE_LSTM_Tuner.best_joint_params.copy()

    def tune_lstm(self, data, change_probs, cpflags, mus):
        # initialize random seed
        seed_random()
        gamma = 1.0 # currently disabled # 0.99 # discount factor (same as RL)
        transaction_cost = 0.0 # currently disabled # 0.0005,   # optional, small transaction cost per trade
        replay_alpha_cp = 0.6      # weight mix: alpha*cp + (1-alpha)*|reward|
        base_action_sigma = 0.1
        state_window = 50
        device = 'cpu'
        best_score, best_params = -np.inf, None
        
        for params in self._grid(self.lstm_space):
            state_returns = [0.0]*(state_window-1)
            state_returns.append(data.iloc[0])
            policy_lstm = LSTMPolicy(input_dim=state_window + params['z_dim'], hidden_dim=params['hidden_dim']).to(device)
            opt_policy = optim.Adam(policy_lstm.parameters(), lr=params['lr'])
            rms = RunningMeanStd()
            buffer = WeightedReplayBuffer(capacity=30000)
            prev_action = torch.tensor(0.0, dtype=torch.float32, device=device)  # tensor on device
            total_policy_loss = 0.0
            total_pnl = 0.0
            discounted_pnl = 0.0

            pnls = []
            for i in range(len(data) - 1):
                rms.update([data.iloc[i]]) # Use iloc for pandas Series
                # --- Policy (LSTM) ---
                state_arr = np.array(state_returns[-state_window:])
                state_norm = (state_arr - rms.mean) / (math.sqrt(rms.var) + 1e-8)
                state_t = torch.tensor(state_norm.astype(np.float32))[None, :].to(device)

                with torch.no_grad():
                    z_t = mus[i]
                inp_t = torch.cat([state_t, z_t.detach()], dim=-1)
                action_t = torch.tanh(policy_lstm(inp_t))  # [-1, 1]
        
                reward = data[i + 1] - data[i]
                # --- PnL computation (with gradient flow) ---
                reward_t = torch.tensor([reward], dtype=torch.float32, device=device)
                pnl_t = action_t * reward_t
                tc = transaction_cost * torch.abs(action_t - prev_action)
                pnl_net_t = pnl_t - tc  # subtract cost
                loss_policy = -pnl_net_t.mean()  # maximize pnl

                opt_policy.zero_grad()
                loss_policy.backward()
                opt_policy.step()

                pnl_scalar = float(pnl_net_t.detach().cpu().numpy().squeeze())
            
                total_policy_loss += float(loss_policy.detach().cpu().numpy())
                total_pnl += pnl_scalar
                pnls.append(pnl_scalar)
                discounted_pnl += (gamma ** i) * pnl_scalar

                prev_action = action_t.detach().clone()  # store detached value

                # compute weight: mix BOCPD surprise and reward magnitude
                w_cp = float(change_probs[i])
                w_ret = abs(pnl_scalar)
                weight = float(replay_alpha_cp * w_cp + (1.0 - replay_alpha_cp) * w_ret + 1e-8)
                # --- Store transition in buffer (for future stability) ---
                buffer.push(state_norm.astype(np.float32),
                            action_t.detach().cpu().numpy().squeeze().astype(np.float32),
                            float(pnl_scalar), #.item(),
                            None, False, weight,
                            inp_t.squeeze(0).cpu().numpy().astype(np.float32))

                # Upweight near detected changes
                if cpflags[i] == 1:
                    buffer.upweight_recent(window=200, multiplier=1.8)
                
                # periodic updates
                if ((buffer.size() >= 256) and (i % 8 == 0)):
                    batch = buffer.sample(128)
                    
                    # prepare tensors
                    states = torch.tensor(np.stack([b.state for b in batch]), dtype=torch.float32, device=device)
                    actions = torch.tensor(np.stack([b.action for b in batch]), dtype=torch.float32, device=device).unsqueeze(-1)
                    rewards = torch.tensor(np.stack([b.reward for b in batch]), dtype=torch.float32, device=device).unsqueeze(-1)
                    sample_weights = [b.weight for b in batch]
                    sample_weights_t = torch.tensor(sample_weights, dtype=torch.float32, device=device).unsqueeze(-1)

                    # compute predicted actions and weighted loss (policy)
                    with torch.no_grad():
                        z_placeholder = torch.zeros(states.size(0), params['z_dim'], device=device)  # if you want to include z, adapt
                    pred_actions = policy_lstm(torch.cat([states, z_placeholder], dim=-1))
                    pred_actions = torch.tanh(pred_actions)

                    # policy loss: -pred_actions * reward (we want actions that produce positive reward)
                    per_sample_loss = - (pred_actions * rewards)  # (N,1)
                    weighted_loss = (per_sample_loss * sample_weights_t).mean()

                    opt_policy.zero_grad()
                    weighted_loss.backward()
                    opt_policy.step()

                    total_policy_loss += float(weighted_loss.detach().cpu().numpy())
                
                # Move window
                state_returns.append(data[i + 1])

            avg_policy = total_policy_loss / len(data)

            score = evaluate_composite_score(pnls, cost_per_trade=0.001)
            print(f'LSTM score = {round(score,4)} :: params = {params}')
            if score > best_score:
                best_score, best_params = score, params

        self.best_lstm_params = best_params
        return best_params, best_score

    def joint_tuning(self, data, best_lstm_params, change_probs, cpflags, mus):
        # initialize random seed
        np.random.seed(42)
        random.seed(42)
        torch.manual_seed(42)
        
        gamma = 1.0 # currently disabled # 0.99 # discount factor (same as RL)
        transaction_cost = 0.0 # currently disabled # 0.0005,   # optional, small transaction cost per trade
        replay_alpha_cp = 0.6      # weight mix: alpha*cp + (1-alpha)*|reward|

        device = 'cpu'
        best_score, best_params = -np.inf, None
        for params in self._grid(self.joint_space):
            if params['sample_batch_size'] >= params['buffer_size_updates']:
                continue
            base_action_sigma = params['base_action_sigma']
            state_window = params['state_window']
            state_returns = [0.0]*(state_window-1)
            state_returns.append(data.iloc[0])
            rms = RunningMeanStd()
            policy_lstm = LSTMPolicy(input_dim=state_window + self.best_lstm_params['z_dim'], hidden_dim=self.best_lstm_params['hidden_dim']).to(device)
            opt_policy = optim.Adam(policy_lstm.parameters(), lr=self.best_lstm_params['lr'])
            buffer = WeightedReplayBuffer(capacity=30000)
            prev_action = torch.tensor(0.0, dtype=torch.float32, device=device)  # tensor on device

            total_policy_loss = 0.0
            total_pnl = 0.0
            discounted_pnl = 0.0
            pnls = []

            for i in range(len(data) - 1):
                rms.update([data.iloc[i]]) # Use iloc for pandas Series
                state_arr = np.array(state_returns[-state_window:])
                state_norm = (state_arr - rms.mean) / (math.sqrt(rms.var) + 1e-8)

                state_t = torch.tensor(state_norm.astype(np.float32))[None, :].to(device)
                with torch.no_grad():
                    z_t = mus[i]
                inp_t = torch.cat([state_t, z_t.detach()], dim=-1)
                action_t = torch.tanh(policy_lstm(inp_t))  # [-1, 1]

                reward = data[i + 1] - data[i]
                # --- PnL computation (with gradient flow) ---
                reward_t = torch.tensor([reward], dtype=torch.float32, device=device)
                pnl_t = action_t * reward_t
                tc = transaction_cost * torch.abs(action_t - prev_action)
                pnl_net_t = pnl_t - tc  # subtract cost
                loss_policy = -pnl_net_t.mean()  # maximize pnl

                opt_policy.zero_grad()
                loss_policy.backward()
                opt_policy.step()

                pnl_scalar = float(pnl_net_t.detach().cpu().numpy().squeeze())
            
                total_policy_loss += float(loss_policy.detach().cpu().numpy())
                total_pnl += pnl_scalar
                pnls.append(pnl_scalar)
                discounted_pnl += (gamma ** i) * pnl_scalar

                prev_action = action_t.detach().clone()  # store detached value

                # compute weight: mix BOCPD surprise and reward magnitude
                w_cp = float(change_probs[i])
                w_ret = abs(pnl_scalar)
                weight = float(replay_alpha_cp * w_cp + (1.0 - replay_alpha_cp) * w_ret + 1e-8)
                # --- Store transition in buffer (for future stability) ---
                buffer.push(state_norm.astype(np.float32),
                            action_t.detach().cpu().numpy().squeeze().astype(np.float32),
                            float(pnl_scalar), #.item(),
                            None, False, weight,
                            inp_t.squeeze(0).cpu().numpy().astype(np.float32))#.astype(np.float32))

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
                        z_placeholder = torch.zeros(states.size(0), self.best_lstm_params['z_dim'], device=device)  # if you want to include z, adapt
                    pred_actions = policy_lstm(torch.cat([states, z_placeholder], dim=-1))
                    pred_actions = torch.tanh(pred_actions)

                    # policy loss: -pred_actions * reward (we want actions that produce positive reward)
                    per_sample_loss = - (pred_actions * rewards)  # (N,1)
                    weighted_loss = (per_sample_loss * sample_weights_t).mean()

                    opt_policy.zero_grad()
                    weighted_loss.backward()
                    opt_policy.step()

                    total_policy_loss += float(weighted_loss.detach().cpu().numpy())

                state_returns.append(data[i + 1])

            score = evaluate_composite_score(pnls, cost_per_trade=0.001)
            print(f'Joint Tuning score = {round(score,4)} :: params = {params}')
            if score > best_score:
                best_score, best_params = score, params

        self.best_joint_params = best_params
        return best_params, best_score

    def tune(self, data):
        best_bocpd_params, best_bocpd_score, cps, cpflags, runtime_len = self.tune_bocpd(data)
        print(f"Best BOCPD parameters: {best_bocpd_params}")
        print(f"Best BOCPD score: {round(best_bocpd_score,3)}")

        best_vae_params, best_vae_core, z_t = self.tune_vae(data, cps)
        print(f"Best vae parameters: {best_vae_params}")
        print(f"Best vae score: {round(best_vae_core,4)}")

        best_lstm_params, best_lstm_score = self.tune_lstm(data, cps, cpflags, z_t)
        print(f"Best LSTM parameters: {best_lstm_params}")
        print(f"Best LSTM score: {round(best_lstm_score,4)}")

        best_joint_tuning_params, best_joint_tuning_score = self.joint_tuning(data, best_lstm_params, cps, cpflags, z_t)
        print(f"Best Joint Tuning parameters: {best_joint_tuning_params}")
        print(f"Best Joint Tuning score: {round(best_joint_tuning_score,4)}")


    @property
    def best_params(self):
        return self.best_bocpd_params, self.best_vae_params, self.best_lstm_params, self.best_joint_params
