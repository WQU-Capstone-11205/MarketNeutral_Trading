import numpy as np
from typing import Dict, List, Any
from itertools import product

from util.running_mean_std import RunningMeanStd
from util.metrics import evaluate_composite_score
from ml_dl_models.actor_critic import Actor
from ml_dl_models.actor_critic import Critic
from util.weighted_replay_buffer import WeightedReplayBuffer
from tuning.bocpd_vae_tuner import BOCPD_VAE_Tuner

class BOCPD_VAE_RL_Tuner(BOCPD_VAE_Tuner):
    default_rl_space = {
        "state_dim": [16],
        "action_dim": [1],
        "hidden_dim": [64, 128, 256, 512],
        "lr": [1e-4, 5e-5, 1e-5, 5e-6, 1e-6]
    }

    default_joint_space = {
        "state_window": [25, 50, 100, 200],
        "base_action_sigma": [0.01, 0.1, 0.3, 0.5, 0.7],
        "wt_multplier": [1.5, 1.8, 2.0, 5.0],
        "buffer_size_updates": [16, 64, 128, 256, 512],
        "sample_batch_size": [8, 16, 64, 128, 256]
    }

    best_rl_params = {
        "state_dim": [16],
        "action_dim": [1],
        "hidden_dim": [128], #64
        "lr": [1e-5]
    }

    best_joint_params = {
        "state_window": [50],
        "base_action_sigma": [0.3],
        "wt_multplier": [1.5],
        "buffer_size_updates": [256],
        "sample_batch_size": [128]
    }

    def __init__(self, custom_bocpd_space: Dict[str, List[Any]]=None,
                 custom_vae_space: Dict[str, List[Any]]=None,
                 custom_rl_space: Dict[str, List[Any]]=None,
                 custom_joint_space: Dict[str, List[Any]]=None):
        """
        Args:
            custom_rl_space: dict of RL hyperparameter ranges
            custom_joint_space: dict for joint tuning of BOCPD, VAE, and RL
        """
        super().__init__(custom_bocpd_space, custom_vae_space)
        self.rl_space = BOCPD_VAE_RL_Tuner.default_rl_space.copy()
        if custom_rl_space:
            self.rl_space.update(custom_rl_space)
        self.joint_space = BOCPD_VAE_RL_Tuner.default_joint_space.copy()
        if custom_joint_space:
            self.joint_space.update(custom_joint_space)
        self.best_rl_params = BOCPD_VAE_RL_Tuner.best_rl_params.copy()
        self.best_joint_params = BOCPD_VAE_RL_Tuner.best_joint_params.copy()

    def tune_rl(self, data, change_probs, cpflags, z_t):
        # initialize random seed
        np.random.seed(42)
        random.seed(42)
        torch.manual_seed(42)

        base_action_sigma = 0.1
        state_window = 50
        device = 'cpu'
        best_score, best_params = -np.inf, None
        for params in self._grid(self.rl_space):
            state_returns = [0.0]*(state_window-1)
            state_returns.append(data.iloc[0])
            rms = RunningMeanStd()
            actor = Actor(state_dim=state_window, z_dim=params['state_dim'], hidden_dim=params['hidden_dim'], action_dim=params['action_dim']).to(device)
            critic = Critic(state_dim=state_window, z_dim=params['state_dim'], hidden_dim=params['hidden_dim']).to(device)
            actor_opt = optim.Adam(actor.parameters(), lr=params['lr'])
            critic_opt = optim.Adam(critic.parameters(), lr=params['lr'])
            buffer = WeightedReplayBuffer(capacity=30000)
            trades = []
            for i in range(len(data) - 1):
                rms.update([data.iloc[i]]) # Use iloc for pandas Series
                state_arr = np.array(state_returns[-state_window:])
                state_norm = (state_arr - rms.mean) / (math.sqrt(rms.var) + 1e-8)

                state_t = torch.tensor(state_norm.astype(np.float32))[None, :].to(device)
                with torch.no_grad():
                    z_t_det = z_t
                    action_mean = actor(state_t, z_t_det).cpu().numpy().squeeze()
                # exploration scale increases with change_prob
                noise_sigma = base_action_sigma * (1.0 + 5.0 * change_probs[i])  # alpha=5 scaling, cp is now 1D
                action = action_mean + np.random.normal(scale=noise_sigma, size=action_mean.shape)
                action = np.clip(action, -1.0, 1.0)
                next_ret = data.iloc[i + 1] # Use iloc for pandas Series
                reward = float(action * next_ret)
                trades.append(reward)
                            # store transition in buffer with initial weight 1.0
                buffer.push(
                    state_norm.astype(np.float32),
                    action.astype(np.float32),
                    reward,
                    None, False,
                    1.0,
                    None  # shape (seq_len, input_dim)
                )

                # if change_prob large, upweight recent transitions
                if cpflags[i] == 1:
                    buffer.upweight_recent(window=200, multiplier=1.8)

                # periodic updates
                if ((buffer.size() >= 256) and (i % 8 == 0)):
                    batch = buffer.sample(128)

                    # prepare tensors for actor/critic
                    states = torch.tensor(np.stack([b.state for b in batch]), dtype=torch.float32).to(device)
                    actions = torch.tensor(np.stack([b.action for b in batch]), dtype=torch.float32).to(device)
                    rewards = torch.tensor(np.stack([b.reward for b in batch]), dtype=torch.float32).unsqueeze(-1).to(device)

                    # critic update
                    values = critic(states, torch.zeros(states.size(0), params['state_dim']).to(device))  # critic input placeholder z
                    targets = rewards
                    critic_loss = nn.MSELoss()(values, targets)
                    critic_opt.zero_grad(); critic_loss.backward(); critic_opt.step()

                    # actor update
                    with torch.no_grad():
                        adv = (rewards - values).detach()
                    pred_actions = actor(states, torch.zeros(states.size(0), params['state_dim']).to(device))
                    actor_loss = - (pred_actions * adv).mean()
                    actor_opt.zero_grad(); actor_loss.backward(); actor_opt.step()

                state_returns.append(next_ret)

            score = evaluate_composite_score(trades, cost_per_trade=0.001)
            print(f'RL score = {round(score,4)} :: params = {params}')
            if score > best_score:
                best_score, best_params = score, params

        self.best_rl_params = best_params
        return best_params, best_score

    def joint_tuning(self, data, best_rl_params, change_probs, cpflags, z_t):
        # initialize random seed
        np.random.seed(42)
        random.seed(42)
        torch.manual_seed(42)
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
            actor = Actor(state_dim=state_window, z_dim=best_rl_params['state_dim'], hidden_dim=best_rl_params['hidden_dim'], action_dim=best_rl_params['action_dim']).to(device)
            critic = Critic(state_dim=state_window, z_dim=best_rl_params['state_dim'], hidden_dim=best_rl_params['hidden_dim']).to(device)
            actor_opt = optim.Adam(actor.parameters(), lr=best_rl_params['lr'])
            critic_opt = optim.Adam(critic.parameters(), lr=best_rl_params['lr'])
            buffer = WeightedReplayBuffer(capacity=30000)
            trades = []
            for i in range(len(data) - 1):
                rms.update([data.iloc[i]]) # Use iloc for pandas Series
                state_arr = np.array(state_returns[-state_window:])
                state_norm = (state_arr - rms.mean) / (math.sqrt(rms.var) + 1e-8)

                state_t = torch.tensor(state_norm.astype(np.float32))[None, :].to(device)
                with torch.no_grad():
                    z_t_det = z_t
                    action_mean = actor(state_t, z_t_det).cpu().numpy().squeeze()
                # exploration scale increases with change_prob
                noise_sigma = base_action_sigma * (1.0 + 5.0 * change_probs[i])  # alpha=5 scaling, cp is now 1D
                action = action_mean + np.random.normal(scale=noise_sigma, size=action_mean.shape)
                action = np.clip(action, -1.0, 1.0)
                next_ret = data.iloc[i + 1] # Use iloc for pandas Series
                reward = float(action * next_ret)
                trades.append(reward)
                            # store transition in buffer with initial weight 1.0
                buffer.push(
                    state_norm.astype(np.float32),
                    action.astype(np.float32),
                    reward,
                    None, False,
                    1.0,
                    None  # shape (seq_len, input_dim)
                )

                # if change_prob large, upweight recent transitions
                if cpflags[i] == 1:
                    buffer.upweight_recent(window=200, multiplier=params['wt_multplier'])

                # periodic updates
                if ((buffer.size() >= params['buffer_size_updates']) and (i % 8 == 0)):
                    batch = buffer.sample(params['sample_batch_size'])

                    # prepare tensors for actor/critic
                    states = torch.tensor(np.stack([b.state for b in batch]), dtype=torch.float32).to(device)
                    actions = torch.tensor(np.stack([b.action for b in batch]), dtype=torch.float32).to(device)
                    rewards = torch.tensor(np.stack([b.reward for b in batch]), dtype=torch.float32).unsqueeze(-1).to(device)

                    # critic update
                    values = critic(states, torch.zeros(states.size(0), best_rl_params['state_dim']).to(device))  # critic input placeholder z
                    targets = rewards
                    critic_loss = nn.MSELoss()(values, targets)
                    critic_opt.zero_grad(); critic_loss.backward(); critic_opt.step()

                    # actor update
                    with torch.no_grad():
                        adv = (rewards - values).detach()
                    pred_actions = actor(states, torch.zeros(states.size(0), best_rl_params['state_dim']).to(device))
                    actor_loss = - (pred_actions * adv).mean()
                    actor_opt.zero_grad(); actor_loss.backward(); actor_opt.step()

                state_returns.append(next_ret)

            score = evaluate_composite_score(trades, cost_per_trade=0.001)
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

        best_rl_params, best_rl_score = self.tune_rl(data, cps, cpflags, z_t)
        print(f"Best RL parameters: {best_rl_params}")
        print(f"Best RL score: {round(best_rl_score,4)}")

        best_joint_tuning_params, best_joint_tuning_score = self.joint_tuning(data, best_rl_params, cps, cpflags, z_t)
        print(f"Best Joint Tuning parameters: {best_joint_tuning_params}")
        print(f"Best Joint Tuning score: {round(best_joint_tuning_score,4)}")


    @property
    def best_params(self):
        return self.best_bocpd_params, self.best_vae_params, self.best_rl_params, self.best_joint_params
