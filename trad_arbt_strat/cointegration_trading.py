import math
import pandas as pd
import numpy as np

def cointegration_trading(data, beta, entry_threshold=2, exit_threshold=0.5):
    """Generates trading signals based on cointegration spread strategy."""
    # Calculate the spread
    spread = data.iloc[:, 1] - beta * data.iloc[:, 0]
    # Calculate z-score of the spread
    z_score = (spread - spread.mean()) / spread.std()

    # Generate signals and positions
    signals = pd.Series(0, index=data.index)
    signals[z_score > entry_threshold] = -1 # Short the spread
    signals[z_score < -entry_threshold] = 1  # Long the spread
    signals[(z_score < exit_threshold) & (z_score > -exit_threshold)] = 0 # Exit positions

    # Backtest
    # --- Hybrid model style reward calculation ---
    rms_mean, rms_var = 0.0, 0.0
    n = 0
    rewards = []
    cumulative_pnl = []
    pnls = []
    cum_pnl = 0.0
    eps = 1e-8
    
    for i in range(1, len(spread)):
        cur_ret = spread.iloc[i - 1]
        next_ret = spread.iloc[i]
        action = signals.iloc[i - 1]

        # compute return (change in spread)
        ret = next_ret - cur_ret
    
        # Update running mean/variance (Welford)
        n += 1
        delta = ret - rms_mean
        rms_mean += delta / n
        rms_var += delta * (ret - rms_mean)

        var = rms_var / max(n - 1, 1)

        # RL-style reward: delta spread × position, normalized by variance
        raw_reward = float(action * ret)
        if raw_reward == 0:
            raw_reward = eps
        reward = raw_reward / (math.sqrt(var) + eps)

        pnls.append(raw_reward)
        rewards.append(reward)

    cumulative_pnl = np.cumsum(pnls)
    rewards = pd.Series(rewards, index=data.index[1:])
    cumulative_pnl = pd.Series(cumulative_pnl, index=data.index[1:])

    return rewards, cumulative_pnl
