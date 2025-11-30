import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

# ============================================
# Simple Baseline: Z-Score Pairs Trading
# ============================================
def zscore_trading(spread, window=50, entry=2.0, exit=0.5):
    #spread = df["A"] - df["B"]
    eps = 1e-8
    mu = spread.rolling(window).mean()
    sigma = spread.rolling(window).std()
    zscore = (spread - mu) / sigma
    position = np.where(zscore > entry, -1,
                np.where(zscore < -entry, 1,
                np.where(abs(zscore) < exit, 0, np.nan)))
    position = pd.Series(position).ffill().fillna(0) #0

    pos_shift = position.shift(1)
    pos_shift.iloc[0] = 0
    sprd_diff = [0.0]*len(spread)
    rms_mean, rms_var = 0.0, 0.0
    n = 0
    for i in range(1, len(spread)):
        cur_ret = spread.iloc[i - 1]
        next_ret = spread.iloc[i]
        # compute return (change in sprd_diff)
        ret = next_ret - cur_ret
    
        # Update running mean/variance (Welford)
        n += 1
        delta = ret - rms_mean
        rms_mean += delta / n
        rms_var += delta * (ret - rms_mean)
        var = rms_var / max(n - 1, 1)

        # RL-style reward: delta spread × position, normalized by variance
        raw_reward = float(pos_shift.iloc[i] * ret)
        if raw_reward == 0:
            raw_reward = eps
        reward = raw_reward / (math.sqrt(var) + eps)
        sprd_diff[i] = reward
           
    pnl = pd.Series(np.array(sprd_diff), index = spread.index)
    
    pnl.index = pd.to_datetime(pnl.index)
    pnl.plot(title="Baseline PnL (Z-Score Pairs Trading)")
    plt.show()
    pnl = pnl.dropna()
    pnlcum = pnl.cumsum()
    pnlcum.plot(title="Cumulative PnL (Z-Score Pairs Trading)")
    pnlcum.index = pd.to_datetime(pnlcum.index)
    plt.show()
    return pnlcum, pnl, position
