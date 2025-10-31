import math
import pandas as pd
import numpy as np

def backtest_strategy(data, beta, entry_threshold=2, exit_threshold=0.5):
    """Generates trading signals based on z-score and backtests the strategy."""
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
    # Calculate daily portfolio returns
    returns = pd.Series(0.0, index=data.index)
    returns = signals.shift(1) * (data.pct_change().iloc[:, 1] - beta * data.pct_change().iloc[:, 0])
    returns = returns.dropna()

    cumulative_returns = returns.cumsum().apply(np.exp)

    return returns, cumulative_returns

