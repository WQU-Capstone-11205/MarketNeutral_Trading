import numpy as np
import matplotlib.pyplot as plt

def evaluate_strategy(pnl_series, risk_free_rate=0.0, freq=252):
    """
    Evaluate performance metrics for a trading strategy.
    
    pnl_series: np.array or list of per-step returns (not cumulative)
    risk_free_rate: daily risk-free rate (if annualized freq=252)
    freq: trading periods per year (252 for daily)
    """
    pnl_series = np.asarray(pnl_series)
    cumulative_returns = np.cumsum(pnl_series)
    total_return = cumulative_returns[-1]

    mean_ret = np.mean(pnl_series)
    std_ret = np.std(pnl_series) + 1e-8

    # Annualized Sharpe Ratio
    sharpe = ((mean_ret - risk_free_rate) / std_ret) * np.sqrt(freq)
    
    return {
        "total_return": total_return,
        "sharpe_ratio": sharpe,
        "cumulative_returns": cumulative_returns
    }
