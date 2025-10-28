import numpy as np

def sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252):
    excess_returns = returns - risk_free_rate
    mean = np.mean(excess_returns)
    std = np.std(excess_returns) + 1e-8
    return (mean / std) * np.sqrt(periods_per_year)
      
def compute_max_drawdown(equity_curve):
      equity_curve = np.asarray(equity_curve)
      if len(equity_curve) < 2:
          return 0.0
      
      # Compute running maximum
      running_max = np.maximum.accumulate(equity_curve)
      # Compute drawdowns
      drawdowns = (running_max - equity_curve) / (running_max + 1e-8)
      # Maximum drawdown
      mdd = np.max(drawdowns)
      return mdd

def evaluate_composite_score(trades, cost_per_trade, freq_per_year=252):
      """
      trades: realized returns (after the agent’s actions).
      cost_per_trade: proportional transaction cost (e.g., 0.001 for 10 bps).
      freq_per_year: periods per year (default 252 for daily).
      """
      r = np.asarray(trades)
      if len(r) < 2:
          return -np.inf
      
      # Sharpe ratio (risk-adjusted return)
      mean_r = np.mean(r)
      std_r = np.std(r)
      sharpe = np.sqrt(freq_per_year) * mean_r / (std_r + 1e-8)
      
      # max drawdown, which is equivalent to risk
      max_dd = compute_max_drawdown(np.cumsum(r))
      
      # adaptive cost penalty
      total_cost = np.sum(np.abs(trades)) * cost_per_trade
      net_pnl = np.sum(r)
      cost_penalty = total_cost / (abs(net_pnl) + 1e-8)
      
      # composite score (with weights)
      score = sharpe - 0.5 * max_dd - 0.2 * cost_penalty
      return score
