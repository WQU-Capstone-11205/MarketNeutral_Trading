import math
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from sklearn.linear_model import LinearRegression

def check_cointegration_and_hedge_ratio(series1, series2):
    """
      Performs the Engle-Granger two-step 
      cointegration test and finds the 
      hedge ratio.
    """
    # Combine the series and drop rows with NaN values
    combined_series = pd.concat([series1, series2], axis=1).dropna()
    series1_cleaned = combined_series.iloc[:, 0]
    series2_cleaned = combined_series.iloc[:, 1]
    
    # Add a constant to the independent variable for regression
    X = sm.add_constant(series1_cleaned)
    # Perform linear regression to find the hedge ratio (beta)
    model = sm.OLS(series2_cleaned, X).fit()
    beta = model.params.iloc[1]
    
    # Calculate the spread (residuals)
    spread = series2_cleaned - beta * series1_cleaned
    
    # Perform Augmented Dickey-Fuller (ADF) test on the spread to check stationarity
    coint_t, p_value, crit_value = coint(series1_cleaned, series2_cleaned)

    if p_value < 0.05:
        print(f"\nPair is cointegrated with p-value {p_value:.4f}. Hedge ratio (beta): {beta:.4f}")
        return beta, spread
    else:
        print(f"\nPair is not cointegrated with p-value {p_value:.4f}. Cannot trade.")
        return None, None

def alpha_beta(strategy_returns, benchmark_returns, freq=252):
    """
    Computes annual Alpha and Beta for strategy returns with respect 
    to benchmark returns.

    Returns: Alpha, Beta
    """
    # Convert pandas Series to numpy arrays if they are not already
    if isinstance(benchmark_returns, pd.Series):
        benchmark_returns = benchmark_returns.values
    if isinstance(strategy_returns, pd.Series):
        strategy_returns = strategy_returns.values

    sz = len(benchmark_returns)
    X = benchmark_returns.reshape(-1, 1)
    X = X[:sz-1]
    y = strategy_returns
    reg = LinearRegression().fit(X, y)
    beta = reg.coef_[0]
    alpha = (reg.intercept_) * freq  # annualized
    return alpha, beta

def annual_volatility(returns, freq=252):
    # Computes annual volatility for input returns
    return np.std(returns) * np.sqrt(freq)

def sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252):
    # Computes annual Sharpe Ratio for input data
    excess_returns = returns - risk_free_rate
    mean = np.mean(excess_returns)
    std = np.std(excess_returns) + 1e-8
    return (mean / std) * np.sqrt(periods_per_year)

def sortino_ratio(returns, freq=252):
    # Computes annual Sortino Ratio for input data
    mean_ret = np.mean(returns)
    downside = np.std(returns[returns < 0])
    return (mean_ret / downside) * np.sqrt(freq)
    
def compute_max_drawdown(equity_curve):
      # Computes percentage Max drawdown for input data
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

      Returns: composite score
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

def calculate_performance_metrics(returns, cumulative_returns, risk_free_rate=0.01, benchmark_returns=None):
    """
        Calculates Sharpe Ratio, Sortino Ratio, 
        Max Drawdown, Annual Volatility, 
        Alpha, and Beta.
    """

    # Annualized Returns
    annual_returns = cumulative_returns.iloc[-1]**(252/len(returns)) - 1

    # Annual Volatility
    annual_volatility = returns.std() * np.sqrt(252)

    # Sharpe Ratio
    sharpe_ratio = (annual_returns - risk_free_rate) / annual_volatility

    # Sortino Ratio
    downside_returns = returns[returns < 0]
    downside_deviation = downside_returns.std() * np.sqrt(252)
    sortino_ratio = (annual_returns - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0

    # Max Drawdown
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()

    # Alpha and Beta (requires a benchmark)
    alpha, beta = None, None
    if benchmark_returns is not None:
        # Align returns and benchmark returns
        aligned_returns, aligned_benchmark_returns = returns.align(benchmark_returns, join='inner', axis=0)
        aligned_returns = aligned_returns.dropna()
        aligned_benchmark_returns = aligned_benchmark_returns.dropna()

        if not aligned_returns.empty and not aligned_benchmark_returns.empty:
            X = sm.add_constant(aligned_benchmark_returns)
            model = sm.OLS(aligned_returns, X).fit()
            beta = model.params.iloc[1]
            alpha = model.params.iloc[0] * 252 # Annualize alpha (daily alpha * 252)

    return {
        "Annual Returns": annual_returns,
        "Annual Volatility": annual_volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Sortino Ratio": sortino_ratio,
        "Max Drawdown": max_drawdown,
        "Alpha": alpha,
        "Beta": beta,
    }
