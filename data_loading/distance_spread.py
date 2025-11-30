import yfinance as yf
import pandas as pd
import datetime as dt
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

# Calculate spread with hedge ratio of means 
def distance_spread(tickers, start_date=None, end_date=None):
    if end_date is None:
        end_date = dt.date.today()
    if start_date is None:
        start_date = end_date - dt.timedelta(days=365*2)  # ~2 years
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False, multi_level_index=False)['Adj Close']
    first = tickers[0]
    second = tickers[1]
    hedge_ratio = data[first].mean() / data[second].mean()
    spread = data[first] - hedge_ratio * data[second]
    spread = spread.dropna()
    return spread;

def cointegration_spread(tickers, start_date=None, end_date=None):
    if end_date is None:
        end_date = dt.date.today()
    if start_date is None:
        start_date = end_date - dt.timedelta(days=365*2)  # ~2 years
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False, multi_level_index=False)['Adj Close']
        
    # Download adjusted close prices
    first, second = tickers[0], tickers[1]
    data = data.dropna()

    # Step 1: Regress first on second to get hedge ratio (β)
    X = sm.add_constant(data[second])
    model = sm.OLS(data[first], X).fit()
    beta = model.params[1]           # hedge ratio
    beta_pval = model.pvalues[1]     # significance test of beta ≠ 0

    # Step 2: Compute the spread using OLS beta
    spread = data[first] - beta * data[second]

    # Step 3: Cointegration test (Engle-Granger)
    # Null hypothesis: no cointegration
    coint_t, coint_pval, crit_vals = coint(data[first], data[second])

    return spread, beta, beta_pval, coint_pval

