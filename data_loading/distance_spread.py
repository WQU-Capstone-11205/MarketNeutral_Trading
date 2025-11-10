import yfinance as yf
import pandas as pd
import datetime as dt

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
    first = tickers[0]
    second = tickers[1]
    hedge_ratio = data[first].mean() / data[second].mean()
    spread = data[first] - hedge_ratio * data[second]
    spread = spread.dropna()
    return spread;

