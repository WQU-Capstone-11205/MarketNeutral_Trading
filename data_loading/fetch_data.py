import yfinance as yf
import pandas as pd
import datetime as dt

# Download yfinance data for the given tickers
def fetch_from_yfinance(tickers, start_date=None, end_date=None):
    if end_date is None:
        end_date = dt.date.today()
    if start_date is None:
        start_date = end_date - dt.timedelta(days=365*2)  # ~2 years
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False, multi_level_index=False)['Adj Close']
    return data;
