import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from pandas_datareader import data as pdr
from pandas_datareader.famafrench import FamaFrenchReader

def get_ff_benchmark_returns(start_date, end_date, freq='D'):
    """
    Download the Fama‑French factors of daily
    Input:
        start_date: starting date of the FF benchmark 
        end_date: ending date of the FF benchmark
        freq: Daily ('D')

    Output:
        Fama‑French factors returns
    """
    # Download Fama‑French factors
    # Example: “F‑F_Research_Data_Factors” gets monthly; you can also use daily versions
    ff = FamaFrenchReader('F-F_Research_Data_Factors_Daily', # or _Daily if you need daily data
                          start=start_date, end=end_date).read()[0]
    # Convert percent to decimals (if needed)
    ff = ff.div(100)
    # The column “Mkt‑RF” is the market excess return
    # Add back risk‑free if you just want the market return: Mkt = (Mkt‑RF) + RF
    ff['Mkt'] = ff['Mkt-RF'] + ff['RF']
    # Choose the benchmark you want. Here we pick market excess or market.
    benchmark = ff['Mkt-RF'].copy()
    benchmark.index = pd.to_datetime(benchmark.index)
    return benchmark
