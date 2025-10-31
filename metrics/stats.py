import math
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

def check_cointegration_and_hedge_ratio(series1, series2):
    """
      Performs the Engle-Granger two-step 
      cointegration test and finds the 
      hedge ratio.
    """
    # Add a constant to the independent variable for regression
    X = sm.add_constant(series1)
    # Perform linear regression to find the hedge ratio (beta)
    model = sm.OLS(series2, X).fit()
    beta = model.params.iloc[1]

    # Calculate the spread (residuals)
    spread = series2 - beta * series1

    # Perform Augmented Dickey-Fuller (ADF) test on the spread to check stationarity
    coint_t, p_value, crit_value = coint(series1, series2)

    if p_value < 0.05:
        print(f"\nPair is cointegrated with p-value {p_value:.4f}. Hedge ratio (beta): {beta:.4f}")
        return beta, spread
    else:
        print(f"\nPair is not cointegrated with p-value {p_value:.4f}. Cannot trade.")
        return None, None

