import numpy as np
import matplotlib.pyplot as plt

def compare_trends_plot(input_spread, results):
    """
    Comparison trend plots of Profit and Loss with 
    input spread
    """
    fig, ax1 = plt.subplots(figsize=(12, 5))

    ax1.plot(input_spread.index, input_spread, label='Spread', color='gray')
    ax1.set_ylabel('USD/pair trading', color='gray')
    ax1.tick_params(axis='y', labelcolor='gray')

    ax2 = ax1.twinx()
    ax2.plot(input_spread.index[:-1], results, label='Profit & Loss', color='blue')
    ax2.set_ylabel('USD/pair trading', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax1.set_xlabel('Date')
    plt.title('Profit & Loss (results) vs Spread (input)')
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    plt.show()

def drawdown_plot(returns):
    """
    Drawdown plot input data
    """
    cum_pnl = np.cumsum(returns)
    rolling_max = np.maximum.accumulate(cum_pnl)
    drawdown = (rolling_max - cum_pnl) # / (rolling_max + 1e-8)
    plt.plot(returns.index, drawdown, label="Profit & Loss", color="blue")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.title("Portfolio returns drawdown (Traditional)")
    plt.show()
