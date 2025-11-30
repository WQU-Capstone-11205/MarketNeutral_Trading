import numpy as np
import matplotlib.pyplot as plt

from structural_break.hazard import ConstantHazard
from structural_break.distribution import StudentT
from structural_break.bocpd import BOCPD

def plot_change_points(spread):
    """
    Plot of change points marked on the input spread.
    """
    # Initialize object
    bc = BOCPD(ConstantHazard(300), StudentT(mu=0, kappa=1, alpha=1, beta=1))
    spread_data = spread.copy()
    # Online estimation and get the maximum likelihood r_t at each time point
    change_point_flags = []
    prev_rt = 0
    for i, d in enumerate(spread_data.values):
        bc.update(d)
        if bc.rt < prev_rt:
            change_point_flags.append(1)
        else:
            change_point_flags.append(0)
        prev_rt = bc.rt

    # Plot data with estimated change points in it
    plt.plot(spread_data.index, spread_data.values, alpha=0.5)
    plt.xlabel("Time")
    plt.ylabel("Spread")
    plt.title("Spread vs Time with change points")
    plt.legend() # No handles with labels found to put in legend.
    cp_index = spread_data.index[np.array(change_point_flags).astype(bool)]
    plt.scatter(cp_index, spread_data.values[np.array(change_point_flags).astype(bool)], c='green', label="change point")
    plt.legend()

def plot_rt_change_probs(data, rt_mle, change_probs):
    """
    Plot of change probability, most likely run length, 
    along with input spread
    """
    sz = len(data)
    # --- Plot results ---
    fig, ax1 = plt.subplots(figsize=(12, 5))
    
    ax1.plot(data.index[:sz-1], data[:sz-1], label='Original Spread', color='gray')
    ax1.set_ylabel('USD/pair trading', color='gray')
    ax1.tick_params(axis='y', labelcolor='gray')

    ax2 = ax1.twinx()
    ax2.plot(data.index[:sz-1],change_probs, label='Change Probability', color='red')
    ax2.set_ylabel('Change Probability', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # Optional: Plot most likely run length
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60)) # Offset the third y-axis
    ax3.plot(data.index[:sz-1],rt_mle, label='Most Likely Run Length', color='blue', linestyle='--')
    ax3.set_ylabel('Most Likely Run Length', color='blue')
    ax3.tick_params(axis='y', labelcolor='blue')
    
    ax1.set_xlabel('Date')
    plt.title('BOCPD: Change Probability Over Time (for Test spread)')
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    plt.show()

def vae_plot(input_spread, recon_spread):
    """
    Plot of reconstructed spread superimposed on input spread.
    """
    plt.plot(input_spread.index, input_spread, label="Original Spread")
    plt.plot(input_spread.index, recon_spread, label="Reconstructed Spread")
    plt.xlabel("Date")
    plt.ylabel("USD/pair trading")
    plt.legend();
    plt.title("VAE: Original vs Reconstructed Spread")
    plt.show()
  
def compare_trends_plot(input_spread, results):
    """
    Plot of Profit and Loss along with the input spread.
    """
    fig, ax1 = plt.subplots(figsize=(12, 5))

    # --- compute common limits ---
    y_min = min(input_spread.min(), np.min(results))
    y_max = max(input_spread.max(), np.max(results))

    # --- plot distance spread ---
    ax1.plot(input_spread.index, input_spread, label='Spread', color='gray')
    ax1.set_ylabel('USD/pair trading', color='gray')
    ax1.tick_params(axis='y', labelcolor='gray')

    # Set unified scale
    ax1.set_ylim(y_min, y_max)

    # --- secondary axis (same scale) ---
    ax2 = ax1.twinx()
    ax2.plot(input_spread.index[:-1], results, label='Profit & Loss', color='blue')
    ax2.set_ylabel('USD/pair trading', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')

    # Make ax2 use the same limits
    ax2.set_ylim(y_min, y_max)

    ax1.set_xlabel('Date')
    plt.title('Profit & Loss (results) vs Spread (input)')

    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    plt.show()

def drawdown_plot(returns):
    """
    Drawdown plot of the input data
    """
    cum_pnl = np.cumsum(returns)
    rolling_max = np.maximum.accumulate(cum_pnl)
    drawdown = (rolling_max - cum_pnl) / (rolling_max + 1e-8)
    plt.plot(returns.index, drawdown, label="Profit & Loss", color="blue")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.title("Portfolio returns drawdown")
    plt.show()

def compare_drawdown_plot(spread_returns, pnl_returns):
    """
    Drawdown comparison plots of profit and loss, 
    and input spread 
    """
    cum_spread = np.cumsum(spread_returns)
    rolling_max_spread = np.maximum.accumulate(cum_spread)
    drawdown_spread = (rolling_max_spread - cum_spread) / (rolling_max_spread + 1e-8)
    
    cum_pnl = np.cumsum(pnl_returns)
    rolling_max_pnl = np.maximum.accumulate(cum_pnl)
    drawdown_pnl = (rolling_max_pnl - cum_pnl) / (rolling_max_pnl + 1e-8)
    
    plt.plot(drawdown_spread, label="Spread", color="red")
    plt.plot(drawdown_pnl, label="Profit & Loss", color="blue")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.title("Input spread vs Portfolio returns Drawdown")
    plt.show()
