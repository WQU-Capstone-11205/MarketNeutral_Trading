import numpy as np
import matplotlib.pyplot as plt

from structural_break.hazard import ConstantHazard
from structural_break.distribution import StudentT
from structural_break.bocpd import BOCPD

def plot_change_points(spread):
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
