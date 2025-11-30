"""
The following code is referred from: https://github.com/y-bar/bocd/blob/master/bocd/bocd.py
Author: teramonagi (Nagi Teramo)
"""
import math
import numpy as np
import pandas as pd

# -------------------------
# BOCPD implementation
# -------------------------
class BOCPD:
    def __init__(self, hazard, distribution):
        """
        Initialize the BOCPD model.
        
        Args:
            hazard: A callable function returning the hazard probability H(r) for a given run length.
            distribution: A distribution object with pdf() and update_params() methods.
        
        Initializes internal variables:
            T: Current time step
            beliefs: Matrix representing the run length posterior probabilities
            change_probs: List of change probabilities
            cp_flags: List of changepoint flags
            rt_mle: List of most likely run lengths
        """
        self.hazard = hazard
        self.distribution = distribution
        self.T = 0
        self.beliefs = np.zeros((1, 2))
        self.beliefs[0, 0] = 1.0
        self.change_probs = []
        self.cp_flags = []
        self.rt_mle = []

    def reset_params(self):
        # Reset the model to its initial state
        self.T = 0
        self.beliefs = np.zeros((1, 2))
        self.beliefs[0, 0] = 1.0

    def _expand_belief_matrix(self):
        # Expand the belief matrix for a new time step
        # Adds a row for the next run length probabilities
        rows = np.zeros((1, 2))
        self.beliefs = np.concatenate((self.beliefs, rows), axis=0)

    def _shift_belief_matrix(self):
        # Moves the new probabilities to the prior column and clears the next column for updates
        self.beliefs[:, 0] = self.beliefs[:, 1]
        self.beliefs[:, 1] = 0.0

    def update(self, x):
        self._expand_belief_matrix()

        # Evaluate Predictive Probability (3 in Algorithm 1)
        pi_t = self.distribution.pdf(x)

        # Calculate H(r_{t-1})
        h = self.hazard(self.rt)

        # Calculate Growth Probability (4 in Algorithm 1)
        self.beliefs[1 : self.T + 2, 1] = self.beliefs[: self.T + 1, 0] * pi_t * (1 - h)

        # Calculate Changepoint Probabilities (5 in Algorithm 1)
        self.beliefs[0, 1] = (self.beliefs[: self.T + 1, 0] * pi_t * h).sum()

        # Determine Run length Distribution (7 in Algorithm 1)
        self.beliefs[:, 1] = self.beliefs[:, 1] / self.beliefs[:, 1].sum()

        # Update sufficient statistics (8 in Algorithm 8)
        self.distribution.update_params(x)

        # Update internal state
        self._shift_belief_matrix()
        self.T += 1

        # Update results
        curr_rt = self.rt[0]
        cp_flag = 1 if ((len(self.rt_mle) > 0) and (curr_rt < self.rt_mle[-1])) else 0
        self.cp_flags.append(cp_flag)
        self.rt_mle.append(curr_rt)
        change_prob = self.beliefs.T[0][curr_rt]
        self.change_probs.append(self.beliefs.T[0][curr_rt])
        return change_prob, cp_flag

    @property
    def results(self):
        return self.change_probs, self.rt_mle, self.cp_flags

    @property
    def rt(self):
        return np.where(self.beliefs[:, 0] == self.beliefs[:, 0].max())[0]

    @property
    def belief(self):
        return self.beliefs[:, 0]
