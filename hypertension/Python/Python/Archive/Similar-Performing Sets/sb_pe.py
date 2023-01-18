# ===================================================
# Estimating value functions using the SBPE algorithm
# ===================================================

# Loading modules
import numpy as np


# Simulation-based policy ealuation algorithm
def sbpe(tp, healthy, Q_hat_next, discount, obs):

    # Extracting parameters
    numtrt = tp.shape[1]  # number of treatment choices

    # Initializing Q-value observations
    Q = np.full(numtrt, np.nan)

    # Generating state transitions
    np.random.seed(obs)  # seeds for pseudo-random number generator
    u = np.random.rand(numtrt, 1)  # generating len(tp)*numtrt uniform random numbers (additinal axis added for comparison with cummulative probabilities)
    prob = tp.T; cum_prob = prob.cumsum(axis=1)  # estimating cummulative probabilities
    h_next = (u < cum_prob).argmax(axis=1) # generating next states

    for j in range(numtrt): # each treatment
        # Estimating Q-values (based on future action)
        if h_next[j] == healthy: # only the healthy state has rewards associated
            Q[j] = 1 + discount*Q_hat_next
        else: # no reward if patient is not healthy
            Q[j] = 0

    return Q

