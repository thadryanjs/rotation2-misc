# ====================================================================================
# Estimating Set of Near-Optimal Choices using the SBMCC Algorithm
# ====================================================================================

# Loading modules
import numpy as np


# Simulation-based multiple comparison with a control algorithm for policy evaluation
def sbmcc(Q_bar, Q_hat, sigma2_bar, a_ctrl, obs, rep, byrep=False):

    # Extracting parameters
    numtrt = Q_hat.shape[0]

    if byrep is True:
        # Arrays to store results
        abs_psi = np.full((numtrt, rep+1), np.nan)

        # Calculating root statistic for each repication until rep
        # Estimating Q-values with rep number of replications
        Q_hat_rep = np.nanmean(Q_bar[:, :(rep+1)], axis=1) # estimated Q-values

        for j in range(numtrt): # each treatment
            for r in range(rep+1): # each replication
                abs_psi[j, r] = np.abs((Q_bar[a_ctrl, r]-Q_bar[j, r]-(Q_hat_rep[a_ctrl]-Q_hat_rep[j])))/\
                                np.sqrt((sigma2_bar[a_ctrl, r]+sigma2_bar[j, r])/obs)

        # Obtaining maximum abs_psi
        psi_max = np.amax(abs_psi, axis=0)

    else: # calculate max_psi only once for reps number of replications
        # Arrays to store results
        abs_psi = np.full(numtrt, np.nan)

        # Calculating root statistic
        for j in range(numtrt): # each treatment
            abs_psi[j] = np.abs((Q_bar[a_ctrl, rep]-Q_bar[j, rep]-(Q_hat[a_ctrl]-Q_hat[j])))/\
                             np.sqrt((sigma2_bar[j, rep]+sigma2_bar[a_ctrl, rep])/obs)

        # Obtaining maximum abs_psi
        psi_max = np.amax(abs_psi)

    return psi_max
