# ====================================
# Estimating transition probabilities
# ====================================

# Loading modules
import numpy as np

# Transition probabilities function
def TP(periodrisk, chddeath, strokedeath, alldeath, pretrtsbp, pretrtdbp, sbpmin, sbpmax, dbpmin,
       sbp_reduction, dbp_reduction, rel_risk_chd, rel_risk_stroke, numhealth):
    # Calculating probability of health states transitions given ASCVD risk, fatality likelihoods,
    # and probability of mortality due to non-ASCVD events

    # Inputs
    ##periodrisk: 1-year risk of CHD and stroke
    ##chddeath: likelihood of death given a CHD events
    ##strokedeath: likelihood of death given a stroke events
    ##alldeath: likelihood of death due to non-ASCVD events
    ##riskslope: relative risk estimates of CHD and stroke events
    ##trtvector: vector of treatments to consider
    ##pretrtsbp: pre-treatment SBP
    ##pretrtdbp: pre-treatment DBP
    ##sbpmin (sbpmax): Minimum (maximum) SBP allowed (clinical constraint)
    ##dbpmin: minimum DBP allowed (clinical constraint)

    # Line for debugging purposes
    # t = h = j = 1; trt_effects = trt_effects[0:4]; periodrisk = ascvdrisk1

    # Extracting parameters
    years = periodrisk.shape[0]  # number of non-stationary stages
    events = periodrisk.shape[1]  # number of events
    numtrt = len(sbp_reduction) # number of treatment choices

    # Storing feasibility indicators
    feasible = np.full((years, numtrt), np.nan) # indicators of whether the treatment is feasible at each time

    # Storing risk and TP calculations
    risk = np.full((years, events, numtrt), np.nan)  # stores post-treatment risks
    ptrans = np.zeros((numhealth, years, numtrt))  # state transition probabilities (default of 0, to reduce computations)

    for t in range(years):  # each stage (year)
        for j in range(numtrt):  # each treatment
            if j == 0:  # the do nothing treatment
                if pretrtsbp[t] > sbpmax:  # cannot do nothing when pre-treatment SBP is too high
                    feasible[t, j] = 0
                else:
                    feasible[t, j] = 1  # otherwise, do nothing is always feasible
            else:  # prescibe >0 drugs
                newsbp = pretrtsbp[t] - sbp_reduction[j]
                newdbp = pretrtdbp[t] - dbp_reduction[j]

                # Violates minimum allowable SBP constraint or min allowable DBP constrain and
                # the do nothing option is feasible
                if (newsbp < sbpmin or newdbp < dbpmin) and feasible[t, 0] == 1:
                    feasible[t, j] = 0
                else:  # does not violate min BP
                    feasible[t, j] = 1

            for k in range(events):  # each event type
                # Calculating post-treatment risks
                if k == 0: #CHD events
                    risk[t, k, j] = rel_risk_chd[j]*periodrisk[t, k]
                elif k == 1: #stroke events
                    risk[t, k, j] = rel_risk_stroke[j]*periodrisk[t, k]

            # Health condition transition probabilities: allows for both CHD and stroke in same period (starting from healthy condition)
            quits = 0
            while quits == 0:  # compute transition probabilities, using a "break" command if you've exceeded 1 (this never happens!)
                ptrans[3, t, j] = min(1, chddeath.iloc[t] * risk[t, 0, j])  # likelihood of death from CHD event
                cumulprob = ptrans[3, t, j]

                ptrans[4, t, j] = min(1, strokedeath.iloc[t] * risk[t, 1, j])  # likelihood of death from stroke
                if cumulprob + ptrans[4, t, j] >= 1:  # check for invalid probabilities
                    ptrans[4, t, j] = 1 - cumulprob
                    break  # all other probabilities should be left as 0 [as initialized before loop]
                cumulprob = cumulprob + ptrans[4, t, j]

                ptrans[5, t, j] = min(1, alldeath.iloc[t])  # likelihood of death from non CVD cause
                if cumulprob + ptrans[5, t, j] >= 1:  # check for invalid probabilities
                    ptrans[5, t, j] = 1 - cumulprob
                    break  # all other probabilities should be left as 0 [as initialized before loop]
                cumulprob = cumulprob + ptrans[5, t, j]

                ptrans[1, t, j] = min(1, (1 - chddeath.iloc[t]) * risk[t, 0, j])  # likelihood of having CHD and surviving
                if cumulprob + ptrans[1, t, j] >= 1:  # check for invalid probabilities
                    ptrans[1, t, j] = 1 - cumulprob
                    break  # all other probabilities should be left as 0 [as initialized before loop]
                cumulprob = cumulprob + ptrans[1, t, j]

                ptrans[2, t, j] = min(1, (1 - strokedeath.iloc[t]) * risk[t, 1, j])  # likelihood of having stroke and surviving
                if cumulprob + ptrans[2, t, j] >= 1:  # check for invalid probabilities
                    ptrans[2, t, j] = 1 - cumulprob
                    break  # all other probabilities should be left as 0 [as initialized before loop]
                cumulprob = cumulprob + ptrans[2, t, j]

                ptrans[0, t, j] = 1 - cumulprob  # otherwise, patient is still healthy
                break  # computed all probabilities, now quit

    return feasible, ptrans
