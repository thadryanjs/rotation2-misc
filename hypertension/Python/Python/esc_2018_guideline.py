# ==========================================================================
# 2018 ESC/ESH Guidelines for the management of arterial hypertension
# ==========================================================================

# Loading modules
import numpy as np

# Function to obtain policy according to the AHA's guideline
def esc_guideline(pretrtrisk, pretrtsbp, pretrtdbp, targetrisk, sbpmin, dbpmin,
                  sbp_reduction, dbp_reduction, rel_risk_chd, rel_risk_stroke, age):

    # Extracting parameters
    years = pretrtrisk.shape[0]
    numtrt = sbp_reduction.shape[0] # number of treatment choices

    # Arrays to store results (initializing with no treatmnet)
    policy = np.empty(years); policy[:] = np.nan

    # Array of treament options
    allmeds = np.arange(numtrt)  # index for possible treatment options

    # Determining action per stage
    for t in range(years):

        #-# Determine Target Blood Pressures (JNC 8 has different targets depending on patient age)
        if age[t] >= 65:
            targetsbp = 139.999 #-# target is a range (130-139), so used 139.999 to get close to end of range
            targetdbp = 79.999
        elif age[t] < 65:
            targetsbp = 130
            targetdbp = 79.999

        #-# Determine Threshold Blood Pressures:
        if age[t] < 80:
            thresholdsbp = 140
            thresholddbp = 90
        elif age[t] >= 80:
            thresholdsbp = 160
            thresholddbp = 90

        # Identifying patient's past treatment
        if t == min(range(years)):
            past_trt = 0  # start with no treatment
        else:
            past_trt = policy[t-1]  # evaluate last patient's treatment first

        # Calculating post-treatment risk and BP with past treatment
        # post_trt_risk = rel_risk_chd[int(past_trt)]*pretrtrisk[t, 0] + rel_risk_stroke[int(past_trt)]*pretrtrisk[t, 1]
        post_trt_sbp = pretrtsbp[t] - sbp_reduction[int(past_trt)]
        post_trt_dbp = pretrtdbp[t] - dbp_reduction[int(past_trt)]

        #-# Check to see if threshold is reached, to enter treatment loop
        if (post_trt_sbp >= thresholdsbp or post_trt_dbp >= thresholddbp):

            # Simulating 1-month evaluations within each year
            month = 1  # initial month
            while month <= 12 and (post_trt_sbp >= targetsbp or post_trt_dbp >= targetdbp):  # BP not on target with current medication within the same year

                # Attempting to increase treatment
                if (past_trt + 1) > np.amax(allmeds):
                    new_trt = past_trt # cannot give more than 5 medications
                else:
                    new_trt = past_trt + 1 # increase medication intensity

                # Calculating post-treatment BP with new potential treatment
                post_trt_sbp = pretrtsbp[t] - sbp_reduction[int(new_trt)]
                post_trt_dbp = pretrtdbp[t] - dbp_reduction[int(new_trt)]

                # Evaluating the feasibility of new treatment
                if post_trt_sbp < sbpmin or post_trt_dbp < dbpmin:
                    policy[t] = past_trt # new treatment is not feasible
                else:
                    policy[t] = new_trt  # new treatment is feasible

                past_trt = policy[t] # next month's evaluation
                month += 1 # next month's evaluation

        else: # BP already on target keeping past year's treatment
            policy[t] = past_trt # keep current treatment

    return policy
