# ==========================================================================
# The Seventh Report of the Joint National Committee on Prevention,
# Detection, Evaluation, and Treatment of High Blood Pressure (JNC 7)
# ==========================================================================

# Loading modules
import numpy as np

# Function to obtain policy according to the AHA's guideline
def jnc_7_guideline(pretrtrisk, pretrtsbp, pretrtdbp, targetrisk, sbpmin, dbpmin,
                  sbp_reduction, dbp_reduction, rel_risk_chd, rel_risk_stroke, diabetic):

    # Extracting parameters
    years = pretrtrisk.shape[0]
    numtrt = sbp_reduction.shape[0] # number of treatment choices

    # Arrays to store results (initializing with no treatment)
    policy = np.empty(years); policy[:] = np.nan

    # Array of treatment options
    allmeds = np.arange(numtrt)  # index for possible treatment options

    #-# Determine Target Blood Pressures (JNC 7 uses different goals for diabetic and non-diabetic pts)
    if diabetic: #-# Targets for patients with diabetes or CKD
        targetsbp = 130
        targetdbp = 80
    elif not diabetic: #-# Targets for patients with no compelling indications
        targetsbp = 140
        targetdbp = 90

    #-# Determine Threshold Blood Pressures:
    if diabetic: #-# Thresholds for patients with diabetes or CKD
        thresholdsbp = 120
        thresholddbp = 80
    elif not diabetic: #-# Thresholds for patients with no compelling indications
        thresholdsbp = 140
        thresholddbp = 90

    # Determining action per stage
    for t in range(years):
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
            if (post_trt_sbp >= 130 or post_trt_dbp >= 80) and (post_trt_sbp < 140 or post_trt_dbp < 90): # High risk with stage 1 hypertension
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
            elif post_trt_sbp >= 140 or post_trt_dbp >= 90: # Stage 2 hypertension
                # Simulating 1-month evaluations within each year
                month = 1  # initial month
                while month <= 12 and (post_trt_sbp >= targetsbp or post_trt_dbp >= targetdbp):  # BP not on target with current medication within the same year

                    # Attempting to increase treatment
                    if (past_trt + 1) > np.amax(allmeds):
                        new_trt = past_trt # cannot give more than 5 medications
                    elif past_trt < 3:
                        new_trt = 3 # patients with stage 2 hypertension should be treated with at least two agents
                    else:
                        new_trt = past_trt + 1 # increase medication intensity

                    # Calculating post-treatment BP with new potential treatment
                    post_trt_sbp = pretrtsbp[t] - sbp_reduction[int(new_trt)]
                    post_trt_dbp = pretrtdbp[t] - dbp_reduction[int(new_trt)]

                    # Evaluating the feasibility of new treatment
                    if post_trt_sbp < sbpmin or post_trt_dbp < dbpmin:
                        policy[t] = past_trt # new treatment is not feasible
                    else:
                        policy[t] = new_trt # new treatment is feasible

                    past_trt = policy[t]  # next month's evaluation
                    month += 1 # next month's evaluation
        else: # BP already on target keeping past year's treatment
            policy[t] = past_trt # keep current treatment

    return policy
