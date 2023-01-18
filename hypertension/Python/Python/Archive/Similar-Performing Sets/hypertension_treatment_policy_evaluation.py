# ******************************************************
# Policy Evaluation Case Study - Hypertension Treatment
# ******************************************************

# Loading modules
import os  # directory changes
import pandas as pd  # data frame operations
import numpy as np  # array operations
import time as tm  # timing code
import multiprocessing as mp  # parallel computations
import pickle as pk  # saving results
from bp_med_effects import med_effects # medication parameters
from ascvd_risk import rev_arisk  # risk calculations
from transition_probabilities import TP  # transition probability calculations
from policy_evaluation import evaluate_pi, evaluate_events # policy evaluation
from aha_2017_guideline import aha_guideline # 2017 AHA's guideline for hypertension treatment
import sb_pe  # simulation-based backwards induction
import sb_mcc_pe # simulation-based multiple comparison with a control algorithm

# Establishing directories
if os.name == 'posix': # name for linux system (for Dartmouth Babylons)
    home_dir = os.path.abspath(os.environ['HOME'] + '/Documents/Policy Evaluation/Python')
    data_dir = os.path.abspath(os.environ['HOME'] + '/Documents/Policy Evaluation/Data')
    results_dir = os.path.abspath(os.environ['HOME'] + '/Documents/Policy Evaluation/Python/Results')
    fig_dir = os.path.abspath(os.environ['HOME']+'/Documents/Policy Evaluation/Python/Figures')
else:
    # home_dir = os.path.abspath(os.environ['USERPROFILE'] + '\\My Drive\\Research\\Current Projects\\Policy Evaluation\\Python')
    # data_dir = os.path.abspath(os.environ['USERPROFILE'] + '\\My Drive\\Research\\Current Projects\\Policy Evaluation\\Data')
    # results_dir = os.path.abspath(os.environ['USERPROFILE'] + '\\My Drive\\Research\\Current Projects\\Policy Evaluation\\Python\\Results')
    # fig_dir = os.path.abspath(os.environ['USERPROFILE'] + '\\My Drive\\Research\\Current Projects\\Policy Evaluation\\Python\\Figures')

    #-# Changed directories to match my filesystem -Mateo
    data_dir = os.path.abspath('G:\\.shortcut-targets-by-id\\1Tc06k-1aEZmEU8eFQzJLLaiJ7asR3-rf\\ASURE\\Data')
    home_dir = os.path.abspath('G:\\.shortcut-targets-by-id\\1Tc06k-1aEZmEU8eFQzJLLaiJ7asR3-rf\\ASURE\\Python\\Similar-Performing Sets\\Output')
    results_dir = os.path.abspath('G:\\.shortcut-targets-by-id\\1Tc06k-1aEZmEU8eFQzJLLaiJ7asR3-rf\\ASURE\\Python\\Similar-Performing Sets\\Output\\Results\\Babylon Results')
    fig_dir = os.path.abspath('G:\\.shortcut-targets-by-id\\1Tc06k-1aEZmEU8eFQzJLLaiJ7asR3-rf\\ASURE\\Python\\Similar-Performing Sets\\Output\\Figures')

# =======================
# Initializing parameters
# =======================

# Selecting number of cores for parallel processing
if os.name == 'posix': # name for linux system (for Dartmouth Babylons)
    cores = 20
else:
    cores = mp.cpu_count() - 1

# Risk parameters
ascvd_hist_mult = [3, 3]  # multiplier to account for history of CHD and stroke, respectively

# MDP parameters
##Discounting factor
discount = 0.97

# Transition probability parameters
numhealth = 10  # Number of health states
years = 10  # Number of years (non-stationary stages)
events = 2  # Number of events considered in model
numeds = 5 # Maximum number of drugs combined

## Terminal reward
### Terminal QoL weights
QoLterm = {"40-44": [1, 0.9348, 0.8835, 0.9348*0.8835, 0.9348, 0.8835, 0, 0, 0, 0],
           "45-54": [1, 0.9374, 0.8835, 0.9374*0.8835, 0.9374, 0.8835, 0, 0, 0, 0],
           "55-64": [1, 0.9376, 0.8835, 0.9376*0.8835, 0.9376, 0.8835, 0, 0, 0, 0],
           "65-74": [1, 0.9372, 0.8835, 0.9372*0.8835, 0.9372, 0.8835, 0, 0, 0, 0],
           "75-84": [1, 0.9364, 0.8835, 0.9363*0.8835, 0.9364, 0.8835, 0, 0, 0, 0]
           }

### Terminal condition standardized mortality rate (first list males, second list females)
mortality_rates = {"Males <2 CHD events":    [1, 1/1.6, 1/2.3, (1/1.6)*(1/2.3), 1/1.6, 1/2.3, 0, 0, 0, 0],
                   "Females <2 CHD events":  [1, 1/2.1, 1/2.3, (1/2.1)*(1/2.3), 1/2.1, 1/2.3, 0, 0, 0, 0],
                   "Males >=2 CHD events":   [1, 1/3.4, 1/2.3, (1/3.4)*(1/2.3), 1/3.4, 1/2.3, 0, 0, 0, 0],
                   "Females >=2 CHD events": [1, 1/2.5, 1/2.3, (1/2.5)*(1/2.3), 1/2.5, 1/2.3, 0, 0, 0, 0]
                   }

# Simulation parameters
#-# reps was 500, changed to 300 to decrease simulation runtime -Mateo
reps = int(300) # number of simulation replications  # see number of batches analysis result
beta = 0.01 # significance level for adaptive sample size
delta = 0.5 # tolerance level for difference in estimated and true value functions

# MCC parameters
alpha = 0.05 # significance level of simultaneous confidence intervals

# Treatment parameters
# #BP clinical constraints
sbpmin = 120  # minimum allowable SBP (rev 1/15/20 after Sussman's and Hayward's feedback)
sbpmax = 150  # maximum allowable SBP
dbpmin = 55  # minimum allowable DBP (rev 1/15/20 after Sussman's and Hayward's feedback)

#AHA's guideline parameters
targetrisk = 0.1
targetsbp = 130
targetdbp = 80

# #Half dosages compared to standard dosages
hf_red_frac = 2/3 # fraction of BP and risk reduction
hf_disut_frac = 1/2 # fraction of disutility

# #Estimated change in BP by dosage (assuming absolute BP reductions and linear reductions with respect to dose)
sbp_drop_std = 5.5 # average SBP reduction per medication at standard dose in BPLTCC trials according to Burke
sbp_drop_hf = sbp_drop_std*hf_red_frac # average SBP reduction per medication at half dose (assumption by Sussman and Hayward (1/15/2020)
dbp_drop_std = 3.3 # average DBP reduction per medication at standard dose in BPLTCC trials according to Burke
dbp_drop_hf = dbp_drop_std*hf_red_frac # average DBP reduction per medication at half dose (assumption by Sussman and Hayward (1/15/2020)

# Estimated change in risk by dosage (assuming absolute risk reductions)
rel_risk_chd_std = 0.87 # estimated change in risk for CHD events per medication at standard dose in BPLTCC trials according to Burke
rel_risk_stroke_std = 0.79 # estimated change in risk for stroke events per medication at standard dose in BPLTCC trials according to Burke
rel_risk_chd_hf = 1-((1-rel_risk_chd_std)*hf_red_frac) # estimated change in risk for CHD events per medication at half dose (assumption by Sussman and Hayward (1/15/2020)
rel_risk_stroke_hf = 1-((1-rel_risk_stroke_std)*hf_red_frac) # estimated change in risk for stroke events per medication at half dose (assumption by Sussman and Hayward (1/15/2020)

# #Estimated treatment disutility by dosage
disut_std = 0.002 # treatment disutility per medication at standard dose (assumption by Sussman and Hayward (1/15/2020)
disut_hf = disut_std*hf_disut_frac # treatment disutility per medication at half dose (assumption by Sussman and Hayward (1/15/2020)

# #Treatment choices (21 trts: no treatment plus 1 to 5 drugs at standard and half dosages)
allmeds = list(range(21))  # index for possible treatment options
numtrt = len(allmeds)  # number of treatment choices

# State ordering
## Initial order: (healthy, surviving a CHD, surviving a stroke, dying from a CHD, dying from a stroke,
# dying from non-ASCVD related cause, and death or history of ASCVD)
order = [6, 5, 4, 3, 2, 1, 0]  # ordering states in nondecreasing order of rewards
healthy = int(np.where(np.array(order) == 0)[0]) # indentification of healthy state
event_id = [x for _, x in sorted(zip(order, [0, 1, 1, 1, 1, 0, 0]))] # identification of states in where ASCVD events happen

# #Treatment effects (SBP reductions, DBP reductions, post-treatment relative risk for CHD events,
# post-treatment relative risk for CHD events, and treatment related disutilities)
sbp_reduction, dbp_reduction, rel_risk_chd, rel_risk_stroke, disutility, meds = med_effects(hf_red_frac, sbp_drop_std,
                                                                                            sbp_drop_hf, dbp_drop_std,
                                                                                            dbp_drop_hf, rel_risk_chd_std,
                                                                                            rel_risk_chd_hf, rel_risk_stroke_std,
                                                                                            rel_risk_stroke_hf, disut_std,
                                                                                            disut_hf, numtrt,
                                                                                            "nondecreasing") # "nonincreasing"

# =============
# Loading data
# =============

# Loading life expectancy and death likelihood data
# (first column age, second column male, third column female)
os.chdir(data_dir)
lifedata = pd.read_csv('lifedata.csv', header=None)
strokedeathdata = pd.read_csv('strokedeathdata.csv', header=None)
chddeathdata = pd.read_csv('chddeathdata.csv', header=None)
alldeathdata = pd.read_csv('alldeathdata.csv', header=None)

# Loading risk slopes (first column age, second column CHD, third column stroke)
riskslopedata = pd.read_csv('riskslopes.csv', header=None)

# Loading 2009-2016 Continuous NHANES dataset (ages 40-60)
# ptdata = pd.read_csv('Complete Forecasted Patient Profiles.csv') # complete clinical scenarios (all genders, races, diabetes status, smoking status, BP levels, and cholesterol levels)
ptdata = pd.read_csv('Continuous NHANES 50-54 Dataset.csv') #-# working with 50-54 NHANES dataset now -Mateo
ptdata['black'] = pd.Series(np.where(ptdata.race.values == 1, 0, 1), ptdata.index)

# ------------------
# Patient simulation
# ------------------

# Objects to store results of patient simulation
risk1 = [] # (save only for debugging purposes)
risk10 = [] # (save only for debugging purposes)
transitions = [] # (save only for debugging purposes)
bi_Q = [] # true action-value functions
sb_Q = [] # estimated action-value functions
sb_sigma2 = [] # estimated variance of action-value functions
action_set = [] # sets of similar-performing actions
medication_set = [] # sets of similar-performing treatment alternatives
pt_sim = pd.DataFrame() # results of patient simulation
ci_width_list = [] # store only for number of replications analysis

# Running simulation
os.chdir(home_dir)
id_seq = range(0, len(ptdata.id.unique())) # Sequence of patient id's to evaluate       #-# iterate through each patient from 0 to n -Mateo

#-# Added timer to estimate time remaining in run -Mateo
time = tm.time() #-# Init timer and elapsed time -Mateo
elapsedTime = 0
elapsedTimesList = np.array([]) #-# Save elapsed times in an array in order to get avg time per loop -Mateo

if __name__ == '__main__':
    for i in id_seq:

        # Keeping track of progress
        elapsedTimesList = np.append(elapsedTimesList, elapsedTime) #-# add last elapsed time to array of previous elapsed times -Mateo
        #-# Need conditional statement here to catch edge case on first loop. Finding the average of an empty array throws an error -Mateo
        if i == 0:
            avgElapsedTime = 0
        else:
            avgElapsedTime = np.average(elapsedTimesList[1:len(elapsedTimesList)]) #-# don't include initial elapsed time of 0 in average calculation -Mateo

        estimatedTimeLeft = (len(id_seq)-i)*avgElapsedTime #-# Estimate total time left (in seconds) by multiplying the number of patients left and the average elapsed time -Mateo
        estimatedTimeLeft_s = estimatedTimeLeft % 60            #-# Convert total estimated time to seconds remaining
        estimatedTimeLeft_m = estimatedTimeLeft % 3600 // 60    #-# Convert total estimated time to minutes remaining
        estimatedTimeLeft_h = estimatedTimeLeft // 3600         #-# Convert total estimated time to hours remaining
        print("Estimated time remaining "+f"{estimatedTimeLeft_h:.0f}"+"h "+f"{estimatedTimeLeft_m:.0f}"+"m "+f"{estimatedTimeLeft_s:.0f}"+"s") #-# Print estimated time remining in "_h _m _s" format
        print("Estimated completion time "+tm.asctime(tm.localtime(tm.time()+estimatedTimeLeft))[:-5]) #-# Add estimated time remaining to current time to predict estimated time of completion in "Day Month Date H:M:S Year" format -Mateo
        print(tm.asctime(tm.localtime(tm.time()))[:-5], "Evaluating patient", i)
        elapsedTime = tm.time() - time #-# subtract current time from time since last cycle, i.e. get new elapsed time -Mateo
        time = tm.time() #-# Start new timer to use in next iteration -Mateo
        # Extracting patient's data from larger data matrix
        patientdata = ptdata[ptdata.id == i]

        # life expectancy and death likelihood data index
        if patientdata.sex.iloc[0] == 0:  # male
            sexcol = 1  # column in deathdata corresponding to male
        else:
            sexcol = 2  # column in deathdata corresponding to female

        # Death rates
        chddeath = chddeathdata.iloc[list(np.where([j in patientdata.age.values
                                                    for j in list(chddeathdata.iloc[:, 0])])[0]), sexcol]
        strokedeath = strokedeathdata.iloc[list(np.where([j in patientdata.age.values
                                                          for j in list(strokedeathdata.iloc[:, 0])])[0]), sexcol]
        alldeath = alldeathdata.iloc[list(np.where([j in patientdata.age.values
                                                    for j in list(alldeathdata.iloc[:, 0])])[0]), sexcol]

        # Estimating terminal conditions
        Lterm = lifedata.iloc[np.where(patientdata.age.iloc[max(range(years))] == lifedata.iloc[:, 0])[0][0], sexcol] # healthy life expectancy

        ## Storing risk calculations
        ascvdrisk1 = np.full((years, events), np.nan)  # 1-year CHD and stroke risk (for transition probabilities)
        ascvdrisk10 = np.full((years, events), np.nan)  # 10-year CHD and stroke risk (for AHA's guidelines)

        ## Calculating risk for healthy state only (before ordering of states)
        for t in range(years): # each age
            for k in range(events): # each event type

                # 1-year ASCVD risk calculation (for transition probabilities)
                ascvdrisk1[t, k] = rev_arisk(k, patientdata.sex.iloc[t], patientdata.black.iloc[t], patientdata.age.iloc[t],
                                             patientdata.sbp.iloc[t], patientdata.smk.iloc[t], patientdata.tc.iloc[t],
                                             patientdata.hdl.iloc[t], patientdata.diab.iloc[t], 0, 1)

                # 10-year ASCVD risk calculation (for AHA's guidelines)
                ascvdrisk10[t, k] = rev_arisk(k, patientdata.sex.iloc[t], patientdata.black.iloc[t], patientdata.age.iloc[t],
                                              patientdata.sbp.iloc[t], patientdata.smk.iloc[t], patientdata.tc.iloc[t],
                                              patientdata.hdl.iloc[t], patientdata.diab.iloc[t], 0, 10)

        # Calculating transition probabilities (from healthy states only)
        feas, tp = TP(ascvdrisk1, chddeath, strokedeath, alldeath, patientdata.sbp.values, patientdata.dbp.values,
                      sbpmin, sbpmax, dbpmin, sbp_reduction, dbp_reduction, rel_risk_chd, rel_risk_stroke, numhealth)

        # Sorting transition probabilities to satisfy stochastic ordering with respect to states
        tp = tp[order, :, :]

        # Extracting list of feasible actions per decision epoch (from healthy states only)
        feasible = []  # stores index of feasible actions
        for tt in range(feas.shape[0]):
            feasible.append(list(np.where(feas[tt, :] == 1)[0]))
        del feas

        # Calculating necessary observations for a beta confidence level of a difference smaller than delta between estimates and true values
        obs = int(np.ceil(((np.sum(discount**np.arange(years-1)*1)+(discount**9)*Lterm - 0)**2)*np.log(2/beta)/(2*delta**2))) # using life expectancy of the 50-54 patient (always larger than the life expectancy of the 70-74 patient)

        # Initializing objects to store SBPE results (only for healthy state)
        Q_bar = np.full((years, numtrt, reps), np.nan)  # initializing estimate of Q-values per simulation replicate
        Q_hat = np.full((years, numtrt), np.nan)  # initializing overall estimate of Q-values
        sigma2_bar = np.full((years, numtrt, reps), np.nan)  # initializing estimate of the variance of Q-values per simulation replicate
        sigma2_hat = np.full((years, numtrt), np.nan) # initializing estimate of the variance of the average Q-values per simulation replicate

        ## Initializing objects to store SBMCC results
        d_alpha = np.full(years, np.nan)  # array to store empirical 1-epsilon quantiles at all decision epochs across sensitivity analysis scenarios
        Pi = pd.DataFrame()  # list to store ranges of actions at all decision epochs across sensitivity analysis scenarios
        Pi_meds = Pi.copy() # list to store ranges of medications at all decision epochs across sensitivity analysis scenarios

        # Calculating policies according to 2017 AHA's guidelines (controls)
        pi_aha = aha_guideline(ascvdrisk10, patientdata.sbp.values, patientdata.dbp.values,
                               targetrisk, targetsbp, targetdbp, sbpmin, dbpmin,
                               sbp_reduction, dbp_reduction, rel_risk_chd, rel_risk_stroke)

        for t in reversed(range(years)): # number of decisions remaining
            # Extracting transition probabilities and feasibility checks from appropriate scenarios at current year
            print("Simulating year", t+1) #-# Added printout here to better monitor progress -Mateo
            tp_t = tp[:, t, :]
            feasible_t = feasible[t]

            # Q-value at next period
            if t == max(range(years)):
                Q_hat_next = Lterm # expected lifetime as terminal reward
            else:
                Q_hat_next = Q_hat[t+1, pi_aha[t+1].astype(int)] # Q-value associated with next year's action

            # Running Simulation-based policy evaluation algorithm (only for healthy state)
            ## Running health trajectories in parallel (only for healthy state)
            with mp.Pool(cores) as pool:  # creating pool of parallel workers
                Q_sim = pool.starmap_async(sb_pe.sbpe, [(tp_t, healthy, Q_hat_next, discount, r)
                                                        for r in range(obs*reps)]).get()

            ## Converting results into array (with appropriate dimensions)
            Q_sim = np.array(np.split(np.stack(Q_sim), reps, axis=0)).T  # Splitting results into reps number of batches of obs number of observations (and transposing array to match destination)

            # Calculating estimates of Q-values and their variances per replication
            Q_bar[t, :, :] = np.nanmean(Q_sim, axis=1)  # estimated Q-value at each replication
            sigma2_bar[t, :, :] = np.nanvar(Q_sim, axis=1, ddof=1)  # estimated variance per replication

            # Calculating estimates of Q-values and their variances across replications (for all scenarios)
            Q_bar[t, ...][Q_bar[t, ...] < 0] = 0  # making sure that rewards are non-negative (assuming that there is nothing worse than death)
            Q_hat[t, :] = np.nanmean(Q_bar[t, :, :], axis=1)  # overall estimated Q-value (excluding initial batch)
            sigma2_hat[t, :] = np.nanvar(Q_bar[t, :, :], axis=1, ddof=1)  # estimated variance of the replication (batch) average (excluding initial batch)

            # Running simulation-based multiple comparison with a control algorithm
            ## Calculating root statistic in parallel (only in healthy state)
            with mp.Pool(cores) as pool:  # creating pool of parallel workers
                max_psi = pool.starmap_async(sb_mcc_pe.sbmcc, [(Q_bar[t, :, :], Q_hat[t, :],
                                                                sigma2_bar[t, :, :], pi_aha[t].astype(int),
                                                                obs, rep, False)
                                                               for rep in range(reps)]).get() # replase "False" by "True" in the last argument for number of batches analysis

            # ## Estimating width of confidence intervals (if byrep=True) # run only for number of batches analysis
            # d_alpha_reps = [np.percentile(max_psi[x], axis=0, q=(1-alpha)*100, interpolation="nearest") for x in range(1, reps)] # calculating 1-alpha quantile for each replication
            # sigma2_hat_reps = [np.nanvar(Q_bar[t, :, :(y+1)], axis=1, ddof=1) for y in range(1, reps)] # calculating estimate of variance of the relication average
            # ci_width = [np.amax(d_alpha_reps[x]*np.sqrt((sigma2_hat_reps[x][feasible_t] + sigma2_hat_reps[x][pi_aha[t].astype(int)])/(x+1)))
            #             for x in range(1, reps-1)] # calculating width (excluding results with a single replication)  ### can we use range(1, reps)????
            # ci_width_list.append(ci_width)

            ## Converting results to an array (if byrep=False)
            max_psi = np.array(max_psi)

            ## Calculating quantile values
            d_alpha[t] = np.apply_along_axis(np.percentile, axis=0, arr=max_psi,
                                             q=(1-alpha)*100, interpolation="nearest")

            ## Identifying set of actions that are not significantly different from the approximately optimal action (in the set of feasible actions)
            Pi_epoch = np.where(Q_hat[t, pi_aha[t].astype(int)]-Q_hat[t, feasible_t] <=
                                d_alpha[t]*np.sqrt((sigma2_hat[t, feasible_t] +
                                                           sigma2_hat[t, pi_aha[t].astype(int)])/reps))[0]

            ### Making sure that we at least get one element in the set (if there is no variation in Q-values Pi_epoch = [])
            if Pi_epoch.shape[0] == 0:
                Pi_epoch = [pi_aha[t].astype(int)]

            ### Saving set of similar-performing actions
            Pi = pd.concat([pd.DataFrame(Pi_epoch, columns=[str(t)]), Pi], axis=1)

            del Q_sim, Q_hat_next # deleting unnecessary variables (making sure that they are not recycled from next year)

        # Evaluating policy from AHA's guidelines in true transition probabilities
        V_pi_aha = evaluate_pi(pi_aha.astype(int), tp, healthy, Lterm, disutility, discount)

        # Evaluating policy from AHA's guidelines in true transition probabilities in terms of ASCVD events
        aha_evt, aha_time_evt = evaluate_events(pi_aha.astype(int), tp, event_id)

        # Sampling best possible action in similar-performing sets
        pi_best_range = np.full(years, np.nan)  # stores best action in range
        for t in range(Pi.shape[1]):
            if Pi.iloc[:, t].shape[0] > 0:  # Making sure that there are elements in the range of actions
                pi_best_range[t] = Pi[str(t)][np.argmax(Q_hat[t, Pi[str(t)].dropna().astype(int)])].astype(int) # best action in next year's range
            else: # Only the control action is part of the range
                pi_best_range[t] = pi_aha[t].astype(int)

        ## Evaluating policy from range with the best treatment in true transition probabilities
        V_pi_best_range = evaluate_pi(pi_best_range.astype(int), tp, healthy, Lterm, disutility, discount)

        ## Evaluating policy from range with the best treatment in true transition probabilities in terms of ASCVD events
        best_range_evt, best_range_time_evt = evaluate_events(pi_best_range.astype(int), tp, event_id)

        # Sampling median number of medications (rounding up, if necessary) at each year
        pi_med_range = np.full(years, np.nan)  # stores median action in range
        for t in range(Pi.shape[1]):
            if Pi.iloc[:, t].shape[0] > 0:  # Making sure that there are elements in the range of actions
                pi_med_range[t] = np.ceil(Pi[str(t)].median()).astype(int) # median action in range
            else: # Only the control action is part of the range
                pi_med_range[t] = pi_aha[t].astype(int)

        ## Evaluating policy from range with the median number of medications in true transition probabilities
        V_pi_med_range = evaluate_pi(pi_med_range.astype(int), tp, healthy, Lterm, disutility, discount)

        ## Evaluating policy from range with the median number of medications in true transition probabilities in terms of ASCVD events
        med_range_evt, med_range_time_evt = evaluate_events(pi_med_range.astype(int), tp, event_id)

        # Sampling fewest number of medications at each year
        pi_fewest_range = np.full(years, np.nan)  # stores smallest action in range
        for t in range(Pi.shape[1]):
            if Pi.iloc[:, t].shape[0] > 0:  # Making sure that there are elements in the range of actions
                pi_fewest_range[t] = Pi[str(t)].min().astype(int) # sampling smallest action
            else: # Only the control action is part of the range
                pi_fewest_range[t] = pi_aha[t].astype(int)

        ## Evaluating policy from range with the fewest number of medications in true transition probabilities
        V_pi_fewest_range = evaluate_pi(pi_fewest_range.astype(int), tp, healthy, Lterm, disutility, discount)

        ## Evaluating policy from range with the fewest number of medications in true transition probabilities in terms of ASCVD events
        fewest_range_evt, fewest_range_time_evt = evaluate_events(pi_fewest_range.astype(int), tp, event_id)

        # Creating data frame of range of medications
        Pi_meds = pd.DataFrame(np.select([Pi.isna()] + [Pi == x for x in range(numtrt)], np.append(np.nan, meds)))

        # Evaluating no treatment in true transition probabilities
        V_no_trt = evaluate_pi(np.zeros(years, dtype=int), tp, healthy, Lterm, disutility, discount)

        # Evaluating no treatment in true transition probabilities in terms of ASCVD events
        notrt_evt, notrt_time_evt = evaluate_events(np.zeros(years, dtype=int), tp, event_id)

        ## Data frame of results for a single patient (single result per patient-year)
        ptresults = pd.concat([pd.Series(np.repeat(i, years), name='id'),
                                   pd.Series(np.arange(years), name='year'),

                                   pd.Series(V_no_trt, name='V_notrt'), # patient's true value functions for no treatment
                                   pd.Series(V_pi_aha, name='V_aha'), # patient's true value functions under AHA's guidelines
                                   pd.Series(V_pi_best_range, name='V_best_range'), # patient's true value functions using the best treatment option in range
                                   pd.Series(V_pi_med_range, name='V_med_range'), # patient's true value functions using the median treatment option in range
                                   pd.Series(V_pi_fewest_range, name='V_fewest_range'), # patient's true value functions using worst treatment option in range

                                   pd.Series(pi_aha, name='pi_aha'), # patient's policy according to AHA's guidelines
                                   pd.Series(pi_best_range, name='pi_best_range'), # patient's policy using the best treatment option in range
                                   pd.Series(pi_med_range, name='pi_med_range'), # patient's policy using the median treatment option in range
                                   pd.Series(pi_fewest_range, name='pi_fewest_range'), # patient's policy from the worst treatment option in range

                                   pd.Series(notrt_evt, name='evt_notrt'), # expected number of events under no treatment
                                   pd.Series(aha_evt, name='evt_aha'), # expected number of events under AHA's guideline
                                   pd.Series(best_range_evt, name='evt_best_range'), # expected number of events using the best treatment option in range
                                   pd.Series(med_range_evt, name='evt_med_range'), # expected number of events using the median treatment option in range
                                   pd.Series(fewest_range_evt, name='evt_fewest_range'), # expected number of events using worst treatment option in range

                                   pd.Series(notrt_time_evt, name='time_evt_notrt'), # expected years until an adverse event (including non-ASCVD related death) under no treatment
                                   pd.Series(aha_time_evt, name='time_evt_aha'), # expected years until an adverse event (including non-ASCVD related death) under AHA's guideline
                                   pd.Series(best_range_time_evt, name='time_evt_best_range'), # expected years until an adverse event (including non-ASCVD related death) using the median treatment option in range
                                   pd.Series(med_range_time_evt, name='time_evt_med_range'), # expected years until an adverse event (including non-ASCVD related death) using the median treatment option in range
                                   pd.Series(fewest_range_time_evt, name='time_evt_fewest_range') # expected years until an adverse event (including non-ASCVD related death) using worst treatment option in range
                               ], axis=1)

        # Merging single patient data in data frame with data from of all patients
        pt_sim = pt_sim.append(ptresults, ignore_index=True)
        ptresults = np.nan # making sure values are not recycled

        # Saving patient-level results (for healthy state only)
        ## List of results (multiple results per patient-year)
        # risk1.append(ascvdrisk1)  # patient's 1-year risk calculations (save only for debugging purposes)
        # risk10.append(ascvdrisk10)  # patient's 10-year risk calculations (save only for debugging purposes)
        # transitions.append(tp)  # patient's transition probabilities (save only for debugging purposes)
        sb_Q.append(Q_hat)  # patient's estimates of Q-values
        sb_sigma2.append(sigma2_hat) # variance of avergae Q-values per replication
        action_set.append(Pi)  # patient's ranges of near-optimal actions (for healthy state)
        medication_set.append(Pi_meds)  # patient's ranges of near-optimal treatment choices (for healthy state)

        # Saving all results (saving each time a patient is evaluated)
        os.chdir(home_dir)
        if not os.path.isdir("Results"):
            os.mkdir("Results")
        os.chdir(results_dir)
        with open('Results for patients ' + str(min(id_seq)) + ' to ' + str(max(id_seq)) + ' until patient ' + str(i) + ' using adaptive observations and ' + str(reps) +
                  ' batches.pkl', 'wb') as f: # full simulation
            pk.dump([#risk1, risk10, transitions,
                     sb_Q, sb_sigma2, action_set, medication_set,
                     pt_sim], f, protocol=3)
        # with open('Results for patient 317 in different clinical scenarios using adaptive observations and '+str(reps)+
        #           ' batches-stage 2 hypertension and different ages.pkl', 'wb') as f: #simulation of a single patient
        #     pk.dump([bi_Q, sb_Q, sb_sigma2, action_set, medication_set,
        #              pt_sim], f, protocol=3)
        # with open('Number of batches analysis for patients ' + str(min(id_seq)) + ' to ' + str(i) + '.pkl', 'wb') as f:  # number of batches analysis
        #     pk.dump(ci_width_list, f, protocol=3)
