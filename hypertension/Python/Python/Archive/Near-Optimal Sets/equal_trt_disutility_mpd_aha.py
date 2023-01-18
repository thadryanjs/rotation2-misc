# ===========================================================================================================
# Finding disutility value so that the MDP gives the same number of medications as the 2017 AHA's guidelines
# ===========================================================================================================

# Loading modules
import os  # directory changes
import pandas as pd  # data frame operations
import numpy as np  # matrix operations
import pickle as pk  # saving results
import multiprocessing as mp  # parallel computations
from ascvd_risk import rev_arisk  # risk calculations
from transition_probabilities import TP  # transition probability calculations
from aha_2017_guideline import aha_guideline # 2017 AHA's guideline for hypertension treatment
from backwards_induction_mdp import backwards_induction  # solving MDP using backwards induction

# Importing parameters from main module
from hypertension_treatment_sbbi_sbmcc import cores, home_dir, ptdata, numhealth, years, numtrt, events, chddeathdata, \
    strokedeathdata, alldeathdata, sbpmin, sbpmax, dbpmin, sbp_reduction, dbp_reduction, rel_risk_chd, rel_risk_stroke, \
    healthy, order, QoL, QoLterm, lifedata, mortality_rates, targetrisk, targetsbp, targetdbp, disutility, disut_std, discount, meds

# Ptient simulation function
def patient_sim(ptid, disut):
    # Extracting patient's data from larger data matrix
    patientdata = ptdata[ptdata.id == ptid]

    # Assume that the patient has the same pre-treatment SBP/DBP no matter the health status
    pretrtsbp = np.ones([numhealth, years])*np.array(patientdata.sbp)
    pretrtdbp = np.ones([numhealth, years])*np.array(patientdata.dbp)

    # Storing risk calculations
    ascvdrisk1 = np.zeros((numhealth, years, events))  # 1-year CHD and stroke risk (for transition probabilities)
    ascvdrisk10 = np.zeros((numhealth, years, events))  # 10-year CHD and stroke risk (for AHA's guidelines)

    # Calculating risk for healthy state only (before ordering of states)
    for t in range(years):  # each age
        for k in range(events):  # each event type

            # 1-year ASCVD risk calculation (for transition probabilities)
            ascvdrisk1[0, t, k] = rev_arisk(k, patientdata.sex.iloc[t], patientdata.black.iloc[t],
                                            patientdata.age.iloc[t],
                                            patientdata.sbp.iloc[t],
                                            patientdata.smk.iloc[t], patientdata.tc.iloc[t],
                                            patientdata.hdl.iloc[t],
                                            patientdata.diab.iloc[t], 0, 1)

            # 10-year ASCVD risk calculation (for AHA's guidelines)
            ascvdrisk10[0, t, k] = rev_arisk(k, patientdata.sex.iloc[t], patientdata.black.iloc[t],
                                             patientdata.age.iloc[t],
                                             patientdata.sbp.iloc[t],
                                             patientdata.smk.iloc[t], patientdata.tc.iloc[t],
                                             patientdata.hdl.iloc[t],
                                             patientdata.diab.iloc[t], 0, 10)

    # life expectancy and death likelihood data index
    if patientdata.sex.iloc[0] == 0:  # male
        sexcol = 1  # column in deathdata corresponding to male
    else:
        sexcol = 2  # column in deathdata corresponding to female

    # Death rates
    chddeath = chddeathdata.iloc[list(np.where([j in list(patientdata.age)
                                                for j in list(chddeathdata.iloc[:, 0])])[0]), sexcol]
    strokedeath = strokedeathdata.iloc[list(np.where([j in list(patientdata.age)
                                                      for j in list(strokedeathdata.iloc[:, 0])])[0]), sexcol]
    alldeath = alldeathdata.iloc[list(np.where([j in list(patientdata.age)
                                                for j in list(alldeathdata.iloc[:, 0])])[0]), sexcol]

    # Calculating transition probabilities
    feas, tp = TP(ascvdrisk1, chddeath, strokedeath, alldeath, pretrtsbp, pretrtdbp,
                  sbpmin, sbpmax, dbpmin, sbp_reduction, dbp_reduction, rel_risk_chd, rel_risk_stroke)

    # Sorting transition probabilities and feasibility checks to satisfy stochastic ordering with respect to states
    tp = tp[order, :, :, :]; tp = tp[:, order, :, :]; feas = feas[order, :, :]

    # Making treatment infeasible in non-healthy states (and making sure no treatment is always feasible)
    feas[np.arange(feas.shape[0]) != healthy, :, :] = 0
    feas[np.arange(feas.shape[0]) != healthy, :, 0] = 1

    # Extracting list of feasible actions per state and decision epoch
    feasible = []  # stores index of feasible actions
    for h in range(feas.shape[0]):
        tmp = []
        for t in range(feas.shape[1]):
            tmp.append(list(np.where(feas[h, t, :] == 1)[0]))
        feasible.append(tmp); del tmp
    del feas

    # Inmediate rewards by age
    qol = None
    if 40 <= patientdata.age.iloc[0] <= 44:
        qol = QoL.get("40-44")
    elif 45 <= patientdata.age.iloc[0] <= 54:
        qol = QoL.get("45-54")
    elif 55 <= patientdata.age.iloc[0] <= 64:
        qol = QoL.get("55-64")
    elif 65 <= patientdata.age.iloc[0] <= 74:
        qol = QoL.get("65-74")
    elif 75 <= patientdata.age.iloc[0] <= 84:
        qol = QoL.get("75-84")

    # Estimating terminal conditions
    # #Healthy life expectancy
    healthy_lifexp = lifedata.iloc[np.where(patientdata.age.iloc[max(range(years))] == lifedata.iloc[:, 0])[0][0], sexcol]

    # #Mortality rates by gender
    if patientdata.sex.iloc[0] == 0:  # Male mortality rates
        SMR = mortality_rates.get("Males <2 CHD events")
    else:  # Female mortality rates
        SMR = mortality_rates.get("Females <2 CHD events")

    # #Terminal QALYs
    Qterm = None
    if 40 <= patientdata.age.iloc[max(range(years))] <= 44:
        Qterm = [j*k*healthy_lifexp for j, k in zip(SMR, QoLterm.get("40-44"))]
    elif 45 <= patientdata.age.iloc[max(range(years))] <= 54:
        Qterm = [j*k*healthy_lifexp for j, k in zip(SMR, QoLterm.get("45-54"))]
    elif 55 <= patientdata.age.iloc[max(range(years))] <= 64:
        Qterm = [j*k*healthy_lifexp for j, k in zip(SMR, QoLterm.get("55-64"))]
    elif 65 <= patientdata.age.iloc[max(range(years))] <= 74:
        Qterm = [j*k*healthy_lifexp for j, k in zip(SMR, QoLterm.get("65-74"))]
    elif 75 <= patientdata.age.iloc[max(range(years))] <= 84:
        Qterm = [j*k*healthy_lifexp for j, k in zip(SMR, QoLterm.get("75-84"))]

    # Calculating policies according to 2017 AHA's guidelines
    aha = aha_guideline(ascvdrisk10[0, :, :], pretrtsbp[0, :], pretrtdbp[0, :], targetrisk, targetsbp, targetdbp,
                        sbpmin, dbpmin, sbp_reduction, dbp_reduction, rel_risk_chd, rel_risk_stroke)

    # Calculating policies using backwards induction (to compare results)
    _, _, pi = backwards_induction(tp, qol, Qterm, disut, discount, feasible)

    # Converting policy to medications
    pi_meds = np.full(years, np.nan)
    aha_meds = np.full(years, np.nan)
    for m in range(numtrt):
        pi_meds = np.where(pi[healthy, :] == m, meds[m], pi_meds)
        aha_meds = np.where(aha == m, meds[m], aha_meds)

    return np.sum(pi_meds), np.sum(aha_meds)


# Disutility values to be evaluated
disut_list = np.arange(0.0183, 0.0184, 0.00001) # 0.01836 gives a difference of 1 medication at standard dosage

# Objects to store results of simulation
med_sum = []

# Running simulation in parallel
os.chdir(home_dir)
id_seq = range(len(ptdata.id.unique())) # Sequence of patient id's to evaluate
if __name__ == '__main__':
    with mp.Pool(cores) as pool:  # Creating pool of parallel workers
        for d in range(len(disut_list)):
            # Changing original distulity to disutility scenario
            trtharm = disutility/disut_std*disut_list[d]

            # Parallel computation
            par_results = pool.starmap_async(patient_sim, [(i, trtharm) for i in id_seq]).get()

            # Converting results to a data frame
            df = pd.DataFrame(par_results, columns=['BI', 'AHA'])

            # Storing complete data frame
            med_sum.append(df)

            # Printing overall results
            print("Scenario:", disut_list[d], "Backwards Induction:", df['BI'].sum().round(2),
                  "AHA's Guideline:", df['AHA'].sum().round(2),
                  "Difference", (df['BI'].sum() - df['AHA'].sum()).round(2))
