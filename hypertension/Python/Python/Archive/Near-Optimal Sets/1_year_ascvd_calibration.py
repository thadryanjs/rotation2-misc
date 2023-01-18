# ===========================================================================================
# Estimating 1-year ASCVD risk from revised 10-year ASCVD risk calculator in Yadlowsky (2018)
# ===========================================================================================

# Loading modules
import os  # directory changes
import pandas as pd  # data frame operations
import numpy as np  # matrix operations
import time as tm  # timing code
import pickle as pk  # saving results
from ascvd_risk import arisk, rev_arisk  # risk calculations
from gurobipy import *  # Note: copy and paste module files to appropiate interpreter's site-packages folder

# Importing parameters from main module
from hypertension_treatment_sbbi_sbmcc import data_dir

# Loading dataset
os.chdir(data_dir + '\\Continuous NHANES')
ptdata = pd.read_csv('Continuous NHANES 50-54 Dataset.csv') # 2009-2013 Continuous NHANES dataset adults ages 50-54
# ptdata = pd.read_csv('Continuous NHANES 40-75 Dataset.csv') # 2009-2013 Continuous NHANES dataset adults ages 40-75

# Calculating 10-year ASCVD risk for each patient at every year
ptdata['risk10'] = ptdata.apply(lambda row: rev_arisk(0, row['sex'], np.where(row['race'] == 1, 0, 1), row['age'],
                                                      row['sbp'], row['smk'], row['tc'], row['hdl'], row['diab'],
                                                      0, 10) + rev_arisk(1, row['sex'], np.where(row['race'] == 1, 0, 1),
                                                                         row['age'], row['sbp'], row['smk'], row['tc'],
                                                                         row['hdl'], row['diab'], 0, 10), axis=1)

## Extrating parameters
Y = np.count_nonzero(ptdata.id==0) # number of forecasted years per patient
P = int(ptdata.shape[0]/Y) # number of (unique) patients
delta = 0.03 # allowable difference between sum of ten 1-year risks and the 10-year risk per patient (selected by trial and error) #use delta = 0.03 for ages 50-54 and delta = 0.049 for ages 40-75

# Extracting first year data
ptdata1 = ptdata.groupby('id').first().reset_index()

# Extracting risk information over planning horizon
riskdf = ptdata[['id', 'risk10']]
riskdf = riskdf.assign(year=pd.Series(np.tile(np.arange(Y), P), index=riskdf.index))
riskdf = riskdf.pivot(index='id', columns='year', values='risk10')

# Linear program to find optimal multiplier to convert from 10-year risk to 1-year risk
## Creating lists of states and actions
years = list(np.arange(0, Y, 1))
patients = list(np.arange(0, P, 1))

## Creating Gurobi model object
m = Model()

## Adding decision variable to model (the only decision variable is the mutiplier)
abs_diff = m.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS)
mult = m.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS)

## Declaring model objective
m.setObjective(abs_diff, GRB.MINIMIZE)

## Adding constraints
### Absolute value constraints (aggrgated constraints)
const1 = m.addConstr(quicksum((quicksum(riskdf.iloc[p, y]*mult for y in years) - ptdata1.risk10[p]) for p in patients) <= abs_diff)
const2 = m.addConstr(quicksum((-quicksum(riskdf.iloc[p, y]*mult for y in years) + ptdata1.risk10[p]) for p in patients) <= abs_diff)

### Tolerance constraints
const3 = m.addConstrs((ptdata1.risk10[p] - quicksum(riskdf.loc[p, y]*mult for y in years) <= delta for p in patients))
const4 = m.addConstrs((quicksum(riskdf.loc[p, y]*mult for y in years) - ptdata1.risk10[p] <= delta for p in patients))

# Processing model specifications
m.update()

# Surpressing output
m.setParam('OutputFlag', False)

# Setting time limit to 2 hours
m.setParam('TimeLimit', 7200)

m.setParam('DualReductions', 0)

# Optimizing model
m.optimize()

# Extracting optimal multiplier
if m.Status == 2: # Model was solved to optimality

    # Storing optimal multiplier
    opt_mult = mult.X # 0.082 for 50-54 # 0.088 for 40-75 years old
    print(opt_mult)

    # Storing objevtive value
    obj_val = m.objVal
    print(obj_val)

    # Storing shadow prices
    sp1 = const1.Pi
    sp2 = const2.Pi
    sp3 = np.empty(len(patients)); sp3[:] = np.nan
    sp4 = np.empty(len(patients)); sp4[:] = np.nan
    for p in patients:
        sp3[p] = const3[p].Pi
        sp4[p] = const4[p].Pi

    # Estimating 10-year risk from the summation of ten 1-year risks
    risk_est = ptdata[['id', 'risk10']]
    risk_est.risk10 = risk_est.risk10*mult.X
    risk_est = risk_est.groupby('id').sum().reset_index()

    # Comparing the summation of ten 1-year risk to the 10-year risk at year 1
    ad = np.abs(risk_est.risk10-riskdf.loc[:, 0])
    print(np.round([np.mean(ad), np.median(ad), np.min(ad), np.max(ad)], 3))

else:
    opt_mult = np.nan
    print("Optimal solution was not found - status code:", m.Status)

