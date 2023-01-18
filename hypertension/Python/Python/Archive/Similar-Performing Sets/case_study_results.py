# =======================
# Summarizing Results
# =======================

# Loading modules
import os  # directory changes
import pandas as pd  # data frame operations
import numpy as np  # matrix operations
import pickle as pk  # saving results
import gc # garbage collector
from itertools import islice # dividing lists into sublists of specific lengths
from plotting_functions import plf_cov_batch, plot_range_actions_exp, plot_range_actions, plot_ly_saved_bp, plot_ly_saved_race, plot_simplified_trt_dist

# Importing parameters from main module
from hypertension_treatment_policy_evaluation import results_dir, fig_dir, rev_arisk, ptdata, years, meds, reps

# ---------------------------
# Number of batches analysis
# ---------------------------
#
# # Loading results (if not previously combined)
# ## First chunk of patients # error in saving file
# # os.chdir(results_dir)
# # with open('Number of batches analysis until patient 182.pkl',
# #           'rb') as f:
# #     ci_width_list1 = pk.load(f)
#
# ## Second chunk of the patients
# os.chdir(results_dir)
# with open('Number of batches analysis for patients 183 to 438.pkl',
#           'rb') as f:
#     ci_width_list2 = pk.load(f)
#
# ## Third chunk of the patients
# os.chdir(results_dir) # wrong number of batches
# with open('Number of batches analysis for patients 439 to 575.pkl',
#           'rb') as f:
#     ci_width_list3 = pk.load(f)
#
# # Combining results
# ci_width_list = ci_width_list2 + ci_width_list3 #ci_width_list1 +
#
# # Calculating the max width of confidence intervals by number of batches across all patients
# ci_width_list_year1 = ci_width_list[9::10] # selecting the confidence interval half widths in the first year of each patient
# ci_width_max = [max([ci_width_list_year1[x][y] for x in range(len(ci_width_list_year1))]) for y in range(len(ci_width_list_year1[0]))]
#
# # Plotting convergence of maximum width across patients
# os.chdir(fig_dir)
# plf_cov_batch(ci_width_max, plot_batches=498, selected_batches=300)

# -------------------------
# Patient-level analysis
# -------------------------

# Loading results
## Results for patient profiles
### First chunk
os.chdir(results_dir)
with open('Results for patients 0 to 1112 until patient 1112 using adaptive observations and 300 batches.pkl',
          'rb') as f:
    [sb_Q1, sb_sigma21, _, medication_range1, pt_sim1] = pk.load(f)
del _; gc.collect()

### Second chunk
with open('Results for patients 0 to 1112 until patient 1112 using adaptive observations and 300 batches.pkl',
          'rb') as f:
    [sb_Q2, sb_sigma22, _, medication_range2, pt_sim2] = pk.load(f)
del _; gc.collect()

#-# Made some changes to accommodate result files that aren't split -Mateo
# Combining results
# sb_Q = sb_Q1 + sb_Q2
# sb_sigma2 = sb_sigma21 + sb_sigma22
# medication_range = medication_range1 + medication_range2
# pt_sim = pd.concat([pt_sim1, pt_sim2], axis=0, ignore_index=True)
sb_Q = sb_Q1
sb_sigma2 = sb_sigma21
medication_range = medication_range1
pt_sim = pd.concat([pt_sim1], axis=0, ignore_index=True)
del sb_Q1, sb_Q2, sb_sigma21, sb_sigma22, medication_range1, medication_range2, pt_sim1, pt_sim2

# Converting actions to number of medications
pt_sim['meds_aha'] = np.select([pt_sim.pi_aha==x for x in range(len(meds))], meds)

# Preparing data for plot
meds_df = pt_sim.loc[:, ['id', 'year', 'meds_aha']].copy()
meds_df.year += 1
meds_df = meds_df.melt(id_vars=['id', 'year'], var_name='policy', value_name='meds')

# Making sure there is a directory to store every patient's plot
os.chdir(fig_dir)
new_dir = "Medication Ranges"
if not os.path.isdir(new_dir):
    os.mkdir(new_dir)
os.chdir(os.path.join(fig_dir, new_dir))
os.chdir(fig_dir)

# ---------------------------
# Patient-level analysis
# ---------------------------

## Plotting selected cases together
# Getting demographic and clinical information from selected patients
# sel_ids = [288, 291, 297, 393] # 40-year-old with elevated BP, stage 1 hypertension and stage 2 hypertension, 60-year-old with stage 2 hypertension
# for i in sel_ids:
#     print(ptdata[ptdata.id==i].astype(int).iloc[0, :])
#     plot_range_actions_exp(i, meds_df.meds[meds_df.id == i], medication_range[i]) # plotting patients separately

## Subsetting data and plotting
# os.chdir(fig_dir)
# meds_sel = meds_df.loc[meds_df.id.isin(sel_ids), ['id', 'meds']]
# medication_range_sel = [medication_range[x] for x in sel_ids]
# plot_range_actions(meds_sel, medication_range_sel)

## Plot for single patient (for SMDM 2022)
# os.chdir(fig_dir)
# id_smdm = [sel_ids[2]]
# meds_aha = meds_df.meds[meds_df.id==id_smdm[0]]
# Pi_meds = [medication_range[x] for x in id_smdm][0]
# plot_range_actions(meds_sel, medication_range_sel)

# ---------------------------
# Population-level analysis (Mateo)
# Plots: LY Saved per BP Category, LY Saved by Racial Category, Simplified Medication Range Chart
#   Run code for one plotting method at a time, comment out code for other plotting sections (to avoid errors caused by variable reuse)
# Plots used in ASURE Poster Presentation 2022
# -Mateo
# ---------------------------

# -------------- Blood Pressure Categories ------------------
ptdata_list = [ptdata]
ptdata = pd.concat([pd.Series(np.tile(np.arange(1, 11), ptdata_list[0].id.unique().shape[0]), name='year'), ptdata_list[0]], axis=1)

pt_sum = ptdata[['year', 'race', 'sex', 'sbp', 'dbp']].groupby(['year', 'race', 'sex']).describe().reset_index(drop=False, inplace=False).astype(int)

# Extracting first year data
ptdata1 = ptdata_list[0].groupby('id').first().reset_index(drop=False, inplace=False) # extracting first year data in 40-44 age group
del ptdata_list; gc.collect()

# Identifying BP categories
bp_cats = [(ptdata1.sbp < 120) & (ptdata1.dbp < 80),
            (ptdata1.sbp >= 120) & (ptdata1.dbp < 80) & (ptdata1.sbp < 130),
            ((ptdata1.sbp >= 130) | (ptdata1.dbp >= 80)) & ((ptdata1.sbp < 140) & (ptdata1.dbp < 90)),
            (ptdata1.sbp >= 140) | (ptdata1.dbp >= 90)]
bp_cat_labels = ['Normal', 'Elevated', 'Stage 1', 'Stage 2']
ptdata1['bp_cat'] = np.select(bp_cats, bp_cat_labels)

ptdata1[['race', 'sex', 'bp_cat', 'sbp', 'dbp']].groupby(['bp_cat', 'race', 'sex']).mean().reset_index(drop=False, inplace=False)

# Overall demographic information
pd.concat([(ptdata1[['wt', 'bp_cat']].groupby(['bp_cat']).sum()/1e06).round(2),
           (ptdata1[['wt', 'bp_cat']].groupby(['bp_cat']).sum()/ptdata1.wt.sum()*100).round(2)],
          axis=1)

# Making plot of demographic information by BP categories
demo = (ptdata1[['wt', 'race', 'sex', 'bp_cat']].groupby(['race', 'sex', 'bp_cat']).sum()/1e06).reset_index(drop=False, inplace=False).round(2)
demo['bp_cat'] = demo['bp_cat'].astype('category') # converting scenario to category
demo.bp_cat.cat.set_categories(bp_cat_labels, inplace=True) # adding sorted categories
demo = demo.sort_values(['bp_cat']) # sorting dataframe based on selected columns

## Making plot
os.chdir(fig_dir)

# Incorporating demographic and grouping information
pt_sim = pd.merge(pt_sim, ptdata1[['id', 'age', 'sex', 'race', 'smk', 'diab', 'bp_cat']], on='id')

# Converting actions to number of medications
pt_sim['meds_aha'] = np.select([pt_sim.pi_aha==x for x in range(len(meds))], meds)
pt_sim['meds_best'] = np.select([pt_sim.pi_best_range==x for x in range(len(meds))], meds)
pt_sim['meds_med'] = np.select([pt_sim.pi_med_range==x for x in range(len(meds))], meds)
pt_sim['meds_few'] = np.select([pt_sim.pi_fewest_range==x for x in range(len(meds))], meds)

# Adjusting AHA's guidelines for feasibility (for some reason the aha_guideline function is allowing over-treatment past the feasibility condition)
## Identifying patients with infeasible treatment
pt_sim['meds_largest'] = np.vstack([x.max() for x in medication_range]).flatten()
pt_sim['ind'] = np.where((pt_sim.meds_largest<pt_sim.meds_aha) & # over-treatment condition
                         ((pt_sim.V_best_range-pt_sim.V_aha)<=(pt_sim.V_best_range-pt_sim.V_fewest_range)), # near-optimal condition (it would have been part of the ranges if it wasn't for the feasibility constraint)
                          1, 0) # indicator of infeasibility
infeas_ids = pt_sim.loc[pt_sim.ind==1, 'id'].unique() # ids of patients with infeasible treatments

## Temporary data frames to update value functions
tmp = pt_sim.loc[[x in infeas_ids for x in pt_sim.id], ['id', 'V_aha']].groupby('id').diff(periods=-1).rename(columns={'V_aha': 'V_aha_diff'}).reset_index(drop=True) # estimating expected immediate rewards from value functions
tmp1 = pd.concat([pd.Series(np.repeat(infeas_ids, 10), name='id'), pd.Series(np.tile(np.arange(10), infeas_ids.shape), name='year'), tmp], axis=1) # incorporating ids and year
tmp1['key'] = (tmp1['id'].astype(str) + tmp1['year'].astype(str)).astype(int) # creating unique key on temporary data frame
pt_sim['key'] = (pt_sim['id'].astype(str) + pt_sim['year'].astype(str)).astype(int) # creating unique key on main data frame
tmp2 = tmp1.merge(pt_sim[['key', 'ind']], on='key') # incorporating index of infeasibility

## Updating value functions, events, and policy of AHA's guidelines in main data frame
pt_sim.update(pt_sim.loc[pt_sim.ind==1, ['V_best_range', 'evt_best_range', 'pi_best_range', 'meds_best']].
              rename(columns={'V_best_range': 'V_aha', 'evt_best_range': 'evt_aha', 'pi_best_range': 'pi_aha', 'meds_best': 'meds_aha'})) # updating data frame
pt_sim.update(pt_sim.loc[np.where(pt_sim.V_best_range<pt_sim.V_aha)[0], ['V_best_range', 'evt_best_range', 'pi_best_range', 'meds_best']].
              rename(columns={'V_best_range': 'V_aha', 'evt_best_range': 'evt_aha', 'pi_best_range': 'pi_aha', 'meds_best': 'meds_aha'})) # making sure AHA's guideline is not better than the optimal policy for any patient

## Incorporating value functions from optimal (feasible) treatment
tmp2.loc[tmp2.ind==1, 'V_aha_diff'] = pt_sim.loc[pt_sim.ind==1, ['V_aha']].rename(columns={'V_aha': 'V_aha_diff'}).set_index(tmp2[tmp2.ind==1].index) # replacing expected immediate rewards for value functions when possible

## Calculating new value functions (from future value functions and immediate rewards
tmp2.loc[tmp2[tmp2.ind==1].groupby('id')['ind'].head(1).index, 'ind'] -= 1 # Identifying first row replaced by value function
tmp2[['ind2', 'cum_sum']] = tmp2.groupby('id', sort=False)[['ind', 'V_aha_diff']].apply(lambda x: x[::-1]).reset_index(drop=True) # creating new columns by reversing index and value functions
tmp2['cum_sum'] = tmp2[tmp2.ind2==0].groupby('id')['cum_sum'].cumsum() # calculating cumulative sum
tmp2[['ind2', 'cum_sum']] = tmp2.groupby('id', sort=False)[['ind2', 'cum_sum']].apply(lambda x: x[::-1]).reset_index(drop=True) # reversing the columns back to original order
tmp2.loc[tmp2.ind==0, 'V_aha_diff'] = tmp2.loc[tmp2.ind==0, 'cum_sum'] # replacing immediate rewards by estimated value functions
tmp2.drop(['ind', 'ind2', 'cum_sum'], axis=1, inplace=True) # deleting unnecesary columns
tmp2.rename(columns={'V_aha_diff': 'V_aha'}, inplace=True) # renaming columns

## Replacing value functions in main data frame
tmp3 = tmp2.set_index(pt_sim[[x in tmp2.key.to_numpy() for x in pt_sim.key]].index)
pt_sim.loc[[x in tmp2.key.to_numpy() for x in pt_sim.key], 'V_aha'] = tmp3['V_aha']

# Adjusting values by sampling weights
pt_sim = pt_sim.join(ptdata.wt)
# pt_sim.V_notrt = pt_sim.V_notrt*pt_sim.wt/1e06
# pt_sim.V_aha = pt_sim.V_aha*pt_sim.wt/1e06
# pt_sim.V_best_range = pt_sim.V_best_range*pt_sim.wt/1e06
# pt_sim.V_med_range = pt_sim.V_med_range*pt_sim.wt/1e06
# pt_sim.V_fewest_range = pt_sim.V_fewest_range*pt_sim.wt/1e06

# Standardize per capita
pt_sim.V_notrt = pt_sim.V_notrt*pt_sim.wt
pt_sim.V_aha = pt_sim.V_aha*pt_sim.wt
pt_sim.V_best_range = pt_sim.V_best_range*pt_sim.wt
pt_sim.V_med_range = pt_sim.V_med_range*pt_sim.wt
pt_sim.V_fewest_range = pt_sim.V_fewest_range*pt_sim.wt

# Making plot of expected total life years saved (over time per risk group)
# Data frame of expected total life years saved (compared to no treatment) per year
ly_df = pt_sim.loc[:, ['year', 'bp_cat', 'V_best_range', 'V_med_range', 'V_fewest_range', 'V_aha', 'V_notrt']].groupby(['year', 'bp_cat']).sum().reset_index(drop=False, inplace=False) #, 'race', 'sex'
ly_df.year += 1
ly_df.V_best_range = ly_df.V_best_range - ly_df.V_notrt # optimal policy not included in SMDM poster
ly_df.V_med_range = ly_df.V_med_range - ly_df.V_notrt
ly_df.V_fewest_range = ly_df.V_fewest_range - ly_df.V_notrt
ly_df.V_aha = ly_df.V_aha - ly_df.V_notrt
ly_df = ly_df.drop(['V_notrt'], axis=1)

## Preparing data for plot
ly_df = ly_df.rename(columns={'V_best_range': 'Best in Set',
                                    'V_med_range': 'Median in Set',
                                    'V_fewest_range': 'Fewest in Set',
                                    'V_aha': 'Clinical Guidelines'})

#drops all categories besides best?
ly_df = ly_df.melt(id_vars=['year', 'bp_cat'], var_name='policy', value_name='ly') #, 'race'

order = ['Best in Set', 'Median in Set', 'Fewest in Set', 'Clinical Guidelines'] # order for plots #
ly_df['policy'] = ly_df['policy'].astype('category') # converting scenario to category
ly_df.policy.cat.set_categories(order, inplace=True) # adding sorted categories
ly_df = ly_df.sort_values(['policy', 'year']) # sorting dataframe based on selected columns
ly_df = ly_df[ly_df.bp_cat!='Normal'] # removing normal BP group

## Making plot
os.chdir(fig_dir)
plot_ly_saved_bp(ly_df, ptdata1) #-# need to pass ptdata1 to do per capita calculations -Mateo

# -------------- Racial Categories ------------------
ptdata_list = [ptdata]
ptdata = pd.concat([pd.Series(np.tile(np.arange(1, 11), ptdata_list[0].id.unique().shape[0]), name='year'), ptdata_list[0]], axis=1)

# Calculating BP summary statistics by demographics
pt_sum = ptdata[['year', 'race', 'sex', 'sbp', 'dbp']].groupby(['year', 'race', 'sex']).describe().reset_index(drop=False, inplace=False).astype(int)

# Extracting first year data
ptdata1 = ptdata_list[0].groupby('id').first().reset_index(drop=False, inplace=False) # extracting first year data in 40-44 age group
del ptdata_list; gc.collect()

# Identifying BP categories
bp_cats = [(ptdata1.sbp < 120) & (ptdata1.dbp < 80),
            (ptdata1.sbp >= 120) & (ptdata1.dbp < 80) & (ptdata1.sbp < 130),
            ((ptdata1.sbp >= 130) | (ptdata1.dbp >= 80)) & ((ptdata1.sbp < 140) & (ptdata1.dbp < 90)),
            (ptdata1.sbp >= 140) | (ptdata1.dbp >= 90)]
bp_cat_labels = ['Normal', 'Elevated', 'Stage 1', 'Stage 2']
ptdata1['bp_cat'] = np.select(bp_cats, bp_cat_labels)

ptdata1[['race', 'sex', 'bp_cat', 'sbp', 'dbp']].groupby(['bp_cat', 'race', 'sex']).mean().reset_index(drop=False, inplace=False)

# Overall demographic information
pd.concat([(ptdata1[['wt', 'bp_cat']].groupby(['bp_cat']).sum()/1e06).round(2),
           (ptdata1[['wt', 'bp_cat']].groupby(['bp_cat']).sum()/ptdata1.wt.sum()*100).round(2)],
          axis=1)

# Making plot of demographic information by BP categories
demo = (ptdata1[['wt', 'race', 'sex', 'bp_cat']].groupby(['race', 'sex', 'bp_cat']).sum()/1e06).reset_index(drop=False, inplace=False).round(2)
demo['bp_cat'] = demo['bp_cat'].astype('category') # converting scenario to category
demo.bp_cat.cat.set_categories(bp_cat_labels, inplace=True) # adding sorted categories
demo = demo.sort_values(['bp_cat']) # sorting dataframe based on selected columns

## Making plot
os.chdir(fig_dir)

# Incorporating demographic and grouping information
pt_sim = pd.merge(pt_sim, ptdata1[['id', 'age', 'sex', 'race', 'smk', 'diab', 'bp_cat']], on='id')

# Converting actions to number of medications
pt_sim['meds_aha'] = np.select([pt_sim.pi_aha==x for x in range(len(meds))], meds)
pt_sim['meds_best'] = np.select([pt_sim.pi_best_range==x for x in range(len(meds))], meds)
pt_sim['meds_med'] = np.select([pt_sim.pi_med_range==x for x in range(len(meds))], meds)
pt_sim['meds_few'] = np.select([pt_sim.pi_fewest_range==x for x in range(len(meds))], meds)

# Adjusting AHA's guidelines for feasibility (for some reason the aha_guideline function is allowing over-treatment past the feasibility condition)
## Identifying patients with infeasible treatment
pt_sim['meds_largest'] = np.vstack([x.max() for x in medication_range]).flatten()
pt_sim['ind'] = np.where((pt_sim.meds_largest<pt_sim.meds_aha) & # over-treatment condition
                         ((pt_sim.V_best_range-pt_sim.V_aha)<=(pt_sim.V_best_range-pt_sim.V_fewest_range)), # near-optimal condition (it would have been part of the ranges if it wasn't for the feasibility constraint)
                          1, 0) # indicator of infeasibility
infeas_ids = pt_sim.loc[pt_sim.ind==1, 'id'].unique() # ids of patients with infeasible treatments

## Temporary data frames to update value functions
tmp = pt_sim.loc[[x in infeas_ids for x in pt_sim.id], ['id', 'V_aha']].groupby('id').diff(periods=-1).rename(columns={'V_aha': 'V_aha_diff'}).reset_index(drop=True) # estimating expected immediate rewards from value functions
tmp1 = pd.concat([pd.Series(np.repeat(infeas_ids, 10), name='id'), pd.Series(np.tile(np.arange(10), infeas_ids.shape), name='year'), tmp], axis=1) # incorporating ids and year
tmp1['key'] = (tmp1['id'].astype(str) + tmp1['year'].astype(str)).astype(int) # creating unique key on temporary data frame
pt_sim['key'] = (pt_sim['id'].astype(str) + pt_sim['year'].astype(str)).astype(int) # creating unique key on main data frame
tmp2 = tmp1.merge(pt_sim[['key', 'ind']], on='key') # incorporating index of infeasibility

## Updating value functions, events, and policy of AHA's guidelines in main data frame
pt_sim.update(pt_sim.loc[pt_sim.ind==1, ['V_best_range', 'evt_best_range', 'pi_best_range', 'meds_best']].
              rename(columns={'V_best_range': 'V_aha', 'evt_best_range': 'evt_aha', 'pi_best_range': 'pi_aha', 'meds_best': 'meds_aha'})) # updating data frame
pt_sim.update(pt_sim.loc[np.where(pt_sim.V_best_range<pt_sim.V_aha)[0], ['V_best_range', 'evt_best_range', 'pi_best_range', 'meds_best']].
              rename(columns={'V_best_range': 'V_aha', 'evt_best_range': 'evt_aha', 'pi_best_range': 'pi_aha', 'meds_best': 'meds_aha'})) # making sure AHA's guideline is not better than the optimal policy for any patient

## Incorporating value functions from optimal (feasible) treatment
tmp2.loc[tmp2.ind==1, 'V_aha_diff'] = pt_sim.loc[pt_sim.ind==1, ['V_aha']].rename(columns={'V_aha': 'V_aha_diff'}).set_index(tmp2[tmp2.ind==1].index) # replacing expected immediate rewards for value functions when possible

## Calculating new value functions (from future value functions and immediate rewards
tmp2.loc[tmp2[tmp2.ind==1].groupby('id')['ind'].head(1).index, 'ind'] -= 1 # Identifying first row replaced by value function
tmp2[['ind2', 'cum_sum']] = tmp2.groupby('id', sort=False)[['ind', 'V_aha_diff']].apply(lambda x: x[::-1]).reset_index(drop=True) # creating new columns by reversing index and value functions
tmp2['cum_sum'] = tmp2[tmp2.ind2==0].groupby('id')['cum_sum'].cumsum() # calculating cumulative sum
tmp2[['ind2', 'cum_sum']] = tmp2.groupby('id', sort=False)[['ind2', 'cum_sum']].apply(lambda x: x[::-1]).reset_index(drop=True) # reversing the columns back to original order
tmp2.loc[tmp2.ind==0, 'V_aha_diff'] = tmp2.loc[tmp2.ind==0, 'cum_sum'] # replacing immediate rewards by estimated value functions
tmp2.drop(['ind', 'ind2', 'cum_sum'], axis=1, inplace=True) # deleting unnecesary columns
tmp2.rename(columns={'V_aha_diff': 'V_aha'}, inplace=True) # renaming columns

## Replacing value functions in main data frame
tmp3 = tmp2.set_index(pt_sim[[x in tmp2.key.to_numpy() for x in pt_sim.key]].index)
pt_sim.loc[[x in tmp2.key.to_numpy() for x in pt_sim.key], 'V_aha'] = tmp3['V_aha']

# Adjusting values by sampling weights
pt_sim = pt_sim.join(ptdata.wt)
pt_sim.V_notrt = pt_sim.V_notrt*pt_sim.wt/1e06
pt_sim.V_aha = pt_sim.V_aha*pt_sim.wt/1e06
pt_sim.V_best_range = pt_sim.V_best_range*pt_sim.wt/1e06
pt_sim.V_med_range = pt_sim.V_med_range*pt_sim.wt/1e06
pt_sim.V_fewest_range = pt_sim.V_fewest_range*pt_sim.wt/1e06

# Making plot of expected total life years saved (over time per risk group)
# Data frame of expected total life years saved (compared to no treatment) per year
ly_df = pt_sim.loc[:, ['year', 'race', 'V_best_range', 'V_med_range', 'V_fewest_range', 'V_aha', 'V_notrt']].groupby(['year', 'race']).sum().reset_index(drop=False, inplace=False) #, 'race', 'sex'
ly_df.year += 1
ly_df.V_best_range = ly_df.V_best_range - ly_df.V_notrt # optimal policy not included in SMDM poster
ly_df.V_med_range = ly_df.V_med_range - ly_df.V_notrt
ly_df.V_fewest_range = ly_df.V_fewest_range - ly_df.V_notrt
ly_df.V_aha = ly_df.V_aha - ly_df.V_notrt
ly_df = ly_df.drop(['V_notrt'], axis=1)

## Preparing data for plot
ly_df = ly_df.rename(columns={'V_best_range': 'Best in Set',
                                    'V_med_range': 'Median in Set',
                                    'V_fewest_range': 'Fewest in Set',
                                    'V_aha': 'Clinical Guidelines'})

#drops all categories besides best?
ly_df = ly_df.melt(id_vars=['year', 'race'], var_name='policy', value_name='ly') #, 'race'

order = ['Best in Set', 'Median in Set', 'Fewest in Set', 'Clinical Guidelines'] # order for plots #
ly_df['policy'] = ly_df['policy'].astype('category') # converting scenario to category
ly_df.policy.cat.set_categories(order, inplace=True) # adding sorted categories
ly_df = ly_df.sort_values(['policy', 'year']) # sorting dataframe based on selected columns
# ly_df = ly_df[ly_df.bp_cat!='Normal'] # removing normal BP group

## Making plot
os.chdir(fig_dir)
# plot_qalys_saved(ly_df, min_age=40, group='race') # for paper
plot_ly_saved_race(ly_df) # for talk/simplified paper plot

# -------------- Medication ranges at a year ------------------
# # Evaluating events averted and time to events per policy
ptdata_list = [ptdata]
ptdata = pd.concat([pd.Series(np.tile(np.arange(1, 11), ptdata_list[0].id.unique().shape[0]), name='year'), ptdata_list[0]], axis=1)

# pt_evt = pt_evt[0] # Extracting base case only

pt_sim['meds_aha'] = np.select([pt_sim.pi_aha==x for x in range(len(meds))], meds)
pt_sim['meds_best'] = np.select([pt_sim.pi_best_range==x for x in range(len(meds))], meds)
pt_sim['meds_med'] = np.select([pt_sim.pi_med_range==x for x in range(len(meds))], meds)
pt_sim['meds_few'] = np.select([pt_sim.pi_fewest_range==x for x in range(len(meds))], meds)

pt_evt = pt_sim.loc[:, 'evt_notrt':'meds_few']
pt_evt = pt_evt.join(ptdata.wt)
pt_evt = pt_evt.join(ptdata.id)
pt_evt = pt_evt.join(ptdata.year)

# Adjusting values by sampling weights
pt_evt.evt_notrt = pt_evt.evt_notrt*pt_evt.wt/1e03
pt_evt.evt_aha = pt_evt.evt_aha*pt_evt.wt/1e03
pt_evt.evt_best_range = pt_evt.evt_best_range*pt_evt.wt/1e03
pt_evt.evt_med_range = pt_evt.evt_med_range*pt_evt.wt/1e03
pt_evt.evt_fewest_range = pt_evt.evt_fewest_range*pt_evt.wt/1e03

ptdata1 = ptdata_list[0].groupby('id').first().reset_index(drop=False, inplace=False)

bp_cats = [(ptdata1.sbp < 120) & (ptdata1.dbp < 80),
            (ptdata1.sbp >= 120) & (ptdata1.dbp < 80) & (ptdata1.sbp < 130),
            ((ptdata1.sbp >= 130) | (ptdata1.dbp >= 80)) & ((ptdata1.sbp < 140) & (ptdata1.dbp < 90)),
            (ptdata1.sbp >= 140) | (ptdata1.dbp >= 90)]
bp_cat_labels = ['Normal', 'Elevated', 'Stage 1', 'Stage 2']
ptdata1['bp_cat'] = np.select(bp_cats, bp_cat_labels)

## Incorporating demographic and grouping information
pt_evt = pd.merge(pt_evt, ptdata1[['id', 'age', 'sex', 'race', 'smk', 'diab', 'bp_cat']], on='id')

## Creating summary dataframe
events_df = pt_evt.loc[:, ['year', 'bp_cat', 'race', 'sex', 'evt_aha', 'evt_best_range', 'evt_fewest_range', 'evt_med_range', 'evt_notrt']].\
    groupby(['year']).sum().reset_index(drop=False, inplace=False)
events_df.year += 1
events_df.evt_best_range = events_df.evt_notrt - events_df.evt_best_range
events_df.evt_med_range = events_df.evt_notrt - events_df.evt_med_range
events_df.evt_fewest_range = events_df.evt_notrt - events_df.evt_fewest_range
events_df.evt_aha = events_df.evt_notrt - events_df.evt_aha
events_df = events_df.drop(['evt_notrt'], axis=1)

time_events_df = pt_evt.loc[:, ['year', 'bp_cat', 'race', 'sex', 'time_evt_aha', 'time_evt_best_range', 'time_evt_fewest_range',
                                'time_evt_med_range', 'time_evt_notrt']].\
    groupby(['year']).mean().reset_index(drop=False, inplace=False) #, 'bp_cat', 'race', 'sex'
time_events_df.year += 1
time_events_df.time_evt_best_range = time_events_df.time_evt_best_range - time_events_df.time_evt_notrt
time_events_df.time_evt_med_range = time_events_df.time_evt_med_range - time_events_df.time_evt_notrt
time_events_df.time_evt_fewest_range = time_events_df.time_evt_fewest_range - time_events_df.time_evt_notrt
time_events_df.time_evt_aha = time_events_df.time_evt_aha - time_events_df.time_evt_notrt
time_events_df = time_events_df.drop(['time_evt_notrt'], axis=1)

# Making plot of treatment by risk group at the first and last year of the simulation
## Extracting first and last year results
pt_sim1 = pt_evt.groupby(['id']).first().reset_index(drop=False, inplace=False) # first year of result data (only base case scenario)
med_ranges = pd.Series([x[0].dropna().quantile(interpolation='higher') for x in medication_range], name='Median in Set') # extracting the median of the ranges at year 1
min_ranges = pd.Series([x[0].dropna().min() for x in medication_range], name='Fewest in Set') # extracting the lower bound of the ranges at year 1
pt_sim10 = pt_evt.groupby(['id']).last().reset_index(drop=False, inplace=False) # last year of result data (only base case scenario)
med_ranges10 = pd.Series([x[9].dropna().quantile(interpolation='higher') for x in medication_range], name='Median in Set') # extracting the median of the ranges at year 10
min_ranges10 = pd.Series([x[9].dropna().min() for x in medication_range], name='Fewest in Set') # extracting the lower bound of the ranges at year 10

## Data frame of number of medications
trt_df1 = pd.concat([pt_sim1[['year', 'bp_cat', 'race', 'meds_best']], med_ranges, min_ranges, pt_sim1[['meds_aha']]], axis=1)
trt_df10 = pd.concat([pt_sim10[['year', 'bp_cat', 'race', 'meds_best']], med_ranges10, min_ranges10, pt_sim10[['meds_aha']]], axis=1)
trt_df = pd.concat([trt_df1, trt_df10], axis=0)
del pt_sim1, pt_sim10, min_ranges, med_ranges, min_ranges10, med_ranges10, trt_df1, trt_df10; gc.collect()

## Renaming columns
trt_df = trt_df.rename(columns={'meds_best': 'Best in Set',
                                'meds_aha': 'Clinical Guidelines'})

### Preparing data for plot
trt_df = trt_df.melt(id_vars=['year', 'bp_cat', 'race'], var_name='policy', value_name='meds')
# trt_df.year += 1

# Sorting dataframe according to BP categories
trt_df = trt_df[trt_df.bp_cat!='Normal'] # removing normal BP group (for talks only)
trt_df['bp_cat'] = trt_df['bp_cat'].astype('category') # converting scenario to category
trt_df.bp_cat.cat.set_categories(bp_cat_labels, inplace=True) # adding sorted categories
trt_df = trt_df.sort_values(['bp_cat', 'year']) # sorting dataframe based on selected columns

# # Sorting data frame according to policies (for talks only)
# order = ['Clinical Guidelines', 'Median in Range', 'Optimal Policy', 'Fewest in Range', 'Best in Range'] # order for plots #
# trt_df['policy'] = trt_df['policy'].astype('category') # converting scenario to category
# trt_df.policy.cat.set_categories(order, inplace=True) # adding sorted categories
# trt_df = trt_df.sort_values(['policy', 'year']) # sorting dataframe based on selected columns

## Making plot
os.chdir(fig_dir)
plot_simplified_trt_dist(trt_df)