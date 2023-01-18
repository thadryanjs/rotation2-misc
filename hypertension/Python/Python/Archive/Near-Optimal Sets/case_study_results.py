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
from plotting_functions import plot_range_actions#, plot_dist_qvalues, plf_cov_batch, plot_demo_bp, plot_qalys_saved,  \
# plot_trt_dist, plot_prop_covered, plot_ranges_len_meds, plot_prop_mis  # plot generation functions

# Importing parameters from main module
from hypertension_treatment_sbbi_sbmcc import results_dir, fig_dir, rev_arisk, ptdata_list, years, meds, \
    future_action, disut_list, risk_mult_list, trt_ben_mult_list, \
    known_dist, age_list, reps, ctrl_reps, sample_df

# # ---------------------------
# # Number of batches analysis
# # ---------------------------
#
# # Loading results
# os.chdir(results_dir)
# with open('Number of batches analysis for 1113 patients.pkl','rb') as f:
#     ci_width_list = pk.load(f)
#
# # Calculating the max width of confidence intervals by number of batches across all patients
# ci_width_max = [max([ci_width_list[x][y] for x in range(len(ci_width_list))]) for y in range(len(ci_width_list[0]))]
#
# Fixing wrong multiplier in simulation (only run if dividing standard error by \sqrt(MK) instead of \sqrt(K)
# ci_width_max = [x/np.sqrt(m) for m, x in enumerate(ci_width_max, 1)]
#
# # Plotting convergence of maximum width across patients
# os.chdir(fig_dir)
# ci_width = ci_width_max[:1000]
# plf_cov_batch(ci_width, plot_batches=500, selected_batches=300)
#
# # General performance for 400 batches
# pd.DataFrame([ci_width_list[x][399] for x in range(len(ci_width_list))]).describe()
#
# ---------------------------
# Population-level analysis
# ---------------------------

# Calculating BP summary statistics by demographics
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
plot_demo_bp(demo)

# Loading results (if previously combined)
os.chdir(results_dir)
with open('Results for patients 0 to 1112 until patient 1112 using adaptive observations and 301 batches.pkl',
          'rb') as f:
    [_, _, _, _, medication_range, pt_sim] = pk.load(f)

# # # Loading results (if not previously combined)
# ## First chunk of patients (if not previously combined)
# os.chdir(results_dir)
# with open('Results for patients 0 to 556 until patient 556 using adaptive observations and 301 batches.pkl',
#           'rb') as f:
#     [#risk1, risk10, transitions,
#      bi_Q, sb_Q, sb_sigma2, action_range, medication_range,
#      pt_sim] = pk.load(f)
#
# ## Second chunk of the patients
# with open('Results for patients 557 to 1112 until patient 1112 using adaptive observations and 301 batches.pkl',
#           'rb') as f:
#     [#risk1, risk10, transitions,
#      bi_Q1, sb_Q1, sb_sigma21, action_range1, medication_range1,
#      pt_sim1] = pk.load(f)
#
# # Combining results
# bi_Q =[item + bi_Q1[i] for i, item in enumerate(bi_Q)]
# sb_Q =[item + sb_Q1[i] for i, item in enumerate(sb_Q)]
# sb_sigma2 =[item + sb_sigma21[i] for i, item in enumerate(sb_sigma2)]
# action_range =[item + action_range1[i] for i, item in enumerate(action_range)]
# medication_range =[item + medication_range1[i] for i, item in enumerate(medication_range)]
# pt_sim =[pd.concat([item, pt_sim1[i]], axis=0, ignore_index=True) for i, item in enumerate(pt_sim)]
# del bi_Q1, sb_Q1, sb_sigma21, action_range1, medication_range1, pt_sim1
#
# # # Making sure that NAs in action ranges are NAs in medication ranges (run only if not done in main analysis)
# # medication_range = [[pd.DataFrame(np.select([action_range[sc][i].isna()] + [action_range[sc][i] == x
# #                                                                             for x in range(meds.shape[0])], np.append(np.nan, meds)))
# #                      for i in range(len(action_range[0]))] for sc in range(len(action_range))]
#
# # Saving combined results
# os.chdir(results_dir)
# with open('Results for patients 0 to 1112 until patient 1112 using adaptive observations and 301 batches.pkl', 'wb') as f:  # full simulation
#     pk.dump([  # risk1, risk10, transitions,
#         bi_Q, sb_Q, sb_sigma2, action_range, medication_range,
#         pt_sim], f, protocol=3)
#
# # Saving small subset for initial computations
# bi_Q =[item[:100] for item in bi_Q]
# sb_Q =[item[:100] for item in sb_Q]
# sb_sigma2 =[item[:100] for item in sb_sigma2]
# action_range =[item[:100] for item in action_range]
# medication_range =[item[:100] for item in medication_range]
# pt_sim =[item.loc[item['id'] < 100] for item in pt_sim]
# with open('Results for patients 0 to 1112 until patient 100 using adaptive observations and 301 batches.pkl', 'wb') as f:  # full simulation
#     pk.dump([  # risk1, risk10, transitions,
#         bi_Q, sb_Q, sb_sigma2, action_range, medication_range,
#         pt_sim], f, protocol=3)

# Extracting base case results (discarding the rest of the results)
medication_range = medication_range[0]
pt_sim = pt_sim[0]
del _; gc.collect()

# Incorporating demographic and grouping information
pt_sim = pd.merge(pt_sim, ptdata1[['id', 'age', 'sex', 'race', 'smk', 'diab', 'bp_cat']], on='id')

# Converting actions to number of medications
pt_sim['meds_aha'] = np.select([pt_sim.pi_aha==x for x in range(len(meds))], meds)
pt_sim['meds_opt'] = np.select([pt_sim.pi_opt==x for x in range(len(meds))], meds)
pt_sim['meds_apr'] = np.select([pt_sim.pi_apr==x for x in range(len(meds))], meds)

# Adjusting AHA's guidelines for feasibility (for some reason the aha_guideline function is allowing over-treatment past the feasibility condition)
## Identifying patients with infeasible treatment
pt_sim['meds_largest'] = np.vstack([x.max() for x in medication_range]).flatten()
pt_sim['ind'] = np.where((pt_sim.meds_largest<pt_sim.meds_aha) & # over-treatment condition
                         ((pt_sim.V_opt-pt_sim.V_aha)<=(pt_sim.V_opt-pt_sim.V_fewest_range)), # near-optimal condition (it would have been part of the ranges if it wasn't for the feasibility constraint)
                          1, 0) # indicator of infeasibility
infeas_ids = pt_sim.loc[pt_sim.ind==1, 'id'].unique() # ids of patients with infeasible treatments

## Temporary data frames to update value functions
tmp = pt_sim.loc[[x in infeas_ids for x in pt_sim.id], ['id', 'V_aha']].groupby('id').diff(periods=-1).rename(columns={'V_aha': 'V_aha_diff'}).reset_index(drop=True) # estimating expected immediate rewards from value functions
tmp1 = pd.concat([pd.Series(np.repeat(infeas_ids, 10), name='id'), pd.Series(np.tile(np.arange(10), infeas_ids.shape), name='year'), tmp], axis=1) # incorporating ids and year
tmp1['key'] = (tmp1['id'].astype(str) + tmp1['year'].astype(str)).astype(int) # creating unique key on temporary data frame
pt_sim['key'] = (pt_sim['id'].astype(str) + pt_sim['year'].astype(str)).astype(int) # creating unique key on main data frame
tmp2 = tmp1.merge(pt_sim[['key', 'ind']], on='key') # incorporating index of infeasibility

## Updating value functions, events, and policy of AHA's guidelines in main data frame
pt_sim.update(pt_sim.loc[pt_sim.ind==1, ['V_opt', 'evt_opt', 'pi_opt', 'meds_opt']].
              rename(columns={'V_opt': 'V_aha', 'evt_opt': 'evt_aha', 'pi_opt': 'pi_aha', 'meds_opt': 'meds_aha'})) # updating data frame
pt_sim.update(pt_sim.loc[np.where(pt_sim.V_opt<pt_sim.V_aha)[0], ['V_opt', 'evt_opt', 'pi_opt', 'meds_opt']].
              rename(columns={'V_opt': 'V_aha', 'evt_opt': 'evt_aha', 'pi_opt': 'pi_aha', 'meds_opt': 'meds_aha'})) # making sure AHA's guideline is not better than the optimal policy for any patient

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
pt_sim.V_notrt = pt_sim.V_notrt*pt_sim.wt/1e06
pt_sim.V_aha = pt_sim.V_aha*pt_sim.wt/1e06
pt_sim.V_apr = pt_sim.V_apr*pt_sim.wt/1e06
pt_sim.V_med_range = pt_sim.V_med_range*pt_sim.wt/1e06
pt_sim.V_fewest_range = pt_sim.V_fewest_range*pt_sim.wt/1e06
pt_sim.V_opt = pt_sim.V_opt*pt_sim.wt/1e06

# Making plot of expected total life years saved (over time per risk group)
# Data frame of expected total life years saved (compared to no treatment) per year
ly_df = pt_sim.loc[:, ['year', 'bp_cat', 'V_opt', 'V_apr', 'V_med_range', 'V_fewest_range', 'V_aha', 'V_notrt']].groupby(['year', 'bp_cat']).sum().reset_index(drop=False, inplace=False) #, 'race', 'sex'
ly_df.year += 1
ly_df.V_opt = ly_df.V_opt - ly_df.V_notrt # optimal policy not included in SMDM poster
ly_df.V_apr = ly_df.V_apr - ly_df.V_notrt
ly_df.V_med_range = ly_df.V_med_range - ly_df.V_notrt
ly_df.V_fewest_range = ly_df.V_fewest_range - ly_df.V_notrt
ly_df.V_aha = ly_df.V_aha - ly_df.V_notrt
ly_df = ly_df.drop(['V_notrt'], axis=1)

## Preparing data for plot
ly_df = ly_df.rename(columns={'V_opt': 'Optimal Policy',
                                    'V_apr': 'Best in Range',
                                    'V_med_range': 'Median in Range',
                                    'V_fewest_range': 'Fewest in Range',
                                    'V_aha': 'Clinical Guidelines'})

ly_df = ly_df.melt(id_vars=['year', 'bp_cat'], var_name='policy', value_name='ly') #, 'race'

order = ['Clinical Guidelines', 'Median in Range', 'Optimal Policy', 'Fewest in Range', 'Best in Range'] # order for plots #
ly_df['policy'] = ly_df['policy'].astype('category') # converting scenario to category
ly_df.policy.cat.set_categories(order, inplace=True) # adding sorted categories
ly_df = ly_df.sort_values(['policy', 'year']) # sorting dataframe based on selected columns
ly_df = ly_df[ly_df.bp_cat!='Normal'] # removing normal BP group

## Making plot
os.chdir(fig_dir)
# plot_qalys_saved(ly_df, min_age=40, group='race') # for paper
plot_ly_saved(ly_df) # for talk/simplified paper plot

# # Evaluating events averted and time to events per policy
# ## Loading results
# os.chdir(results_dir)
# with open('Event results for patients 0 to 1112 until patient 1112 using adaptive observations and 301 batches.pkl',
#           'rb') as f:
#     [pt_evt] = pk.load(f)
# pt_evt = pt_evt[0] # Extracting base case only
#
# # Adjusting values by sampling weights
# pt_evt.evt_notrt = pt_evt.evt_notrt*pt_evt.wt/1e03
# pt_evt.evt_aha = pt_evt.evt_aha*pt_evt.wt/1e03
# pt_evt.evt_apr = pt_evt.evt_apr*pt_evt.wt/1e03
# pt_evt.evt_med_range = pt_evt.evt_med_range*pt_evt.wt/1e03
# pt_evt.evt_fewest_range = pt_evt.evt_fewest_range*pt_evt.wt/1e03
# pt_evt.evt_opt = pt_evt.evt_opt*pt_evt.wt/1e03
#
# ## Incorporating demographic and grouping information
# pt_evt = pd.merge(pt_evt, ptdata1[['id', 'age', 'sex', 'race', 'smk', 'diab', 'bp_cat']], on='id')
#
# ## Creating summary dataframe
# events_df = pt_evt.loc[:, ['year', 'bp_cat', 'race', 'sex', 'evt_aha', 'evt_apr', 'evt_fewest_range', 'evt_med_range', 'evt_notrt', 'evt_opt']].\
#     groupby(['year']).sum().reset_index(drop=False, inplace=False)
# events_df.year += 1
# events_df.evt_opt = events_df.evt_notrt - events_df.evt_opt
# events_df.evt_apr = events_df.evt_notrt - events_df.evt_apr
# events_df.evt_med_range = events_df.evt_notrt - events_df.evt_med_range
# events_df.evt_fewest_range = events_df.evt_notrt - events_df.evt_fewest_range
# events_df.evt_aha = events_df.evt_notrt - events_df.evt_aha
# events_df = events_df.drop(['evt_notrt'], axis=1)
#
# time_events_df = pt_evt.loc[:, ['year', 'bp_cat', 'race', 'sex', 'time_evt_aha', 'time_evt_apr', 'time_evt_fewest_range',
#                                 'time_evt_med_range', 'time_evt_notrt', 'time_evt_opt']].\
#     groupby(['year']).mean().reset_index(drop=False, inplace=False) #, 'bp_cat', 'race', 'sex'
# time_events_df.year += 1
# time_events_df.time_evt_opt = time_events_df.time_evt_opt - time_events_df.time_evt_notrt
# time_events_df.time_evt_apr = time_events_df.time_evt_apr - time_events_df.time_evt_notrt
# time_events_df.time_evt_med_range = time_events_df.time_evt_med_range - time_events_df.time_evt_notrt
# time_events_df.time_evt_fewest_range = time_events_df.time_evt_fewest_range - time_events_df.time_evt_notrt
# time_events_df.time_evt_aha = time_events_df.time_evt_aha - time_events_df.time_evt_notrt
# time_events_df = time_events_df.drop(['time_evt_notrt'], axis=1)
#
# # Making plot of treatment by risk group at the first and last year of the simulation
# ## Extracting first and last year results
# pt_sim1 = pt_sim.groupby(['id']).first().reset_index(drop=False, inplace=False) # first year of result data (only base case scenario)
# med_ranges = pd.Series([x[0].dropna().quantile(interpolation='higher') for x in medication_range], name='Median in Range') # extracting the median of the ranges at year 1
# min_ranges = pd.Series([x[0].dropna().min() for x in medication_range], name='Fewest in Range') # extracting the lower bound of the ranges at year 1
# pt_sim10 = pt_sim.groupby(['id']).last().reset_index(drop=False, inplace=False) # last year of result data (only base case scenario)
# med_ranges10 = pd.Series([x[9].dropna().quantile(interpolation='higher') for x in medication_range], name='Median in Range') # extracting the median of the ranges at year 10
# min_ranges10 = pd.Series([x[9].dropna().min() for x in medication_range], name='Fewest in Range') # extracting the lower bound of the ranges at year 10
#
# ## Data frame of number of medications
# trt_df1 = pd.concat([pt_sim1[['year', 'bp_cat', 'race', 'meds_opt', 'meds_apr']], med_ranges, min_ranges, pt_sim1[['meds_aha']]], axis=1)
# trt_df10 = pd.concat([pt_sim10[['year', 'bp_cat', 'race', 'meds_opt', 'meds_apr']], med_ranges10, min_ranges10, pt_sim10[['meds_aha']]], axis=1)
# trt_df = pd.concat([trt_df1, trt_df10], axis=0)
# del pt_sim1, pt_sim10, min_ranges, med_ranges, min_ranges10, med_ranges10, trt_df1, trt_df10; gc.collect()
#
# ## Renaming columns
# trt_df = trt_df.rename(columns={'meds_opt': "Optimal Policy", 'meds_apr': 'Best in Range',
#                                 'meds_aha': 'Clinical Guidelines'})
#
# ### Preparing data for plot
# trt_df = trt_df.melt(id_vars=['year', 'bp_cat', 'race'], var_name='policy', value_name='meds')
# trt_df.year += 1
#
# # Sorting dataframe according to BP categories
# trt_df['bp_cat'] = trt_df['bp_cat'].astype('category') # converting scenario to category
# trt_df.bp_cat.cat.set_categories(bp_cat_labels, inplace=True) # adding sorted categories
# trt_df = trt_df.sort_values(['bp_cat', 'year']) # sorting dataframe based on selected columns
#
# # # Sorting data frame according to policies (for talks only)
# # trt_df = trt_df[trt_df.bp_cat!='Normal'] # removing normal BP group (for talks only)
# # order = ['Clinical Guidelines', 'Median in Range', 'Optimal Policy', 'Fewest in Range', 'Best in Range'] # order for plots #
# # trt_df['policy'] = trt_df['policy'].astype('category') # converting scenario to category
# # trt_df.policy.cat.set_categories(order, inplace=True) # adding sorted categories
# # trt_df = trt_df.sort_values(['policy', 'year']) # sorting dataframe based on selected columns
#
# ## Making plot
# os.chdir(fig_dir)
# plot_trt_dist(trt_df)
#
#
# # Calculating proportion of people that receive no treatment with normal BP over time
# for y in range(years):
#     tmp = pt_sim.groupby(['id']).nth(y).reset_index(drop=False, inplace=False)
#     print(tmp['wt'][np.where((tmp.pi_opt!=0) & (tmp.bp_cat=='Normal'))[0]].sum()/tmp.wt.sum()*100)
#
#
# # Calculating average treatment across demographics
# tmp = pt_sim[['year', 'bp_cat', 'sex', 'race', 'pi_opt', 'pi_fewest_range', 'pi_aha']].groupby(['year', 'bp_cat', 'sex', 'race']).describe().reset_index(drop=False, inplace=False)
# tmp.columns = pd.Index([e[0] + e[1] for e in tmp.columns.tolist()]) # changing column names from multiindex to single index
# tmp.year += 1
# t1 = tmp[(tmp.race==0) & (tmp.sex==0) & (tmp.bp_cat=='Elevated')][['pi_opt', 'pi_fewest_range', 'pi_aha']].astype(int)
# t2 = tmp[(tmp.race==0) & (tmp.sex==1) & (tmp.bp_cat=='Elevated')][['pi_opt', 'pi_fewest_range', 'pi_aha']].astype(int)
# t3 = tmp[(tmp.race==1) & (tmp.sex==0) & (tmp.bp_cat=='Elevated')][['pi_opt', 'pi_fewest_range', 'pi_aha']].astype(int)
# t4 = tmp[(tmp.race==1) & (tmp.sex==1) & (tmp.bp_cat=='Elevated')][['pi_opt', 'pi_fewest_range', 'pi_aha']].astype(int)
#
# # Calculating summary statistics for range width across demographics
# width_ranges = pd.concat([x.nunique() for x in medication_range], axis=0).reset_index(drop=True, inplace=False) # width of ranges
# width_ranges.rename('width', inplace=True) # renaming columns
# width_ranges = pd.concat([pt_sim.loc[:, ['year', 'bp_cat', 'sex', 'race']], width_ranges], axis=1).\
#     groupby(['year', 'bp_cat', 'sex', 'race']).describe().reset_index(drop=False, inplace=False) # adding patient information
# width_ranges.columns = pd.Index([e[0] + e[1] for e in width_ranges.columns.tolist()]) # changing column names from multiindex to single index
# width_ranges.year += 1
# width_ranges.drop(['widthcount', 'widthstd'], axis=1, inplace=True) # deleting unnecesary columns
#
# ## Examamining demographic groups
# t1 = width_ranges[(width_ranges.race==0) & (width_ranges.sex==0) & (width_ranges.bp_cat=='Elevated')][['widthmean', 'widthmin', 'width25%', 'width50%', 'width75%', 'widthmax']].astype(int)
# t2 = width_ranges[(width_ranges.race==0) & (width_ranges.sex==1) & (width_ranges.bp_cat=='Elevated')][['widthmean', 'widthmin', 'width25%', 'width50%', 'width75%', 'widthmax']].astype(int)
# t3 = width_ranges[(width_ranges.race==1) & (width_ranges.sex==0) & (width_ranges.bp_cat=='Elevated')][['widthmean', 'widthmin', 'width25%', 'width50%', 'width75%', 'widthmax']].astype(int)
# t4 = width_ranges[(width_ranges.race==1) & (width_ranges.sex==1) & (width_ranges.bp_cat=='Elevated')][['widthmean', 'widthmin', 'width25%', 'width50%', 'width75%', 'widthmax']].astype(int)
#
# Making plot of proportion of patient-years of policies covered in range
## Indicators of whether or not action was contained in range
# ind_opt = pd.Series(np.array([pt_sim.meds_opt[pt_sim.id==k].reset_index(drop=True)[y].round(3) in x[y].dropna().to_numpy().round(3)
#                               for k, x in enumerate(medication_range) for y in range(years)]).astype(int), name='Optimal Policy')
# ind_apr = pd.Series(np.array([pt_sim.meds_apr[pt_sim.id==k].reset_index(drop=True)[y].round(3) in x[y].dropna().to_numpy().round(3)
#                               for k, x in enumerate(medication_range) for y in range(years)]).astype(int), name='Best in Range')
# ind_aha = pd.Series(np.array([pt_sim.meds_aha[pt_sim.id==k].reset_index(drop=True)[y].round(3) in x[y].dropna().to_numpy().round(3)
#                               for k, x in enumerate(medication_range) for y in range(years)]).astype(int), name='Clinical Practice')
# ptcount = pd.Series(np.repeat(1, ind_aha.shape), name="counter") # counter of total number of patients (for groups)
#
# ## Creating dataframe of proportion of patient-years covered in range
# prop_df = pd.concat([pt_sim.loc[:, ['year', 'bp_cat', 'sex', 'race']], ind_aha, ptcount], axis=1).\
#     groupby(['year', 'bp_cat', 'sex', 'race']).sum().reset_index(drop=False, inplace=False) # , 'sex', 'race' # ind_opt, ind_apr, #including only clinical practice as the rest of the proportions are always 1
# prop_df = prop_df[prop_df.bp_cat!='Normal'] # removing normal BP group
# # del ind_aha; gc.collect() #ind_opt, ind_apr,
#
# ## Calculating proportions
# # prop_df['Optimal Policy'] /= prop_df['counter'] # proportion is always 1
# # prop_df['Best in Range'] /= prop_df['counter'] # proportion is always 1
# prop_df['Clinical Practice'] /= prop_df['counter']
#
# ## Preparing data for plot
# prop_df.year += 1
# prop_df.drop('counter', axis=1, inplace=True) # deleting counter column
# prop_df = prop_df.melt(id_vars=['year', 'bp_cat', 'sex', 'race'], var_name='policy', value_name='prop_cv') #, 'sex', 'race'
# prop_df.drop('policy', axis=1, inplace=True) # deleting policy name column
#
# # Sorting dataframe according to BP categories
# prop_df['bp_cat'] = prop_df['bp_cat'].astype('category') # converting scenario to category
# prop_df.bp_cat.cat.set_categories(bp_cat_labels[1:], inplace=True) # adding sorted categories
# prop_df = prop_df.sort_values(['bp_cat', 'year']) # sorting dataframe based on selected columns
#
# ## Making plot
# os.chdir(fig_dir)
# plot_prop_covered(prop_df)
#
# # -------------------------
# # Sensitivity analyses
# # -------------------------
#
# # List of sensivity scenarios
# mis_est_list = ['Half Risk', 'Double Risk', '50% Nonadherence'] # considering three misestimation scenarios
# disut_list = [np.unique((np.array(x[1:])/np.array(disut_list[0][1:])).round(3))[0] for x in disut_list[1:]] # obtaining multipliers from disutility lists
# sens_list = [age_list, risk_mult_list[1:], trt_ben_mult_list[1:], disut_list, known_dist[1:], future_action[1:]] # combining scenarios in a single list (excluding misestimation scenarios)
# sens_len = [len(x) for x in sens_list] # number of scenarios per analysis
# tmp = iter(range(sum(sens_len))) # scenario ids
# sens_id = [list(islice(tmp, elem)) for elem in sens_len]; del tmp # dividing list of scenario ids in to sublists of apropiate lengths
# del future_action, disut_list, risk_mult_list, trt_ben_mult_list, known_dist, age_list, mis_est_list; gc.collect()
#
# # Loading results
# os.chdir(results_dir)
# with open('Results for patients 0 to 1112 until patient 1112 using adaptive observations and 301 batches.pkl',
#           'rb') as f:
#     [_, _, _, _, medication_range, pt_sim] = pk.load(f)
#
# del _; gc.collect()
#
# # Preparing data for plots/tables
# ## Merging list of dataframes to a single dataframe
# [df.insert(loc=0, column='scenario', value=np.repeat(sc, df.shape[0])) for sc, df in enumerate(pt_sim)] # adding scenario identifier
# pt_sim = pd.concat(pt_sim).reset_index(drop=True, inplace=False) # merging dataframes
#
# ## Modifying ids of 70-74 age group
# pt_sim.loc[pt_sim.scenario==1, 'id'] = pt_sim.loc[pt_sim.scenario==1, 'id'] + pt_sim.id.max() + 1
#
# ## Converting actions to number of medications
# pt_sim['meds_aha'] = np.select([pt_sim.pi_aha==x for x in range(len(meds))], meds)
# pt_sim['meds_opt'] = np.select([pt_sim.pi_opt==x for x in range(len(meds))], meds)
# pt_sim['meds_apr'] = np.select([pt_sim.pi_apr==x for x in range(len(meds))], meds)
#
# # Adjusting value functions by sampling weights
# pt_sim.V_notrt = pt_sim.V_notrt*pt_sim.wt/1e06
# pt_sim.V_aha = pt_sim.V_aha*pt_sim.wt/1e06
# pt_sim.V_apr = pt_sim.V_apr*pt_sim.wt/1e06
# pt_sim.V_med_range = pt_sim.V_med_range*pt_sim.wt/1e06
# pt_sim.V_fewest_range = pt_sim.V_fewest_range*pt_sim.wt/1e06
# pt_sim.V_opt = pt_sim.V_opt*pt_sim.wt/1e06
#
# ## Calculating statistics of ranges width
# ### Note: patients with a width of 0 in the range are records added for dimension purposes on the 70-74 age group (do not include these records in the results)
# medication_range[1] = [pd.DataFrame(np.full((1, years), np.nan)) if len(df.index) == 0 else df for df in medication_range[1]] # adding empty dataframes to identify records added for dimension purposes on the 70-74 age group
# sum_width_ranges = pd.concat([x.nunique() for y in medication_range for x in y], axis=0).reset_index(drop=False, inplace=False) # number of choices in ranges of every patient per scenario
# sum_width_ranges.insert(loc=0, column='id', value=np.tile(np.repeat(np.arange(len(medication_range[0])), years), sum(sens_len))) # adding patient id (added to debug)
# # sum_width_ranges.loc[sum_width_ranges.scenario==1, 'id'] = sum_width_ranges.loc[sum_width_ranges.scenario==1, 'id'] + sum_width_ranges.id.max() + 1 # modifying ids for 70-74 age group
# sum_width_ranges.insert(loc=0, column='scenario', value=np.repeat(np.arange(sum(sens_len)), len(medication_range[0])*years)) # adding scenario indicator
# sum_width_ranges.rename(columns={'index': 'year', 0: 'width'}, inplace=True) # renaming columns
# sum_width_ranges = sum_width_ranges[['scenario', 'year', 'width']].groupby(['scenario', 'year']).describe(percentiles=[0.05, 0.95]).reset_index(drop=False, inplace=False)
# tmp = pd.Index([e[0] + e[1] for e in sum_width_ranges.columns.tolist()]); sum_width_ranges.columns = tmp # changing column names from multiindex to single index
# sum_width_ranges = sum_width_ranges[['scenario', 'year', 'widthmean', 'width5%', 'width95%']] # selecting relevant columns
# sum_width_ranges = sum_width_ranges.rename(columns={'widthmean': 'mean', 'width5%': 'min', 'width95%': 'max'}) # renaming columns
# sum_width_ranges.year += 1
#
# ## Calculating statistics of number of medications in range
# sum_meds_ranges = pd.concat([pd.concat(x, axis=0).describe(percentiles=[0.05, 0.95]).iloc[[1, 4, 6]].T for x in medication_range], axis=0).reset_index(drop=False, inplace=False)
# sum_meds_ranges.insert(loc=0, column='scenario', value=np.repeat(np.arange(sum(sens_len)), years)) # adding scenario dientifyer
# sum_meds_ranges.rename(columns={'index': 'year', '5%': 'min', '95%': 'max'}, inplace=True) # renaming columns
# sum_meds_ranges.year += 1
#
# ## Medications in range per scenario at year 1
# sens_tbl_meds = sum_meds_ranges[sum_meds_ranges.year==1].reset_index(drop=False, inplace=False)
# sens_tbl_meds = sens_tbl_meds.rename(columns={'mean': 'mean_med', 'min': 'min_med',
#                                               'max': 'max_med'})
#
# ## Range width per scenario at year 1
# sens_tbl_width = sum_width_ranges[sum_width_ranges.year==1].reset_index(drop=False, inplace=False)
# sens_tbl_width = sens_tbl_width.rename(columns={'mean': 'mean_width', 'min': 'min_width',
#                                                 'max': 'max_width'})
#
# # Making table of QALYs saved, average (5%-95%) treatment in range, and average (5%-95%) range width
# ## QALYs saved (compared to no treatment) per scenario at year 1
# sens_tbl_qalys = pt_sim.loc[pt_sim.year==0, ['scenario', 'V_opt', 'V_apr', 'V_med_range', 'V_fewest_range', 'V_aha', 'V_notrt']].groupby(['scenario']).sum().reset_index(drop=False, inplace=False)
# sens_tbl_qalys.V_opt = sens_tbl_qalys.V_opt - sens_tbl_qalys.V_notrt
# sens_tbl_qalys.V_apr = sens_tbl_qalys.V_apr - sens_tbl_qalys.V_notrt
# sens_tbl_qalys.V_med_range = sens_tbl_qalys.V_med_range - sens_tbl_qalys.V_notrt
# sens_tbl_qalys.V_fewest_range = sens_tbl_qalys.V_fewest_range - sens_tbl_qalys.V_notrt
# sens_tbl_qalys.V_aha = sens_tbl_qalys.V_aha - sens_tbl_qalys.V_notrt
# sens_tbl_qalys = sens_tbl_qalys.drop(['V_notrt'], axis=1)
#
# ## Complete table
# sens_tbl = pd.concat([sens_tbl_qalys, sens_tbl_meds[['mean_med', 'min_med', 'max_med']],
#                       sens_tbl_width[['mean_width', 'min_width', 'max_width']]], axis=1)
# del sens_tbl_qalys, sens_tbl_meds, sens_tbl_width; gc.collect()
# # sens_tbl.to_clipboard() # copying table to clipboard
#
# # Making figure to compare range width and medications in ranges across sensitivity analysis scenarios (base case, normal Q-values, Median in Range future action, and Fewest in Range future action)
# ## Dataframe of width of ranges over time
# sens_df_width = sum_width_ranges[(sum_width_ranges.scenario==0) | (sum_width_ranges.scenario==9) | (sum_width_ranges.scenario==10) | (sum_width_ranges.scenario==11)].copy() # extracting base case, normal Q-values, Median in Range future action, and Fewest in Range future action scenarios
# order = np.sort(sens_df_width.scenario.unique())[::-1] # descending order of scenarios
# sens_df_width['scenario'] = sens_df_width['scenario'].astype('category') # converting scenario to category
# sens_df_width.scenario.cat.set_categories(order, inplace=True) # adding sorted categories
# sens_df_width = sens_df_width.sort_values(['scenario', 'year']) # sorting dataframe based on selected columns
#
# ## Dataframe of medications over time
# sens_df_meds = sum_meds_ranges[(sum_meds_ranges.scenario==0) | (sum_meds_ranges.scenario==9) | (sum_meds_ranges.scenario==10) | (sum_meds_ranges.scenario==11)].copy() # extracting base case, normal Q-values, Median in Range future action, and Fewest in Range future action scenarios
# order = np.sort(sens_df_meds.scenario.unique())[::-1] # descending order of scenarios
# sens_df_meds['scenario'] = sens_df_meds['scenario'].astype('category') # converting scenario to category
# sens_df_meds.scenario.cat.set_categories(order, inplace=True) # adding sorted categories
# sens_df_meds = sens_df_meds.sort_values(['scenario', 'year']) # sorting dataframe based on selected columns
# del order; gc.collect()
#
# ## Making plot
# os.chdir(fig_dir)
# plot_ranges_len_meds(sens_df_width, sens_df_meds)
#
# # Making plot of QALYs saved in the 70-74 age group
# ## Extracting first year data
# ptdata1 = ptdata_list[1].groupby('id').first().reset_index(drop=False, inplace=False) # extracting first year data in 70-74 age group
# ptdata1['id'] = ptdata1['id'] + ptdata_list[0].id.max() + 1 # modifying ids of 70-74 age group
#
# # Subsetting results
# pt_sim70 = pt_sim[pt_sim.scenario==1].reset_index(drop=True, inplace=False).copy()
#
# # Identifying BP categories
# bp_cats = [(ptdata1.sbp < 120) & (ptdata1.dbp < 80),
#             (ptdata1.sbp >= 120) & (ptdata1.dbp < 80) & (ptdata1.sbp < 130),
#             ((ptdata1.sbp >= 130) | (ptdata1.dbp >= 80)) & ((ptdata1.sbp < 140) | (ptdata1.dbp < 90)),
#             (ptdata1.sbp >= 140) | (ptdata1.dbp >= 90)]
# bp_cat_labels = ['Normal', 'Elevated', 'Stage 1', 'Stage 2']
# ptdata1['bp_cat'] = np.select(bp_cats, bp_cat_labels)
#
# # Incorporating demographic and grouping information
# pt_sim70 = pd.merge(pt_sim70, ptdata1[['id', 'age', 'sex', 'race', 'smk', 'diab', 'bp_cat']], on='id')
#
# # Making plot of expected total QALYs saved (over time per BP group)
# ## Data frame of expected total QALYs saved (compared to no treatment) per year
# qalys_df = pt_sim70.loc[:, ['year', 'bp_cat', 'sex', 'V_opt', 'V_apr', 'V_med_range', 'V_fewest_range', 'V_aha', 'V_notrt']].groupby(['year', 'bp_cat', 'sex']).sum().reset_index(drop=False, inplace=False) #'race', 'sex',
# qalys_df.year += 1
# qalys_df.V_opt = qalys_df.V_opt - qalys_df.V_notrt
# qalys_df.V_apr = qalys_df.V_apr - qalys_df.V_notrt
# qalys_df.V_med_range = qalys_df.V_med_range - qalys_df.V_notrt
# qalys_df.V_fewest_range = qalys_df.V_fewest_range - qalys_df.V_notrt
# qalys_df.V_aha = qalys_df.V_aha - qalys_df.V_notrt
# qalys_df = qalys_df.drop(['V_notrt'], axis=1)
#
# ## Preparing data for plot
# qalys_df = qalys_df.rename(columns={'V_opt': 'Optimal Policy',
#                                     'V_apr': 'Best in Range',
#                                     'V_med_range': 'Median in Range',
#                                     'V_fewest_range': 'Fewest in Range',
#                                     'V_aha': 'Clinical Guidelines'})
#
# qalys_df = qalys_df.melt(id_vars=['year', 'bp_cat', 'sex'], var_name='policy', value_name='qalys') #, 'sex' # , 'race'
#
# order = ['Clinical Guidelines', 'Median in Range', 'Optimal Policy', 'Fewest in Range', 'Best in Range'] # order for plots
# qalys_df['policy'] = qalys_df['policy'].astype('category') # converting scenario to category
# qalys_df.policy.cat.set_categories(order, inplace=True) # adding sorted categories
# qalys_df = qalys_df.sort_values(['policy', 'year']) # sorting dataframe based on selected columns
# qalys_df = qalys_df[qalys_df.bp_cat!='Normal'] # removing normal BP group
#
# ## Making plot
# os.chdir(fig_dir)
# plot_qalys_saved(qalys_df, min_age=70, group='sex')
#
# # Making plot of proportion of patient-years of optimal policies covered in range with misestimation of parameters
# medication_range = medication_range[0] # only considering base case range
#
# ## Indicators of whether or not an optimal action was contained in range
# ###Optimal treatment
# ind_opt_base_case = pd.Series(np.array([pt_sim.meds_opt[(pt_sim.id==k) & (pt_sim.scenario==0)].reset_index(drop=True)[y].round(3) in x[y].dropna().to_numpy().round(3)
#                                         for k, x in enumerate(medication_range) for y in range(years)]).astype(int), name='Base Case')
# ind_opt_half_risk = pd.Series(np.array([pt_sim.meds_opt[(pt_sim.id==k) & (pt_sim.scenario==2)].reset_index(drop=True)[y].round(3) in x[y].dropna().to_numpy().round(3)
#                                         for k, x in enumerate(medication_range) for y in range(years)]).astype(int), name='Half Event Rates')
# ind_opt_double_risk = pd.Series(np.array([pt_sim.meds_opt[(pt_sim.id==k) & (pt_sim.scenario==3)].reset_index(drop=True)[y].round(3) in x[y].dropna().to_numpy().round(3)
#                                           for k, x in enumerate(medication_range) for y in range(years)]).astype(int), name='Double Event Rates')
# ind_opt_half_trt_ben = pd.Series(np.array([pt_sim.meds_opt[(pt_sim.id==k) & (pt_sim.scenario==4)].reset_index(drop=True)[y].round(3) in x[y].dropna().to_numpy().round(3)
#                                            for k, x in enumerate(medication_range) for y in range(years)]).astype(int), name='Half Treatment Benefit')
#
# ### Counter of total number of patients (for groups)
# ptcount = pd.Series(np.repeat(1, ind_opt_base_case.shape), name="counter")
#
# ### Creating dataframe of proportion of patient-years covered in range
# prop_df = pd.concat([pt_sim.loc[pt_sim.scenario==0, ['year']],
#                      ind_opt_base_case, ind_opt_half_risk, ind_opt_double_risk, ind_opt_half_trt_ben,
#                      ptcount],
#                      axis=1).groupby(['year']).sum().reset_index(drop=False, inplace=False)
# # del ind_opt_half_risk, ind_opt_double_risk, ind_opt_half_trt_ben; gc.collect()
#
# ### Calculating proportions
# prop_df['Base Case'] /= prop_df['counter']
# prop_df['Half Event Rates'] /= prop_df['counter']
# prop_df['Double Event Rates'] /= prop_df['counter']
# prop_df['Half Treatment Benefit'] /= prop_df['counter']
#
# ### Melting dataframe
# prop_df.drop('counter', axis=1, inplace=True) # deleting counter column
# prop_df = prop_df.melt(id_vars=['year'], var_name='misestimation', value_name='prop')
# prop_df.insert(loc=1, column='policy', value=np.repeat('Optimal Treatment', prop_df.shape[0]))
#
# ### AHA's guidelines
# #### Adjusting AHA's guidelines for feasibility (for some reason the aha_guideline function is allowing over-treatment past the feasibility condition)
# pt_sim['meds_largest'] = np.tile(np.vstack([x.max() for x in medication_range]).flatten(), pt_sim.scenario.unique().shape)
# pt_sim['ind'] = np.where((pt_sim.meds_largest<pt_sim.meds_aha) & # over-treatment condition
#                          ((pt_sim.V_opt-pt_sim.V_aha)<=(pt_sim.V_opt-pt_sim.V_fewest_range)), # near-optimal condition (it would have been part of the ranges if it wasn't for the feasibility constraint)
#                           1, 0) # indicator of infeasibility
#
# #### Updating value functions, events, and policy of AHA's guidelines in main data frame
# pt_sim.update(pt_sim.loc[pt_sim.ind==1, ['V_opt', 'evt_opt', 'pi_opt', 'meds_opt']].
#               rename(columns={'V_opt': 'V_aha', 'evt_opt': 'evt_aha', 'pi_opt': 'pi_aha', 'meds_opt': 'meds_aha'})) # updating data frame
# pt_sim.update(pt_sim.loc[np.where(pt_sim.V_opt<pt_sim.V_aha)[0], ['V_opt', 'evt_opt', 'pi_opt', 'meds_opt']].
#               rename(columns={'V_opt': 'V_aha', 'evt_opt': 'evt_aha', 'pi_opt': 'pi_aha', 'meds_opt': 'meds_aha'})) # making sure AHA's guideline is not better than the optimal policy for any patient
#
# #### Counting total number of prescriptions covered in the ranges
# ind_aha_base_case = pd.Series(np.array([pt_sim.meds_aha[(pt_sim.id==k) & (pt_sim.scenario==0)].reset_index(drop=True)[y].round(3) in x[y].dropna().to_numpy().round(3)
#                                         for k, x in enumerate(medication_range) for y in range(years)]).astype(int), name='Base Case')
# ind_aha_half_risk = pd.Series(np.array([pt_sim.meds_aha[(pt_sim.id==k) & (pt_sim.scenario==2)].reset_index(drop=True)[y].round(3) in x[y].dropna().to_numpy().round(3)
#                                         for k, x in enumerate(medication_range) for y in range(years)]).astype(int), name='Half Event Rates')
# ind_aha_double_risk = pd.Series(np.array([pt_sim.meds_aha[(pt_sim.id==k) & (pt_sim.scenario==3)].reset_index(drop=True)[y].round(3) in x[y].dropna().to_numpy().round(3)
#                                           for k, x in enumerate(medication_range) for y in range(years)]).astype(int), name='Double Event Rates')
# ind_aha_half_trt_ben = pd.Series(np.array([pt_sim.meds_aha[(pt_sim.id==k) & (pt_sim.scenario==4)].reset_index(drop=True)[y].round(3) in x[y].dropna().to_numpy().round(3)
#                                            for k, x in enumerate(medication_range) for y in range(years)]).astype(int), name='Half Treatment Benefit')
#
# ### Counter of total number of patients (for groups)
# ptcount = pd.Series(np.repeat(1, ind_aha_base_case.shape), name="counter")
#
# ### Creating dataframe of proportion of patient-years covered in range
# prop_df1 = pd.concat([pt_sim.loc[pt_sim.scenario==0, ['year']],
#                      ind_aha_base_case, ind_aha_half_risk, ind_aha_double_risk, ind_aha_half_trt_ben,
#                      ptcount],
#                      axis=1).groupby(['year']).sum().reset_index(drop=False, inplace=False)
# # del medication_range, ind_aha_half_risk, ind_aha_double_risk, ind_aha_half_trt_ben; gc.collect()
#
# ### Calculating proportions
# prop_df1['Base Case'] /= prop_df1['counter']
# prop_df1['Half Event Rates'] /= prop_df1['counter']
# prop_df1['Double Event Rates'] /= prop_df1['counter']
# prop_df1['Half Treatment Benefit'] /= prop_df1['counter']
#
# ### Melting dataframe
# prop_df1.drop('counter', axis=1, inplace=True) # deleting counter column
# prop_df1 = prop_df1.melt(id_vars=['year'], var_name='misestimation', value_name='prop')
# prop_df1.insert(loc=1, column='policy', value=np.repeat('Clinical Guidelines', prop_df1.shape[0]))
#
# ## Preparing data for plot
# prop_df = pd.concat([prop_df1, prop_df], axis=0)
# prop_df.year += 1
# del prop_df1
#
# ## Making plot
# os.chdir(fig_dir)
# plot_prop_mis(prop_df)
#
# # -------------------------
# # Patient-level analysis
# # -------------------------
#
# # Loading results
# # ## Results for the whole population
# # os.chdir(results_dir)
# # with open('Results for patients 0 to 1112 until patient 1112 using adaptive observations and 301 batches.pkl',
# #           'rb') as f:
# #     [_, sb_Q, sb_sigma2, _, medication_range, pt_sim] = pk.load(f)
# # del _; gc.collect()
#
# ## Results for selected patient profiles
# os.chdir(results_dir)
# with open('Results for patient 783 in different clinical scenarios using adaptive observations and 301 batches.pkl',
#           'rb') as f:
#     [medication_range, pt_sim] = pk.load(f)
#
# ## Calculating 10-year risk (to better understand results)
# sample_df['risk10'] = sample_df.apply(lambda row: rev_arisk(0, row['sex'], row['black'], row['age'], row['sbp'], row['smk'],
#                                                             row['tc'], row['hdl'], row['diab'], 0, 10)
#                                                   + rev_arisk(1, row['sex'], row['black'], row['age'], row['sbp'], row['smk'],
#                                                               row['tc'], row['hdl'], row['diab'], 0, 10), axis=1)
#
# # Converting actions to number of medications
# pt_sim['Clinical Guidelines'] = np.select([pt_sim.pi_aha==x for x in range(len(meds))], meds)
# pt_sim['Optimal Policy'] = np.select([pt_sim.pi_opt==x for x in range(len(meds))], meds)
# pt_sim['Best in Range'] = np.select([pt_sim.pi_apr==x for x in range(len(meds))], meds)
#
# # # Removing patient profiles (see hypertension_treatment_sbbi_sbmcc.py file) # for paper
# # pt_sim = pt_sim[(pt_sim.id!=5) & (pt_sim.id!=6) & (pt_sim.id!=7) & (pt_sim.id!=8)].reset_index(drop=True, inplace=False)
# # pt_sim.id = pd.Series(np.repeat(np.arange(6), years))
# # medication_range = [i for j, i in enumerate(medication_range) if j not in [5, 6, 7, 8]]
#
# # Removing patient profiles (see hypertension_treatment_sbbi_sbmcc.py file) # for talk
# pt_sim = pt_sim[(pt_sim.id!=1) & (pt_sim.id!=4) & (pt_sim.id!=5) & (pt_sim.id!=6) & (pt_sim.id!=7) & (pt_sim.id!=8)].reset_index(drop=True, inplace=False)
# pt_sim.id = pd.Series(np.repeat(np.arange(6), years))
# medication_range = [i for j, i in enumerate(medication_range) if j not in [1, 4, 5, 6, 7, 8]]
#
# ## Preparing data for plot
# meds_df = pt_sim.loc[:, ['id', 'year', 'Clinical Guidelines', 'Optimal Policy', 'Best in Range']].copy() # 'Clinical Guidelines', 'Optimal Policy', 'Best in Range'
# meds_df.year += 1
# meds_df = meds_df.melt(id_vars=['id', 'year'], var_name='policy', value_name='meds')
#
# # Directory for plotting selected cases
# os.chdir(fig_dir)
# plot_range_actions(meds_df, medication_range)
#
# # # Making sure there is a directory to store every patient's plot
# # os.chdir(fig_dir)
# # new_dir = "Medication Ranges"
# # if not os.path.isdir(new_dir):
# #     os.mkdir(new_dir)
# # os.chdir(os.path.join(fig_dir, new_dir))
# #
# # # Plotting range of near-optimal treatment choices for each patient in the dataset (identifying "interesting" patients)
# # all_ids = range(pt_sim.id.unique().shape[0]) #[783]
# # for i in all_ids:
# #     plot_range_actions(i, pt_sim.meds_apr[pt_sim.id==pt_sim.id.unique()[i]], medication_range[i],
# #                        pt_sim.meds_aha[pt_sim.id==pt_sim.id.unique()[i]], pt_sim.meds_opt[pt_sim.id==pt_sim.id.unique()[i]])
# #
# # # Getting demographic and clinical information from selected patients
# # sel_id = [783]
# # for i in sel_id:
# #     print(ptdata_list[0][ptdata_list[0].id==i].astype(int).iloc[0, :])
