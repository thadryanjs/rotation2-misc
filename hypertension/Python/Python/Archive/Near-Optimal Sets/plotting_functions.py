# ==================================================
# Creating Policy and Action Value Function Plots
# ==================================================

# Loading modules
import os  # directory changes
import numpy as np  # matrix operations
import pandas as pd  # data frame operations
# import matplotlib # plotting configurations
# matplotlib.use("Agg") # making sure interactive mode is off (use "TkAgg" for interactive mode)
import matplotlib.pyplot as plt #base plots
# plt.ioff() # making sure interactive mode is off (use plt.ion() for interactive mode)
import seaborn as sns #plots

# Plotting parameters
sns.set_style("ticks")
paper_rc = {'lines.linewidth': 1, 'lines.markersize': 6, 'font.size': 12}
sns.set_context("paper", rc=paper_rc) # use also for talks
# poster_rc = {'lines.linewidth': 1, 'lines.markersize': 14, 'font.size': 10}
# sns.set_context("poster", rc=poster_rc)

# # Main paper functions
# ## Function to plot the distribution of Q-values
# def plot_dist_qvalues(Q, healthy, bins, rep0, rep1):
#
#     # Preparing data for plots
#     plot_data = pd.DataFrame(Q[healthy, 0, 0, :, :])
#
#     # Histogram or density plot
#     for j in [rep0, rep1]:
#         sns.distplot(plot_data.iloc[:, j], hist=True, bins=bins, kde=False,
#                      label='Replication '+str(j))  # histogram
#         # sns.distplot(plot_data.iloc[:, j], hist=False, kde=True, label='Replication ' + str(i)) #kernel density
#         # sns.kdeplot(plot_data.iloc[:, j], label='Replication ' + str(i), shade=True) #shaded kernel density
#
#     # #Modifying axes and legend
#     plt.xlabel('Q-value')
#     plt.ylabel('Frequency')
#     plt.legend(facecolor='white', framealpha=1)
#
#     # #Saving plot
#     plt.savefig('Distribution of Sample Q-values-Hypertension Treatment.pdf', bbox_inches='tight')
#     plt.close()
#
#     # Q-Q plot
#     plt.figure()
#     plt.scatter(np.sort(plot_data.iloc[:, rep0]), np.sort(plot_data.iloc[:, rep1]))
#
#     # #Modifying axes
#     plt.xlabel('Replication '+str(rep0))
#     plt.ylabel('Replication 1'+str(rep1))
#
#     # #Adding diagonal line
#     ax = plt.gca()
#     ax.plot(ax.get_xlim(), ax.get_ylim(), color='r')
#
#     # #Saving plot
#     plt.savefig('Q-Q Plot-Hypertension Treatment.pdf', bbox_inches='tight')
#     plt.close()
#
# ## Function to plot convergence of confidence interval width with different number of batches and a fix number of observations per batch
# def plf_cov_batch(ci_width, plot_batches, selected_batches):
#
#     # Figure parameters
#     axes_size = 14 # font size of axes labels
#     tick_size = 12 # font size for tick labels
#
#     # Making figure
#     plt.figure()
#     plt.plot(ci_width[:plot_batches], color='black')
#
#     # Modifying Axes
#     plt.xlabel('Number of Batches', fontsize=axes_size, fontweight='semibold')
#     plt.ylabel('Confidence Interval Width', fontsize=axes_size, fontweight='semibold')
#     plt.xticks([1, selected_batches, plot_batches], fontsize=tick_size)
#     plt.yticks(ticks=[int(np.floor(min(ci_width)*1000))/1000, int(np.floor(max(ci_width)*10))/10], fontsize=tick_size)
#     plt.ylim(0, int(np.floor(max(ci_width)*10))/10+0.05)
#
#     # Adding horizontal line for reference number of batches
#     plt.hlines(y=ci_width[-1], xmin=1, xmax=plot_batches, color='red', alpha=0.60, zorder=100)
#     plt.text(x=selected_batches, y=ci_width[selected_batches]+0.01, s=str(ci_width[selected_batches].round(2)), color='gray', fontsize=tick_size)
#     plt.vlines(x=selected_batches, ymin=-0.05, ymax=ci_width[selected_batches], color='gray', linestyle='--', zorder=200)
#
#     # Saving plot
#     plt.savefig('Number of Batches Analysis.pdf', bbox_inches='tight')
#     plt.close()
#
# ## Function to plot ranges of nearly optimal actions
# def plot_range_actions(meds_df, medication_range):
#
#     # Figure parameters
#     n_colors = 6  # number of colors in palette
#     x_ticks = range(2, 12, 2)
#     xlims = [0.5, 10.5] # limit of the x-axis
#     axes_size = 10 # font size of axes labels
#     subtitle_size = 9 # font size for subplot titles
#     tick_size = 8 # font size for tick labels
#     legend_size = 8 # font size for legend labels
#     mks = ['D', 'o', 'X'] # markers for plots
#     marker_size = 24 # marker size
#
#     # Making figure
#     fig, axes = plt.subplots(nrows=2, ncols=2)
#     sns.scatterplot(x="year", y="meds", hue="policy", data=meds_df[meds_df.id==0], markers=mks,
#                     style="policy", s=marker_size, linewidth=0.5, palette=sns.color_palette("Greys", n_colors)[2:-1],
#                     ci=None, ax=axes[0, 0], zorder=100)
#     sns.scatterplot(x="year", y="meds", hue="policy", data=meds_df[meds_df.id==1], markers=mks,
#                     style="policy", s=marker_size, linewidth=0.5, palette=sns.color_palette("Greys", n_colors)[2:-1],
#                     ci=None, legend=False, ax=axes[0, 1], zorder=100)
#     sns.scatterplot(x="year", y="meds", hue="policy", data=meds_df[meds_df.id==2], markers=mks,
#                     style="policy", s=marker_size, linewidth=0.5, palette=sns.color_palette("Greys", n_colors)[2:-1],
#                     ci=None, legend=False, ax=axes[1, 0], zorder=100)
#     sns.scatterplot(x="year", y="meds", hue="policy", data=meds_df[meds_df.id==3], markers=mks,
#                     style="policy", s=marker_size, linewidth=0.5, palette=sns.color_palette("Greys", n_colors)[2:-1],
#                     ci=None, legend=False, ax=axes[1, 1], zorder=100)
#     # sns.scatterplot(x="year", y="meds", hue="policy", data=meds_df[meds_df.id==4], markers=mks,
#     #                 style="policy", s=marker_size, linewidth=0.5, palette=sns.color_palette("Greys", n_colors)[2:-1],
#     #                 ci=None, legend=False, ax=axes[2, 0], zorder=100)
#     # sns.scatterplot(x="year", y="meds", hue="policy", data=meds_df[meds_df.id==5], markers=mks,
#     #                 style="policy", s=marker_size, linewidth=0.5, palette=sns.color_palette("Greys", n_colors)[2:-1],
#     #                 ci=None, legend=False, ax=axes[2, 1], zorder=100)
#     # sns.scatterplot(x="year", y="meds", hue="policy", data=meds_df[meds_df.id==6], markers=mks,
#     #                 style="policy", s=marker_size, linewidth=0.5, palette=sns.color_palette("Greys", n_colors)[2:-1],
#     #                 ci=None, legend=False, ax=axes[2, 0], zorder=100)
#     # sns.scatterplot(x="year", y="meds", hue="policy", data=meds_df[meds_df.id==7], markers=mks,
#     #                 style="policy", s=marker_size, linewidth=0.5, palette=sns.color_palette("Greys", n_colors)[2:-1],
#     #                 ci=None, legend=False, ax=axes[2, 1], zorder=100)
#     # sns.scatterplot(x="year", y="meds", hue="policy", data=meds_df[meds_df.id==8], markers=mks,
#     #                 style="policy", s=marker_size, linewidth=0.5, palette=sns.color_palette("Greys", n_colors)[2:-1],
#     #                 ci=None, legend=False, ax=axes[2, 2], zorder=100)
#
#     ## Adding subtitles
#     axes[0, 0].set_title('54-year-old White Male', fontsize=subtitle_size, fontweight='semibold')
#     axes[0, 1].set_title('54-year-old White Female', fontsize=subtitle_size, fontweight='semibold')
#     axes[1, 0].set_title('54-year-old White Male Smoker', fontsize=subtitle_size, fontweight='semibold')
#     axes[1, 1].set_title('70-year-old White Male', fontsize=subtitle_size, fontweight='semibold')
#     # axes[0, 1].set_title('54-year-old Black Male', fontsize=subtitle_size, fontweight='semibold')
#     # axes[2, 0].set_title('54-year-old White Male Smoker', fontsize=subtitle_size, fontweight='semibold')
#     # axes[1, 2].set_title('54-year-old Black Male with EBP', fontsize=subtitle_size, fontweight='semibold')
#     # axes[2, 0].set_title('54-year-old White Male with S2H', fontsize=subtitle_size, fontweight='semibold')
#     # axes[2, 1].set_title('60-year-old White Male with S1H', fontsize=subtitle_size, fontweight='semibold')
#     # axes[2, 2].set_title('70-year-old White Male with S1H', fontsize=subtitle_size, fontweight='semibold')
#
#     # Figure Configuration
#     ## Overall labels
#     fig.add_subplot(111, frameon=False)
#     plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
#     plt.grid(False)
#     plt.xlabel('Year', fontsize=axes_size, fontweight='semibold')
#     plt.ylabel('Antihypertensive Medications\n\n\n', fontsize=axes_size, fontweight='semibold')
#
#     ## Values for y-axes
#     all_meds = np.delete(np.round(np.arange(0, 5, 0.333333333), 2), 1)
#     all_labels = ['0 SD/0 HD', '0 SD/1 HD', '1 SD/0 HD',
#                   '0 SD/2 HD', '1 SD/1 HD', '2 SD/0 HD', # '2 SD/0 HD' = '0 SD/3 HD'
#                   '1 SD/2 HD', '2 SD/1 HD', '3 SD/0 HD', # '2 SD/1 HD' = '0 SD/4 HD'  # '3 SD/0 HD' = '1 SD/3 HD'
#                   '2 SD/2 HD', '3 SD/1 HD', '4 SD/0 HD', # '2 SD/2 HD' = '0 SD/5 HD'  # '3 SD/1 HD' = '1 SD/4 HD' # '4 SD/0 HD' = '2 SD/3 HD'
#                   '3 SD/2 HD', '4 SD/1 HD', '5 SD/0 HD']
#
#     ## Axes configuration for every subplot
#     plt.setp(axes, xlim=xlims, xticks=x_ticks, xlabel='', ylabel='')
#     fig.subplots_adjust(bottom=0.1, wspace=0.4, hspace=0.4)
#     obs_meds = np.concatenate([meds_df.meds.to_numpy(), np.concatenate([x.min().to_numpy() for x in medication_range]),
#                                np.concatenate([x.max().to_numpy() for x in medication_range])]) # same y axis across all plots (comment for individual axes and uncomment line inside loop)
#     for k, ax in list(enumerate(fig.axes))[:-1]:
#         plt.sca(ax)
#
#         # shaded area
#         plt.fill_between(x=np.arange(1, 11), y1=np.amin(medication_range[k], axis=0), y2=np.amax(medication_range[k], axis=0),
#                          color=sns.color_palette("Greys", n_colors)[0], zorder=0)
#
#         # x-axis
#         plt.xticks(fontsize=tick_size)
#
#         # y-axis
#         # obs_meds = np.concatenate([meds_df[meds_df.id==k].meds.to_numpy(), medication_range[k].min().to_numpy(), medication_range[k].max().to_numpy()]) # individual axes (comment for smae y-axis across all plots)
#         yticks = np.round(np.arange(np.amin(obs_meds), np.amax(obs_meds), 0.333333333), 2)
#         yticks = yticks[yticks != 0.33] # Making sure that there is no tick between no treatment and a medication at half dosage
#         if yticks.shape[0] == 0:
#             yticks = [np.amin(obs_meds)]
#         ind = np.where([j in list(yticks) for j in list(all_meds)])[0]
#         ylabels = [all_labels[j] for j in ind.astype(int)]
#         ax.set_yticks(ticks=yticks)
#         ax.set_yticklabels(labels=ylabels, fontsize=tick_size)
#         ax.set_ylim((np.amin(yticks)-0.15, np.amax(yticks)+0.15))
#
#     # Modifying Legend
#     handles, labels = axes[0, 0].get_legend_handles_labels()
#     axes[0, 0].legend(loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(1.1, -1.8),
#                       handles=handles[1:], labels=labels[1:], prop={'size': legend_size}) # for 6 profiles: bbox_to_anchor=(1.1, -3.2) # for 4 profiles: bbox_to_anchor=(1.1, -1.8)
#
#     #Saving plot
#     fig.set_size_inches(6.5, 4) # for 6 profiles: fig.set_size_inches(6.5, 6) for 4 profiles: fig.set_size_inches(6.5, 4)
#     plt.savefig('Ranges for Patient Profiles.pdf', bbox_inches='tight')
#     plt.close()
#
# ## Function to plot demographic summary by BP category
# def plot_demo_bp(demo):
#
#     # Figure parameters
#     n_colors = 6 # number of colors in palette
#     xlims = [-0.5, 3.5] # limit of the x-axis
#     ylims = [0, 4] # limits of the y-axis
#     y_ticks = np.arange(6)
#     data_labe_size = 8 # font size for data labels
#     axes_size = 10 # font size of axes labels
#     subtitle_size = 9 # font size for subplot titles
#     tick_size = 8 # font size for tick labels
#     legend_size = 8 # font size for legend labels
#     cat_labels = demo.bp_cat.unique()  # BP categories
#
#     # Making plot
#     fig, axes = plt.subplots(nrows=1, ncols=2)
#     g1 = sns.barplot(x="bp_cat", y="wt", hue="sex", data=demo[demo.race == 0], palette=sns.color_palette("Greys", n_colors)[3:-1], ax=axes[0])
#     g2 = sns.barplot(x="bp_cat", y="wt", hue="sex", data=demo[demo.race == 1], palette=sns.color_palette("Greys", n_colors)[3:-1], ax=axes[1])
#
#     # Adding data labels
#     for p in g1.patches:
#         g1.annotate(format(p.get_height(), '.2f'), (p.get_x()+p.get_width()/2., p.get_height()), ha='center',
#                        va='center', xytext=(0, 10), textcoords='offset points', fontsize=data_labe_size)
#
#     for p in g2.patches:
#         g2.annotate(format(p.get_height(), '.2f'), (p.get_x()+p.get_width()/2., p.get_height()), ha='center',
#                        va='center', xytext=(0, 10), textcoords='offset points', fontsize=data_labe_size)
#
#     # Figure Configuration
#     ## Configuration for the panel plot
#     fig.add_subplot(111, frameon=False)
#     plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
#     plt.grid(False)
#     plt.xlabel('BP Catgory', fontsize=axes_size, fontweight='semibold')
#     plt.ylabel('Number of People (in Millions)', fontsize=axes_size, fontweight='semibold')
#
#     ## Axes configuration for every subplot
#     plt.setp(axes, xlim=xlims, xlabel='',
#              ylim=ylims, yticks=y_ticks, ylabel='')
#     fig.subplots_adjust(bottom=0.12)
#
#     ## Adding subtitles
#     axes[0].set_title('Race: Black', fontsize=subtitle_size, fontweight='semibold')
#     axes[1].set_title('Race: White', fontsize=subtitle_size, fontweight='semibold')
#
#     for ax in fig.axes:
#         plt.sca(ax)
#         # plt.xticks(rotation=15, fontsize=tick_size-1, ticks=np.arange(len(risk_labels)), labels=risk_labels, ha='center')
#         plt.xticks(fontsize=tick_size-1, ticks=np.arange(len(cat_labels)), labels=cat_labels,
#                    ha='center')  # rotation=15,
#         plt.yticks(fontsize=tick_size)
#
#     # Modifying Legend
#     handles, _ = axes[0].get_legend_handles_labels()
#     labels = ['Females', 'Males']
#     axes[0].legend(loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(1.1, -0.12),
#                       handles=handles, labels=labels, prop={'size': legend_size})
#     axes[1].get_legend().remove()
#
#     fig.set_size_inches(6.5, 3.66)
#     plt.savefig("Demographics by BP Category.pdf", bbox_inches='tight')
#     plt.close()
#
# ## Function to plot total QALYs saved over planning horizon by risk group
# def plot_qalys_saved(qalys_df, min_age=40, group='sex'):
#
#     # Figure parameters
#     mks = ['D', 's', 'o', 'P', 'X']
#     n_colors = 7  # number of colors in palette
#     xlims = [0.5, 10.5] # limit of the x-axis
#     x_ticks = range(2, 12, 2)
#     y_ul = int(np.ceil(qalys_df['qalys'].max()*10))/10 # upper label in the y-axis
#     ylims = [-0.1, y_ul+0.1] # limits of the y-axis (poster)
#     y_ticks = np.arange(0, y_ul+0.4, 0.4)
#     axes_size = 10 # font size of axes labels
#     subtitle_size = 9 # font size for subplot titles
#     tick_size = 8 # font size for tick labels
#     legend_size = 8 # font size for legend labels
#     marker_size = 5 # marker size
#     line_width = 0.5  # width for lines in plots
#     cat_labels = qalys_df.bp_cat.unique()
#
#     # Making figure
#     fig, axes = plt.subplots(nrows=2, ncols=3)
#     sns.lineplot(x="year", y="qalys", hue="policy", data=qalys_df[(qalys_df['bp_cat']==cat_labels[0]) & (qalys_df[group]==0)], style="policy", markers=mks,
#                  markersize=marker_size, markeredgewidth=0.5, dashes=False, linewidth=line_width, palette=sns.color_palette("Greys", n_colors)[1:-1], ci=None, ax=axes[0, 0])
#     sns.lineplot(x="year", y="qalys", hue="policy", data=qalys_df[(qalys_df['bp_cat']==cat_labels[1]) & (qalys_df[group]==0)], style="policy", markers=mks,
#                  markersize=marker_size, markeredgewidth=0.5, dashes=False, linewidth=line_width, palette=sns.color_palette("Greys", n_colors)[1:-1], ci=None, legend=False, ax=axes[0, 1])
#     sns.lineplot(x="year", y="qalys", hue="policy", data=qalys_df[(qalys_df['bp_cat']==cat_labels[2]) & (qalys_df[group]==0)], style="policy", markers=mks,
#                  markersize=marker_size, markeredgewidth=0.5, dashes=False, linewidth=line_width, palette=sns.color_palette("Greys", n_colors)[1:-1], ci=None, legend=False, ax=axes[0, 2])
#     sns.lineplot(x="year", y="qalys", hue="policy", data=qalys_df[(qalys_df['bp_cat']==cat_labels[0]) & (qalys_df[group]==1)], style="policy", markers=mks,
#                  markersize=marker_size, markeredgewidth=0.5, dashes=False, linewidth=line_width, palette=sns.color_palette("Greys", n_colors)[1:-1], ci=None, legend=False, ax=axes[1, 0])
#     sns.lineplot(x="year", y="qalys", hue="policy", data=qalys_df[(qalys_df['bp_cat']==cat_labels[1]) & (qalys_df[group]==1)], style="policy", markers=mks,
#                  markersize=marker_size, markeredgewidth=0.5, dashes=False, linewidth=line_width, palette=sns.color_palette("Greys", n_colors)[1:-1], ci=None, legend=False, ax=axes[1, 1])
#     sns.lineplot(x="year", y="qalys", hue="policy", data=qalys_df[(qalys_df['bp_cat']==cat_labels[2]) & (qalys_df[group]==1)], style="policy", markers=mks,
#                  markersize=marker_size, markeredgewidth=0.5, dashes=False, linewidth=line_width, palette=sns.color_palette("Greys", n_colors)[1:-1], ci=None, legend=False, ax=axes[1, 2])
#
#     # Figure Configuration
#     ## Configuration for the panel plot
#     fig.add_subplot(111, frameon=False)
#     plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
#     plt.grid(False)
#     plt.xlabel('Year', fontsize=axes_size, fontweight='semibold')
#     plt.ylabel('Expected Life-Years Saved (in Millions)', fontsize=axes_size, fontweight='semibold')
#
#     ## Axes configuration for every subplot
#     plt.setp(axes, xlim=xlims, xticks=x_ticks, xlabel='',
#              ylim=ylims, yticks=y_ticks, ylabel='')
#     fig.subplots_adjust(bottom=0.15, wspace=0.3, hspace=0.35)
#     for ax in fig.axes:
#         plt.sca(ax)
#         plt.xticks(fontsize=tick_size)
#         plt.yticks(fontsize=tick_size)
#
#     ## Adding subtitles
#     if group=='sex':
#         axes[0, 0].set_title('Females: Elevated', fontsize=subtitle_size, fontweight='semibold')
#         axes[0, 1].set_title('Females: Stage 1', fontsize=subtitle_size, fontweight='semibold')
#         axes[0, 2].set_title('Females: Stage 2', fontsize=subtitle_size, fontweight='semibold')
#         axes[1, 0].set_title('Males: Elevated', fontsize=subtitle_size, fontweight='semibold')
#         axes[1, 1].set_title('Males: Stage 1', fontsize=subtitle_size, fontweight='semibold')
#         axes[1, 2].set_title('Males: Stage 2', fontsize=subtitle_size, fontweight='semibold')
#     elif group=='race':
#         axes[0, 0].set_title('Black Race: Elevated', fontsize=subtitle_size, fontweight='semibold')
#         axes[0, 1].set_title('Black Race: Stage 1', fontsize=subtitle_size, fontweight='semibold')
#         axes[0, 2].set_title('Black Race: Stage 2', fontsize=subtitle_size, fontweight='semibold')
#         axes[1, 0].set_title('White Race: Elevated', fontsize=subtitle_size, fontweight='semibold')
#         axes[1, 1].set_title('White Race: Stage 1', fontsize=subtitle_size, fontweight='semibold')
#         axes[1, 2].set_title('White Race: Stage 2', fontsize=subtitle_size, fontweight='semibold')
#
#     # Modifying Legend
#     handles, labels = axes[0, 0].get_legend_handles_labels()
#     order = [0, 5, 3, 1, 4, 2]
#     handles = [x for _, x in sorted(zip(order, handles))]
#     labels = [x for _, x in sorted(zip(order, labels))]
#     lgnd = axes[0, 0].legend(loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(1.8, -1.60),
#                              handles=handles[1:], labels=labels[1:], prop={'size': legend_size})
#     for l in lgnd.legendHandles:
#         l._legmarker.set_markersize(marker_size)
#
#     fig.set_size_inches(6.5, 4.5)
#     if min_age==40:
#         plt.savefig("QALYs Saved per Policy - 50-54 - "+group+".pdf", bbox_inches='tight') # for pape
#     elif min_age==70:
#         plt.savefig("QALYs Saved per Policy - 70-74 - "+group+".pdf", bbox_inches='tight')
#     plt.close()
#
# ## Function to plot distribution of treatment by risk group at year 1 and 10
# def plot_trt_dist(trt_df):
#
#     # Figure parameters
#     n_colors = 6 # number of colors in palette
#     xlims = [-0.5, 3.5] # limit of the x-axis
#     ylims = [-0.5, 5.5] # limits of the y-axis
#     y_ticks = np.arange(6)
#     axes_size = 10 # font size of axes labels
#     subtitle_size = 9 # font size for subplot titles
#     tick_size = 8 # font size for tick labels
#     legend_size = 8 # font size for legend labels
#     cat_labels = trt_df.bp_cat.unique() # BP categories
#     flierprops = dict(marker='o', markerfacecolor='none', markeredgecolor='black', markeredgewidth=0.5, markersize=2,
#                       linestyle='none') # outliers properties
#
#     # Overall
#     # fig, axes = plt.subplots(nrows=1, ncols=2)
#     # sns.boxplot(x="bp_cat", y="meds", hue="policy", data=trt_df[(trt_df.year==1)],
#     #             palette=sns.color_palette("Greys", n_colors), linewidth=0.75, flierprops=flierprops, ax=axes[0])
#     # sns.boxplot(x="bp_cat", y="meds", hue="policy", data=trt_df[(trt_df.year==10)],
#     #             palette=sns.color_palette("Greys", n_colors), linewidth=0.75, flierprops=flierprops, ax=axes[1])
#
#     # #By sex
#     # fig, axes = plt.subplots(nrows=2, ncols=2)
#     # sns.boxplot(x="bp_cat", y="meds", hue="policy", data=trt_df[(trt_df.year==1) & (trt_df.sex==0)],
#     #             palette=sns.color_palette("Greys", n_colors), linewidth=0.75, flierprops=flierprops, ax=axes[0, 0])
#     # sns.boxplot(x="bp_cat", y="meds", hue="policy", data=trt_df[(trt_df.year==10) & (trt_df.sex==0)],
#     #             palette=sns.color_palette("Greys", n_colors), linewidth=0.75, flierprops=flierprops, ax=axes[0, 1])
#     # sns.boxplot(x="bp_cat", y="meds", hue="policy", data=trt_df[(trt_df.year==1) & (trt_df.sex==1)],
#     #             palette=sns.color_palette("Greys", n_colors), linewidth=0.75, flierprops=flierprops, ax=axes[1, 0])
#     # sns.boxplot(x="bp_cat", y="meds", hue="policy", data=trt_df[(trt_df.year==10) & (trt_df.sex==1)],
#     #             palette=sns.color_palette("Greys", n_colors), linewidth=0.75, flierprops=flierprops, ax=axes[1, 1])
#
#     #By race
#     fig, axes = plt.subplots(nrows=2, ncols=2)
#     sns.boxplot(x="bp_cat", y="meds", hue="policy", data=trt_df[(trt_df.year==1) & (trt_df.race==0)],
#                 palette=sns.color_palette("Greys", n_colors), linewidth=0.75, flierprops=flierprops, ax=axes[0, 0])
#     sns.boxplot(x="bp_cat", y="meds", hue="policy", data=trt_df[(trt_df.year==10) & (trt_df.race==0)],
#                 palette=sns.color_palette("Greys", n_colors), linewidth=0.75, flierprops=flierprops, ax=axes[0, 1])
#     sns.boxplot(x="bp_cat", y="meds", hue="policy", data=trt_df[(trt_df.year==1) & (trt_df.race==1)],
#                 palette=sns.color_palette("Greys", n_colors), linewidth=0.75, flierprops=flierprops, ax=axes[1, 0])
#     sns.boxplot(x="bp_cat", y="meds", hue="policy", data=trt_df[(trt_df.year==10) & (trt_df.race==1)],
#                 palette=sns.color_palette("Greys", n_colors), linewidth=0.75, flierprops=flierprops, ax=axes[1, 1])
#
#     # Figure Configuration
#     ## Configuration for the panel plot
#     fig.add_subplot(111, frameon=False)
#     plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
#     plt.grid(False)
#     plt.xlabel('BP Category', fontsize=axes_size, fontweight='semibold') #'Risk Group'
#     plt.ylabel('Number of Medications', fontsize=axes_size, fontweight='semibold')
#
#     ## Axes configuration for every subplot
#     plt.setp(axes, xlim=xlims, xlabel='',
#              ylim=ylims, yticks=y_ticks, ylabel='')
#     fig.subplots_adjust(bottom=0.3, wspace=0.2, hspace=0.45)
#
#     ## Adding subtitles
#     ### Overall
#     # bp_labels = ['Normal BP', 'Elevated BP', 'Stage 1\nHypertension', 'Stage 2\nHypertension']
#     # axes[0].set_title('Year 1', fontsize=subtitle_size, fontweight='semibold')
#     # axes[1].set_title('Year 10', fontsize=subtitle_size, fontweight='semibold')
#     # fig.subplots_adjust(bottom=0.3)  # for overal plots
#
#     # ### By sex
#     # axes[0, 0].set_title('Females: Year 1', fontsize=subtitle_size, fontweight='semibold')
#     # axes[0, 1].set_title('Females: Year 10', fontsize=subtitle_size, fontweight='semibold')
#     # axes[1, 0].set_title('Males: Year 1', fontsize=subtitle_size, fontweight='semibold')
#     # axes[1, 1].set_title('Males: Year 10', fontsize=subtitle_size, fontweight='semibold')
#
#     ### By race
#     axes[0, 0].set_title('Black Race: Year 1', fontsize=subtitle_size, fontweight='semibold')
#     axes[0, 1].set_title('Black Race: Year 10', fontsize=subtitle_size, fontweight='semibold')
#     axes[1, 0].set_title('White Race: Year 1', fontsize=subtitle_size, fontweight='semibold')
#     axes[1, 1].set_title('White Race: Year 10', fontsize=subtitle_size, fontweight='semibold')
#
#     for ax in fig.axes:
#         plt.sca(ax)
#         # plt.xticks(rotation=15, fontsize=tick_size-1, ticks=np.arange(len(bp_labels)), labels=bp_labels, ha='center') # for overall plots
#         plt.xticks(fontsize=tick_size-1, ticks=np.arange(len(cat_labels)), labels=cat_labels, ha='center') #rotation=15, # for plots by group
#         plt.yticks(fontsize=tick_size)
#
#     # # Adding patterns to plots
#     # hatches = ["-", "/", "+", "\\", "."]
#     # for ax in fig.axes:
#     #     for hatch, patch in zip(hatches, ax.artists):
#     #         patch.set_hatch(hatch)
#
#     # Modifying Legend
#     # ## For overall plot
#     # handles, labels = axes[0].get_legend_handles_labels()
#     # axes[0].legend(loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(1.15, -0.35),
#     #                   handles=handles, labels=labels, prop={'size': legend_size})
#     # axes[1].get_legend().remove()
#
#     ## For group plots
#     handles, labels = axes[0, 0].get_legend_handles_labels()
#     axes[0, 0].legend(loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(1.15, -1.9),
#                       handles=handles, labels=labels, prop={'size': legend_size})
#     axes[0, 1].get_legend().remove()
#     axes[1, 0].get_legend().remove()
#     axes[1, 1].get_legend().remove()
#
#     # fig.set_size_inches(6.5, 3.5) # overall
#     # plt.savefig("Treatment per Policy.pdf", bbox_inches='tight')
#     fig.set_size_inches(6.5, 5.5)  # by sex/race
#     plt.savefig("Treatment per Policy-race.pdf", bbox_inches='tight')
#     plt.close()
#
# ## Function to plot proportion of actions by policy convered in range by risk group
# def plot_prop_covered(prop_df):
#
#     # Figure parameters
#     n_colors = 4  # number of colors in palette
#     xlims = [0.5, 10.5] # limit of the x-axis
#     x_ticks = range(2, 12, 2) #[1, 5, 10]
#     ylims = [-0.1, 1.1] # limits of the y-axis
#     y_ticks = [0, 0.5, 1]
#     axes_size = 10 # font size of axes labels
#     subtitle_size = 9 # font size for subplot titles
#     tick_size = 8 # font size for tick labels
#     legend_size = 8 # font size for legend labels
#     line_width = 1.5 # width for lines in plots
#
#     # Making figure
#     fig, axes = plt.subplots(nrows=2, ncols=2)
#     sns.lineplot(x="year", y="prop_cv", hue="bp_cat",
#                  data=prop_df[(prop_df['race'] == 0) & (prop_df['sex'] == 0)],
#                  style="bp_cat", markers=False, dashes=True, palette=sns.color_palette("Greys", n_colors)[1:], ci=None,
#                  ax=axes[0, 0])
#     sns.lineplot(x="year", y="prop_cv", hue="bp_cat",
#                  data=prop_df[(prop_df['race'] == 0) & (prop_df['sex'] == 1)],
#                  style="bp_cat", markers=False, dashes=True, palette=sns.color_palette("Greys", n_colors)[1:], ci=None,
#                  legend=False, ax=axes[0, 1])
#     sns.lineplot(x="year", y="prop_cv", hue="bp_cat",
#                  data=prop_df[(prop_df['race'] == 1) & (prop_df['sex'] == 0)],
#                  style="bp_cat", markers=False, dashes=True, palette=sns.color_palette("Greys", n_colors)[1:], ci=None,
#                  legend=False, ax=axes[1, 0])
#     sns.lineplot(x="year", y="prop_cv", hue="bp_cat",
#                  data=prop_df[(prop_df['race'] == 1) & (prop_df['sex'] == 1)],
#                  style="bp_cat", markers=False, dashes=True, palette=sns.color_palette("Greys", n_colors)[1:], ci=None,
#                  legend=False, ax=axes[1, 1])
#
#     # Figure Configuration
#     ## Configuration for the panel plot
#     fig.add_subplot(111, frameon=False)
#     plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
#     plt.grid(False)
#     plt.xlabel('Year', fontsize=axes_size, fontweight='semibold')
#     plt.ylabel('Proportion of Patient-Years Covered in Range', fontsize=axes_size, fontweight='semibold')
#
#     ## Axes configuration
#     plt.setp(axes, xlim=xlims, xticks=x_ticks, xlabel='',
#              ylim=ylims, yticks=y_ticks, ylabel='')
#     fig.subplots_adjust(wspace=0.3, hspace=0.5)
#     for ax in fig.axes:
#         plt.sca(ax)
#         plt.xticks(fontsize=tick_size)
#         plt.yticks(fontsize=tick_size)
#         plt.setp(ax.lines, linewidth=line_width)
#
#     ## Adding subtitles
#     axes[0, 0].set_title('Black Females', fontsize=subtitle_size, fontweight='semibold')
#     axes[0, 1].set_title('Black Males', fontsize=subtitle_size, fontweight='semibold')
#     axes[1, 0].set_title('White Females', fontsize=subtitle_size, fontweight='semibold')
#     axes[1, 1].set_title('White Males', fontsize=subtitle_size, fontweight='semibold')
#
#     ## Formatting y-axis as percentages
#     for ax in fig.axes:
#         ax.set_yticklabels(['{:,.0%}'.format(x) for x in ax.get_yticks()])
#
#     # Modifying Legend
#     handles, labels = axes[0, 0].get_legend_handles_labels()
#     axes[0, 0].legend(loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(1.15, -1.9),
#                       handles=handles[1:], labels=labels[1:], prop={'size': legend_size})
#
#     fig.set_size_inches(6.5, 4.25)
#     plt.savefig("Proportion of Patient-Years Covered in Range.pdf", bbox_inches='tight')
#     plt.close()
#
# ## Function to plot range average (min-max) width and number of medications
# def plot_ranges_len_meds(sens_df_width, sens_df_meds):
#
#     # Figure parameters
#     n_colors = 6  # number of colors in palette
#     x_ticks = range(2, 12, 2)
#     y_ul = int(np.ceil(sens_df_width['max'].max()/10))*10  # upper label in the y-axis
#     ylims = [[0, y_ul+1], [-0.5, 5.5]] # limits of the y-axis
#     y_ticks = [np.linspace(start=0, stop=y_ul, num=6), np.arange(6)]
#     y_labels = ['Range Width', 'Number of Medications']
#     axes_size = 10 # font size of axes labels
#     subtitle_size = 9 # font size for subplot titles
#     tick_size = 8 # font size for tick labels
#     legend_size = 8 # font size for legend labels
#     # mk_size = 4 # marker size for bound on error bars
#
#     # # Adding jitter to year to ease the visualization of the error bars (error bars look weird)
#     # jitt = [-0.3, -0.1, 0.1, 0.3]
#     # for y, x in enumerate(sens_df_width.scenario.unique()):
#     #     sens_df_width.loc[sens_df_width.scenario == x, 'year'] += jitt[y]
#     #     sens_df_meds.loc[sens_df_meds.scenario == x, 'year'] += jitt[y]
#
#     # Making figure
#     fig, axes = plt.subplots(nrows=1, ncols=2)
#     sns.scatterplot(x="year", y="mean", hue="scenario", data=sens_df_width, markers=['H', 'v', '^', 'X'],
#                     style="scenario", palette=np.sort(sns.color_palette("Greys", n_colors)[1:-1])[::-1].tolist(), linewidth=0.5,
#                     ci=None, ax=axes[0], zorder=100) # s=25,
#     sns.scatterplot(x="year", y="mean", hue="scenario", data=sens_df_meds, markers=['H', 'v', '^', 'X'],
#                     style="scenario", palette=np.sort(sns.color_palette("Greys", n_colors)[1:-1])[::-1].tolist(), linewidth=0.5,
#                     ci=None, legend=False, ax=axes[1], zorder=100) #s=25,
#
#     # Figure Configuration
#     ## Configuration for the panel plot
#     fig.add_subplot(111, frameon=False)
#     plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
#     plt.grid(False)
#     plt.xlabel('Year', fontsize=axes_size, fontweight='semibold')
#     plt.ylabel('', fontsize=axes_size, fontweight='semibold')
#
#     ## Axes configuration for every subplot
#     fig.subplots_adjust(bottom=0.2, wspace=0.3)
#     for k, ax in list(enumerate(fig.axes))[:-1]:
#         plt.sca(ax)
#         plt.xlabel('')
#         plt.xticks(ticks=x_ticks, fontsize=tick_size)
#         plt.ylim(ylims[k])
#         plt.yticks(ticks=y_ticks[k], fontsize=tick_size)
#         plt.ylabel(y_labels[k], fontsize=subtitle_size, fontweight='semibold')
#
#     # Adding error band (or bars)
#     colors = sns.color_palette("Greys", n_colors).as_hex()[1:]
#     for j, k, l in [(0, 0, 10), (1, 9, 20)]: # excluding random and worst future actions because they are the same as in best future action # enumerate(sens_df_width.scenario.unique())
#         ## Range width plot
#         # axes[0].fill_between(x=sens_df_width.loc[sens_df_meds.scenario==k, 'year'],
#         #                      y1=sens_df_width.loc[sens_df_meds.scenario==k, 'min'],
#         #                      y2=sens_df_width.loc[sens_df_meds.scenario==k, 'max'], color=colors[j],
#         #                      alpha=0.2, zorder=0) # error bands
#         # axes[0].vlines(x=sens_df_width.loc[sens_df_width.scenario==k, 'year'],
#         #                ymin=sens_df_width.loc[sens_df_width.scenario==k, 'min'],
#         #                ymax=sens_df_width.loc[sens_df_width.scenario==k, 'max'],
#         #                color=colors[j], linestyle='solid', linewidth=0.6, zorder=0) # error bars
#         axes[0].plot(sens_df_width.loc[sens_df_width.scenario==k, 'year'],
#                      sens_df_width.loc[sens_df_width.scenario==k, 'min'],
#                      marker='', linestyle='dashed', color=colors[j], zorder=l)  # for error bars: marker='_', markersize=mk_size, linestyle=''
#         axes[0].plot(sens_df_width.loc[sens_df_width.scenario==k, 'year'],
#                      sens_df_width.loc[sens_df_width.scenario==k, 'max'],
#                      marker='', linestyle='dashed', color=colors[j], zorder=l) # for error bars: marker='_', markersize=mk_size, linestyle=''
#
#         # Number of medications plot
#         # axes[1].fill_between(x=sens_df_meds.loc[sens_df_meds.scenario==k, 'year'],
#         #                      y1=sens_df_meds.loc[sens_df_meds.scenario==k, 'min'],
#         #                      y2=sens_df_meds.loc[sens_df_meds.scenario==k, 'max'], color=colors[j],
#         #                      alpha=0.2, zorder=0) # error bands
#         # axes[1].vlines(x=sens_df_meds.loc[sens_df_meds.scenario==k, 'year'],
#         #                ymin=sens_df_meds.loc[sens_df_meds.scenario==k, 'min'],
#         #                ymax=sens_df_meds.loc[sens_df_meds.scenario==k, 'max'],
#         #                color=colors[j], linestyle='solid', linewidth=0.6, zorder=0) # error bars
#         axes[1].plot(sens_df_meds.loc[sens_df_meds.scenario==k, 'year'],
#                      sens_df_meds.loc[sens_df_meds.scenario==k, 'min'],
#                      marker='', linestyle='dashed', color=colors[j], zorder=l)  # for error bars: marker='_', markersize=mk_size, linestyle=''
#         axes[1].plot(sens_df_meds.loc[sens_df_meds.scenario==k, 'year'],
#                      sens_df_meds.loc[sens_df_meds.scenario==k, 'max'],
#                      marker='', linestyle='dashed', color=colors[j], zorder=l)  # for error bars: marker='_', markersize=mk_size, linestyle=''
#
#     # Modifying Legend
#     labels = ['Base Case', 'Assuming Normality', "Median in Next Year's Range", "Fewest in Next Year's Range"]
#     handles, _ = axes[0].get_legend_handles_labels() # reversing order in handles (to follow the order of labels)
#     axes[0].legend(loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(1.15, -0.2),
#                    handles=handles[1:][::-1], labels=labels, prop={'size': legend_size})
#
#     fig.set_size_inches(6.5, 3)
#     plt.savefig("Range Width and Medications.pdf", bbox_inches='tight')
#     plt.close()
#
# ## Function to plot proportion of actions by policy convered in range by misestimation scenario
# def plot_prop_mis(prop_df):
#
#     # Figure parameters
#     mks = ['D', 'o']
#     n_colors = 3  # number of colors in palette
#     xlims = [0.5, 10.5] # limit of the x-axis
#     x_ticks = range(2, 12, 2) #[1, 5, 10]
#     ylims = [-0.1, 1.1] # limits of the y-axis
#     y_ticks = [0, 0.5, 1]
#     axes_size = 10 # font size of axes labels
#     subtitle_size = 9 # font size for subplot titles
#     tick_size = 8 # font size for tick labels
#     legend_size = 8 # font size for legend labels
#     line_width = 1.5 # width for lines in plots
#
#     # Making figure
#     fig, axes = plt.subplots(nrows=2, ncols=2)
#     sns.lineplot(x="year", y="prop", hue="policy", data=prop_df[prop_df['misestimation']=='Base Case'], style="policy", markers=mks, #markeredgewidth=0.5,
#                  dashes=False, ci=None, palette=sns.color_palette("Greys", n_colors)[1:], ax=axes[0, 0]) #sns.color_palette("Greys", n_colors)[1:] # "viridis"
#     sns.lineplot(x="year", y="prop", hue="policy", data=prop_df[prop_df['misestimation']=='Half Event Rates'], style="policy", markers=mks, #markeredgewidth=0.5,
#                  dashes=False, ci=None, palette=sns.color_palette("Greys", n_colors)[1:], legend=False, ax=axes[0, 1])
#     sns.lineplot(x="year", y="prop", hue="policy", data=prop_df[prop_df['misestimation']=='Double Event Rates'], style="policy", markers=mks, #markeredgewidth=0.5,
#                  dashes=False, ci=None, palette=sns.color_palette("Greys", n_colors)[1:], legend=False, ax=axes[1, 0])
#     sns.lineplot(x="year", y="prop", hue="policy", data=prop_df[prop_df['misestimation']=='Half Treatment Benefit'], style="policy", markers=mks, #markeredgewidth=0.5,
#                  dashes=False, ci=None, palette=sns.color_palette("Greys", n_colors)[1:], legend=False, ax=axes[1, 1])
#
#     # Figure Configuration
#     ## Configuration for the panel plot
#     fig.add_subplot(111, frameon=False)
#     plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
#     plt.grid(False)
#     plt.xlabel('Year', fontsize=axes_size, fontweight='semibold')
#     plt.ylabel('Proportion of Patients Covered in Range', fontsize=axes_size, fontweight='semibold')
#
#     ## Axes configuration
#     plt.setp(axes, xlim=xlims, xticks=x_ticks, xlabel='',
#              ylim=ylims, yticks=y_ticks, ylabel='')
#     fig.subplots_adjust(wspace=0.3, hspace=0.5, bottom=0.2)
#     for ax in fig.axes:
#         plt.sca(ax)
#         plt.xticks(fontsize=tick_size)
#         plt.yticks(fontsize=tick_size)
#         # plt.setp(ax.lines, linewidth=line_width)
#
#     ## Adding subtitles
#     axes[0, 0].set_title('Base Case', fontsize=subtitle_size, fontweight='semibold')
#     axes[0, 1].set_title('Half Event Rates', fontsize=subtitle_size, fontweight='semibold')
#     axes[1, 0].set_title('Double Event Rates', fontsize=subtitle_size, fontweight='semibold')
#     axes[1, 1].set_title('Half Treatment Benefit', fontsize=subtitle_size, fontweight='semibold')
#
#     ## Formatting y-axis as percentages
#     for ax in fig.axes:
#         ax.set_yticklabels(['{:,.0%}'.format(x) for x in ax.get_yticks()])
#
#     # Modifying Legend
#     handles, labels = axes[0, 0].get_legend_handles_labels()
#     axes[0, 0].legend(loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(1.15, -1.8),
#                       handles=handles[1:], labels=labels[1:], prop={'size': legend_size})
#
#     # For paper
#     fig.set_size_inches(6.5, 4.5)
#     plt.savefig("Proportion of Patient-Years Covered in Range with Misestimated Parameters.pdf", bbox_inches='tight')
#     # For talks
#     # fig.set_size_inches(8, 4.5)
#     # plt.savefig('Proportion of Patient-Years Covered in Range with Misestimated Parameters.png', bbox_inches='tight', dpi=600)
#     plt.close()
#
# Poster functions
# # Function to plot ranges of nearly optimal actions (for posters)
# def plot_range_actions(ptid, meds_apr, Pi_meds, meds_aha, meds_opt):
#
#     # # Line for debugging purposes
#     # ptid = 8; Pi_meds = medication_range[ptid]; meds_aha = pt_sim.meds_aha[pt_sim.id==ptid]; meds_apr = pt_sim.meds_apr[pt_sim.id==ptid]
#
#     # Plotting parameters
#     n_colors = 7
#     palette = sns.color_palette("Greys", n_colors).as_hex()[1:-1]
#     axes_size = 26 # font size of axes labels
#     tick_size = 22 # font size for tick labels
#     legend_size = 22  # font size for legend labels
#
#     # Plotting policies
#     fig_name_policy = 'Policy for Patient ' + str(ptid)
#     Pi_meds_melt = Pi_meds.melt()
#     Pi_meds_melt.columns = ["Year", "Range"]
#
#     # #Creating plot
#     fig = plt.figure()
#     lines = plt.plot(range(10), meds_apr, 'X', range(10), meds_aha, 'D', range(10), np.amin(Pi_meds, axis=0), "",
#                      range(10), np.amax(Pi_meds, axis=0), "")
#     plt.setp(lines[0], color=palette[3], zorder=3)
#     plt.setp(lines[1], color=palette[-1], zorder=2)
#     plt.setp(lines[2], linewidth=1, color="darkgray", linestyle='dashed', zorder=1, label='_nolegend_')
#     plt.setp(lines[3], linewidth=1, color="darkgray", linestyle='dashed', zorder=1, label='_nolegend_')
#
#     ### Modifying title
#     # plt.title("Figure 1: Ranges of treatment choices for a 53-year old Black male smoker\n               with stage 1 hypertension.\n",
#     #           loc="left") # use for SMDM abstract
#     # plt.title("Patient " + str(ptid))
#
#     ### Modifying axes
#     plt.xticks(range(10), range(1, 11))
#     obs_meds = np.array([meds_aha, meds_apr, np.array(Pi_meds.min()), np.array(Pi_meds.max())])
#     yticks = np.round(np.arange(obs_meds.min(), obs_meds.max(), 0.333333333), 2)
#     yticks = yticks[yticks != 0.33] # Making sure that there is no tick between no treatment and a medication at half dosage
#     if len(yticks) == 0:
#         yticks = [obs_meds.min()]
#     all_labels = ['0 SD/0 HD', '0 SD/1 HD', '1 SD/0 HD',
#                   '0 SD/2 HD', '1 SD/1 HD', '2 SD/0 HD', # '2 SD/0 HD' = '0 SD/3 HD'
#                   '1 SD/2 HD', '2 SD/1 HD', '3 SD/0 HD', # '2 SD/1 HD' = '0 SD/4 HD'  # '3 SD/0 HD' = '1 SD/3 HD'
#                   '2 SD/2 HD', '3 SD/1 HD', '4 SD/0 HD', # '2 SD/2 HD' = '0 SD/5 HD'  # '3 SD/1 HD' = '1 SD/4 HD' # '4 SD/0 HD' = '2 SD/3 HD'
#                   '3 SD/2 HD', '4 SD/1 HD', '5 SD/0 HD']
#
#     #### Converting medications to index of labels
#     all_meds = np.delete(np.round(np.arange(0, 5, 0.333333333), 2), 1)
#     ind = np.where([j in list(yticks) for j in list(all_meds)])[0]
#     ylabels = [all_labels[j] for j in ind.astype(int)]
#
#     # # Manual ticks and labels
#     # yticks = np.array([2, 2.33, 2.67, 3., 3.33, 3.67])
#     # ylabels = ['2 SD/0 HD', '1 SD/2 HD', '2 SD/1 HD', '3 SD/0 HD', '2 SD/2 HD', '3 SD/1 HD']
#
#     plt.xticks(fontsize=tick_size)
#     plt.yticks(yticks, labels=ylabels, fontsize=tick_size)
#     plt.ylim((np.amin(yticks)-0.1, np.amax(yticks)+0.1))
#     ax = plt.gca()
#     plt.xlabel('Year', fontsize=axes_size, fontweight='semibold')
#     # ax.xaxis.set_label_coords(4.5, -0.07, transform=ax.xaxis.get_ticklabels()[0].get_transform())
#     plt.ylabel('Antihypertensive Medications', fontsize=axes_size, fontweight='semibold')
#     # ax.yaxis.set_label_coords(0, 3, transform=ax.yaxis.get_ticklabels()[0].get_transform())
#
#     ### Adding shaded area
#     plt.fill_between(x=range(10), y1=np.amin(Pi_meds, axis=0), y2=np.amax(Pi_meds, axis=0), color=palette[2],
#                      alpha=0.25)
#
#     #### Modifying Legend
#     ax = plt.gca()
#     ax.legend(("Best in Range", "Clinical Guidelines"),
#               loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False, prop={'size': legend_size})
#
#     #Saving plot
#     # fig.set_size_inches(6.5, 4.25) # for paper
#     fig.set_size_inches(11, 7)  # for poster
#     plt.savefig(fig_name_policy + '.png', bbox_inches='tight', dpi=600)
#     plt.close()
#
# # Function to plot total QALYs saved over planning horizon by risk group (for posters)
# def plot_qalys_saved(qalys_df, min_age=40):
#
#     # Figure parameters
#     mks = ['D', 's', 'P', 'X'] #['D', 's', 'o', 'P', 'X']
#     n_colors = 7  # number of colors in palette
#     xlims = [0.5, 10.5] # limit of the x-axis
#     x_ticks = range(2, 12, 2)#[1, 5, 10]
#     y_ul = int(np.ceil(qalys_df['qalys'].max())) # upper label in the y-axis
#     ylims = [-0.1, y_ul+0.1] # limits of the y-axis (poster)
#     y_ticks = np.arange(0, y_ul+0.5, 0.5) # (poster)
#     axes_size = 26 # font size of axes labels
#     subtitle_size = 24 # font size for subplot titles
#     tick_size = 22 # font size for tick labels
#     legend_size = 22  # font size for legend labels
#     cat_labels = ['Normal', 'Elevated', 'Stage 1', 'Stage 2']
#
#     # Making figure
#     fig, axes = plt.subplots(nrows=1, ncols=2)
#     sns.lineplot(x="year", y="qalys", hue="policy", data=qalys_df[qalys_df['bp_cat']==cat_labels[2]], style="policy", markers=mks,
#                  dashes=False, palette=sns.color_palette("Greys", n_colors)[2:-1], ci=None, ax=axes[0])
#     sns.lineplot(x="year", y="qalys", hue="policy", data=qalys_df[qalys_df['bp_cat']==cat_labels[3]], style="policy", markers=mks,
#                  dashes=False, palette=sns.color_palette("Greys", n_colors)[2:-1], ci=None, legend=False, ax=axes[1])
#
#     # Figure Configuration
#     ## Configuration for the panel plot
#     fig.add_subplot(111, frameon=False)
#     plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
#     plt.grid(False)
#     plt.xlabel('Year', fontsize=axes_size, fontweight='semibold')
#     plt.ylabel('Expected Life-Years Saved\n(in Millions)', fontsize=axes_size, fontweight='semibold')
#
#     ## Axes configuration for every subplot
#     plt.setp(axes, xlim=xlims, xticks=x_ticks, xlabel='',
#              ylim=ylims, yticks=y_ticks, ylabel='')
#     fig.subplots_adjust(bottom=0.2, wspace=0.3)
#     for ax in fig.axes:
#         plt.sca(ax)
#         plt.xticks(fontsize=tick_size)
#         plt.yticks(fontsize=tick_size)
#
#     axes[0].set_title('Stage 1 Hypertension', fontsize=subtitle_size, fontweight='semibold')
#     axes[1].set_title('Stage 2 Hypertension', fontsize=subtitle_size, fontweight='semibold')
#
#     # Modifying Legend
#     handles, labels = axes[0].get_legend_handles_labels()
#     order = [0, 1, 3, 2, 4] # [0, 1, 3, 5, 2, 4]
#     handles = [x for _, x in sorted(zip(order, handles))]
#     labels = [x for _, x in sorted(zip(order, labels))]
#     axes[0].legend(loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(1.15, -0.15),
#                       handles=handles[1:], labels=labels[1:], prop={'size': legend_size})
#
#     # Printing plot
#     fig.set_size_inches(14, 7)
#     plt.savefig("QALYs Saved per Policy - 50-54.png", bbox_inches='tight', dpi=600)
#     plt.close()
#
#
# Talk figures (use seaborn paper context)
# ## Function to plot ranges of nearly optimal actions
# def plot_range_actions(meds_df, medication_range):
#
#     # Figure parameters
#     n_colors = 6  # number of colors in palette
#     x_ticks = range(2, 12, 2)
#     xlims = [0.5, 10.5] # limit of the x-axis
#     axes_size = 14 # font size of axes labels
#     subtitle_size = 12 # font size for subplot titles
#     tick_size = 10 # font size for tick labels
#     legend_size = 10 # font size for legend labels
#     mks = ['D', 'o', 'X'] # markers for plots # 'D', 'o', 'X'
#     marker_size = 30 # marker size
#     titles = ['54-year-old Male', '54-year-old Female', '54-year-old Male Smoker', '70-year-old Male'] # 70-year-old Male
#
#     # Making figure
#     for i in meds_df.id.unique():
#         fig, axes = plt.subplots(nrows=1, ncols=1)
#         sns.scatterplot(x="year", y="meds", hue="policy", data=meds_df[meds_df.id==i], markers=mks,
#                         style="policy", s=marker_size, linewidth=0.5, palette="viridis", #sns.color_palette("Greys", n_colors)[2:-1]
#                         ci=None, zorder=100) #, legend=False
#
#         ## Adding subtitles
#         axes.set_title(titles[i], fontsize=subtitle_size, fontweight='semibold')
#
#         # Figure Configuration
#         plt.xlabel('Year', fontsize=axes_size, fontweight='semibold')
#         plt.ylabel('Antihypertensive Medications', fontsize=axes_size, fontweight='semibold')
#
#         # fig.add_subplot(111, frameon=False)
#         # plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
#         # plt.grid(False)
#
#         ## Values for y-axes
#         all_meds = np.delete(np.round(np.arange(0, 5, 0.333333333), 2), 1)
#         all_labels = ['0 SD/0 HD', '0 SD/1 HD', '1 SD/0 HD',
#                       '0 SD/2 HD', '1 SD/1 HD', '2 SD/0 HD', # '2 SD/0 HD' = '0 SD/3 HD'
#                       '1 SD/2 HD', '2 SD/1 HD', '3 SD/0 HD', # '2 SD/1 HD' = '0 SD/4 HD'  # '3 SD/0 HD' = '1 SD/3 HD'
#                       '2 SD/2 HD', '3 SD/1 HD', '4 SD/0 HD', # '2 SD/2 HD' = '0 SD/5 HD'  # '3 SD/1 HD' = '1 SD/4 HD' # '4 SD/0 HD' = '2 SD/3 HD'
#                       '3 SD/2 HD', '4 SD/1 HD', '5 SD/0 HD']
#
#         ## Axes configuration for every subplot
#         fig.subplots_adjust(bottom=0.15)
#
#         # shaded area
#         plt.fill_between(x=np.arange(1, 11), y1=np.amin(medication_range[i], axis=0), y2=np.amax(medication_range[i], axis=0),
#                          color="lightskyblue", alpha=0.2, zorder=0) #sns.color_palette("Greys", n_colors)[-1]
#
#         # x-axis
#         plt.xticks(fontsize=tick_size)
#
#         # y-axis
#         obs_meds = np.concatenate([meds_df[meds_df.id==i].meds.to_numpy(), medication_range[i].min().to_numpy(), medication_range[i].max().to_numpy()])
#         yticks = np.round(np.arange(np.amin(obs_meds), np.amax(obs_meds), 0.333333333), 2)
#         yticks = yticks[yticks != 0.33] # Making sure that there is no tick between no treatment and a medication at half dosage
#         if yticks.shape[0] == 0:
#             yticks = [np.amin(obs_meds)]
#         ind = np.where([j in list(yticks) for j in list(all_meds)])[0]
#         ylabels = [all_labels[j] for j in ind.astype(int)]
#         axes.set_yticks(ticks=yticks)
#         axes.set_yticklabels(labels=ylabels, fontsize=tick_size)
#         axes.set_ylim((np.amin(yticks)-0.15, np.amax(yticks)+0.15))
#
#         # Modifying Legend
#         handles, labels = axes.get_legend_handles_labels()
#         axes.legend(loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.15),
#                     handles=handles[0:], labels=labels[0:], prop={'size': legend_size})
#
#         #Saving plot
#         fig.set_size_inches(6, 4)
#         plt.savefig('Ranges for Patient Profile '+str(i)+'.svg', bbox_inches='tight', dpi=600)
#         plt.close()

## Function to plot total life years saved over planning horizon by risk group (for talks and paper)
def plot_ly_saved(ly_df):

    # Figure parameters
    mks = ['D', 's', 'o', 'P', 'X']
    n_colors = 7  # number of colors in palette
    xlims = [0.5, 10.5] # limit of the x-axis
    x_ticks = range(2, 12, 2)#[1, 5, 10]
    y_ll = 0 # lower label in the y-axis
    y_ul = int(np.ceil(ly_df['ly'].max())) # upper label in the y-axis
    ylims = [y_ll-0.2, y_ul+0.2] # limits of the y-axis
    y_ticks = np.arange(y_ll, y_ul+0.5, 0.5)
    axes_size = 14 # font size of axes labels
    subtitle_size = 12 # font size for subplot titles
    tick_size = 10 # font size for tick labels
    legend_size = 10 # font size for legend labels
    cat_labels = ['Normal', 'Elevated', 'Stage 1', 'Stage 2']

    # Making figure
    # ## Single plot
    # fig, axes = plt.subplots(nrows=1, ncols=1)
    # sns.lineplot(x="year", y="ly", hue="policy", data=ly_df, style="policy", markers=mks,
    #              dashes=False, palette="viridis", ci=None) #sns.color_palette("Greys", n_colors)[1:-1]

    ## Panel plot
    fig, axes = plt.subplots(nrows=1, ncols=3)
    sns.lineplot(x="year", y="ly", hue="policy", data=ly_df[ly_df['bp_cat']==cat_labels[1]], style="policy", markers=mks,
                 dashes=False, palette=sns.color_palette("Greys", n_colors)[1:-1], ci=None, ax=axes[0]) #
    sns.lineplot(x="year", y="ly", hue="policy", data=ly_df[ly_df['bp_cat']==cat_labels[2]], style="policy", markers=mks,
                 dashes=False, palette=sns.color_palette("Greys", n_colors)[1:-1], ci=None, legend=False, ax=axes[1])
    sns.lineplot(x="year", y="ly", hue="policy", data=ly_df[ly_df['bp_cat']==cat_labels[3]], style="policy", markers=mks,
                 dashes=False, palette=sns.color_palette("Greys", n_colors)[1:-1], ci=None, legend=False, ax=axes[2])

    # Figure Configuration
    ## Configuration for the panel plot
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel('Year', fontsize=axes_size, fontweight='semibold')
    plt.ylabel('Expected Life-Years Saved\n(in Millions)', fontsize=axes_size, fontweight='semibold')

    ## Axes configuration for every subplot
    plt.setp(axes, xlim=xlims, xticks=x_ticks, xlabel='',
             ylim=ylims, yticks=y_ticks, ylabel='')
    fig.subplots_adjust(bottom=0.2, wspace=0.3)
    for ax in fig.axes:
        plt.sca(ax)
        plt.xticks(fontsize=tick_size)
        plt.yticks(fontsize=tick_size)

    axes[0].set_title('Elevated BP', fontsize=subtitle_size, fontweight='semibold')
    axes[1].set_title('Stage 1 Hypertension', fontsize=subtitle_size, fontweight='semibold')
    axes[2].set_title('Stage 2 Hypertension', fontsize=subtitle_size, fontweight='semibold')

    # Modifying Legend
    # ## Single plot
    # handles, labels = axes.get_legend_handles_labels()
    # order = [0, 1, 3, 5, 2, 4]
    # handles = [x for _, x in sorted(zip(order, handles))]
    # labels = [x for _, x in sorted(zip(order, labels))]
    # axes.legend(loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.15),
    #              handles=handles[1:], labels=labels[1:], prop={'size': legend_size})

    ## Panel plot
    handles, labels = axes[0].get_legend_handles_labels()
    order = [0, 1, 3, 5, 2, 4]
    handles = [x for _, x in sorted(zip(order, handles))]
    labels = [x for _, x in sorted(zip(order, labels))]
    axes[0].legend(loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(1.7, -0.2),
                      handles=handles[1:], labels=labels[1:], prop={'size': legend_size})

    # Printing plot
    fig.set_size_inches(7.5, 3) # for simplified panel plots in paper
    plt.savefig("Life Years Saved per Policy.pdf", bbox_inches='tight')
    # fig.set_size_inches(6, 4) # for single plot in talks
    # fig.set_size_inches(9, 4) # for panel plots in talks
    # plt.savefig("Life Years Saved per Policy.png", bbox_inches='tight', dpi=600)
    plt.close()

# ## Function to plot proportion of actions by policy convered in range by BP group (for talks)
# def plot_prop_covered(prop_df):
#
#     # Figure parameters
#     n_colors = 4  # number of colors in palette
#     xlims = [0.5, 10.5] # limit of the x-axis
#     x_ticks = range(2, 12, 2) #[1, 5, 10]
#     ylims = [-0.1, 1.1] # limits of the y-axis
#     y_ticks = [0, 0.5, 1]
#     axes_size = 14 # font size of axes labels
#     subtitle_size = 12 # font size for subplot titles
#     tick_size = 10 # font size for tick labels
#     legend_size = 10 # font size for legend labels
#     line_width = 1.5 # width for lines in plots
#
#     # Making figure
#     fig, ax = plt.subplots()
#     sns.lineplot(x="year", y="prop_cv", hue="bp_cat",
#                  data=prop_df, style="bp_cat", markers=False, dashes=True,
#                  palette=sns.color_palette("Greys", n_colors)[1:], ci=None) #"viridis"
#
#     # Figure Configuration
#     plt.xlabel('Year', fontsize=axes_size, fontweight='semibold')
#     plt.ylabel('Proportion of Patient-Years\nCovered in Range', fontsize=axes_size, fontweight='semibold')
#
#     for axes in fig.axes:
#         axes.set_yticklabels(['{:,.0%}'.format(x) for x in ax.get_yticks()])
#         plt.sca(axes)
#         plt.xticks(fontsize=tick_size)
#         plt.yticks(fontsize=tick_size)
#         plt.setp(axes.lines, linewidth=line_width)
#
#     # Modifying Legend
#     fig.subplots_adjust(bottom=0.15)
#     handles, _ = ax.get_legend_handles_labels()
#     labels = ['bp_cat', 'Elevated BP', 'Stage 1 Hypertension', 'Stage 2 Hypertension']
#     ax.legend(loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.15),
#                       handles=handles[1:], labels=labels[1:], prop={'size': legend_size})
#
#     fig.set_size_inches(6.5, 4.25) # for simplified paper plot
#     plt.savefig("Proportion of Patient-Years Covered in Range.pdf", bbox_inches='tight')
#     # fig.set_size_inches(6.5, 4.25) # for talks
#     # plt.savefig("Proportion of Patient-Years Covered in Range.png", bbox_inches='tight', dpi=600)
#     plt.close()
#
# ## Function to plot distribution of treatment at year 1 and 10 (for talks)
# def plot_trt_dist(trt_df):
#     # Figure parameters
#     n_colors = 6  # number of colors in palette
#     xlims = [-0.5, 4.5]  # limit of the x-axis
#     ylims = [-0.5, 5.5]  # limits of the y-axis
#     y_ticks = np.arange(6)
#     axes_size = 14  # font size of axes labels
#     subtitle_size = 12  # font size for subplot titles
#     tick_size = 10  # font size for tick labels
#     flierprops = dict(marker='o', markerfacecolor='none', markeredgecolor='black', markeredgewidth=0.5, markersize=2,
#                       linestyle='none')  # outliers properties
#
#     fig, axes = plt.subplots(nrows=1, ncols=1)
#     sns.boxplot(x="policy", y="meds", data=trt_df[(trt_df.year == 10)],
#                 palette=sns.color_palette("viridis", n_colors), linewidth=0.75, flierprops=flierprops)
#
#     # fig, axes = plt.subplots(nrows=1, ncols=2)
#     # sns.boxplot(x="policy", y="meds", data=trt_df[(trt_df.year == 1)],
#     #             palette=sns.color_palette("viridis", n_colors), linewidth=0.75, flierprops=flierprops, ax=axes[0])
#     # sns.boxplot(x="policy", y="meds", data=trt_df[(trt_df.year == 10)],
#     #             palette=sns.color_palette("viridis", n_colors), linewidth=0.75, flierprops=flierprops, ax=axes[1])
#
#     # Figure Configuration
#     ## Configuration for the panel plot
#     fig.add_subplot(111, frameon=False)
#     plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
#     plt.grid(False)
#     plt.xlabel('\nTreatment Strategy', fontsize=axes_size, fontweight='semibold')  # 'Risk Group'
#     plt.ylabel('Number of Medications', fontsize=axes_size, fontweight='semibold')
#
#     ## Axes configuration for every subplot
#     plt.setp(axes, xlim=xlims, xlabel='',
#              ylim=ylims, yticks=y_ticks, ylabel='')
#
#     ## Adding subtitles
#     ### Overall
#     # axes.set_title('Year 1', fontsize=subtitle_size, fontweight='semibold')
#     axes.set_title('Year 10', fontsize=subtitle_size, fontweight='semibold')
#
#     for ax in fig.axes:
#         plt.sca(ax)
#         plt.xticks(rotation=15, fontsize=tick_size-1, ha='center')
#         plt.yticks(fontsize=tick_size)
#
#     fig.set_size_inches(6.5, 4.25)
#     plt.savefig("Treatment per Policy-Year 10.png", bbox_inches='tight', dpi=600)
#     plt.close()
#
# # Functions for paper competitions
# # Function to plot ranges of nearly optimal actions (for paper competitions)
# def plot_range_actions_comp(meds_df, medication_range):
#
#     # Figure parameters
#     n_colors = 6  # number of colors in palette
#     x_ticks = range(2, 12, 2)
#     xlims = [0.5, 10.5] # limit of the x-axis
#     axes_size = 8 # font size of axes labels
#     subtitle_size = 7 # font size for subplot titles
#     tick_size = 7 # font size for tick labels
#     legend_size = 6 # font size for legend labels
#     mks = ['D', 'o', 'X'] # markers for plots
#     marker_size = 18 # marker size
#
#     # Making figure
#     fig, axes = plt.subplots(nrows=1, ncols=4)
#     sns.scatterplot(x="year", y="meds", hue="policy", data=meds_df[meds_df.id==0], markers=mks,
#                     style="policy", s=marker_size, linewidth=0.5, palette=sns.color_palette("Greys", n_colors)[2:-1],
#                     ci=None, ax=axes[0], zorder=100)
#     sns.scatterplot(x="year", y="meds", hue="policy", data=meds_df[meds_df.id==1], markers=mks,
#                     style="policy", s=marker_size, linewidth=0.5, palette=sns.color_palette("Greys", n_colors)[2:-1],
#                     ci=None, legend=False, ax=axes[1], zorder=100)
#     sns.scatterplot(x="year", y="meds", hue="policy", data=meds_df[meds_df.id==2], markers=mks,
#                     style="policy", s=marker_size, linewidth=0.5, palette=sns.color_palette("Greys", n_colors)[2:-1],
#                     ci=None, legend=False, ax=axes[2], zorder=100)
#     sns.scatterplot(x="year", y="meds", hue="policy", data=meds_df[meds_df.id==3], markers=mks,
#                     style="policy", s=marker_size, linewidth=0.5, palette=sns.color_palette("Greys", n_colors)[2:-1],
#                     ci=None, legend=False, ax=axes[3], zorder=100)
#
#     ## Adding subtitles
#     axes[0].set_title('54-year-old\nWhite Male', fontsize=subtitle_size, fontweight='semibold')
#     axes[1].set_title('54-year-old\nWhite Female', fontsize=subtitle_size, fontweight='semibold')
#     axes[2].set_title('54-year-old\nWhite Male Smoker', fontsize=subtitle_size, fontweight='semibold')
#     axes[3].set_title('70-year-old\nWhite Male', fontsize=subtitle_size, fontweight='semibold')
#
#     # Figure Configuration
#     ## Overall labels
#     fig.add_subplot(111, frameon=False)
#     plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
#     plt.grid(False)
#     plt.xlabel('Year', fontsize=axes_size, fontweight='semibold')
#     plt.ylabel('Antihypertensive\nMedications\n\n\n', fontsize=axes_size, fontweight='semibold')
#
#     ## Values for y-axes
#     all_meds = np.delete(np.round(np.arange(0, 5, 0.333333333), 2), 1)
#     all_labels = ['0 SD/0 HD', '0 SD/1 HD', '1 SD/0 HD',
#                   '0 SD/2 HD', '1 SD/1 HD', '2 SD/0 HD', # '2 SD/0 HD' = '0 SD/3 HD'
#                   '1 SD/2 HD', '2 SD/1 HD', '3 SD/0 HD', # '2 SD/1 HD' = '0 SD/4 HD'  # '3 SD/0 HD' = '1 SD/3 HD'
#                   '2 SD/2 HD', '3 SD/1 HD', '4 SD/0 HD', # '2 SD/2 HD' = '0 SD/5 HD'  # '3 SD/1 HD' = '1 SD/4 HD' # '4 SD/0 HD' = '2 SD/3 HD'
#                   '3 SD/2 HD', '4 SD/1 HD', '5 SD/0 HD']
#
#     ## Axes configuration for every subplot
#     plt.setp(axes, xlim=xlims, xticks=x_ticks, xlabel='', ylabel='')
#     fig.subplots_adjust(bottom=0.5)
#     obs_meds = np.concatenate([meds_df.meds.to_numpy(), np.concatenate([x.min().to_numpy() for x in medication_range]),
#                                np.concatenate([x.max().to_numpy() for x in medication_range])]) # same y axis across all plots (comment for individual axes and uncomment line inside loop)
#     for k, ax in list(enumerate(fig.axes))[:-1]:
#         plt.sca(ax)
#
#         # shaded area
#         plt.fill_between(x=np.arange(1, 11), y1=np.amin(medication_range[k], axis=0), y2=np.amax(medication_range[k], axis=0),
#                          color=sns.color_palette("Greys", n_colors)[0], zorder=0)
#
#         # x-axis
#         plt.xticks(fontsize=tick_size)
#
#         # y-axis
#         if k == 0:
#             yticks = np.round(np.arange(np.amin(obs_meds), np.amax(obs_meds), 0.333333333), 2)
#             yticks = yticks[yticks != 0.33] # Making sure that there is no tick between no treatment and a medication at half dosage
#             if yticks.shape[0] == 0:
#                 yticks = [np.amin(obs_meds)]
#             ind = np.where([j in list(yticks) for j in list(all_meds)])[0]
#             ylabels = [all_labels[j] for j in ind.astype(int)]
#
#             ax.set_yticks(ticks=yticks)
#             ax.set_yticklabels(labels=ylabels, fontsize=tick_size)
#             ax.set_ylim((np.amin(yticks)-0.15, np.amax(yticks)+0.15))
#         else:
#             ax.set_yticks(ticks=[])
#             ax.set_yticklabels(labels=[], fontsize=tick_size)
#             ax.set_ylim((np.amin(yticks)-0.15, np.amax(yticks)+0.15))
#
#     # Modifying Legend
#     handles, labels = axes[0].get_legend_handles_labels()
#     axes[0].legend(loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(2.5, -0.5),
#                       handles=handles[1:], labels=labels[1:], prop={'size': legend_size})
#
#     #Saving plot
#     fig.set_size_inches(6.5, 2.5)
#     plt.savefig('Ranges for Patient Profiles - competitions.pdf', bbox_inches='tight')
#     plt.close()
#
# # Function to plot total QALYs saved over planning horizon (for paper competitions)
# def plot_qalys_saved_comp(qalys_df):
#
#     # Figure parameters
#     mks = ['D', 's', 'o', 'P', 'X']
#     n_colors = 7  # number of colors in palette
#     xlims = [0.5, 10.5] # limit of the x-axis
#     x_ticks = range(2, 12, 2)#[1, 5, 10]
#     y_ll = 0 # lower label in the y-axis
#     y_ul = int(np.ceil(qalys_df['qalys'].max())) # upper label in the y-axis
#     ylims = [y_ll-0.2, y_ul+0.2] # limits of the y-axis
#     y_ticks = np.arange(y_ll, y_ul+0.5, 0.5)
#     axes_size = 9 # font size of axes labels
#     subtitle_size = 8 # font size for subplot titles
#     tick_size = 8 # font size for tick labels
#     legend_size = 7 # font size for legend labels
#     cat_labels = ['Normal', 'Elevated', 'Stage 1', 'Stage 2']
#     marker_size = 5  # marker size
#
#     # Making figure
#     # fig, axes = plt.subplots(nrows=1, ncols=1)
#     # sns.lineplot(x="year", y="qalys", hue="policy", data=qalys_df, style="policy", markers=mks,
#     #              dashes=False, palette="viridis", ci=None) #sns.color_palette("Greys", n_colors)[1:-1]
#
#     fig, axes = plt.subplots(nrows=1, ncols=3)
#     sns.lineplot(x="year", y="qalys", hue="policy", data=qalys_df[qalys_df['bp_cat']==cat_labels[1]], style="policy", markers=mks, markersize=marker_size, markeredgewidth=0.5,
#                  dashes=False, palette=sns.color_palette("Greys", n_colors)[1:-1], ci=None, ax=axes[0]) #
#     sns.lineplot(x="year", y="qalys", hue="policy", data=qalys_df[qalys_df['bp_cat']==cat_labels[2]], style="policy", markers=mks, markersize=marker_size, markeredgewidth=0.5,
#                  dashes=False, palette=sns.color_palette("Greys", n_colors)[1:-1], ci=None, legend=False, ax=axes[1])
#     sns.lineplot(x="year", y="qalys", hue="policy", data=qalys_df[qalys_df['bp_cat']==cat_labels[3]], style="policy", markers=mks, markersize=marker_size, markeredgewidth=0.5,
#                  dashes=False, palette=sns.color_palette("Greys", n_colors)[1:-1], ci=None, legend=False, ax=axes[2])
#
#     # Figure Configuration
#     ## Configuration for the panel plot
#     fig.add_subplot(111, frameon=False)
#     plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
#     plt.grid(False)
#     plt.xlabel('Year', fontsize=axes_size, fontweight='semibold')
#     plt.ylabel('Expected Life-Years Saved\n(in Millions)', fontsize=axes_size, fontweight='semibold')
#
#     ## Axes configuration for every subplot
#     plt.setp(axes, xlim=xlims, xticks=x_ticks, xlabel='',
#              ylim=ylims, yticks=y_ticks, ylabel='')
#     fig.subplots_adjust(bottom=0.2)
#     for k, ax in list(enumerate(fig.axes))[:-1]:
#         plt.sca(ax)
#         if k == 0:
#             plt.xticks(fontsize=tick_size)
#             plt.yticks(fontsize=tick_size)
#         else:
#             plt.xticks(fontsize=tick_size)
#             ax.set_yticks(ticks=[])
#             ax.set_yticklabels(labels=[], fontsize=tick_size)
#
#     axes[0].set_title('Elevated BP', fontsize=subtitle_size, fontweight='semibold')
#     axes[1].set_title('Stage 1 Hypertension', fontsize=subtitle_size, fontweight='semibold')
#     axes[2].set_title('Stage 2 Hypertension', fontsize=subtitle_size, fontweight='semibold')
#
#     # Modifying Legend
#     handles, labels = axes[0].get_legend_handles_labels()
#     order = [0, 1, 3, 5, 2, 4]
#     handles = [x for _, x in sorted(zip(order, handles))]
#     labels = [x for _, x in sorted(zip(order, labels))]
#     axes[0].legend(loc='upper center', ncol=5, frameon=False, bbox_to_anchor=(1.7, -0.3),
#                       handles=handles[1:], labels=labels[1:], prop={'size': legend_size})
#
#     # Printing plot
#     fig.set_size_inches(6.5, 2) # for simplified panel plots in paper
#     plt.savefig("QALYs Saved per Policy - competitions.pdf", bbox_inches='tight')
#     plt.close()
