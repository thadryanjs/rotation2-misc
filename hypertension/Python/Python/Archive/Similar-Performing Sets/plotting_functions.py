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
from hypertension_treatment_policy_evaluation import numhealth

# Plotting parameters
sns.set_style("ticks")
paper_rc = {'lines.linewidth': 1, 'lines.markersize': 6, 'font.size': 12}
sns.set_context("paper", rc=paper_rc) # use also for talks

## Function to plot convergence of confidence interval width with different number of batches and a fix number of observations per batch
def plf_cov_batch(ci_width_max, plot_batches, selected_batches):

    # Figure parameters
    axes_size = 10 # font size of axes labels
    tick_size = 8 # font size for tick labels

    # Making figure
    fig = plt.figure()
    plt.plot(ci_width_max[:plot_batches], color='black')

    # Modifying Axes
    plt.xlabel('Number of Batches', fontsize=axes_size, fontweight='semibold')
    plt.ylabel('Confidence Interval\nHalf Width', fontsize=axes_size, fontweight='semibold')
    plt.xticks([1, selected_batches, 500], fontsize=tick_size)
    plt.yticks(ticks=[int(np.floor(min(ci_width_max)*1000))/1000, int(np.floor(max(ci_width_max)*100))/100], fontsize=tick_size)
    plt.ylim(0, int(np.floor(max(ci_width_max)*10))/10+0.05)

    # Adding horizontal line for reference number of batches
    plt.hlines(y=ci_width_max[-1], xmin=1, xmax=plot_batches-1, color='red', alpha=0.60, zorder=100)
    plt.text(x=selected_batches, y=ci_width_max[selected_batches]+0.01, s=str(ci_width_max[selected_batches].round(3)), color='gray', fontsize=tick_size)
    plt.vlines(x=selected_batches, ymin=-0.05, ymax=ci_width_max[selected_batches], color='gray', linestyle='--', zorder=200)

    # Saving plot
    fig.set_size_inches(3, 2.5)
    plt.savefig('Number of Batches Analysis.pdf', bbox_inches='tight')
    plt.close()

## Function to plot sets of similar-performing actions for all patient profiles (exploring to identify "nice" looking plots)
def plot_range_actions_exp(ptid, meds_aha, Pi_meds):

    # # Line for debugging purposes
    # ptid = 291; Pi_meds = medication_range[ptid]; meds_aha = meds_df.meds[meds_df.id==ptid]

    # Plotting parameters
    n_colors = 7
    palette = sns.color_palette("Greys", n_colors).as_hex()[1:-1]
    axes_size = 26 # font size of axes labels
    tick_size = 22 # font size for tick labels
    legend_size = 22  # font size for legend labels

    # Plotting policies
    fig_name_policy = 'Policy for Patient ' + str(ptid)
    Pi_meds_melt = Pi_meds.melt()
    Pi_meds_melt.columns = ["Year", "Set"]

    # Ensuring that AHA's guidelines' recommedations are feasible (fix for bug in aha_2017_guideline.py)
    max_meds = np.amax(Pi_meds, axis=0).to_numpy(); min_meds = np.amin(Pi_meds, axis=0).to_numpy()
    meds_aha = np.where(meds_aha > max_meds, max_meds, meds_aha)
    meds_aha = np.where(meds_aha < min_meds, min_meds, meds_aha)

    # Creating plot
    fig = plt.figure()
    lines = plt.plot(range(10), meds_aha, 'X', range(10), min_meds, "",
                     range(10), max_meds, "")
    plt.setp(lines[0], color=palette[3], zorder=3)
    plt.setp(lines[1], linewidth=1, color="darkgray", linestyle='dashed', zorder=1, label='_nolegend_')
    plt.setp(lines[2], linewidth=1, color="darkgray", linestyle='dashed', zorder=1, label='_nolegend_')

    ### Modifying title
    plt.title("Patient " + str(ptid), fontsize=axes_size)

    ### Modifying axes
    plt.xticks(range(10), range(1, 11))
    obs_meds = np.array([meds_aha, np.array(Pi_meds.min()), np.array(Pi_meds.max())])
    yticks = np.round(np.arange(obs_meds.min(), obs_meds.max(), 0.333333333), 2)
    yticks = yticks[yticks != 0.33] # Making sure that there is no tick between no treatment and a medication at half dosage
    if len(yticks) == 0:
        yticks = [obs_meds.min()]
    all_labels = ['0 SD/0 HD', '0 SD/1 HD', '1 SD/0 HD',
                  '0 SD/2 HD', '1 SD/1 HD', '2 SD/0 HD', # '2 SD/0 HD' = '0 SD/3 HD'
                  '1 SD/2 HD', '2 SD/1 HD', '3 SD/0 HD', # '2 SD/1 HD' = '0 SD/4 HD'  # '3 SD/0 HD' = '1 SD/3 HD'
                  '2 SD/2 HD', '3 SD/1 HD', '4 SD/0 HD', # '2 SD/2 HD' = '0 SD/5 HD'  # '3 SD/1 HD' = '1 SD/4 HD' # '4 SD/0 HD' = '2 SD/3 HD'
                  '3 SD/2 HD', '4 SD/1 HD', '5 SD/0 HD']

    #### Converting medications to index of labels
    all_meds = np.delete(np.round(np.arange(0, 5, 0.333333333), 2), 1)
    ind = np.where([j in list(yticks) for j in list(all_meds)])[0]
    ylabels = [all_labels[j] for j in ind.astype(int)]

    plt.xticks(fontsize=tick_size)
    plt.yticks(yticks, labels=ylabels, fontsize=tick_size)
    plt.ylim((np.amin(yticks)-0.1, np.amax(yticks)+0.1))
    ax = plt.gca()
    plt.xlabel('Year', fontsize=axes_size, fontweight='semibold')
    plt.ylabel('Antihypertensive\nMedications', fontsize=axes_size, fontweight='semibold')

    ### Adding shaded area
    plt.fill_between(x=range(10), y1=np.amin(Pi_meds, axis=0), y2=np.amax(Pi_meds, axis=0), color=palette[2],
                     alpha=0.25)

    #Saving plot
    fig.set_size_inches(6.5, 4.25) # for paper
    plt.savefig(fig_name_policy + '.png', bbox_inches='tight', dpi=1200)
    plt.close()

## Function to plot ranges of similar-performing actions for selected profiles
def plot_range_actions(meds_sel, medication_range_sel):

    # Figure parameters
    n_colors = 6  # number of colors in palette
    x_ticks = range(2, 12, 2)
    xlims = [0.5, 10.5] # limit of the x-axis
    axes_size = 8 # font size of axes labels
    tick_size = 6 # font size for tick labels
    palette = sns.color_palette("Greys", n_colors).as_hex()[1:-1]
    marker_size = 3 # marker size
    subtitles = ['40-Year-Old Patient with\nElevated BP', '40-Year-Old Patient with\nStage 1 Hypertension',
                 '40-Year-Old Patient with\nStage 2 Hypertension', '60-Year-Old Patient with\nStage 2 Hypertension']
    subtitle_size = 7  # font size for subplot titles
    all_meds = np.delete(np.round(np.arange(0, 5, 0.333333333), 2), 1)
    all_labels = ['0 SD/0 HD', '0 SD/1 HD', '1 SD/0 HD',
                  '0 SD/2 HD', '1 SD/1 HD', '2 SD/0 HD',  # '2 SD/0 HD' = '0 SD/3 HD'
                  '1 SD/2 HD', '2 SD/1 HD', '3 SD/0 HD',  # '2 SD/1 HD' = '0 SD/4 HD'  # '3 SD/0 HD' = '1 SD/3 HD'
                  '2 SD/2 HD', '3 SD/1 HD', '4 SD/0 HD', # '2 SD/2 HD' = '0 SD/5 HD'  # '3 SD/1 HD' = '1 SD/4 HD' # '4 SD/0 HD' = '2 SD/3 HD'
                  '3 SD/2 HD', '4 SD/1 HD', '5 SD/0 HD']
    ids = np.unique(meds_sel.id)

    # Making figure
    fig, axes = plt.subplots(nrows=1, ncols=4)

    # Figure Configuration
    ## Overall labels
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel('Year', fontsize=axes_size, fontweight='semibold')
    plt.ylabel('Antihypertensive\nMedications\n\n', fontsize=axes_size, fontweight='semibold')

    ## Axes configuration for every subplot
    plt.setp(axes, xlim=xlims, xticks=x_ticks, xlabel='', ylabel='')
    fig.subplots_adjust(bottom=0.5)
    obs_meds = np.concatenate([meds_sel.meds.to_numpy(), np.concatenate([x.min().to_numpy() for x in medication_range_sel]),
                               np.concatenate([x.max().to_numpy() for x in medication_range_sel])]) # same y axis across all plots (comment for individual axes and uncomment line inside loop)
    for k, ax in list(enumerate(fig.axes))[:-1]:
        plt.sca(ax)

        # Ensuring that AHA's guidelines' recommedations are feasible (fix for bug in aha_2017_guideline.py)
        meds_aha = meds_sel.loc[meds_sel.id==ids[k], 'meds'].to_numpy()
        max_meds = np.amax(medication_range_sel[k], axis=0).to_numpy()
        min_meds = np.amin(medication_range_sel[k], axis=0).to_numpy()
        meds_aha = np.where(meds_aha > max_meds, max_meds, meds_aha)
        meds_aha = np.where(meds_aha < min_meds, min_meds, meds_aha)

        # Creating plot
        lines = axes[k].plot(range(1, 11), meds_aha, 'o', range(1, 11), min_meds, "", range(1, 11), max_meds, "")
        plt.setp(lines[0], color=palette[3], zorder=3, markersize=marker_size)
        plt.setp(lines[1], linewidth=0.75, color="darkgray", linestyle='dashed', zorder=1, label='_nolegend_')
        plt.setp(lines[2], linewidth=0.75, color="darkgray", linestyle='dashed', zorder=1, label='_nolegend_')

        #Adding subtitles
        axes[k].set_title(subtitles[k], fontsize=subtitle_size, fontweight='semibold')

        # shaded area
        plt.fill_between(x=np.arange(1, 11), y1=np.amin(medication_range_sel[k], axis=0), y2=np.amax(medication_range_sel[k], axis=0),
                         color=sns.color_palette("Greys", n_colors)[0], zorder=0)

        # x-axis
        plt.xticks(fontsize=tick_size)

        # y-axis
        if k == 0:
            yticks = np.round(np.arange(np.amin(obs_meds), np.amax(obs_meds), 0.333333333), 2)
            yticks = yticks[yticks != 0.33] # Making sure that there is no tick between no treatment and a medication at half dosage
            if yticks.shape[0] == 0:
                yticks = [np.amin(obs_meds)]
            ind = np.where([j in list(yticks) for j in list(all_meds)])[0]
            ylabels = [all_labels[j] for j in ind.astype(int)]

            ax.set_yticks(ticks=yticks)
            ax.set_yticklabels(labels=ylabels, fontsize=tick_size)
            ax.set_ylim((np.amin(yticks)-0.15, np.amax(yticks)+0.15))
        else:
            ax.set_yticks(ticks=[])
            ax.set_yticklabels(labels=[], fontsize=tick_size)
            ax.set_ylim((np.amin(yticks)-0.15, np.amax(yticks)+0.15))

    #Saving plot
    fig.set_size_inches(6.5, 3)
    plt.savefig('Sets for Patient Profiles.pdf', bbox_inches='tight')
    plt.close()

## Function to plot sets of similar-performing actions for a single patient profile (for SMDM 2022 abstract)
def plot_range_actions_smdm(meds_aha, Pi_meds):

    # Plotting parameters
    n_colors = 7
    palette = sns.color_palette("Greys", n_colors).as_hex()[1:-1]
    axes_size = 12 # font size of axes labels
    tick_size = 10 # font size for tick labels
    legend_size = 10  # font size for legend labels

    # Plotting policies
    Pi_meds_melt = Pi_meds.melt()
    Pi_meds_melt.columns = ["Year", "Set"]

    # Ensuring that AHA's guidelines' recommedations are feasible (fix for bug in aha_2017_guideline.py)
    max_meds = np.amax(Pi_meds, axis=0).to_numpy(); min_meds = np.amin(Pi_meds, axis=0).to_numpy()
    meds_aha = np.where(meds_aha > max_meds, max_meds, meds_aha)
    meds_aha = np.where(meds_aha < min_meds, min_meds, meds_aha)

    # #Creating plot
    fig = plt.figure()
    lines = plt.plot(range(10), meds_aha, 'X', range(10), min_meds, "",
                     range(10), max_meds, "")
    plt.setp(lines[0], color=palette[3], zorder=3)
    plt.setp(lines[1], linewidth=1, color="darkgray", linestyle='dashed', zorder=1, label='_nolegend_')
    plt.setp(lines[2], linewidth=1, color="darkgray", linestyle='dashed', zorder=1, label='_nolegend_')

    ### Modifying title
    plt.title('Figure 1: Sets of Treatment Choices for a 40-Year-Old Patient\nwith Stage 2 Hypertension', fontsize=axes_size,
              fontweight='semibold', loc='left')
    # plt.suptitle('\nGray shaded area represents treatment choices containted in the sets', fontsize=axes_size)

    ### Modifying axes
    plt.xticks(range(10), range(1, 11))
    obs_meds = np.array([meds_aha, np.array(Pi_meds.min()), np.array(Pi_meds.max())])
    yticks = np.round(np.arange(obs_meds.min(), obs_meds.max(), 0.333333333), 2)
    yticks = yticks[yticks != 0.33] # Making sure that there is no tick between no treatment and a medication at half dosage
    if len(yticks) == 0:
        yticks = [obs_meds.min()]
    all_labels = ['3 SD/0 HD', # '2 SD/1 HD' = '0 SD/4 HD'  # '3 SD/0 HD' = '1 SD/3 HD'
                  '2 SD/2 HD', '3 SD/1 HD', '4 SD/0 HD', # '2 SD/2 HD' = '0 SD/5 HD'  # '3 SD/1 HD' = '1 SD/4 HD' # '4 SD/0 HD' = '2 SD/3 HD'
                  '3 SD/2 HD', '4 SD/1 HD'] #, '5 SD/0 HD'

    #### Converting medications to index of labels
    all_meds = np.round(np.arange(3, 5, 0.333333333), 2)
    ind = np.where([j in list(yticks) for j in list(all_meds)])[0]
    ylabels = [all_labels[j] for j in ind.astype(int)]

    plt.xticks(fontsize=tick_size)
    plt.yticks(yticks, labels=ylabels, fontsize=tick_size)
    plt.ylim((np.amin(yticks)-0.1, np.amax(yticks)+0.1))
    plt.xlabel('Year', fontsize=axes_size, fontweight='semibold')
    plt.ylabel('Number of Antihypertensive Medications\nat Standard Dose (SD) and Half Dose (HS)', fontsize=axes_size, fontweight='semibold')

    ### Adding shaded area
    plt.fill_between(x=range(10), y1=np.amin(Pi_meds, axis=0), y2=np.amax(Pi_meds, axis=0), color=palette[2],
                     alpha=0.25)

    #### Modifying Legend
    ax = plt.gca()
    ax.legend(('Hypertension Treatment Guidelines', 'Set of Treatment Choices'), loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False, prop={'size': legend_size})

    #Saving plot
    fig.set_size_inches(6.5, 4.25) # for paper
    plt.savefig('plot_smdm_2022.png', bbox_inches='tight', dpi=300)
    plt.close()

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
    # fig.set_size_inches(7.5, 3) # for simplified panel plots in paper
    # plt.savefig("Life Years Saved per Policy.pdf", bbox_inches='tight')
    # fig.set_size_inches(6, 4) # for single plot in talks
    # fig.set_size_inches(9, 4) # for panel plots in talks
    fig.tight_layout()
    plt.subplots_adjust(left=0.2,
                        bottom=0.3,
                        right=1,
                        top=0.9,
                        wspace=0,
                        hspace=0)
    plt.savefig("Life Years Saved per Policy.png", bbox_inches='tight', dpi=1200)
    plt.close()

#-# Function to plot simplified distribution of treatment by risk group at year 10 -Mateo
#-# Shows number of medications given by Best in Set, Median in Set, Fewest in Set, and Clinical Guideline treatments -Mateo
#-# Used in ASURE Poster Presentation 2022 -Mateo
def plot_simplified_trt_dist(trt_df):
    # Figure parameters
    n_colors = 6  # number of colors in palette
    xlims = [-0.5, 2.5]  # limit of the x-axis
    ylims = [-0.5, 5.5]  # limits of the y-axis
    y_ticks = np.arange(6)
    axes_size = 14  # font size of axes labels
    subtitle_size = 14  # font size for subplot titles
    tick_size = 16  # font size for tick labels
    legend_size = 14  # font size for legend labels
    cat_labels = trt_df.bp_cat.unique()  # BP categories
    flierprops = dict(marker='o', markerfacecolor='none', markeredgecolor='black', markeredgewidth=0.5, markersize=2,
                      linestyle='none')  # outliers properties

    # Overall
    fig, axes = plt.subplots(nrows=1, ncols=1)
    sns.boxplot(x="bp_cat", y="meds", hue="policy", data=trt_df[(trt_df.year == 10)],
                palette=[(0.00392156862745098, 0.45098039215686275, 0.6980392156862745),
                         (0.8705882352941177, 0.5607843137254902, 0.0196078431372549),
                         (0.8352941176470589, 0.3686274509803922, 0.0),
                         (0.00784313725490196, 0.6196078431372549, 0.45098039215686275),
                         (0.8, 0.47058823529411764, 0.7372549019607844),
                         (0.792156862745098, 0.5686274509803921, 0.3803921568627451),
                         (0.984313725490196, 0.6862745098039216, 0.8941176470588236),
                         (0.5803921568627451, 0.5803921568627451, 0.5803921568627451),
                         (0.9254901960784314, 0.8823529411764706, 0.2),
                         (0.33725490196078434, 0.7058823529411765, 0.9137254901960784)],
                linewidth=0.75, flierprops=flierprops, ax=axes, order=cat_labels) #-# Palette manually defined to use colorblind safe colors in correct order (to match other figures)

    # Figure Configuration
    ## Configuration for the panel plot
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel('BP Category', fontsize=axes_size, fontweight='semibold')  # 'Risk Group'
    plt.ylabel('Number of Medications', fontsize=axes_size, fontweight='semibold')

    ## Axes configuration for every subplot
    plt.setp(axes, xlim=xlims, xlabel='',
             ylim=ylims, yticks=y_ticks, ylabel='')
    fig.subplots_adjust(bottom=0.3, wspace=0.2, hspace=0.45)

    ## Adding subtitles
    bp_labels = ['Elevated BP', 'Stage 1\nHypertension', 'Stage 2\nHypertension']
    axes.set_title('Year 10', fontsize=subtitle_size, fontweight='semibold')
    # axes[1].set_title('Year 10', fontsize=subtitle_size, fontweight='semibold')
    fig.subplots_adjust(bottom=0.3)  # for overall plots

    for ax in fig.axes:
        plt.sca(ax)
        # plt.xticks(fontsize=tick_size-1, ticks=np.arange(len(bp_labels)), labels=bp_labels, ha='center', rotation=15) # for overall plots
        plt.xticks(fontsize=tick_size - 1, ticks=np.arange(len(cat_labels)), labels=cat_labels,
                   ha='center')  # rotation=15, # for plots by group
        plt.yticks(fontsize=tick_size + 1)

    # # Adding patterns to plots
    # hatches = ["-", "/", "+", "\\", "."]
    # for ax in fig.axes:
    #     for hatch, patch in zip(hatches, ax.artists):
    #         patch.set_hatch(hatch)

    # Modifying Legend
    # ## For overall plot
    handles, labels = axes.get_legend_handles_labels()
    axes.legend(loc='upper center', ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.15),
                handles=handles, labels=labels, prop={'size': legend_size}) #-# Moving axes to below plot -Mateo
    # axes[1].get_legend().remove()

    # fig.set_size_inches(6.5, 3.5) # overall
    # plt.savefig("Treatment per Policy.pdf", bbox_inches='tight')
    fig.set_size_inches(6.5, 6.5)  # by sex/race
    plt.savefig("Treatment Boxplot.png", bbox_inches='tight', dpi=1200)
    plt.show()
    plt.close()

#-# Function to plot per capita life years saved over planning horizon by risk group -Mateo
#-# Shows LY Saved for Best in Set, Median in Set, Fewest in Set, and Clinical Guideline treatments -Mateo
#-# Used in ASURE Poster Presentation 2022 -Mateo
def plot_ly_saved_bp(ly_df, ptdata1):
    # Figure parameters
    mks = ['P', 'D', 's', '*'] #-# Setting specific markers to avoid overlapping markers on figure -Mateo
    n_colors = 6  # number of colors in palette
    xlims = [0.5, 10.5]  # limit of the x-axis
    x_ticks = range(2, 12, 2)  # [1, 5, 10]
    y_ll = 0  # lower label in the y-axis
    y_ul = int(np.ceil(ly_df['ly'].max()))  # upper label in the y-axis
    ylims = [y_ll - 0.2, y_ul + 0.2]  # limits of the y-axis
    y_ticks = np.arange(y_ll, y_ul + 0.5, 0.5)
    axes_size = 12  # font size of axes labels
    subtitle_size = 12  # font size for subplot titles
    tick_size = 14  # font size for tick labels
    legend_size = 12  # font size for legend labels
    cat_labels = ['Normal', 'Elevated', 'Stage 1', 'Stage 2']

    # Making figure
    # ## Single plot
    # fig, axes = plt.subplots(nrows=1, ncols=1)
    # sns.lineplot(x="year", y="ly", hue="policy", data=ly_df, style="policy", markers=mks,
    #              dashes=False, palette="viridis", ci=None) #sns.color_palette("Greys", n_colors)[1:-1]

    #-# Calculating LY Saved per capita (per 1000) -Mateo
    #-# Sum the weights of each patient in each category to determine the equivalent amount of people each category represents -Mateo
    populationElevated = ptdata1.groupby(by=['bp_cat'])['wt'].sum()['Elevated'] #-# Sum of weights of patients in "Elevated" Category -Mateo
    populationStage1 = ptdata1.groupby(by=['bp_cat'])['wt'].sum()['Stage 1']    #-# Sum of weights of patients in "Stage 1" Category -Mateo
    populationStage2 = ptdata1.groupby(by=['bp_cat'])['wt'].sum()['Stage 2']    #-# Sum of weights of patients in "Stage 2" Category -Mateo

    #-# Divide life years in category by population of that category, multiply by 1000 to get rate per 1k people.
    #-# Serves as scaled y-axis for each category
    y_Elevated = ly_df[ly_df['bp_cat'] == cat_labels[1]]['ly'] / populationElevated * 1000  #-# Scaled y-axis for "Elevated" Category -Mateo
    y_Stage1 = ly_df[ly_df['bp_cat'] == cat_labels[2]]['ly'] / populationStage1 * 1000      #-# Scaled y-axis for "Stage 1" Category -Mateo
    y_Stage2 = ly_df[ly_df['bp_cat'] == cat_labels[3]]['ly'] / populationStage2 * 1000      #-# Scaled y-axis for "Stage 2" Category -Mateo

    ## Panel plot
    fig, axes = plt.subplots(nrows=1, ncols=3)
    #-# Plot "Elevated" Category -Mateo
    sns.lineplot(x="year", y=y_Elevated, hue="policy", data=ly_df[ly_df['bp_cat'] == cat_labels[1]], style="policy",
                 markers=mks, markersize=7,
                 dashes=False, palette=[(0.00392156862745098, 0.45098039215686275, 0.6980392156862745),
                                        (0.8705882352941177, 0.5607843137254902, 0.0196078431372549),
                                        (0.8352941176470589, 0.3686274509803922, 0.0),
                                        (0.00784313725490196, 0.6196078431372549, 0.45098039215686275)],
                 ci=None, ax=axes[0])  #-# Palette manually defined to use colorblind safe colors in correct order (to match other figures) -Mateo
    #-# Plot "Stage 1" Category -Mateo
    sns.lineplot(x="year", y=y_Stage1, hue="policy", data=ly_df[ly_df['bp_cat'] == cat_labels[2]], style="policy",
                 markers=mks, markersize=7,
                 dashes=False, palette=[(0.00392156862745098, 0.45098039215686275, 0.6980392156862745),
                                        (0.8705882352941177, 0.5607843137254902, 0.0196078431372549),
                                        (0.8352941176470589, 0.3686274509803922, 0.0),
                                        (0.00784313725490196, 0.6196078431372549, 0.45098039215686275)], ci=None,
                 legend=False, ax=axes[1]) #-# Palette manually defined to use colorblind safe colors in correct order (to match other figures) -Mateo
    #-# Plot "Stage 2" Category -Mateo
    sns.lineplot(x="year", y=y_Stage2, hue="policy", data=ly_df[ly_df['bp_cat'] == cat_labels[3]], style="policy",
                 markers=mks, markersize=7,
                 dashes=False, palette=[(0.00392156862745098, 0.45098039215686275, 0.6980392156862745),
                                        (0.8705882352941177, 0.5607843137254902, 0.0196078431372549),
                                        (0.8352941176470589, 0.3686274509803922, 0.0),
                                        (0.00784313725490196, 0.6196078431372549, 0.45098039215686275)], ci=None,
                 legend=False, ax=axes[2]) #-# Palette manually defined to use colorblind safe colors in correct order (to match other figures) -Mateo

    # Figure Configuration
    ## Configuration for the panel plot
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel('Year', fontsize=axes_size, fontweight='semibold')
    plt.ylabel('Expected Life-Years Saved\n(per 1000 people)', fontsize=axes_size, fontweight='semibold')

    ## Axes configuration for every subplot
    plt.setp(axes, xlim=xlims)
    plt.setp(axes, xticks=x_ticks)
    plt.setp(axes, xlabel='', ylabel='')

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
    axes[0].legend(loc='upper center', ncol=4, frameon=False, bbox_to_anchor=(1.7, -0.2),
                   handles=handles, labels=labels, prop={'size': legend_size})
    # axes[0].set_box_aspect(1)
    # Printing plot
    fig.set_size_inches(7.5, 3)  # for simplified panel plots in paper
    # plt.savefig("Life Years Saved per Policy.pdf", bbox_inches='tight')
    plt.savefig("Life Years Saved per Policy BP.png", bbox_inches='tight', dpi=1200)
    plt.show()
    # fig.set_size_inches(6, 4) # for single plot in talks
    # fig.set_size_inches(9, 4) # for panel plots in talks
    # plt.savefig("Life Years Saved per Policy.png", bbox_inches='tight', dpi=600)

#-# Function to plot total life years saved over planning horizon by racial group -Mateo
#-# Shows LY Saved for Best in Set, Median in Set, Fewest in Set, and Clinical Guideline treatments -Mateo
#-# Planned to use in ASURE Poster Presentation 2022, but ended up replacing with other figure -Mateo
def plot_ly_saved_race(ly_df):
    # Figure parameters
    mks = ['P', 'D', 's', '*'] #-# Setting specific markers to avoid overlapping markers on figure -Mateo
    n_colors = 6  # number of colors in palette
    xlims = [0.5, 10.5]  # limit of the x-axis
    x_ticks = range(2, 12, 2)  # [1, 5, 10]
    y_ll = 0  # lower label in the y-axis
    y_ul = int(np.ceil(ly_df['ly'].max()))  # upper label in the y-axis
    ylims = [y_ll - 0.2, y_ul + 0.2]  # limits of the y-axis
    y_ticks = np.arange(y_ll, y_ul + 0.5, 0.5)
    axes_size = 14  # font size of axes labels
    subtitle_size = 12  # font size for subplot titles
    tick_size = 10  # font size for tick labels
    legend_size = 10  # font size for legend labels
    cat_labels = ['Black', 'Non-Black']

    # Making figure
    # ## Single plot
    # fig, axes = plt.subplots(nrows=1, ncols=1)
    # sns.lineplot(x="year", y="ly", hue="policy", data=ly_df, style="policy", markers=mks,
    #              dashes=False, palette="viridis", ci=None) #sns.color_palette("Greys", n_colors)[1:-1]

    ## Panel plot
    fig, axes = plt.subplots(nrows=1, ncols=2)
    #-# Plot "Black" Category -Mateo
    sns.lineplot(x="year", y="ly", hue="policy", data=ly_df[ly_df['race'] == 0], style="policy", markers=mks,
                 markersize=7,
                 dashes=False, palette=[(0.00392156862745098, 0.45098039215686275, 0.6980392156862745),
                                        (0.8705882352941177, 0.5607843137254902, 0.0196078431372549),
                                        (0.8352941176470589, 0.3686274509803922, 0.0),
                                        (0.00784313725490196, 0.6196078431372549, 0.45098039215686275)],
                 ci=None, ax=axes[0]) #-# Palette manually defined to use colorblind safe colors in correct order (to match other figures) -Mateo
    #-# Plot "Non-Black" Category -Mateo
    sns.lineplot(x="year", y="ly", hue="policy", data=ly_df[ly_df['race'] == 1], style="policy", markers=mks,
                 markersize=7,
                 dashes=False, palette=[(0.00392156862745098, 0.45098039215686275, 0.6980392156862745),
                                        (0.8705882352941177, 0.5607843137254902, 0.0196078431372549),
                                        (0.8352941176470589, 0.3686274509803922, 0.0),
                                        (0.00784313725490196, 0.6196078431372549, 0.45098039215686275)],
                 ci=None, legend=False, ax=axes[1]) #-# Palette manually defined to use colorblind safe colors in correct order (to match other figures) -Mateo

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

    axes[0].set_title(cat_labels[0], fontsize=subtitle_size, fontweight='semibold')
    axes[1].set_title(cat_labels[1], fontsize=subtitle_size, fontweight='semibold')

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
    axes[0].legend(loc='upper center', ncol=4, frameon=False, bbox_to_anchor=(1, -0.2),
                   handles=handles, labels=labels, prop={'size': legend_size})
    # axes[0].set_box_aspect(1)
    # Printing plot
    fig.set_size_inches(7.5, 3)  # for simplified panel plots in paper
    # plt.savefig("Life Years Saved per Policy.pdf", bbox_inches='tight')
    plt.savefig("Life Years Saved per Policy Race.png", bbox_inches='tight', dpi=1200)
    plt.show()
    # fig.set_size_inches(6, 4) # for single plot in talks
    # fig.set_size_inches(9, 4) # for panel plots in talks
    # plt.savefig("Life Years Saved per Policy.pdf", bbox_inches='tight')
    plt.close()
