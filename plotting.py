"""
Plotting functions
"""
# Authors: Ben Gravell and Tyler Summers

import numpy as np
import numpy.linalg as la
from scipy.stats import trim_mean
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FormatStrFormatter
from matplotlib.cm import get_cmap

from utility.matrixmath import specrad

from copy import copy
from warnings import warn
from time import sleep
import os


def comparison_plot(output_dict, cost_are_true, t_hist, t_start_estimate, plot_fields,
                    plotfun_str='plot', xscale='linear', yscale='symlog',
                    show_mean=True, show_median=True, show_trimmed_mean=True, show_quantiles=True,
                    show_samples=True,
                    sample_idx_list=None, sample_legend=False,
                    trim_mean_quantile=None, quantiles=None, quantile_fill_alpha=0.2, quantile_color='tab:blue',
                    quantile_region='middle', quantile_style='fill', quantile_legend=False,
                    comparative=True, stat_diff_type='diff_of_stat',
                    grid=True, show_guideline=True, zoom=False, match_ylims=True,
                    maximize_window=False, figsize=(12, 10)):
    """
    Plot comparison of results using two different control schemes.
    :param output_dict: Dictionary of data history dictionaries associated with each control scheme
    :param cost_are_true: True optimal ARE cost
    :param t_hist: Time history, 1D numpy array
    :param t_start_estimate: The time when model estimates started being generated
    :param plot_fields: List of data fields to plot
    :param plotfun_str: What line plot style to use. Valid values are 'plot', 'step'
    :param xscale: String to pass to matplotlib.pyplot ax.yscale
    :param yscale: String to pass to matplotlib.pyplot ax.yscale
    :param show_mean: Boolean
    :param show_median: Boolean
    :param show_trimmed_mean: Boolean
    :param show_quantiles: Boolean
    :param trim_mean_quantile: Quantile to use in computing the trimmed mean, float
    :param quantiles: Quantile levels, list of floats between 0 and 1
    :param quantile_fill_alpha: Transparency of quantile fill regions, float
    :param quantile_color: Color of quantile fill regions or lines, string
    :param quantile_region: Set the quantile plot region. Valid values are 'upper', 'lower', 'middle'
    :param quantile_style: Set the quantile plot style. Valid values are 'fill' or 'plot'
    :param quantile_legend: Set whether to include quantiles in legend, boolean
    :param comparative: Set whether to show the difference between the first and second algorithm data, boolean
    :param stat_diff_type: Set the statistic difference calculation method.
                           Valid values are 'diff_of_stat', 'stat_of_diff'
    :param grid: Show the plot grid, boolean
    :param show_guideline: Show a guideline at 0 or 1 depending on plotted field, boolean
    :param zoom: Zoom the plot, useful for clipping large max/min, boolean
    :param match_ylims: Set whether to set y limits the same for both algorithms' plots, boolean
    :param maximize_window: Boolean
    :param figsize: Figure window size, passed to matplotlib.pyplot figure. Ex. 2-tuple (width, height) in inches
    """

    # Number columns in the subplots
    if len(output_dict) == 1:
        ncols = 1
        if comparative:
            warn('Requested comparative plot with single data set, ignoring comparison.')
    elif len(output_dict) == 2 and not comparative:
        ncols = 2
    elif len(output_dict) == 2 and comparative:
        ncols = 3

    # Number columns in the subplots
    nrows = len(plot_fields)

    # Number of Monte Carlo samples
    control_schemes = list(output_dict.keys())
    Ns = output_dict[control_schemes[0]][plot_fields[0]].shape[0]

    # Set sample indices
    if sample_idx_list is None:
        sample_idx_list = []
    elif sample_idx_list == 'all':
        sample_idx_list = [i for i in range(Ns)]

    # Set quantile levels
    if quantiles is None:
        # quantiles = [1.00, 0.975, 0.875, 0.75, 0.625]
        quantiles = [1.00, 0.999, 0.99, 0.975, 0.875, 0.75]
        # quantiles = [1.00, 0.99, 0.95]
    quantiles = np.array(quantiles)

    # Set quantile level for trimmed mean
    if trim_mean_quantile is None:
        trim_mean_quantile = np.max(quantiles[quantiles < 1])

    # Manually compute alphas of overlapping regions for legend patches
    quantile_alphas = []
    for j, quantile in enumerate(quantiles):
        if j > 0:
            quantile_alpha_old = quantile_alphas[j-1]
            quantile_alpha_new = quantile_fill_alpha+(1-quantile_fill_alpha)*quantile_alpha_old
        else:
            quantile_alpha_new = quantile_fill_alpha
        quantile_alphas.append(quantile_alpha_new)

    # Process history data for plotting
    fields_to_normalize_by_cost_are_true = ['cost_future_hist']
    fields_to_mean = ['alpha_hist', 'beta_hist']
    fields_to_absmax = []
    fields_to_vecnorm = ['x_train_hist', 'x_test_hist', 'u_train_hist', 'u_test_hist', 'x_opt_test_hist']
    fields_to_fronorm = ['K_hist']
    fields_to_truncate = ['x_train_hist', 'x_test_hist', 'x_opt_test_hist']

    # Make the list of statistic names
    statistics = ['ydata', 'mean', 'trimmed_mean', 'median']
    for quantile in quantiles:
        statistics.append('quantile_'+str(quantile))
        statistics.append('quantile_'+str(1-quantile))

    # Build the ydata dictionary from output_dict
    ydata_dict = {}
    for control_scheme in control_schemes:
        ydata_dict[control_scheme] = {}
        for field in plot_fields:
            ydata_dict[control_scheme][field] = {}
            # Preprocessing
            if field in fields_to_normalize_by_cost_are_true:
                ydata = output_dict[control_scheme][field] / cost_are_true
            elif field in fields_to_mean:
                ydata = np.mean(output_dict[control_scheme][field], axis=2)
            elif field in fields_to_absmax:
                ydata = np.max(np.abs(output_dict[control_scheme][field]), axis=2)
            elif field in fields_to_vecnorm:
                ydata = la.norm(output_dict[control_scheme][field], axis=2)
            elif field in fields_to_fronorm:
                ydata = la.norm(output_dict[control_scheme][field], ord='fro', axis=(2,3))
            else:
                ydata = output_dict[control_scheme][field]
            if field in fields_to_truncate:
                ydata = ydata[:,:-1]
            # Convert nan to inf
            ydata[np.isnan(ydata)] = np.inf
            # Store processed data
            ydata_dict[control_scheme][field]['ydata'] = ydata
            # Compute statistics
            ydata_dict[control_scheme][field]['mean'] = np.mean(ydata, axis=0)
            ydata_dict[control_scheme][field]['trimmed_mean'] = trim_mean(ydata, proportiontocut=1-trim_mean_quantile, axis=0)
            ydata_dict[control_scheme][field]['median'] = np.median(ydata, axis=0)
            for quantile in quantiles:
                ydata_dict[control_scheme][field]['quantile_'+str(quantile)] = np.quantile(ydata, quantile, axis=0)
                ydata_dict[control_scheme][field]['quantile_'+str(1-quantile)] = np.quantile(ydata, 1-quantile, axis=0)
            # Compute particular samples' data
            for sample_idx in sample_idx_list:
                ydata_dict[control_scheme][field]['sample_'+str(sample_idx)] = ydata[sample_idx]

    # Compute statistic differences
    if len(output_dict) == 2 and comparative:
        control_scheme = control_schemes[0] + ' minus ' + control_schemes[1]
        control_schemes.append(control_scheme)
        ydata_dict[control_scheme] = {}
        for field in plot_fields:
            ydata_dict[control_scheme][field] = {}
            for statistic in statistics:
                # Choose whether to calculate statistics before or after taking the difference
                if stat_diff_type=='diff_of_stat':
                    stat1 = ydata_dict[control_schemes[0]][field][statistic]
                    stat2 = ydata_dict[control_schemes[1]][field][statistic]
                    ydata_dict[control_scheme][field][statistic] =  stat1 - stat2
                elif stat_diff_type=='stat_of_diff':
                    #TODO: reuse statistic computation code above
                    ydata1 = ydata_dict[control_schemes[0]][field]['ydata']
                    ydata2 = ydata_dict[control_schemes[1]][field]['ydata']
                    ydata_diff = ydata1 - ydata2
                    ydata_dict[control_scheme][field]['ydata'] = ydata_diff
                    # Compute statistics
                    ydata_dict[control_scheme][field]['median'] = np.median(ydata_diff, axis=0)
                    ydata_dict[control_scheme][field]['mean'] = np.mean(ydata_diff, axis=0)
                    ydata_dict[control_scheme][field]['trimmed_mean'] = trim_mean(ydata_diff, proportiontocut=1-trim_mean_quantile, axis=0)
                    for quantile in quantiles:
                        ydata_dict[control_scheme][field]['quantile_'+str(quantile)] = np.quantile(ydata_diff, quantile, axis=0)
                        ydata_dict[control_scheme][field]['quantile_'+str(1-quantile)] = np.quantile(ydata_diff, 1-quantile, axis=0)
            # Compute particular samples' data
            for sample_idx in sample_idx_list:
                ydata_dict[control_scheme][field]['sample_'+str(sample_idx)] = ydata_diff[sample_idx]

    # Initialize figure and axes
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex=True, sharey='row')

    # x start index
    x_start_idx = t_start_estimate

    # Plot each data series
    for j_col, control_scheme in enumerate(control_schemes):
        for i_row, field in enumerate(plot_fields):
            quantile_region_restore = copy(quantile_region)
            # if j_col == 2:
            #     quantile_region = 'middle'
            # Choose the plotting function
            if plotfun_str == 'plot':
                plotfun = ax[i_row, j_col].plot
                fbstep = None
            elif plotfun_str == 'step':
                plotfun = ax[i_row, j_col].step
                fbstep = 'pre'
            legend_handles = []
            legend_labels = []

            # Plot mean
            if show_mean:
                artist, = plotfun(t_hist, ydata_dict[control_scheme][field]['mean'], color='k', lw=2, zorder=120)
                legend_handles.append(artist)
                legend_labels.append("Mean")

            # Plot trimmed mean
            if show_trimmed_mean:
                artist, = plotfun(t_hist, ydata_dict[control_scheme][field]['trimmed_mean'], color='tab:grey', lw=2, zorder=130)
                legend_handles.append(artist)
                legend_labels.append("Trimmed mean, middle %.0f%%" % (100*(1-((1-trim_mean_quantile)*2))))

            # Plot median
            if show_median:
                artist, = plotfun(t_hist, ydata_dict[control_scheme][field]['median'], color='b', lw=2, zorder=110)
                legend_handles.append(artist)
                legend_labels.append("Median")

            if field == 'gamma_reduction_hist':
                quantile_region = 'middle'

            # Plot quantiles
            if j_col == 2 and stat_diff_type == 'diff_of_stat':
                quantile_style = 'plot'
            if show_quantiles:
                qi = 0
                if quantile_style == 'fill':
                    for quantile, quantile_alpha in zip(quantiles, quantile_alphas):
                        if quantile_region == 'upper':
                            y_lwr = ydata_dict[control_scheme][field]['median']
                        else:
                            y_lwr = ydata_dict[control_scheme][field]['quantile_'+str(1-quantile)]
                        if quantile_region == 'lower':
                            y_upr = ydata_dict[control_scheme][field]['median']
                        else:
                            y_upr = ydata_dict[control_scheme][field]['quantile_'+str(quantile)]

                        ax[i_row, j_col].fill_between(t_hist, y_lwr, y_upr, step=fbstep,
                                                      color=quantile_color, alpha=quantile_fill_alpha, zorder=qi)
                        if quantile_legend:
                            legend_handles.append(mpatches.Patch(color=quantile_color, alpha=quantile_alpha))
                            if quantile_region == 'middle':
                                legend_label_str = "Middle %5.1f%%" % (100*(1-((1-quantile)*2)))
                            elif quantile_region == 'upper':
                                legend_label_str = "Upper %5.1f%%" % (50*(1-((1-quantile)*2)))
                            elif quantile_region == 'lower':
                                legend_label_str = "Lower %5.1f%%" % (50*(1-((1-quantile)*2)))
                            legend_labels.append(legend_label_str)

                elif quantile_style == 'plot':
                    if quantile_region == 'middle':
                        cmap = get_cmap('coolwarm_r')
                    else:
                        cmap = get_cmap('viridis')
                        # cmap = get_cmap('tab10')
                    qtot = 0
                    if not quantile_region == 'upper':
                        qtot += len(quantiles)
                    if not quantile_region == 'lower':
                        qtot += len(quantiles)
                    qtot -= 1
                    if not quantile_region == 'lower':
                        for quantile, quantile_alpha in zip(quantiles, quantile_alphas):
                            y_upr = ydata_dict[control_scheme][field]['quantile_'+str(quantile)]
                            artist_upr, = plotfun(t_hist, y_upr, color=cmap(qi/qtot), alpha=0.9, zorder=qi)
                            if quantile_legend:
                                legend_handles.append(artist_upr)
                                legend_label_str = "Percentile %5.1f%%" % (100*quantile)
                                legend_labels.append(legend_label_str)
                            qi += 1
                    if not quantile_region == 'upper':
                        for quantile, quantile_alpha in zip(reversed(quantiles), reversed(quantile_alphas)):
                            y_lwr = ydata_dict[control_scheme][field]['quantile_'+str(1-quantile)]
                            artist_lwr, = plotfun(t_hist, y_lwr, color=cmap(qi/qtot), alpha=0.9, zorder=qi)
                            if quantile_legend:
                                legend_handles.append(artist_lwr)
                                legend_label_str = "Percentile %5.1f%%" % (100*(1-quantile))
                                legend_labels.append(legend_label_str)
                            qi += 1

            # Plot particular samples
            if show_samples:
                for sample_idx in sample_idx_list:
                    sample_idx_modten = sample_idx % 10
                    sample_str = 'sample_'+str(sample_idx)
                    artist, = plotfun(t_hist, ydata_dict[control_scheme][field][sample_str],
                                      color='darkgreen', alpha=0.9, lw=2, zorder=20+sample_idx_modten)
                    if sample_legend:
                        legend_handles.append(artist)
                        legend_labels.append(sample_str)

            # Plot guidelines
            if show_guideline:
                x_guide = np.copy(t_hist)
                y_guide = np.zeros(ydata_dict[control_scheme][field]['ydata'].shape[1])
                if j_col < 2:
                    if plot_fields[i_row] == 'specrad_hist' or plot_fields[i_row] == 'gamma_reduction_hist':
                        y_guide = np.ones(ydata_dict[control_scheme][field]['ydata'].shape[1])
                plotfun(x_guide, y_guide, color='tab:purple', lw=1, linestyle='--', zorder=20)

            # Set axes options
            ax[i_row, j_col].set_xscale(xscale)
            ax[i_row, j_col].set_yscale(yscale)
            if j_col == 0:
                ylabel_str = plot_fields[i_row].replace('_', ' ')
                ax[i_row, j_col].set_ylabel(ylabel_str)

            leg = ax[i_row, j_col].legend(handles=legend_handles, labels=legend_labels, loc='upper right')
            leg.set_zorder(1000)

            # Set axes limits
            try:
                if zoom:
                    x_zoom_start_idx_portion = 0.1
                    xlen = len(t_hist)
                    x_zoom_start_idx = int(x_zoom_start_idx_portion*(xlen-x_start_idx))
                    xl = [t_hist[x_zoom_start_idx], t_hist[-1]]
                    ydata_for_lims = ydata_dict[control_scheme][field]['ydata'][x_zoom_start_idx:]
                    ydata_for_lims = ydata_for_lims[np.isfinite(ydata_for_lims)]
                    ylim_quantile = min([max(quantiles), 0.95])
                else:
                    xl = [t_hist[x_start_idx], t_hist[-1]]
                    ydata_for_lims = ydata_dict[control_scheme][field]['ydata'][x_start_idx:]
                    ydata_for_lims = ydata_for_lims[np.isfinite(ydata_for_lims)]
                    ylim_quantile = max(quantiles)
                yl_quantile_lwr = np.quantile(ydata_for_lims, 1-ylim_quantile)
                yl_quantile_upr = np.quantile(ydata_for_lims, ylim_quantile)
                if j_col == 2:
                    # Make y-axis limits symmetric about zero
                    ylqmax = np.max(np.abs([yl_quantile_lwr, yl_quantile_upr]))
                    yl = [-ylqmax, ylqmax]
                else:
                    yl = [np.min(yl_quantile_lwr), np.max(yl_quantile_upr)]
                ax[i_row, j_col].set_xlim(xl)
                # ax[i_row, j_col].set_ylim(yl)
            except:
                pass

            if grid:
                ax[i_row, j_col].grid('on')
            ax[i_row, j_col].set_axisbelow(True)
            quantile_region = copy(quantile_region_restore)
        xlabel_str = 'Time'
        ax[-1, j_col].set_xlabel(xlabel_str)
        title_str = control_scheme.replace('_', ' ').title()
        ax[0, j_col].set_title(title_str)

    if match_ylims:
        for i_row in range(nrows):
            yl_left = ax[i_row, 0].get_ylim()
            yl_right = ax[i_row, 1].get_ylim()
            yl_min = min(yl_left[0], yl_right[0])
            yl_max = max(yl_left[1], yl_right[1])
            ax[i_row, 0].set_ylim(yl_min, yl_max)
            ax[i_row, 1].set_ylim(yl_min, yl_max)

    if maximize_window:
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
    fig.tight_layout()
    return fig, ax


def multi_plot(output_dict, cost_are_true, t_hist, t_start_estimate, legend=True):
    plt.close('all')
    # plot_fields_all = [['regret_hist',
    #                     'cost_adaptive_hist',
    #                     'cost_optimal_hist',
    #                     'cost_future_hist',
    #                     'specrad_hist'],
    #                    ['Aerr_hist',
    #                     'Berr_hist',
    #                     'alpha_hist',
    #                     'beta_hist',
    #                     'gamma_reduction_hist'],
    #                    ['x_train_hist',
    #                     'x_test_hist',
    #                     'x_opt_test_hist',
    #                     'u_train_hist',
    #                     'u_test_hist']]


    # plot_fields_all = [['regret_hist',
    #                     'cost_future_hist',
    #                     'specrad_hist'],
    #                    ['Aerr_hist',
    #                     'Berr_hist'],
    #                    ['alpha_hist',
    #                     'beta_hist',
    #                     'gamma_reduction_hist'],
    #                    ['x_test_hist',
    #                     'u_test_hist',
    #                     'K_hist']]

    plot_fields_all = [['regret_hist',
                        'cost_future_hist',
                        'specrad_hist'],
                       ['regret_hist',
                        'regret_hist_receding',
                        'regret_hist_accum'],
                       ['Aerr_hist',
                        'Berr_hist'],
                       ['alpha_hist',
                        'beta_hist',
                        'gamma_reduction_hist'],
                       ['x_test_hist',
                        'u_test_hist',
                        'K_hist']]

    plot_type = 'summary'
    # plot_type = 'samples'


    if plot_type == 'summary':
        show_quantiles = True
        show_mean = True
        show_trimmed_mean = True
        show_median = True
        sample_idx_list = None

    elif plot_type == 'samples':
        show_quantiles = True
        show_mean = False
        show_trimmed_mean = False
        show_median = False
        # Build sample idx list based on large excursions of the state
        control_schemes = list(output_dict.keys())
        Ns = output_dict[control_schemes[0]]['regret_hist'].shape[0]
        threshold = 1000
        idx_list = []
        for i, control_scheme in enumerate(control_schemes):
            idx_list = idx_list + np.where(np.any(np.abs(output_dict[control_scheme]['x_test_hist']) > threshold, axis=(1,2)))[0].tolist()
        sample_idx_list = np.unique(idx_list)



    figs = []
    for plot_fields in plot_fields_all:
        fig, ax = comparison_plot(output_dict, cost_are_true, t_hist, t_start_estimate, plot_fields,
                                  show_quantiles=show_quantiles,
                                  show_mean=show_mean,
                                  show_trimmed_mean=show_trimmed_mean,
                                  show_median=show_median,
                                  sample_idx_list=sample_idx_list)
        figs.append(fig)
    plt.show()
    return figs


def multi_plot_paper(output_dict, cost_are_true, t_hist, t_start_estimate,
                     plotfun_str='plot', xscale='linear', yscale='symlog',
                     show_mean=True, show_median=True, show_trimmed_mean=False, show_quantiles=True,
                     trim_mean_quantile=None, quantile_fill_alpha=0.2, quantile_color='tab:blue',
                     quantile_region='upper', quantile_style='fill', quantile_legend=True,
                     stat_diff_type='diff_of_stat',
                     grid=True, show_guideline=True, zoom=False,
                     figsize=(4.5, 3), save_plots=True, dirname_out=None):
    plt.close('all')

    # from matplotlib import rc
    # rc('text', usetex=True)

    control_schemes = list(output_dict.keys())
    diff_scheme = control_schemes[0]+' minus '+control_schemes[1]
    plot_fields = ['regret_hist',
                   'regret_hist',
                   'regret_hist',
                   'Aerr_hist',
                   'Berr_hist',
                   'alpha_hist',
                   'beta_hist',
                   'gamma_reduction_hist',
                   'K_hist']
    plot_control_schemes = ['certainty_equivalent',
                            'robust',
                            diff_scheme,
                            'robust',
                            'robust',
                            'robust',
                            'robust',
                            'robust',
                            diff_scheme]

    ylabels = ['Instantaneous regret',
               'Instantaneous regret',
               'Instantaneous regret diff.',
               r'$|\hat{A}-A|$',
               r'$|\hat{B}-B|$',
               r'$\alpha$',
               r'$\beta$',
               r'$c_\gamma$',
               r'$|K|$']
    filenames = ['instant_regret_ce',
                 'instant_regret_rmn',
                 'instant_regret_diff',
                 'Aerr',
                 'Berr',
                 'alpha',
                 'beta',
                 'gamma_scale',
                 'gain']

    quantiles = [1.00, 0.999, 0.99, 0.95, 0.75]
    quantiles = np.array(quantiles)

    # Set quantile level for trimmed mean
    if trim_mean_quantile is None:
        trim_mean_quantile = np.max(quantiles[quantiles < 1])

    # Manually compute alphas of overlapping regions for legend patches
    quantile_alphas = []
    for j, quantile in enumerate(quantiles):
        if j > 0:
            quantile_alpha_old = quantile_alphas[j-1]
            quantile_alpha_new = quantile_fill_alpha+(1-quantile_fill_alpha)*quantile_alpha_old
        else:
            quantile_alpha_new = quantile_fill_alpha
        quantile_alphas.append(quantile_alpha_new)

    # Process history data for plotting
    fields_to_normalize_by_cost_are_true = ['cost_future_hist']
    fields_to_mean = []
    fields_to_absmax = ['alpha_hist', 'beta_hist']
    fields_to_vecnorm = ['x_train_hist', 'x_test_hist', 'u_train_hist', 'u_test_hist', 'x_opt_test_hist']
    fields_to_fronorm = ['K_hist']
    fields_to_squeeze = []
    fields_to_truncate = ['x_train_hist', 'x_test_hist', 'x_opt_test_hist']

    # Make the list of statistic names
    statistics = ['ydata', 'mean', 'trimmed_mean', 'median']
    for quantile in quantiles:
        statistics.append('quantile_'+str(quantile))
        statistics.append('quantile_'+str(1-quantile))

    # Build the ydata dictionary from output_dict
    ydata_dict = {}
    for control_scheme in control_schemes:
        ydata_dict[control_scheme] = {}
        for field in plot_fields:
            ydata_dict[control_scheme][field] = {}
            # Preprocessing
            if field in fields_to_normalize_by_cost_are_true:
                ydata = output_dict[control_scheme][field] / cost_are_true
            elif field in fields_to_mean:
                ydata = np.mean(output_dict[control_scheme][field], axis=2)
            elif field in fields_to_absmax:
                ydata = np.max(np.abs(output_dict[control_scheme][field]), axis=2)
            elif field in fields_to_vecnorm:
                ydata = la.norm(output_dict[control_scheme][field], axis=2)
            elif field in fields_to_fronorm:
                ydata = la.norm(output_dict[control_scheme][field], ord='fro', axis=(2,3))
            else:
                ydata = output_dict[control_scheme][field]
            if field in fields_to_squeeze:
                ydata = np.squeeze(ydata)
            if field in fields_to_truncate:
                ydata = ydata[:,:-1]
            # Convert nan to inf
            ydata[np.isnan(ydata)] = np.inf
            # Store processed data
            ydata_dict[control_scheme][field]['ydata'] = ydata
            # Compute statistics
            ydata_dict[control_scheme][field]['mean'] = np.mean(ydata, axis=0)
            ydata_dict[control_scheme][field]['trimmed_mean'] = trim_mean(ydata, proportiontocut=1-trim_mean_quantile, axis=0)
            ydata_dict[control_scheme][field]['median'] = np.median(ydata, axis=0)
            for quantile in quantiles:
                ydata_dict[control_scheme][field]['quantile_'+str(quantile)] = np.quantile(ydata, quantile, axis=0)
                ydata_dict[control_scheme][field]['quantile_'+str(1-quantile)] = np.quantile(ydata, 1-quantile, axis=0)


    # Compute statistic differences
    control_scheme = control_schemes[0] + ' minus ' + control_schemes[1]
    control_schemes.append(control_scheme)
    ydata_dict[control_scheme] = {}
    for field in plot_fields:
        ydata_dict[control_scheme][field] = {}
        for statistic in statistics:
            # Choose whether to calculate statistics before or after taking the difference
            if stat_diff_type=='diff_of_stat':
                stat1 = ydata_dict[control_schemes[0]][field][statistic]
                stat2 = ydata_dict[control_schemes[1]][field][statistic]
                ydata_dict[control_scheme][field][statistic] =  stat1 - stat2
            elif stat_diff_type=='stat_of_diff':
                #TODO: reuse statistic computation code above
                ydata1 = ydata_dict[control_schemes[0]][field]['ydata']
                ydata2 = ydata_dict[control_schemes[1]][field]['ydata']
                ydata_diff = ydata1 - ydata2
                ydata_dict[control_scheme][field]['ydata'] = ydata_diff
                # Compute statistics
                ydata_dict[control_scheme][field]['median'] = np.median(ydata_diff, axis=0)
                ydata_dict[control_scheme][field]['mean'] = np.mean(ydata_diff, axis=0)
                ydata_dict[control_scheme][field]['trimmed_mean'] = trim_mean(ydata_diff, proportiontocut=1-trim_mean_quantile, axis=0)
                for quantile in quantiles:
                    ydata_dict[control_scheme][field]['quantile_'+str(quantile)] = np.quantile(ydata_diff, quantile, axis=0)
                    ydata_dict[control_scheme][field]['quantile_'+str(1-quantile)] = np.quantile(ydata_diff, 1-quantile, axis=0)


    # x start index
    x_start_idx = t_start_estimate

    for control_scheme, field, ylabel_str, filename in zip(plot_control_schemes, plot_fields, ylabels, filenames):
        # Initialize figure and axes
        fig, ax = plt.subplots(figsize=figsize)

        # Choose the plotting function
        if plotfun_str == 'plot':
            plotfun = ax.plot
            fbstep = None
        elif plotfun_str == 'step':
            plotfun = ax.step
            fbstep = 'pre'


        # Choose the quantiles
        # if control_scheme == diff_scheme and stat_diff_type == 'diff_of_stat':
        #     quantile_style = 'plot'
        # else:
        #     quantile_style = 'fill'

        if field == 'gamma_reduction_hist':
            quantiles = [1.00, 0.95, 0.75]
            quantile_region = 'middle'
        else:
            quantiles = [0.999, 0.99, 0.95]
            quantile_region = 'upper'
        quantiles = np.array(quantiles)

        xl = [t_hist[x_start_idx], t_hist[-1]*1.05]
        # if field == 'alpha_hist' or field == 'beta_hist' or field == 'gamma_reduction_hist':
        #     xl = [t_hist[x_start_idx], 20]

        legend_handles = []
        legend_labels = []

        # Plot mean
        if show_mean:
            artist, = plotfun(t_hist, ydata_dict[control_scheme][field]['mean'], color='k', lw=2, zorder=120)
            legend_handles.append(artist)
            legend_labels.append("Mean")

        # Plot trimmed mean
        if show_trimmed_mean:
            artist, = plotfun(t_hist, ydata_dict[control_scheme][field]['trimmed_mean'], color='tab:grey', lw=2, zorder=130)
            legend_handles.append(artist)
            legend_labels.append("Trimmed mean, middle %.0f%%" % (100*(1-((1-trim_mean_quantile)*2))))

        # Plot median
        if show_median:
            artist, = plotfun(t_hist, ydata_dict[control_scheme][field]['median'], color='b', lw=2, zorder=110)
            legend_handles.append(artist)
            legend_labels.append("Median")

        # Plot quantiles
        if show_quantiles:
            qi = 0
            if quantile_style == 'fill':
                for quantile, quantile_alpha in zip(quantiles, quantile_alphas):
                    if quantile_region == 'upper':
                        y_lwr = ydata_dict[control_scheme][field]['median']
                    else:
                        y_lwr = ydata_dict[control_scheme][field]['quantile_'+str(1-quantile)]
                    if quantile_region == 'lower':
                        y_upr = ydata_dict[control_scheme][field]['median']
                    else:
                        y_upr = ydata_dict[control_scheme][field]['quantile_'+str(quantile)]

                    ax.fill_between(t_hist, y_lwr, y_upr, step=fbstep,
                                                  color=quantile_color, alpha=quantile_fill_alpha, zorder=qi)
                    if quantile_legend:
                        legend_handles.append(mpatches.Patch(color=quantile_color, alpha=quantile_alpha))
                        if quantile_region == 'middle':
                            legend_label_str = "Middle %.1f%%" % (100*(1-((1-quantile)*2)))
                        elif quantile_region == 'upper':
                            legend_label_str = "Upper %.1f%%" % (50*(1-((1-quantile)*2)))
                        elif quantile_region == 'lower':
                            legend_label_str = "Lower %.1f%%" % (50*(1-((1-quantile)*2)))
                        legend_labels.append(legend_label_str)

            elif quantile_style == 'plot':
                if quantile_region == 'middle':
                    cmap = get_cmap('coolwarm_r')
                else:
                    cmap = get_cmap('viridis')
                    # cmap = get_cmap('tab10')
                qtot = 0
                if not quantile_region == 'upper':
                    qtot += len(quantiles)
                if not quantile_region == 'lower':
                    qtot += len(quantiles)
                qtot -= 1
                qcolors = ['C1', 'C2', 'C4', 'C5']
                if not quantile_region == 'lower':
                    for quantile, quantile_alpha in zip(quantiles, quantile_alphas):
                        y_upr = ydata_dict[control_scheme][field]['quantile_'+str(quantile)]
                        artist_upr, = plotfun(t_hist, y_upr, color=qcolors[qi], alpha=0.9, zorder=qi)
                        if quantile_legend:
                            legend_handles.append(artist_upr)
                            legend_label_str = "Percentile %.1f%%" % (100*quantile)
                            legend_labels.append(legend_label_str)
                        qi += 1
                if not quantile_region == 'upper':
                    for quantile, quantile_alpha in zip(reversed(quantiles), reversed(quantile_alphas)):
                        y_lwr = ydata_dict[control_scheme][field]['quantile_'+str(1-quantile)]
                        artist_lwr, = plotfun(t_hist, y_lwr, color=qcolors[qi], alpha=0.9, zorder=qi)
                        if quantile_legend:
                            legend_handles.append(artist_lwr)
                            legend_label_str = "Percentile %.1f%%" % (100*(1-quantile))
                            legend_labels.append(legend_label_str)
                        qi += 1

        ax.set_xlim(xl)

        # Plot guidelines
        if show_guideline:
            x_guide = np.copy(t_hist)
            y_guide = np.zeros(ydata_dict[control_scheme][field]['ydata'].shape[1])
            if field == 'specrad_hist' or field == 'gamma_reduction_hist':
                y_guide = np.ones(ydata_dict[control_scheme][field]['ydata'].shape[1])
            plotfun(x_guide, y_guide, color='tab:purple', lw=1, linestyle='--', zorder=20)

        yscale = 'symlog'
        if field == 'gamma_reduction_hist':
            yscale = 'linear'
        elif field == 'alpha_hist' or field == 'beta_hist' or field == 'Aerr_hist' or field == 'Berr_hist':
            yscale = 'log'

        # Set axes options
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)

        loc = 'best'
        if field == 'regret_hist':
            loc = 'center right'
        elif field == 'gamma_reduction_hist':
            loc = 'lower right'

        leg = ax.legend(handles=legend_handles, labels=legend_labels, loc=loc)
        leg.set_zorder(1000)

        if field == 'regret_hist' and not control_scheme == diff_scheme:
            yl = [-0.1, 1e27]
        else:
            ydata_lim_lwr = ydata_dict[control_scheme][field]['median'][x_start_idx:]
            ydata_lim_upr = ydata_dict[control_scheme][field]['quantile_'+str(max(quantiles))][x_start_idx:]
            ydata_lim_lwr = ydata_lim_lwr[np.isfinite(ydata_lim_lwr)]
            ydata_lim_upr = ydata_lim_upr[np.isfinite(ydata_lim_upr)]
            yl_lwr = np.min(ydata_lim_lwr)
            yl_upr = np.max(ydata_lim_upr)
            yl = [yl_lwr, yl_upr]
        ax.set_ylim(yl)


        if field == 'regret_hist':
            # plt.locator_params(axis='y', numticks=6)
            plt.yticks([0, 1e10, 1e20, 1e30, 1e40, 1e50, 1e60, 1e70, 1e80])
            # pass
        elif field == 'alpha_hist' or field == 'beta_hist':
            # plt.yticks([0, 0.5, 1, 1.5, 2])
            pass
        elif field == 'gamma_reduction_hist':
            plt.yticks([0, 0.25, 0.5, 0.75, 1])


        if grid:
            ax.grid('on')
        ax.set_axisbelow(True)
        xlabel_str = 'Time'
        ax.set_xlabel(xlabel_str, fontsize=12)
        # rot = None
        # ax.set_ylabel(ylabel_str, fontsize=12, rotation=rot)
        ax.set_ylabel(ylabel_str, fontsize=12)
        title_str = ylabel_str+'_'+control_scheme
        title_str = title_str.replace('_', ' ').title()
        # ax.set_title(title_str)

        fig.tight_layout()
        if save_plots:
           filename_out = 'plot_' + filename + '.png'
           path_out = os.path.join(dirname_out, filename_out)
           plt.savefig(path_out, dpi=600, bbox_inches="tight")


def matrix_hist(matrix_data, matrix_center=None, center=False, bin_ratio=0.1, bin_max=100, density=False,
                sharex='none', sharey='none', figsize=None, title_str=None, show_entry_subtitles=False,
                show_centerline=True, save_plots=True, dirname_out='.'):
    """
    Plot sample matrix data in histograms arranged like the matrix
    :param matrix_data: Array of matrix data to plot in histogram
    :type matrix_data: numpy.ndarray of shape (N, n, m)
    :param matrix_center: Matrix defining center points
    :type matrix_center: numpy.ndarray of shape (n, m)
    :param center: Choose whether to center matrix_data about matrix_center
    :type center: Boolean
    :param bin_ratio: Ratio of number of bins to number of samples
    """

    N, n, m = matrix_data.shape

    if matrix_center is not None and center:
        matrix_data = matrix_data - matrix_center
        matrix_center = np.zeros_like(matrix_center)

    fig, ax = plt.subplots(nrows=n, ncols=m, sharex=sharex, sharey=sharey, figsize=figsize)
    for i in range(n):
        for j in range(m):
            # Handle degenerate cases when n or m == 1 for axes indexing
            if n > 1 and m > 1:
                current_ax = ax[i,j]
            elif n > 1 and m == 1:
                current_ax = ax[i]
            elif n == 1 and m > 1:
                current_ax = ax[j]
            elif n == 1 and m == 1:
                current_ax = ax
            current_ax.hist(matrix_data[:, i, j], density=density, bins=min(int(bin_ratio*N), bin_max))
            if show_centerline and matrix_center is not None:
                current_ax.axvline(matrix_center[i, j], color='k', linestyle = '--', alpha=0.5)
            if show_entry_subtitles:
                current_ax.set_title('(%d,%d)-entry' % (i, j))
    if title_str is not None:
        fig.suptitle(title_str)
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    if save_plots:
       filename_out = 'plot_' + title_str.replace(' ', '_') + '.png'
       path_out = os.path.join(dirname_out, filename_out)
       plt.savefig(path_out, dpi=600)
    return fig, ax


def robust_eval_plot(output_dict, A, B,
                     Pare_true, Kare_true, clip_data=True, clip_scale=100,
                     quantiles=None, ylim_quantile=0.975, quantile_color='tab:blue', quantile_fill_alpha=0.2,
                     xscale='log', yscale='log', show_legend=True, figsize=(6, 4), save_plots=False, dirname_out='.'):
    # Plotting
    plt.close('all')

    # Unpack data
    mult_noise_scales = output_dict['mult_noise_scales']
    specrad_true_hist = output_dict['specrad_true_hist']
    c_true_hist = output_dict['c_true_hist']
    Ahat_hist = output_dict['Ahat_hist']
    Bhat_hist = output_dict['Bhat_hist']
    K_hist = output_dict['K_hist']

    Kfro_hist = la.norm(K_hist, axis=(2, 3), ord='fro')

    n = A.shape[1]
    m = B.shape[1]
    if n == 1 and m == 1:
        scalar_case = True
    else:
        scalar_case = False

    if scalar_case:
        K_hist = K_hist.squeeze()

    specrad_opt = specrad(A+np.dot(B, Kare_true))
    c_opt = np.trace(Pare_true)
    Kfro_opt = la.norm(Kare_true, ord='fro')

    if quantiles is None:
        quantiles = [1.00, 0.99, 0.975, 0.875, 0.75]

    # Manually compute alphas of overlapping regions for legend patches
    quantile_alphas = []
    for j, quantile in enumerate(quantiles):
        if j > 0:
            quantile_alpha_old = quantile_alphas[j-1]
            quantile_alpha_new = quantile_fill_alpha+(1-quantile_fill_alpha)*quantile_alpha_old
        else:
            quantile_alpha_new = quantile_fill_alpha
        quantile_alphas.append(quantile_alpha_new)

    if clip_data:
        def clipper(x):
            return np.clip(x, 0, clip_scale*np.max(x[np.isfinite(x)]))

    if scalar_case:
        Kplot_hist = K_hist
        Kplot_title = 'Gain K of true closed-loop system'
        Kplot_opt = Kare_true.squeeze()
        Kyscale = 'symlog'
    else:
        Kplot_hist = Kfro_hist
        Kplot_title = 'Gain (||K||_F) of true closed-loop system'
        Kplot_opt = Kfro_opt

    ydata_all = [specrad_true_hist, c_true_hist/c_opt, Kplot_hist]
    ylabels = ['Spectral radius', 'Cost (normalized)', 'Gain']
    title_strs = ['Spectral radius of true closed-loop system',
                  'Cost (trace(P)/trace(Pare) of true closed-loop system',
                  Kplot_title]
    yguides = [1, 0, 0]
    yguides_opt = [specrad_opt, 1, Kplot_opt]
    ylim_lwr_bools = [False, True, False]

    zipped_data = zip(ydata_all, ylabels, title_strs, yguides, yguides_opt, ylim_lwr_bools)
    for ydata, ylabel, title_str, yguide, yguide_opt, ylim_lwr_bool in zipped_data:
        if clip_data and not ylabel == 'Gain':
            ydata = clipper(ydata)
        fig, ax = plt.subplots(figsize=figsize)
        legend_handles = []
        legend_labels = []
        line, = plt.plot(mult_noise_scales, np.quantile(ydata, 0.5, axis=0), color='b')
        legend_handles.append(line)
        legend_labels.append('Median')
        for i, quantile in enumerate(quantiles):
            quantile_alpha = quantile_alphas[i]
            y_lwr = np.quantile(ydata, 1-quantile, axis=0)
            y_upr = np.quantile(ydata, quantile, axis=0)
            y_upr[np.isinf(y_upr)] = 10*np.max(y_upr[np.isfinite(y_upr)])
            plt.fill_between(mult_noise_scales, y_lwr, y_upr, color=quantile_color, alpha=quantile_fill_alpha)
            legend_handles.append(mpatches.Patch(color=quantile_color, alpha=quantile_alpha))
            legend_label_str = "Middle %.1f%%" % (100*(1-((1-quantile)*2)))
            legend_labels.append(legend_label_str)
        line, = plt.plot(mult_noise_scales, yguide_opt*np.ones_like(mult_noise_scales), 'g--')
        legend_handles.append(line)
        legend_labels.append('True optimal')
        line, = plt.plot(mult_noise_scales, yguide*np.ones_like(mult_noise_scales), 'r--')
        legend_handles.append(line)
        legend_labels.append('Limit')
        plt.xlabel('Multiplicative noise variance')
        plt.ylabel(ylabel)
        if ylim_lwr_bool:
            ylim_lwr = np.min(ydata)
        else:
            ylim_lwr = np.min(np.quantile(ydata, 1-ylim_quantile, axis=0))
        ylim_upr = np.max(np.quantile(ydata, ylim_quantile, axis=0))
        plt.ylim([ylim_lwr, ylim_upr])
        if show_legend:
            plt.legend(handles=legend_handles, labels=legend_labels)
        plt.title(title_str)
        ax.set_xscale(xscale)
        if ylabel == 'Gain':
            ax.set_yscale(Kyscale)
        else:
            ax.set_yscale(yscale)
        plt.tight_layout()
        if save_plots:
            filename_out = 'plot_' + ylabel.lower().replace(' ', '_') + '.png'
            path_out = os.path.join(dirname_out, filename_out)
            plt.savefig(path_out, dpi=600)

    sleep(0.5)
    matrix_hist(matrix_data=Ahat_hist, matrix_center=A, center=False, title_str='A matrix',
                figsize=figsize, save_plots=True, dirname_out=dirname_out)
    matrix_hist(matrix_data=Bhat_hist, matrix_center=B, center=False, title_str='B matrix',
                figsize=figsize, save_plots=True, dirname_out=dirname_out)


# DEV
# Plot bootstrap estimates of a 2x1 B matrix
def bootplot(B, Bhat, Bhat_boot, V, D):
    plt.close('all')
    print(Bhat_boot.shape)
    # One std dev confidence ellipse
    cov_ellipse_t = np.linspace(0, 2*np.pi)
    cov_ellipse_xy = (V*np.sqrt(D))@np.vstack([np.cos(cov_ellipse_t), np.sin(cov_ellipse_t)])
    plt.plot(np.mean(Bhat_boot[:, 0, 0])+cov_ellipse_xy[0],
             np.mean(Bhat_boot[:, 1, 0])+cov_ellipse_xy[1],
             'r--')
    plt.scatter(Bhat_boot[:,0,0], Bhat_boot[:,1,0], s=5, c='k')
    plt.scatter(np.mean(Bhat_boot[:,0,0]), np.mean(Bhat_boot[:,1,0]), s=50)
    plt.scatter(np.median(Bhat_boot[:,0,0]), np.median(Bhat_boot[:,1,0]), s=50)
    plt.scatter(Bhat[0,0], Bhat[1,0], s=50)
    plt.scatter(B[0,0], B[1,0], s=50)

    plt.axis('equal')
    plt.legend(['Covariance ellipse','Bootstrap samples', 'Bootstrap mean', 'Bootstrap median',
                'Estimated model', 'True'])
    return


# DEV
import numpy.random as npr
from problem_data_gen import gen_system_omni
from monte_carlo_comparison import generate_offline_data
from utility.matrixmath import mdot, lstsqb
# Plot the path of least-square estimates towards the true parameters
def least_square_path_plotter(T=100, system_idx=1, system_kwargs=None, seed=1):
    # Seed the random number generator
    npr.seed(seed)

    # Problem data
    n, m, A, B, Q, R, W = gen_system_omni(system_idx, **system_kwargs)

    # Initial state
    x0 = np.zeros(n)

    # Input exploration noise during explore phase
    u_explore_var = 1.0

    # Generate sample trajectory data (pure exploration)
    x_train_hist, u_train_hist, w_hist = generate_offline_data(n, m, A, B, W, 1, T, u_explore_var, x0, seed)

    # Nominal model
    Ahathist = np.zeros([T, n, n])
    Bhathist = np.zeros([T, n, m])

    # Model error
    Aerr_hist = np.full([T], np.inf)
    Berr_hist = np.full([T], np.inf)



    for t in range(T):
        X = x_train_hist[1:t+1]
        Z = np.hstack([x_train_hist[0:t], u_train_hist[0:t]])
        Thetahat = lstsqb(mdot(X.T, Z), mdot(Z.T, Z))
        Ahat = Thetahat[:, 0:n]
        Bhat = Thetahat[:, n:n+m]

        Ahathist[t] = Ahat
        Bhathist[t] = Bhat
        Aerr_hist[t] = la.norm(A-Ahat, 'fro')
        Berr_hist[t] = la.norm(B-Bhat, 'fro')

    plt.close('all')
    import matplotlib.cm as cm
    # plt.plot(Aerr_hist[k])
    # plt.plot(Berr_hist[k])
    plt.scatter(B[0], B[1], s=100, c='tab:blue', zorder=10)
    plt.plot(Bhathist[:, 0], Bhathist[:, 1], 'grey', zorder=1)
    plt.scatter(Bhathist[:, 0], Bhathist[:, 1], s=20, c=cm.viridis(np.linspace(0, 1, T)), zorder=5)
    plt.show()