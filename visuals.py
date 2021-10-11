import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd


def pannel_singlets(folder, dataframe_ts, varfield, dataframe_freq, bymonth, rangedetail, detail=True, flenm='Pannel', ylbls='m3/s',
                    datefield='Date', scale='log', show=False):
    """

    A customized function to plot a pannel for a single time series.


    :param folder: string of folder path. Ex: "C:/project/data"
    :param dataframe_ts: dataframe with the time series data
    :param varfield: string of variable field in the dataframe of time series
    :param dataframe_freq: dataframe of frequency analysis.
    Must have a filed called 'Exeedance' and a field called 'Values'. Recommendation: use the tsa.frequency() function to get it
    :param bymonth: dictionary of dataframes time series separated by month. Keys must be: 'January', 'February', ..., 'December'
    Recommendation: use the resample.group_by_month() function to get it
    :param rangedetail: tuple or list with strings of two range dates for the detail plot. Ex.: ('1990-12-29', '2000-01-30')
    :param detail: boolean to allow the insertion of a detail plot. Default: True
    :param flenm: string for the file name. Default: 'Pannel'
    :param ylbls: string for the Y axis label.
    :param datefield: string for the Date field in all dataframes. Default: 'Date'
    :param scale: string for the scale of Y axis on plot b and plot c. Default: 'log'.
    Options: 'linear', 'log', 'symlog', 'logit', 'function', 'functionlog'
    :param show: boolean to show plot instead of saving to file
    :return: string of file path
    """
    #
    fig = plt.figure(figsize=(10, 6))
    gs = mpl.gridspec.GridSpec(2, 3, wspace=0.4, hspace=0.3, top=0.95, bottom=0.1, left=0.1, right=0.95)
    # get dataframes
    df1 = dataframe_ts.copy()
    df2 = dataframe_freq.copy()
    #
    if detail:
        dt_0 = pd.to_datetime(rangedetail[0])
        dt_1 = pd.to_datetime(rangedetail[1])
        df_inset = df1.query('{} >= "{}" and {} < "{}"'.format(datefield, dt_0, datefield, dt_1))
    #
    # Series plot
    ax = plt.subplot(gs[0, :])
    aux_str = r'$\bf{' + 'a.  ' + '}$' + 'Full time series'
    plt.title(aux_str, fontsize=10, loc='left')
    plt.plot(df1[datefield], df1[varfield])
    plt.ylim(0, (1.3 * np.max(df1[varfield])))
    plt.ylabel(ylbls)
    # plt.xlabel('Time')
    plt.grid(True, 'major', axis='y')
    #
    # detail lines:
    if detail:
        ymax_h = np.max(df_inset[varfield].values)
        ymin_h = np.min(df_inset[varfield].values)
        lines_c = 'tab:orange'
        plt.plot([dt_0, dt_0], [ymin_h, ymax_h], lines_c)
        plt.plot([dt_1, dt_1], [ymin_h, ymax_h], lines_c)
        plt.plot([dt_0, dt_1], [ymin_h, ymin_h], lines_c)
        plt.plot([dt_0, dt_1], [ymax_h, ymax_h], lines_c)
        #
        # Detail plot
        inset = ax.inset_axes([0.05, 0.67, 0.2, 0.3])
        inset.plot(df_inset[datefield], df_inset[varfield])
        len_inset = len(df_inset[datefield].values)
        ticks = [df_inset[datefield].values[0], df_inset[datefield].values[int(len_inset/2)], df_inset[datefield].values[-1]]
        inset.set_xticks(ticks)
        inset.tick_params(axis='both', which='major', labelsize=8)
        inset.grid(True, 'both')
    #
    # Exceedance curve
    ax = plt.subplot(gs[1, 0])
    aux_str = r'$\bf{' + 'b.  ' + '}$' + 'Exceedance Prob. Curve'
    plt.title(aux_str, fontsize=10, loc='left')
    plt.plot(df2['Exeedance'], df2['Values'])
    plt.yscale(scale)
    plt.ylabel(ylbls)
    plt.xlabel('Exceedance probability (%)', fontsize=10)
    plt.grid(True, 'both')
    #
    # Violinplot
    tpl = ('January', 'February', 'March', 'April', 'May', 'June', 'July',
           'August', 'September', 'October', 'November', 'December')
    violin_data = list()
    for def_i in range(len(bymonth)):
        lcl_y = bymonth[tpl[def_i]][varfield].values[:]
        violin_data.append(lcl_y)
    ax = plt.subplot(gs[1, 1:])
    aux_str = r'$\bf{' + 'c.  ' + '}$' + 'Seasonality Analysis'
    plt.title(aux_str, fontsize=10, loc='left')
    plt.ylabel(ylbls)
    plt.yscale(scale)
    ax.violinplot(violin_data, showmedians=True)
    ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    ax.set_xticklabels(list('JFMAMJJASOND'))
    plt.xlabel("Month", fontsize=10)
    #
    if show:
        plt.show()
    else:
        aux_str = folder + '/' + flenm + '.png'
        plt.savefig(aux_str)
        plt.close()
    return aux_str