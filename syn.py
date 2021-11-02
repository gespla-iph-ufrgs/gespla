"""
***** UNIVERSIDADE FEDERAL DO RIO GRANDE DO SUL *****
********** GESPLA IPH/UFGRS PYTHON LIBRARY **********

Website: https://www.ufrgs.br/warp/
Repository: https://github.com/gespla-iph-ufrgs/gespla

This file is under LICENSE: GNU General Public License v3.0
Permissions:
    Commercial use
    Modification
    Distribution
    Patent use
    Private use
Limitations:
    Liability
    Warranty
Conditions:
    License and copyright notice
    State changes
    Disclose source
    Same license

Module description:
-- collection of model functions and convenience functions for synthetic time series generation

Authors:
Ipora Possantti: https://github.com/ipo-exe

First commit: 11 of November of 2021

"""

import numpy as np
import pandas as pd
import resample
import tsa


def transition_matrix(states_series):
    """
    Auxiliar function to compute a transition matrix given a series of states
    :param states_series: 1d numpy array of integer states
    :return: 2d numpy array transition matrix and 1d numpy array of unique states.
    States are sorted in ascending order along the matrix axis
    Example:

    states : [1, 2]

    transition matrix:
            1           2
    1 [[0.06666667 0.93333333]
    2  [0.13084112 0.86915888]]

    """
    # get unique states array
    unique_states = np.unique(states_series)
    # deploy couting matrix
    counting_matrix = np.zeros(shape=(len(unique_states), len(unique_states)))
    # loop in counting matrix
    for i in range(len(unique_states)):
        current_state = unique_states[i]
        for j in range(len(unique_states)):
            next_state = unique_states[j]
            transition_array = np.array([current_state, next_state])
            # scan the series searching for transitions matching
            for t in range(len(states_series) - 1):
                # get checker array (add one to avoid division by zero)
                checker = 1 * ((states_series[t: t + 2] + 1) / (transition_array + 1))
                if np.prod(checker) == 1:
                    counting_matrix[i][j] = counting_matrix[i][j] + 1
    # deploy transition matrix
    trans_matrix = np.zeros(shape=(len(unique_states), len(unique_states)))
    # compute transition matrix
    for i in range(len(unique_states)):
        trans_matrix[i] = counting_matrix[i] / np.sum(counting_matrix[i])
    return trans_matrix, unique_states


def synthetic_states(states, t_matrix, size=1000):
    """
    generate a synthetic series of states
    :param states: 1d numpy array of unique states (integer values)
    :param t_matrix: 2d numpy array of transition probabilities
    :param size: int length of synthetic series
    :return: 1d numpy array of synthetic series of states
    """
    from datetime import datetime
    # deploy random state
    seed = int(str(datetime.now())[-6:])
    np.random.seed(seed)
    # deploy series:
    syn_series = np.zeros(shape=size)
    # set first state:
    syn_series[0] = np.random.choice(a=states, size=1)
    # loop in series
    for t in range(len(syn_series) - 1):
        # find the state index in the matrix
        current_state_index = np.where(states == syn_series[t])[0][0]
        # choose by passing the states and probabilities
        syn_series[t + 1] = np.random.choice(a=states, p=t_matrix[current_state_index])
    return syn_series


def synthetic_values(states_series, cfcs, states):
    """
    Generate a synthetic series based on states and respective Cumulative Frequency Curve (CFC)
    :param states_series: 1d numpy array series of states (integer values)
    :param cfcs: dict of cfcs. Keys must be the string version of the states
    :param states: 1d numpy array of unique states (integer values)
    :return: 1d numpy array of synthetic series
    """
    from datetime import datetime
    # deploy random state
    seed = int(str(datetime.now())[-6:])
    np.random.seed(seed)
    # deploy array of series
    syn_values = np.zeros(shape=len(states_series))
    # loop in states
    for s in states:
        # loop in series
        for t in range(len(states_series)):
            if states_series[t] == s:
                # random choice in cfc
                syn_values[t] = np.random.choice(a=cfcs[str(s)], size=1)[0]
    return syn_values


def syn_prec(fseries, start_date='1970-01-01', end_date='2020-01-01', date_field='Date', var_field='Prec'):
    """
    generate a daily synthetic precipitation series using Markov chain
    :param fseries: string filepath to observed series
    :param start_date: string of starting date format: AAAA-MM-DD
    :param end_date: string of ending date format: AAAA-MM-DD
    :param date_field: string of date field in observed series
    :param var_field: string for precipitation field in observed series
    :return: pandas dataframe of synthetic time series.
    Fields:
    - Date
    - Prec_syn
    - Prec_obs
    """
    # deploy syn series
    syn_df = pd.DataFrame()
    syn_df['Date'] = pd.date_range(start=start_date, end=end_date, freq='D')
    syn_df['Month'] = syn_df['Date'].apply(lambda x: x.strftime('%B'))
    syn_df['Prec_syn'] = 0.0
    #
    # estimate of size:
    size_months = int(1.5 * len(syn_df)/12)
    #
    # read observed series
    series_df = pd.read_csv(fseries, sep=';', parse_dates=['Date'])
    #
    # resample by month
    month_dct = resample.group_by_month(dataframe=series_df, var_field=var_field)
    months_lst = list(month_dct.keys())
    #
    # compute the CFCs for each month
    cfcs_dct = dict()
    for i in range(len(months_lst)):
        lcl_freq = tsa.frequency(dataframe=month_dct[months_lst[i]], var_field=var_field, zero=False, step=0.1)
        if i == 0:
            cfcs_dct['Exceedance'] = lcl_freq['Exceedance'].values
        cfcs_dct[months_lst[i]] = lcl_freq['Values'].values
    cfcs = pd.DataFrame(cfcs_dct)
    #
    # compute a synthetic series for each month:
    syn_months = dict()
    for m in months_lst:
        lcl_prec = month_dct[m]['Prec'].values
        # convert the array to a boolean array to get the rain states
        states_prec = (lcl_prec > 0) * 1
        # compute the transition matrix
        t_matrix, un_states = transition_matrix(states_series=states_prec)
        # compute the synthetic series of states
        lcl_syn_states = synthetic_states(states=un_states, t_matrix=t_matrix, size=size_months)
        #
        # built dict of states CFCs:
        lcl_cfs = {'0': cfcs[m].values * 0, '1': cfcs[m].values}
        # assign values to the states
        lcl_syn = synthetic_values(states_series=lcl_syn_states, cfcs=lcl_cfs, states=un_states)
        syn_months[m] = lcl_syn
    # insert values in list based on month
    for m in months_lst:
        #
        count = 0
        for t in range(len(syn_df)):
            if syn_df['Month'].values[t] == m:
                syn_df['Prec_syn'].values[t] = syn_months[m][count]
                count = count + 1
    # output settings
    syn_df.drop(columns=['Month'], inplace=True)
    syn_df = pd.merge(left=syn_df, right=series_df[[date_field, var_field]], on='Date', how='left')
    syn_df.rename(columns={var_field: 'Prec_obs'}, inplace=True)
    return syn_df
