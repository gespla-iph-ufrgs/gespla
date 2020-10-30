'''
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
-- collection of model functions and convenience functions
for resample time scale

Authors:
Ipora Possantti: https://github.com/ipo-exe

First commit: 30 of October of 2020
'''

import pandas as pd
import numpy as np


def offset_converter(offset):
    """
    Convenience function for converting human readable string to pandas default offsets
    :param offset: string offsets. options:

    hour
    day
    month
    year

    :return: string of pandas default offsets
    """
    def_freq = 'D'
    if offset.strip().lower() == 'day':
        def_freq = 'D'
    elif offset.strip().lower() == 'month':
        def_freq = 'MS'
    elif offset.strip().lower() == 'year':
        def_freq = 'AS'
    elif offset.strip().lower() == 'hour':
        def_freq = 'H'
    else:
        def_freq = offset
    return def_freq


def sma(dataframe, var_field, window=7, date_field='Date', freq='month'):
    """

    This functions performs Simple Moving Average on time series.

    :param dataframe: pandas DataFrame object with time series
    :param var_field: string of variable field
    :param window: int of window time span
    :param date_field: string of date field
    :param freq: string frequency of time scale. Default: 'month' options:

    hour
    day
    month
    year

    :return: DataFrame object with time series with 3 columns:

    'Date' - dates
    'Signal' - observed values of time series
    'SMA' -  Simple Moving Average values

    """
    #
    # get dataframe right
    in_df = dataframe.copy()
    in_df.set_index(date_field, inplace=True)
    in_df.index = pd.to_datetime(in_df.index)
    in_df.index.freq = offset_converter(freq)
    #
    def_lbl = 'SMA'
    in_df[def_lbl] = in_df.rolling(window=window).mean()[var_field]
    #
    # built the output dataframe
    in_df.reset_index(inplace=True)
    out_dct = {'Date': in_df[date_field].values,
               'Signal': in_df[var_field].values,
               def_lbl: in_df[def_lbl].values}
    out_df = pd.DataFrame(out_dct)
    return out_df


def ewma(dataframe, var_field, window=12, date_field='Date', freq='month'):
    """

    This functions performs Exponential Weighting Moving Average on time series.

    :param dataframe: pandas DataFrame object with time series
    :param var_field: string of variable field
    :param window: int of window time span
    :param date_field: string of date field
    :param freq: string frequency of time scale. Default: 'month' options:

    hour
    day
    month
    year

    :return: DataFrame object with time series with 3 columns:

    'Date' - dates
    'Signal' - observed values of time series
    'EWMA' -  Exponential Weighting Moving Average values

    """
    # get dataframe right
    in_df = dataframe.copy()
    in_df.set_index(date_field, inplace=True)
    in_df.index = pd.to_datetime(in_df.index)
    in_df.index.freq = offset_converter(freq)
    #
    in_df['EWMA'] = in_df[var_field].ewm(span=window).mean()
    #
    # built the output dataframe
    in_df.reset_index(inplace=True)
    out_dct = {'Date': in_df[date_field].values,
               'Signal': in_df[var_field].values,
               'EWMA': in_df['EWMA'].values}
    out_df = pd.DataFrame(out_dct)
    return out_df


def hpfilter(dataframe, var_field, lamb=1600, date_field='Date'):
    """

    This functions performs the Hodrick-Prescott Filter on time series.

    :param dataframe: pandas DataFrame object with time series
    :param var_field: string of variable field
    :param lamb: lambda parameter of the Hodrick-Prescott Filter
    :param date_field: string of date field. Default is 1600
    :return: DataFrame object with time series with 3 columns:

    'Date' - dates
    'Signal' - observed values of time series
    'HP-Filter' -  Exponential Weighting Moving Average values

    External dependency: Statsmodels

    """
    # The Hodrick-Prescott Filter:
    from statsmodels.tsa.filters.hp_filter import hpfilter
    #
    # get dataframe right
    in_df = dataframe.copy()
    in_df.set_index(date_field, inplace=True)
    in_df.index = pd.to_datetime(in_df.index)
    #
    gdp_cycle, gdp_trend = hpfilter(in_df[var_field], lamb=lamb)
    #
    # built the output dataframe
    in_df.reset_index(inplace=True)
    out_dct = {'Date': in_df[date_field].values,
               'Signal': in_df[var_field].values,
               'HP-Filter': gdp_trend.values}
    out_df = pd.DataFrame(out_dct)
    return out_df


def ets_decomp(dataframe, var_field, type='additive', date_field='Date'):
    """

    This functions performs the ETS Decomposition on time series.

    :param dataframe: pandas DataFrame object with time series
    :param var_field: string of variable field
    :param type: Type of model. Options:

    'additive'
    'multiplicative'

    :param date_field: string of date field. Default is 1600
    :return: DataFrame object with time series with 3 columns:

    'Date' - dates
    'Signal' - observed values of time series
    'Trend' -  Decomposed trend
    'Season' -  Decomposed seasonality
    'Noise' -  Decomposed residuals (error)
    External dependency: Statsmodels

    """
    from statsmodels.tsa.seasonal import seasonal_decompose
    # get dataframe right
    in_df = dataframe.copy()
    in_df.set_index(date_field, inplace=True)
    in_df.index = pd.to_datetime(in_df.index)
    #
    decomp = seasonal_decompose(in_df[var_field], model=type)
    # get separate components
    series = decomp.observed
    trend = decomp.trend
    season = decomp.seasonal
    resid = decomp.resid
    # built the output dataframe
    in_df.reset_index(inplace=True)
    out_dct = {'Date': in_df[date_field].values,
               'Signal': in_df[var_field].values,
               'Trend': trend.values,
               'Season': season.values,
               'Noise': resid.values}
    out_df = pd.DataFrame(out_dct)
    return out_df


def simple_exp_smooth(dataframe, var_field, date_field='Date', freq='month', span=12):
    """

    :param dataframe:
    :param var_field:
    :param date_field:
    :param freq:
    :param span:
    :return:
    """
    #
    # import dependencies:
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing
    #
    # get dataframe right
    #
    # fit model:

    # retrieve the fittet values
    #


def arma_forecast(dataframe):
    print()


def arima_forecast(dataframe):
    print()


def sarima_forecast(dataframe):
    print()


def sarima_forecast(dataframe):
    print()





