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
--Download data and metadata functions
--Files are saved in .txt format.

Authors:
Marcio Inada: https://github.com/mshigue
Ipora Possantti: https://github.com/ipo-exe

First commit: 20 of October of 2020
'''

import pandas as pd
import numpy as np


def resampler(dataframe, var_field, date_field='Date', type='Month', include_zero=True):
    """
    This function is the resampler function. It takes a time series and resample variables based on a
    type of time scale.


    :param dataframe: pandas DataFrame object
    :param var_field: string head of the variable field.
    :param date_field: string head of the date field. Default: 'Date'
    :param type: time scale type of resampling. Options:

    - 'Month' -  Monthly resample
    - 'Year' - Yearly resample

    :param include_zero: boolean to control if zero value is included or not. Default is to include (True)
    :return: pandas DataFrame object with resampled time series variables:

    - Periods count  - number of periods aggregated
    - Count  - number of valid records aggregated
    - Sum
    - Mean
    - Min
    - Min
    - Max
    - Q25 -  quantile 25
    - Q50 - quantile 50 (median)
    - Q75 - quantile 75

    head field are concatenated with the 'var_field' string parameter. Example:
    Flow_Sum, Flow_Mean, Flow_Min, etc.

    """
    def_df = dataframe.copy()
    def_df.set_index(date_field, inplace=True)
    resam_key = ''
    if type == 'Month':
        resam_key = 'MS'
    if type == 'Year':
        resam_key = 'AS'
    def_out = pd.DataFrame()
    if include_zero:
        na = ''
    else:
        na = 0.0
    def_out['Period_Count'] = def_df.resample(resam_key).count()[var_field]
    def_out[var_field + '_Count'] = def_df.replace(na, np.nan).resample(resam_key).count()[var_field]
    def_out[var_field + '_Sum'] = def_df.replace(na, np.nan).resample(resam_key).sum()[var_field].replace(0.0, np.nan)
    def_out[var_field + '_Mean'] = def_df.replace(na, np.nan).resample(resam_key).mean()[var_field]
    def_out[var_field + '_Min'] = def_df.replace(na, np.nan).resample(resam_key).min()[var_field]
    def_out[var_field + '_Max'] = def_df.replace(na, np.nan).resample(resam_key).max()[var_field]
    def_out[var_field + '_Q25'] = def_df.replace(na, np.nan).resample(resam_key).quantile(0.25)[var_field]
    def_out[var_field + '_Q50'] = def_df.replace(na, np.nan).resample(resam_key).quantile(0.5)[var_field]
    def_out[var_field + '_Q75'] = def_df.replace(na, np.nan).resample(resam_key).quantile(0.75)[var_field]
    def_out.reset_index(inplace=True)
    return def_out


def clear_bad_years(dataframe, var_field, date_field='Date'):
    """
    This function clears a daily time series from 'bad months', which are
    considered months with ANY null record.

    :param dataframe: pandas DataFrame object with the 'dirty' daily series
    :param var_field: string head of the variable field.
    :param date_field: string head of the date field. Default: 'Date'
    :return: pandas DataFrame object with the 'cleared' daily series
    """
    pd.options.mode.chained_assignment = None
    # get DataFrame
    def_df = dataframe.copy()
    # create a helper year-month field
    def_df['Y'] = def_df[date_field].apply(lambda x: x.strftime('%Y'))
    # get all null dates
    dates_null = def_df[def_df[var_field].isnull()]
    # extract all unique months from dates_null
    bad_years = dates_null['Y'].unique()
    # get bad dates:
    for i in range(len(bad_years)):
        bad_dates = def_df['Y'] == bad_years[i]
        def_df[var_field].loc[def_df[bad_dates].index] = np.nan
    def_df.drop('Y', axis='columns', inplace=True)  # drop helper field
    return def_df


def clear_bad_months(dataframe, var_field, date_field='Date'):
    """
    This function clears a daily time series from 'bad months', which are
    considered months with ANY null record.

    :param dataframe: pandas DataFrame object with the 'dirty' daily series
    :param var_field: string head of the variable field.
    :param date_field: string head of the date field. Default: 'Date'
    :return: pandas DataFrame object with the 'cleared' daily series
    """
    pd.options.mode.chained_assignment = None
    # get DataFrame
    def_df = dataframe.copy()
    #
    # create a helper year-month field
    def_df['Y-M'] = def_df[date_field].apply(lambda x: x.strftime('%B-%Y'))
    # get all null dates
    dates_null = def_df[def_df[var_field].isnull()]
    # extract all unique months from dates_null
    bad_months = dates_null['Y-M'].unique()
    # get bad dates:
    for i in range(len(bad_months)):
        bad_dates = def_df['Y-M'] == bad_months[i]
        def_df[var_field].loc[def_df[bad_dates].index] = np.nan
    def_df.drop('Y-M', axis='columns', inplace=True)  # drop helper field
    return def_df


def d2m_prec(dataframe, var_field='Prec', date_field='Date'):
    """
    This functions resamples a precipitation daily time series and returns the
    aggregated monthly time series with sum, mean, max, min and quantiles

    -------
    ** Bad Months **
    For statistical accuracy, months with ANY null record on the daily time series
    are considered 'bad' months and a null value is assigned to it.
    -------

    :param dataframe: pandas DataFrame object with the daily series
    :param date_field: string head of the date field. Default: 'Date'
    :param var_field:  string head of the variable field. Default: 'Flow'
    :return: pandas DataFrame object with the monthly time series. Columns:

     - 'Date' - Month date
     - 'Sum' - Monthly Accumulated Precip (mm/month)
     - 'Avg' - Monthly Average including zero-values.
     - 'Mean' - Monthly mean (exluding zero-values)
     - 'Min' - Monthly minimum (exluding zero-values)
     - 'Max' - Monthly maximum (exluding zero-values)
     - 'Q25' - Monthly 25% Quantile (exluding zero-values)
     - 'Q50' - Monthly 50% Quantile (Median) (exluding zero-values)
     - 'Q75' - Monthly 75% Quantile (exluding zero-values)

    """
    # clear bad months:
    def_df = clear_bad_months(dataframe, var_field=var_field, date_field=date_field)
    # call the resampler function:
    def_out = resampler(def_df, var_field=var_field, date_field=date_field, type='Month', include_zero=False)
    return def_out.copy()


def d2m_flow(dataframe, factor=1.0, var_field='Flow', date_field='Date'):
    """
    his functions resamples a precipitation daily time series and returns the
    aggregated monthly time series with sum, mean, max, min and quantiles

    -------
    ** Bad Months **
    For statistical accuracy, months with ANY null record on the daily time series
    are considered 'bad' months and a null value is assigned to it.

    -------
    ** Flow Units **
    The daily time series unit is considered to be volume/ seconds.
    Therefore, for monthly accumulation, it is converted first in volume/day
    multiplying by 86400.


    :param dataframe: pandas DataFrame object with the daily series
    :param factor: volume unit conversion factor. Default is 1.0 so the volume unit is the same.
    :param date_field: string head of the date field. Default: 'Date'
    :param var_field:  string head of the variable field. Default: 'Flow'
    :return: pandas DataFrame object with the monthly time series. Columns:

     - 'Date' - Month date
     - 'Sum' - Monthly Accumulated Flow (volume units/month)
     - 'Mean' - Monthly mean
     - 'Min' - Monthly minimum
     - 'Max' - Monthly maximum
     - 'Q25' - Monthly 25% Quantile
     - 'Q50' - Monthly 50% Quantile (Median)
     - 'Q75' - Monthly 75% Quantile

    """
    # clear bad months:
    def_df = clear_bad_months(dataframe, var_field=var_field, date_field=date_field)
    #
    # Overwrite the variable field to flow units per day
    def_df[var_field] = def_df[var_field].apply(lambda x: x * 86400 * factor)
    #
    # call the resampler function:
    def_out = resampler(def_df, var_field=var_field, date_field=date_field, type='Month')
    return def_out.copy()


def d2m_stage(dataframe, var_field='Stage', date_field='Date'):
    """
    This functions resamples a stage daily time series and returns the
    aggregated monthly time series with mean, max, min and quantiles

    -------
    ** Bad Months **
    For statistical accuracy, months with ANY null record on the daily time series
    are considered 'bad' months and a null value is assigned to it.
    -------

    :param dataframe: pandas DataFrame object with the daily series
    :param date_field: string head of the date field. Default: 'Date'
    :param var_field:  string head of the variable field. Default: 'Stage'
    :return: pandas DataFrame object with the monthly time series. Columns:

     - 'Date' - Month date
     - 'Mean' - Monthly mean (exluding zero-values)
     - 'Min' - Monthly minimum (exluding zero-values)
     - 'Max' - Monthly maximum (exluding zero-values)
     - 'Q25' - Monthly 25% Quantile (exluding zero-values)
     - 'Q50' - Monthly 50% Quantile (Median) (exluding zero-values)
     - 'Q75' - Monthly 75% Quantile (exluding zero-values)

    """
    # clear bad months:
    def_df = clear_bad_months(dataframe, var_field=var_field, date_field=date_field)
    #
    # call the resampler function:
    def_out = resampler(def_df, var_field=var_field, date_field=date_field, type='Month')
    # drop the Sum field - makes no sense:
    def_out.drop(var_field + '_Sum', axis='columns', inplace=True)
    #
    return def_out.copy()


def d2m_clim(dataframe, var_field, date_field='Date'):
    """
    This functions resamples a climatic daily time series and returns the
    aggregated monthly time series with mean, max, min and quantiles

    By climatic variable we mean:
    Temperature, Relative Humidity, Sunshine Hours, etc
    No accumulated values are valid

    -------
    ** Bad Months **
    For statistical accuracy, months with ANY null record on the daily time series
    are considered 'bad' months and a null value is assigned to it.

    -------
    :param dataframe: pandas DataFrame object with the daily series
    :param date_field: string head of the date field. Default: 'Date'
    :param var_field:  string head of the climatic variable field.
    :return: pandas DataFrame object with the monthly time series. Columns:
     - 'Date' - Month date
     - 'Mean' - Monthly mean (exluding zero-values)
     - 'Min' - Monthly minimum (exluding zero-values)
     - 'Max' - Monthly maximum (exluding zero-values)
     - 'Q25' - Monthly 25% Quantile (exluding zero-values)
     - 'Q50' - Monthly 50% Quantile (Median) (exluding zero-values)
     - 'Q75' - Monthly 75% Quantile (exluding zero-values)

    """
    # clear bad months:
    def_df = clear_bad_months(dataframe, var_field=var_field, date_field=date_field)
    #
    # call the resampler function:
    def_out = resampler(def_df, var_field=var_field, date_field=date_field, type='Month')
    # drop the Sum field - makes no sense:
    def_out.drop(var_field + '_Sum', axis='columns', inplace=True)
    #
    return def_out.copy()


def d2y_prec(dataframe, var_field='Prec', date_field='Date'):
    """
    This functions resamples a precipitation daily time series and returns the
    aggregated yearly time series with sum, mean, max, min and quantiles

    -------
    ** Bad Years **
    For statistical accuracy, years with ANY null record on the daily time series
    are considered 'bad' years and a null value is assigned to it.
    -------

    :param dataframe: pandas DataFrame object with the daily series
    :param date_field: string head of the date field. Default: 'Date'
    :param var_field:  string head of the variable field. Default: 'Flow'
    :return: pandas DataFrame object with the yearly time series. Columns:

     - 'Date' - year date
     - 'Sum' - Yearly Accumulated Precipitation (mm/year)
     - 'Avg' - Yearly Average including zero-values.
     - 'Mean' - Yearly mean (exluding zero-values)
     - 'Min' - Yearly minimum (exluding zero-values)
     - 'Max' - Yearly maximum (exluding zero-values)
     - 'Q25' - Yearly 25% Quantile (exluding zero-values)
     - 'Q50' - Yearly 50% Quantile (Median) (exluding zero-values)
     - 'Q75' - Yearly 75% Quantile (exluding zero-values)

    """
    # clear bad months:
    def_df = clear_bad_years(dataframe, var_field=var_field, date_field=date_field)
    #
    # Finally resamples by year
    # call the resampler function:
    def_out = resampler(def_df, var_field=var_field, date_field=date_field, type='Year', include_zero=False)
    return def_out.copy()


def d2y_flow(dataframe, factor=1.0, var_field='Flow', date_field='Date'):
    """
    his functions resamples a precipitation daily time series and returns the
    aggregated yearly time series with sum, mean, max, min and quantiles

    -------
    ** Bad Years **
    For statistical accuracy, years with ANY null record on the daily time series
    are considered 'bad' years and a null value is assigned to it.

    -------
    ** Flow Units **
    The daily time series unit is considered to be volume/ seconds.
    Therefore, for yearly accumulation, it is converted first in volume/day
    multiplying by 86400.


    :param dataframe: pandas DataFrame object with the daily series
    :param factor: volume unit conversion factor. Default is 1.0 so the volume unit is the same.
    :param date_field: string head of the date field. Default: 'Date'
    :param var_field:  string head of the variable field. Default: 'Flow'
    :return: pandas DataFrame object with the yearly time series. Columns:

     - 'Date' - Year date
     - 'Sum' - Yearly Accumulated Flow (volume units/year)
     - 'Mean' - Yearly mean
     - 'Min' - Yearly minimum
     - 'Max' - Yearly maximum
     - 'Q25' - Yearly 25% Quantile
     - 'Q50' - Yearly 50% Quantile (Median)
     - 'Q75' - Yearly 75% Quantile

    """
    # clear bad months:
    def_df = clear_bad_years(dataframe, var_field=var_field, date_field=date_field)
    #
    # Overwrite the variable field to flow units per day
    def_df[var_field] = def_df[var_field].apply(lambda x: x * 86400 * factor)
    #
    # call the resampler function:
    def_out = resampler(def_df, var_field=var_field, date_field=date_field, type='Year')
    return def_out.copy()


def d2y_stage(dataframe, date_field='Date', var_field='Stage'):
    """
    This functions resamples a stage daily time series and returns the
    aggregated yearly time series with mean, max, min and quantiles

    -------
    ** Bad Years **
    For statistical accuracy, years with ANY null record on the daily time series
    are considered 'bad' years and a null value is assigned to it.
    -------

    :param dataframe: pandas DataFrame object with the daily series
    :param date_field: string head of the date field. Default: 'Date'
    :param var_field:  string head of the variable field. Default: 'Stage'
    :return: pandas DataFrame object with the yearly time series. Columns:

     - 'Date' - Year date
     - 'Mean' - Yearly mean (exluding zero-values)
     - 'Min' - Yearly minimum (exluding zero-values)
     - 'Max' - Yearly maximum (exluding zero-values)
     - 'Q25' - Yearly 25% Quantile (exluding zero-values)
     - 'Q50' - Yearly 50% Quantile (Median) (exluding zero-values)
     - 'Q75' - Yearly 75% Quantile (exluding zero-values)

    """
    # clear bad months:
    def_df = clear_bad_months(dataframe, var_field=var_field, date_field=date_field)
    #
    # call the resampler function:
    def_out = resampler(def_df, var_field=var_field, date_field=date_field, type='Year')
    # drop the Sum field - makes no sense:
    def_out.drop(var_field + '_Sum', axis='columns', inplace=True)
    #
    return def_out.copy()


def d2y_clim(dataframe,  var_field, date_field='Date'):
    """
    This functions resamples a climatic daily time series and returns the
    aggregated yearly time series with mean, max, min and quantiles

    By climatic variable we mean:
    Temperature, Relative Humidity, Sunshine Hours, etc
    No accumulated values are valid

    -------
    ** Bad Years **
    For statistical accuracy, years with ANY null record on the daily time series
    are considered 'bad' years and a null value is assigned to it.
    -------

    :param dataframe: pandas DataFrame object with the daily series
    :param date_field: string head of the date field. Default: 'Date'
    :param var_field:  string head of the variable field.
    :return: pandas DataFrame object with the yearly time series. Columns:

     - 'Date' - Year date
     - 'Mean' - Yearly mean (exluding zero-values)
     - 'Min' - Yearly minimum (exluding zero-values)
     - 'Max' - Yearly maximum (exluding zero-values)
     - 'Q25' - Yearly 25% Quantile (exluding zero-values)
     - 'Q50' - Yearly 50% Quantile (Median) (exluding zero-values)
     - 'Q75' - Yearly 75% Quantile (exluding zero-values)

    """
    # clear bad months:
    def_df = clear_bad_months(dataframe, var_field=var_field, date_field=date_field)
    #
    # call the resampler function:
    def_out = resampler(def_df, var_field=var_field, date_field=date_field, type='Year')
    # drop the Sum field - (makes no sense):
    def_out.drop(var_field + '_Sum', axis='columns', inplace=True)
    #
    return def_out.copy()

