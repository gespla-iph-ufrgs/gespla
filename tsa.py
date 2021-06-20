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
import warnings
#
warnings.filterwarnings('ignore')


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


def compare(dataframes, var_fields, labels, start='full', end='full', date_field='Date', freq='day'):
    """
    Compare fields between dataframes on a single time series
    :param dataframes: iterable of dataframes
    :param var_fields: iterable of strings of fields to compare in dataframes (same length)
    :param labels: iterable of labels to append to each dataframe field (same length)
    :param start: string of start date or 'full' to get the minimal date
    :param end: string of end date or 'full' to get the maximal date
    :param date_field: common date field
    :param freq: string frequency of time scale. Default: 'day' (daily) options:

    hour
    day
    month
    year
    :return: dataframe of compared time series and a 'Deficit' index field of missing data (1 is fully missing)
    """
    if start == 'full':
        start_lst = list()
        for i in range(len(dataframes)):
            lcl_df = dataframes[i]
            start_lst.append(lcl_df[date_field].min())
        start = pd.to_datetime(start_lst).min()
    if end == 'full':
        end_lst = list()
        for i in range(len(dataframes)):
            lcl_df = dataframes[i]
            end_lst.append(lcl_df[date_field].max())
        end = pd.to_datetime(end_lst).max()
    # create the reference date index
    def_freq = offset_converter(freq)
    ref_dates = pd.date_range(start=start, end=end, freq=def_freq)
    # create the reference dataset
    ref_df = pd.DataFrame({'Date': ref_dates})
    for i in range(len(dataframes)):
        lcl_df = dataframes[i].copy()
        lcl_df = lcl_df[[date_field, var_fields[i]]]
        lcl_df.columns = [date_field, var_fields[i] + '_' + labels[i]]
        # left join on datasets
        if i == 0:
            merge = pd.merge(ref_df, lcl_df, how='left', left_on='Date', right_on=date_field)
        else:
            merge = pd.merge(merge, lcl_df, how='left', left_on='Date', right_on=date_field)
    # compute data deficit
    total = len(merge.columns) - 1
    print(total)
    for i in range(1, len(merge.columns)):
        if i == 1:
            accum = 1 * np.isnan(merge[merge.columns.values[i]].values)
        else:
            accum = accum + (1 * np.isnan(merge[merge.columns.values[i]].values))
    merge['Deficit'] = accum / total
    return merge



def insert_gaps(dataframe, date_field='Date', freq='month'):
    """
    This is a convenience function that standardizes a timeseries by inserting the missing gaps as actual records
    :param dataframe: pandas DataFrame object
    :param date_field: string datefield - Default: 'Date'
    :param freq: string frequency of time scale. Default: 'month' (monthly) options:

    hour
    day
    month
    year

    :return: pandas DataFrame object with inserted gaps records
    """
    # get data from DataFrame
    in_df = dataframe.copy()
    # ensure Date field is datetime
    in_df[date_field] = pd.to_datetime(in_df[date_field])
    # create start and end values
    start = in_df[date_field].min()
    end = in_df[date_field].max()
    # create the reference date index
    def_freq = offset_converter(freq)
    ref_dates = pd.date_range(start=start, end=end, freq=def_freq)
    # create the reference dataset
    ref_df = pd.DataFrame({'Date':ref_dates})
    # left join on datasets
    merge = pd.merge(ref_df, in_df, how='left', left_on='Date', right_on=date_field)
    return merge


def eval_prediction(dataframe, obs_field, pred_field):
    """

    This utility function performs the evaluation of observed and predicted arrays values

    :param dataframe: pandas DataFrame object with observed series and predicted series.
    :param obs_field: string head of observed field
    :param pred_field: string head of prediciton field
    :return: dictionary with evaluation metrics

    """
    import statsmodels.tools.eval_measures as metrics
    # get arrays
    in_df = dataframe[[obs_field, pred_field]].copy()
    in_df.dropna(inplace=True)
    obs = in_df[obs_field].values[:]
    pred = in_df[pred_field].values[:]
    #
    mse = metrics.mse(obs, pred)
    rmse = metrics.rmse(obs, pred)
    mabs = metrics.maxabs(obs, pred)
    meanabs = metrics.meanabs(obs, pred)
    out_dct = {'MSE':mse, 'RMSE':rmse, 'Max Abs Error':mabs, 'Mean Abs Error':meanabs}
    return out_dct


def frequency(dataframe, var_field, zero=True):
    """

    This fuction performs a frequency analysis on a given time series.

    :param dataframe: pandas DataFrame object with time series
    :param var_field: string of variable field
    :param zero: boolean control to consider values of zero. Default: True
    :return: pandas DataFrame object with the following columns:

     'Pecentiles' - percentiles in % of array values (from 0 to 100 by steps of 1%)
     'Exeedance' - exeedance probability in % (reverse of percentiles)
     'Frequency' - count of values on the histogram bin defined by the percentiles
     'Probability'- local bin empirical probability defined by frequency/count
     'Values' - values percentiles of bins

    """

    # get dataframe right
    in_df = dataframe[[var_field]].copy()
    in_df = in_df.dropna()
    if zero:
        pass
    else:
        mask = in_df[var_field] != 0
        in_df = in_df[mask]
    def_v = in_df[var_field].values
    ptles = np.arange(0, 101, 1)
    cfc = np.percentile(def_v, ptles)
    exeed = 100 - ptles
    freq = np.histogram(def_v, bins=101)[0]
    prob = freq/np.sum(freq)
    out_dct = {'Percentiles': ptles, 'Exeedance':exeed, 'Frequency': freq, 'Probability': prob, 'Values':cfc}
    out_df = pd.DataFrame(out_dct)
    return out_df


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
    in_df = dataframe[[date_field, var_field]].copy()
    in_df.set_index(date_field, inplace=True)
    in_df.index = pd.to_datetime(in_df.index)
    in_df.index.freq = offset_converter(freq)
    in_df.dropna(inplace=True)
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
    # insert gaps
    out_fill_df = insert_gaps(out_df, date_field=date_field, freq=freq)
    return out_fill_df


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
    in_df = dataframe[[date_field, var_field]].copy()
    in_df.set_index(date_field, inplace=True)
    in_df.index = pd.to_datetime(in_df.index)
    in_df.index.freq = offset_converter(freq)
    in_df.dropna(inplace=True)
    #
    in_df['EWMA'] = in_df[var_field].ewm(span=window).mean()
    #
    # built the output dataframe
    in_df.reset_index(inplace=True)
    out_dct = {'Date': in_df[date_field].values,
               'Signal': in_df[var_field].values,
               'EWMA': in_df['EWMA'].values}
    out_df = pd.DataFrame(out_dct)
    # insert gaps
    out_fill_df = insert_gaps(out_df, date_field=date_field, freq=freq)
    return out_fill_df


def hpfilter(dataframe, var_field, lamb=1600, date_field='Date', freq='month'):
    """

    This functions performs the Hodrick-Prescott Filter on time series.

    :param dataframe: pandas DataFrame object with time series
    :param var_field: string of variable field
    :param lamb: lambda parameter of the Hodrick-Prescott Filter
    :param date_field: string of date field. Default is 1600
    :param freq: string frequency of time scale. Default: 'month' options:

    hour
    day
    month
    year

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
    in_df = dataframe[[date_field, var_field]].copy()
    in_df.set_index(date_field, inplace=True)
    in_df.index = pd.to_datetime(in_df.index)
    in_df.index.freq = offset_converter(freq)
    in_df.dropna(inplace=True)
    #
    gdp_cycle, gdp_trend = hpfilter(in_df[var_field], lamb=lamb)
    #
    # built the output dataframe
    in_df.reset_index(inplace=True)
    out_dct = {'Date': in_df[date_field].values,
               'Signal': in_df[var_field].values,
               'HP-Filter': gdp_trend.values}
    out_df = pd.DataFrame(out_dct)
    # insert gaps
    out_fill_df = insert_gaps(out_df, date_field=date_field, freq=freq)
    return out_fill_df


def ets_decomposition(dataframe, var_field, type='additive', date_field='Date', freq='month'):
    """

    This functions performs the ETS Decomposition on time series.

    :param dataframe: pandas DataFrame object with time series
    :param var_field: string of variable field
    :param type: Type of model. Options:

    'additive'
    'multiplicative'

    :param date_field: string of date field. Default is 1600
    :param freq: string frequency of time scale. Default: 'month' options:

    hour
    day
    month
    year

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
    in_df = dataframe[[date_field, var_field]].copy()
    in_df.set_index(date_field, inplace=True)
    in_df.index = pd.to_datetime(in_df.index)
    def_freq = offset_converter(freq)
    in_df.index.freq = def_freq
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

# exponential smoothing functions:

def ses(dataframe, var_field, date_field='Date', freq='month', span=12):
    """

    This function performs Simple Exponential Smoothing (Holt Linear) on a given time series

    :param dataframe: pandas DataFrame object with time series
    :param var_field: string of variable field
    :param date_field: string of date field. Default is 1600
    :param freq: string frequency of time scale. Default: 'month' options:

    hour
    day
    month
    year

    :return: DataFrame object with time series with 3 columns:

    'Date' - dates
    'Signal' - observed values of time series
    'SES' -  Simple Exponential Smoothing values

    External dependency: Statsmodels

    """
    #
    # import dependencies:
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing
    #
    # get dataframe right
    in_df = dataframe[[date_field, var_field]].copy()
    in_df.set_index(date_field, inplace=True)
    in_df.index = pd.to_datetime(in_df.index)
    def_freq = offset_converter(freq)
    in_df.index.freq = def_freq
    in_df.dropna(inplace=True)
    #
    # fit model:
    model = SimpleExpSmoothing(in_df[var_field])
    # fit the model
    alpha = 2 / (span + 1)
    fitted_model = model.fit(smoothing_level=alpha, optimized=False)
    # retrieve the fittet values
    in_df['SES'] = fitted_model.fittedvalues.shift(-1)
    # retrieve the fittet values
    #print(in_df.to_string())
    # built the output dataframe
    in_df.reset_index(inplace=True)
    out_dct = {'Date': in_df[date_field].values,
               'Signal': in_df[var_field].values,
               'SES': in_df['SES'].values}
    out_df = pd.DataFrame(out_dct)
    # insert gaps
    out_fill_df = insert_gaps(out_df, date_field=date_field, freq=freq)
    return out_fill_df


def des(dataframe, var_field, date_field='Date', trend='add', freq='month'):
    """

    This function performs Double Exponential Smoothing (Holt-Winters Second Order) on a given time series

    :param dataframe: pandas DataFrame object with time series
    :param var_field: string of variable field
    :param date_field: string of date field. Default is 1600
    :param trend: string code for type of trend. Default: 'add'
    options:

    'add' - Additive trend model
    'mul' - Multiplicative trend model

    :param freq: string frequency of time scale. Default: 'month' options:

    hour
    day
    month
    year

    :return: DataFrame object with time series with 3 columns:

    'Date' - dates
    'Signal' - observed values of time series
    'DES' -  Double Exponential Smoothing values

    External dependency: Statsmodels

    """
    #
    # import dependencies:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    #
    # get dataframe right
    in_df = dataframe[[date_field, var_field]].copy()
    in_df.set_index(date_field, inplace=True)
    in_df.index = pd.to_datetime(in_df.index)
    def_freq = offset_converter(freq)
    in_df.index.freq = def_freq
    in_df.dropna(inplace=True)
    #
    # fit model:
    model = ExponentialSmoothing(in_df[var_field], trend=trend)
    fitted_model = model.fit()
    # retrieve the fittet values
    in_df['DES'] = fitted_model.fittedvalues.shift(-1)
    # retrieve the fittet values
    # print(in_df.to_string())
    # built the output dataframe
    in_df.reset_index(inplace=True)
    out_dct = {'Date': in_df[date_field].values,
               'Signal': in_df[var_field].values,
               'DES': in_df['DES'].values}
    out_df = pd.DataFrame(out_dct)
    # insert gaps
    out_fill_df = insert_gaps(out_df, date_field=date_field, freq=freq)
    return out_fill_df


def tes(dataframe, var_field, date_field='Date', trend='add', season='add', season_p=12, freq='month'):
    """

    This function performs Triple Exponential Smoothing (Holt-Winters Third Order) on a given time series

    :param dataframe: pandas DataFrame object with time series
    :param var_field: string of variable field
    :param date_field: string of date field. Default is 1600
    :param trend: string code for type of trend model. Default: 'add'
    options:

    'add' - Additive trend model
    'mul' - Multiplicative trend model

    :param season: string code for type of seasonality model. Default: 'add'
    options:

    'add' - Additive trend model
    'mul' - Multiplicative trend model

    :param season_p: int for number of seasonal periods. Default: 12 (for month seasonality)
    :param freq: string frequency of time scale. Default: 'month'
    options:

    hour
    day
    month
    year

    :return: DataFrame object with time series with 3 columns:

    'Date' - dates
    'Signal' - observed values of time series
    'TES' -  Double Exponential Smoothing values

    External dependency: Statsmodels

    """
    #
    # import dependencies:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    #
    # get dataframe right
    in_df = dataframe[[date_field, var_field]].copy()
    in_df.set_index(date_field, inplace=True)
    in_df.index = pd.to_datetime(in_df.index)
    def_freq = offset_converter(freq)
    in_df.index.freq = def_freq
    in_df.dropna(inplace=True)
    #
    # fit model:
    model = ExponentialSmoothing(in_df[var_field], trend=trend,  seasonal=season, seasonal_periods=12)
    fitted_model = model.fit()
    # retrieve the fittet values
    in_df['TES'] = fitted_model.fittedvalues.shift(-1)
    # retrieve the fittet values
    # built the output dataframe
    in_df.reset_index(inplace=True)
    out_dct = {'Date': in_df[date_field].values,
               'Signal': in_df[var_field].values,
               'TES': in_df['TES'].values}
    out_df = pd.DataFrame(out_dct)
    # insert gaps
    out_fill_df = insert_gaps(out_df, date_field=date_field, freq=freq)
    return out_fill_df


def des_forecast(dataframe, var_field, forecast=1, split=0.8, date_field='Date', trend='add', freq='month'):
    """

    This function performs Double Exponential Smoothing (Holt-Winters Second Order) fit and forecast
     on a given time series

    :param dataframe: pandas DataFrame object with time series
    :param var_field: string of variable field
    :param forecast: multiplier of the testing horizon for forecast horizon. Default=1

    forecast_horizon = test_horizon + forecast * test_horizon

    :param split: fraction of the split (must be less than 1). Default=0.8 (80% for training and 20% for testing)
    :param date_field: string of date field. Default is 1600
    :param trend: string code for type of trend model. Default: 'add'
    options:

    'add' - Additive trend model
    'mul' - Multiplicative trend model

    :param freq: string frequency of time scale. Default: 'month'
    options:

    hour
    day
    month
    year

    :return: DataFrame object with time series with 5 columns:

    'Date' - dates
    'Signal' - observed values of time series
    'Training' -  Double Exponential Smoothing values of the training set
    'Testing' - Double Exponential Smoothing values of the testing set
    'Forecasting' - Double Exponential Smoothing values of the forecasting set

    External dependency: Statsmodels

    """
    #
    # import dependencies:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tools.eval_measures import rmse
    #
    # get dataframe right
    in_df = dataframe[[date_field, var_field]].copy()
    in_df.set_index(date_field, inplace=True)
    in_df.index = pd.to_datetime(in_df.index)
    def_freq = offset_converter(freq)
    in_df.index.freq = def_freq
    in_df.dropna(inplace=True)
    #
    # Spliting
    full_size = len(in_df[var_field])
    split_id = int(full_size * split)
    training_set = in_df.iloc[:split_id]
    testing_set = in_df.iloc[split_id:]
    #
    # fit the training model:
    model = ExponentialSmoothing(training_set[var_field], trend=trend)
    fitted_model = model.fit()
    #
    # prediction on the testing horizon:
    test_horizon = full_size - split_id
    testing_prediction = fitted_model.forecast(test_horizon)
    #
    # forescast on the forecast horizon:
    forecast_horizon = test_horizon + test_horizon * forecast
    forecasting_prediction = fitted_model.forecast(forecast_horizon)
    #
    # *** Output stuff ***
    # set the forecast dataframe
    forecast_df = pd.DataFrame(forecasting_prediction, columns=['Forecasting'])
    forecast_df.index = forecast_df.index.set_names(['Date1'])
    forecast_df.reset_index(inplace=True)
    #
    # set the observed dataframe
    in_df['Training'] = fitted_model.fittedvalues.shift(-1)
    in_df['Testing'] = testing_prediction
    in_df.index = in_df.index.set_names(['Date0'])
    in_df.reset_index(inplace=True)
    #    #
    # set the reference dataframe
    min_tpl = (in_df['Date0'].min(), forecast_df['Date1'].min())
    max_tpl = (in_df['Date0'].max(), forecast_df['Date1'].max())
    full_dates = pd.date_range(start=min(min_tpl), end=max(max_tpl), freq=def_freq)
    ref_df = pd.DataFrame({'Date': full_dates})
    #
    # built the output dataframe
    merged = pd.merge(ref_df, in_df, how='left', left_on='Date', right_on='Date0').drop('Date0', axis='columns')
    merged = pd.merge(merged, forecast_df, how='left', left_on='Date', right_on='Date1').drop('Date1', axis='columns')
    merged.rename(mapper={var_field:'Signal'}, axis=1, inplace=True)
    return merged


def tes_forecast(dataframe, var_field, forecast=1, split=0.8, date_field='Date', trend='add', season='add', season_p=12, freq='month'):
    """

    This function performs Triple Exponential Smoothing (Holt-Winters Third Order) fit and forecast
     on a given time series

    :param dataframe: pandas DataFrame object with time series
    :param var_field: string of variable field
    :param forecast: multiplier of the testing horizon for forecast horizon. Default=1

    forecast_horizon = test_horizon + forecast * test_horizon

    :param split: fraction of the split (must be less than 1). Default=0.8 (80% for training and 20% for testing)
    :param date_field: string of date field. Default is 1600
    :param trend: string code for type of trend model. Default: 'add'
    options:

    'add' - Additive trend model
    'mul' - Multiplicative trend model

    :param season: string code for type of seasonality model. Default: 'add'
    options:

    'add' - Additive trend model
    'mul' - Multiplicative trend model

    :param season_p: int for number of seasonal periods. Default: 12 (for month seasonality)
    :param freq: string frequency of time scale. Default: 'month'
    options:

    hour
    day
    month
    year

    :return: DataFrame object with time series with 5 columns:

    'Date' - dates
    'Signal' - observed values of time series
    'Training' -  Triple Exponential Smoothing values of the training set
    'Testing' - Triple Exponential Smoothing values of the testing set
    'Forecasting' - Triple Exponential Smoothing values of the forecasting set

    External dependency: Statsmodels

    """
    #
    # import dependencies:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tools.eval_measures import rmse
    #
    # get dataframe right
    in_df = dataframe[[date_field, var_field]].copy()
    in_df.set_index(date_field, inplace=True)
    in_df.index = pd.to_datetime(in_df.index)
    def_freq = offset_converter(freq)
    in_df.index.freq = def_freq
    in_df.dropna(inplace=True)
    #
    # Spliting
    full_size = len(in_df[var_field])
    split_id = int(full_size * split)
    training_set = in_df.iloc[:split_id]
    testing_set = in_df.iloc[split_id:]
    #
    # fit the training model:
    model = ExponentialSmoothing(training_set[var_field], trend=trend,  seasonal=season, seasonal_periods=12)
    fitted_model = model.fit()
    #
    # prediction on the testing horizon:
    test_horizon = full_size - split_id
    testing_prediction = fitted_model.forecast(test_horizon)
    #
    # forescast on the forecast horizon:
    forecast_horizon = test_horizon + test_horizon * forecast
    forecasting_prediction = fitted_model.forecast(forecast_horizon)
    #
    # *** Output stuff ***
    # set the forecast dataframe
    forecast_df = pd.DataFrame(forecasting_prediction, columns=['Forecasting'])
    forecast_df.index = forecast_df.index.set_names(['Date1'])
    forecast_df.reset_index(inplace=True)
    #
    # set the observed dataframe
    in_df['Training'] = fitted_model.fittedvalues.shift(-1)
    in_df['Testing'] = testing_prediction
    in_df.index = in_df.index.set_names(['Date0'])
    in_df.reset_index(inplace=True)
    #
    # set the reference dataframe
    min_tpl = (in_df['Date0'].min(), forecast_df['Date1'].min())
    max_tpl = (in_df['Date0'].max(), forecast_df['Date1'].max())
    full_dates = pd.date_range(start=min(min_tpl), end=max(max_tpl), freq=def_freq)
    ref_df = pd.DataFrame({'Date': full_dates})
    #
    # built the output dataframe
    merged = pd.merge(ref_df, in_df, how='left', left_on='Date', right_on='Date0').drop('Date0', axis='columns')
    merged = pd.merge(merged, forecast_df, how='left', left_on='Date', right_on='Date1').drop('Date1', axis='columns')
    merged.rename(mapper={var_field:'Signal'}, axis=1, inplace=True)
    return merged


def arma(dataframe, var_field, p, q, date_field='Date'):
    #
    # import dependencies:
    from statsmodels.tsa.arima_model import ARMA, ARMAResults
    #
    # get dataframe right
    in_df = dataframe[[date_field, var_field]].copy()
    in_df.set_index(date_field, inplace=True)
    in_df.index = pd.to_datetime(in_df.index)
    def_freq = offset_converter(freq)
    in_df.index.freq = def_freq
    in_df.dropna(inplace=True)
    #
    # fit model:
    model = ARMA(in_df[var_field], order=(p, q))
    fitted_model = model.fit()
    #
    # retrieve the fittet values
    in_df['ARIMA'] = fitted_model.fittedvalues.shift(-1)
    #
    #
    # built the output dataframe
    in_df.reset_index(inplace=True)
    out_dct = {'Date': in_df[date_field].values,
               'Signal': in_df[var_field].values,
               'ARIMA': in_df['ARIMA'].values}
    out_df = pd.DataFrame(out_dct)
    # insert gaps
    out_fill_df = insert_gaps(out_df, date_field=date_field, freq=freq)
    return out_fill_df


def arima(dataframe, var_field, date_field='Date'):
    print()


def sarima(dataframe, var_field, date_field='Date'):
    print()


def arma_forecast(dataframe, var_field, forecast=1, date_field='Date'):
    print()


def arima_forecast(dataframe, var_field, forecast=1, date_field='Date'):
    print()


def sarima_forecast(dataframe, var_field, forecast=1, date_field='Date'):
    print()


def sarima_forecast(dataframe):
    print()





