import pandas as pd
import numpy as np


def d2m_prec(dataframe, date_field='Date', var_field='Prec'):
    """
    This functions resamples a precipitation daily time series and returns the
    aggregated monthly time series with sum, mean, max, min and quantiles

    -------
    ** Bad Months **
    For statistical accuracy, months with ANY null record on the daily time series
    are considered 'bad' months and a null value is assigned to it.
    -------

    :param dataframe: pandas DataFrame object with the daily series
    :param factor: volume unit conversion factor. Default is 1.0 so the volume unit is the same.
    :param date_field: string head of the date field. Default: 'Date'
    :param var_field:  string head of the variable field. Default: 'Flow'
    :return: pandas DataFrame object with the monthly time series. Columns:

     - 'Date' - Month date
     - 'Sum' - Monthly Accumulated Flow (volume units/month)
     - 'Avg' - Monthly Average including zero-values.
     - 'Mean' - Monthly mean (exluding zero-values)
     - 'Min' - Monthly minimum (exluding zero-values)
     - 'Max' - Monthly maximum (exluding zero-values)
     - 'Q25' - Monthly 25% Quantile (exluding zero-values)
     - 'Q50' - Monthly 50% Quantile (Median) (exluding zero-values)
     - 'Q75' - Monthly 75% Quantile (exluding zero-values)

    """
    pd.options.mode.chained_assignment = None
    #
    # get DataFrame
    def_df = dataframe.copy()
    #
    # Get the Months where null values happen:
    mask = def_df[var_field].isnull()  # create a boolen mask where null values happen
    dates_null = def_df[mask][date_field]  # series of all null dates
    months_null_all = dates_null.apply(lambda x: x.strftime('%B-%Y'))  # seris of all null months
    months_null = months_null_all.unique()  # numpy array of strings of unique months where null values happen
    #
    # get all dates and months
    dates_all_all = def_df[date_field]
    months_all_all = dates_all_all.apply(lambda x: x.strftime('%B-%Y')).values
    #
    # loop for assigning a boolean mask of 'bad' months
    mask = list()
    for local_month in months_all_all:
        flag = False
        for null_month in months_null:
            if local_month == null_month:
                flag = True
                mask.append(True)
            if flag:
                break
        if flag == False:
            mask.append(False)
    #
    # The Series of bad months:
    mask1 = pd.Series(mask, index=def_df.index)
    #
    # The Series of bad months dates with measured data:
    mask2 = def_df[mask1][var_field].dropna()
    #
    # Then overwrite the bad dates to nan:
    def_df.loc[mask2.index, var_field] = np.nan
    #
    # Finally resamples by month
    def_df.set_index(date_field, inplace=True)
    df_sum = def_df.resample('MS').sum()
    df_avg = def_df.resample('MS').mean()
    df_mean = def_df.replace(0.0, np.nan).resample('MS').mean()
    df_max = def_df.replace(0.0, np.nan).resample('MS').max()
    df_min = def_df.replace(0.0, np.nan).resample('MS').min()
    df_med = def_df.replace(0.0, np.nan).resample('MS').median()
    df_q25 = def_df.replace(0.0, np.nan).resample('MS').quantile(0.25)
    df_q50 = def_df.replace(0.0, np.nan).resample('MS').quantile(0.5)
    df_q75 = def_df.replace(0.0, np.nan).resample('MS').quantile(0.75)
    #
    # built output dataframe
    dct = {'Sum': df_sum[var_field], 'Avg':df_avg[var_field], 'Mean': df_mean[var_field],
           'Max': df_max[var_field], 'Min':df_min[var_field], 'Med':df_med[var_field],
           'Q25':df_q25[var_field], 'Q50':df_q50[var_field], 'Q75':df_q75[var_field]}
    df_out = pd.DataFrame(dct, index=df_sum.index)
    df_out['Sum'].replace(0.0, np.nan, inplace=True)
    df_out['Avg'].replace(0.0, np.nan, inplace=True)
    df_out.reset_index(inplace=True)
    df_out[date_field] = pd.to_datetime(df_out[date_field])
    #print(df_out.head().to_string())
    return df_out.copy()


def d2m_flow(dataframe, factor=1.0, date_field='Date', var_field='Flow'):
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
    pd.options.mode.chained_assignment = None
    #
    # get DataFrame
    def_df = dataframe.copy()
    #
    # Get the Months where null values happen:
    mask = def_df[var_field].isnull()  # create a boolen mask where null values happen
    dates_null = def_df[mask][date_field]  # series of all null dates
    months_null_all = dates_null.apply(lambda x: x.strftime('%B-%Y'))  # seris of all null months
    months_null = months_null_all.unique()  # numpy array of strings of unique months where null values happen
    #
    # get all dates and months
    dates_all_all = def_df[date_field]
    months_all_all = dates_all_all.apply(lambda x: x.strftime('%B-%Y')).values
    #
    # loop for assigning a boolean mask of 'bad' months
    mask = list()
    for local_month in months_all_all:
        flag = False
        for null_month in months_null:
            if local_month == null_month:
                flag = True
                mask.append(True)
            if flag:
                break
        if flag == False:
            mask.append(False)
    #
    # The Series of bad months:
    mask1 = pd.Series(mask, index=def_df.index)
    #
    # The Series of bad months dates with measured data:
    mask2 = def_df[mask1][var_field].dropna()
    #
    # Then overwrite the bad dates to nan:
    def_df.loc[mask2.index, var_field] = np.nan
    #
    # Overwrite the variable field to flow units per day
    def_df[var_field] = def_df[var_field].apply(lambda x: x * 86400 * factor)
    #
    # Finally resamples by month
    def_df.set_index(date_field, inplace=True)
    df_sum = def_df.resample('MS').sum()
    df_mean = def_df.resample('MS').mean()
    df_max = def_df.resample('MS').max()
    df_min = def_df.resample('MS').min()
    df_q25 = def_df.resample('MS').quantile(0.25)
    df_q50 = def_df.resample('MS').quantile(0.5)
    df_q75 = def_df.resample('MS').quantile(0.75)
    #
    # built output dataframe
    dct = {'Sum': df_sum[var_field], 'Mean': df_mean[var_field],
           'Max': df_max[var_field], 'Min': df_min[var_field],
           'Q25': df_q25[var_field], 'Q50': df_q50[var_field], 'Q75': df_q75[var_field]}
    df_out = pd.DataFrame(dct, index=df_sum.index)
    df_out['Sum'].replace(0.0, np.nan, inplace=True)
    df_out.reset_index(inplace=True)
    df_out[date_field] = pd.to_datetime(df_out[date_field])
    # print(df_out.head().to_string())
    return df_out.copy()







