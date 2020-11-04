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
-- collection of model functions and convenience functions for resample time scale

Authors:
Ipora Possantti: https://github.com/ipo-exe

First commit: 20 of October of 2020
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


def cut_edges(dataframe, var_field):
    """

    Utility function to cut off initial and final null records on a given time series

    :param dataframe: pandas DataFrame object
    :param var_field: string head of the variable field.
    :return: pandas DataFrame object
    """
    # get dataframe
    in_df = dataframe.copy()
    def_len = len(in_df)
    # drop first nan lines
    drop_ids = list()
    # loop to collect indexes in the start of series
    for def_i in range(def_len):
        aux = in_df[var_field].isnull().iloc[def_i]
        if aux:
            drop_ids.append(def_i)
        else:
            break
    # loop to collect indexes in the end of series
    for def_i in range(def_len - 1, -1, -1):
        aux = in_df[var_field].isnull().iloc[def_i]
        if aux:
            drop_ids.append(def_i)
        else:
            break
    # loop to drop rows:
    for def_i in range(len(drop_ids)):
        in_df.drop(drop_ids[def_i], inplace=True)
    return in_df


def group_by_month(dataframe, var_field, date_field='Date'):
    """
    This function groups a daily time series into 12 timeseries for each month in the year.

    :param dataframe: pandas DataFrame object. The date field must be in a column, not the index.
    :param var_field: string head of the variable field.
    :param date_field: string head of the date field. Default: 'Date'
    :return: dictionary of dataframes for daily timeseries of each month.

    Keys of dicitonary:
    '1' - January
    '2' - February
    '3' - March

    ...

    '12' - December

    """
    #
    # get data from DataFrame
    in_df = dataframe[[date_field, var_field]].copy()
    #
    # ensure datefield is datetime
    in_df['Date'] = pd.to_datetime(in_df['Date'])
    #
    # create a helper year-month field
    in_df['Month'] = in_df[date_field].apply(lambda x: x.strftime('%B'))
    in_df['Month'] = in_df['Month'].astype('category')
    in_df.dropna(inplace=True)
    # print(in_df.head(10).to_string())
    #
    # built new dataframe:
    aux_df = pd.DataFrame({'Date':in_df[date_field], var_field:in_df[var_field], 'Month':in_df['Month']})
    months = aux_df['Month'].unique()
    def_gb = aux_df.groupby('Month')
    #
    # built output dictionary:
    out_dct = dict()
    for i in range(len(months)):
        #print(months[i])
        out_dct[str(months[i])] = def_gb.get_group(months[i])
    return out_dct


def insert_gaps(dataframe, date_field='Date', freq='day'):
    """
    This is a convenience function that standardizes a timeseries by inserting the missing gaps as actual records
    :param dataframe: pandas DataFrame object
    :param date_field: string datefield - Default: 'Date'
    :param freq: string frequency of time scale. Default: 'day' (daily) options:

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


def interpolate_gaps(dataframe, var_field, size, freq='day', date_field='Date', type='cubic'):
    """
    This function interpolates gaps on a time series. The maximum gap length for interpolation can
    be defined in the size= parameter. The time scale of series are not relevant.

    :param dataframe: pandas DataFrame object. The date field must be in a column, not the index.
    :param var_field: string head of the variable field.
    :param date_field: string head of the date field. Default: 'Date'
    :param size: integer number for maximum gap length to fill. Default is 4.
    :param freq: string of time scale of time series. Options:

    year
    month
    day
    hout

    :param type: string of interpolation tipe (it uses scipy.interpolate.interp1d)
    Default: 'cubic' - cubic spline

    Options (from scipy docs - https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html )
    'linear'
    'nearest'
    'zero'
    'slinear'
    'quadratic'
    'cubic'
    'previous'
    'next'

    Where 'zero', 'slinear', 'quadratic' and 'cubic' refer to a spline interpolation of zeroth, first,
    second or third order; 'previous' and 'next' simply return the previous or next value of the point)
    or as an integer specifying the order of the spline interpolator to use.

    :return: pandas DataFrame object with the following fields:

    'Date' -  datetime of time series
    'Original' - original variable time series
    'Interpolation' - interpolated variable time series


    """
    from scipy.interpolate import interp1d
    #
    # get data from DataFrame
    in_df = dataframe[[date_field, var_field]].copy()
    in_df[date_field] = pd.to_datetime(in_df[date_field])
    #
    # insert all gaps to records
    gap_df = insert_gaps(in_df, date_field=date_field, freq=freq)
    # cut off null values on the edges:
    gap_df = cut_edges(gap_df, var_field)
    # get X and Y from dataframe
    def_x = np.array(gap_df.index)
    def_y = gap_df[var_field].values
    #
    # create a boolean of null values
    ybool = (np.isnan(def_y)) * 1
    #
    # accumulate the null values into an array
    aux_lst = list()
    counter = 0
    for i in range(len(def_y)):
        if ybool[i] == 0:
            counter = 0
            aux_lst.append(counter)
        else:
            counter = counter + 1
            aux_lst.append(counter)
    accum = np.array(aux_lst[:])
    #
    # get only highest values of the accumulated array
    aux_lst.clear()
    aux_lst = list()
    for i in range(0, len(def_y)):
        if i == len(def_y) - 1:
            aux_lst.append(accum[i])
        else:
            if accum[i] == 0:
                aux_lst.append(0)
            else:
                if accum[i + 1] == 0:
                    aux_lst.append(accum[i])
                else:
                    aux_lst.append(0)
    accmhi = np.array(aux_lst[:])
    #
    # load the array to a DataFrame:
    def_df = pd.DataFrame({'Hi': accmhi})
    #
    # overwrite array to get only where the series must be splitted
    def_df = def_df[def_df['Hi'] > size]
    indx_end = np.array(def_df.index + 1)  # end indexes
    indx_start = np.array(def_df.index + 1) - def_df['Hi'].values  # star indexes
    slices_array = np.sort(np.append(indx_start, indx_end))  # merge indexes
    #
    # remove record if is in the end
    def_df = pd.DataFrame({'Slices': slices_array})
    def_df = def_df[def_df['Slices'] < len(def_y)]
    slices_array = def_df['Slices'].values
    sliced_y = np.split(def_y, slices_array)
    sliced_x = np.split(def_x, slices_array)
    #
    # interpolate:
    def_y_new = np.array([])
    for i in range(len(sliced_y)):
        # get local slices
        lcl_y = sliced_y[i]
        lcl_x = sliced_x[i]
        # check if there is null values in y slice
        lcl_bool = np.isnan(sliced_y[i])
        # if all values are null
        if np.sum(lcl_bool) == len(lcl_bool):
            # append all -> is a blank frame according to the size
            def_y_new = np.append(def_y_new, lcl_y)
        # if no value is null,
        elif np.sum(lcl_bool) == 0:
            # append all, is a perfect frame
            def_y_new = np.append(def_y_new, lcl_y)
        # otherwise, it must be interpolated
        else:
            # load to DataFrame:
            def_df = pd.DataFrame({'X': lcl_x, 'Y': lcl_y})
            # drop null values
            def_df = def_df.dropna(how='any')
            # create a custom interpolated function
            interf = interp1d(def_df['X'], def_df['Y'], kind=type)  # create a function
            lcl_y_new = interf(lcl_x)  # interpolate
            # def_df = pd.DataFrame({'X': lcl_x, 'Y': lcl_y_new})
            def_y_new = np.append(def_y_new, lcl_y_new)
            '''# if the last row is null and this is the last frame:
            if last_row_bool and i == len(sliced_y) - 1:
                # load to DataFrame:
                stop = len(lcl_x) - 1
                def_df = pd.DataFrame({'X': lcl_x[:stop], 'Y': lcl_y[:stop]})
                # drop null values
                def_df = def_df.dropna(how='any')
                # create a custom interpolated function
                interf = interp1d(def_df['X'], def_df['Y'], kind=type)  # create a function
                lcl_y_new = interf(lcl_x[:stop])  # interpolate
                #def_df = pd.DataFrame({'X': lcl_x[:stop], 'Y': lcl_y_new})
                def_y_new = np.append(def_y_new, lcl_y_new)
                def_y_new = np.append(def_y_new, np.array(np.nan))  # append a null at the end
            else:
                # load to DataFrame:
                def_df = pd.DataFrame({'X': lcl_x, 'Y': lcl_y})
                # drop null values
                def_df = def_df.dropna(how='any')
                # create a custom interpolated function
                interf = interp1d(def_df['X'], def_df['Y'], kind=type)  # create a function
                lcl_y_new = interf(lcl_x)  # interpolate
                #def_df = pd.DataFrame({'X': lcl_x, 'Y': lcl_y_new})
                def_y_new = np.append(def_y_new, lcl_y_new)'''
    out_dct = {'Date': gap_df['Date'], 'Original':def_y, 'Interpolation': def_y_new}
    out_df = pd.DataFrame(out_dct)
    out_df['Date'] = pd.to_datetime(out_df['Date'])
    return out_df


def resampler(dataframe, var_field, date_field='Date', type='month', include_zero=True):
    """
    This function is the resampler function. It takes a time series and resample variables based on a
    type of time scale.


    :param dataframe: pandas DataFrame object
    :param var_field: string head of the variable field.
    :param date_field: string head of the date field. Default: 'Date'
    :param type: time scale type of resampling. Options:

    - 'month' -  Monthly resample
    - 'year' - Yearly resample

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
    def_df = dataframe[[date_field, var_field]].copy()
    def_df.set_index(date_field, inplace=True)
    resam_key = offset_converter(type)

    def_out = pd.DataFrame()
    if include_zero:
        na = ''
    else:
        na = 0.0
    def_out['Period_Count'] = def_df.resample(resam_key).count()[var_field]
    def_out['Count'] = def_df.replace(na, np.nan).resample(resam_key).count()[var_field]
    def_out['Sum'] = def_df.replace(na, np.nan).resample(resam_key).sum()[var_field].replace(0.0, np.nan)
    def_out['Mean'] = def_df.replace(na, np.nan).resample(resam_key).mean()[var_field]
    def_out['Min'] = def_df.replace(na, np.nan).resample(resam_key).min()[var_field]
    def_out['Max'] = def_df.replace(na, np.nan).resample(resam_key).max()[var_field]
    def_out['Q25'] = def_df.replace(na, np.nan).resample(resam_key).quantile(0.25)[var_field]
    def_out['Q50'] = def_df.replace(na, np.nan).resample(resam_key).quantile(0.5)[var_field]
    def_out['Q75'] = def_df.replace(na, np.nan).resample(resam_key).quantile(0.75)[var_field]
    def_out.reset_index(inplace=True)
    return def_out


def clear_bad_years(dataframe, var_field, date_field='Date'):
    """
    This function clears a daily time series from 'bad years', which are
    considered years with ANY null record.

    :param dataframe: pandas DataFrame object with the 'dirty' daily series
    :param var_field: string head of the variable field.
    :param date_field: string head of the date field. Default: 'Date'
    :return: pandas DataFrame object with the 'cleared' daily series
    """
    pd.options.mode.chained_assignment = None
    # get DataFrame
    def_df = dataframe[[date_field, var_field]].copy()
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
    def_df = dataframe[[date_field, var_field]].copy()
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
    # get data
    in_df = dataframe[[date_field, var_field]].copy()
    # insert gaps
    gaps_df = insert_gaps(in_df, date_field=date_field, freq='D')
    # clear bad months:
    def_df = clear_bad_months(gaps_df, var_field=var_field, date_field=date_field)
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
    # get data
    in_df = dataframe[[date_field, var_field]].copy()
    # insert gaps
    gaps_df = insert_gaps(in_df, date_field=date_field, freq='D')
    # clear bad months:
    def_df = clear_bad_months(gaps_df, var_field=var_field, date_field=date_field)
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
    # get data
    in_df = dataframe[[date_field, var_field]].copy()
    # insert gaps
    gaps_df = insert_gaps(in_df, date_field=date_field, freq='D')
    # clear bad months:
    def_df = clear_bad_months(gaps_df, var_field=var_field, date_field=date_field)
    #
    # call the resampler function:
    def_out = resampler(def_df, var_field=var_field, date_field=date_field, type='Month')
    # drop the Sum field - makes no sense:
    def_out.drop('Sum', axis='columns', inplace=True)
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
    # get data
    in_df = dataframe[[date_field, var_field]].copy()
    # insert gaps
    gaps_df = insert_gaps(in_df, date_field=date_field, freq='D')
    # clear bad months:
    def_df = clear_bad_months(gaps_df, var_field=var_field, date_field=date_field)
    #
    # call the resampler function:
    def_out = resampler(def_df, var_field=var_field, date_field=date_field, type='Month')
    # drop the Sum field - makes no sense:
    def_out.drop('Sum', axis='columns', inplace=True)
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
    # get data
    in_df = dataframe[[date_field, var_field]].copy()
    # insert gaps (ensurance protocol)
    gaps_df = insert_gaps(in_df, date_field=date_field, freq='D')
    # clear bad months:
    def_df = clear_bad_years(gaps_df, var_field=var_field, date_field=date_field)
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
    # get data
    in_df = dataframe[[date_field, var_field]].copy()
    # insert gaps
    gaps_df = insert_gaps(in_df, date_field=date_field, freq='D')
    # clear bad years:
    def_df = clear_bad_years(gaps_df, var_field=var_field, date_field=date_field)
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
    # get data
    in_df = dataframe[[date_field, var_field]].copy()
    # insert gaps
    gaps_df = insert_gaps(in_df, date_field=date_field, freq='D')
    # clear bad years:
    def_df = clear_bad_years(gaps_df, var_field=var_field, date_field=date_field)
    #
    # call the resampler function:
    def_out = resampler(def_df, var_field=var_field, date_field=date_field, type='Year')
    # drop the Sum field - makes no sense:
    def_out.drop('Sum', axis='columns', inplace=True)
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
    # get data
    in_df = dataframe[[date_field, var_field]].copy()
    # insert gaps
    gaps_df = insert_gaps(in_df, date_field=date_field, freq='D')
    # clear bad years:
    def_df = clear_bad_years(gaps_df, var_field=var_field, date_field=date_field)
    #
    # call the resampler function:
    def_out = resampler(def_df, var_field=var_field, date_field=date_field, type='Year')
    # drop the Sum field - (makes no sense):
    def_out.drop('Sum', axis='columns', inplace=True)
    #
    return def_out.copy()

