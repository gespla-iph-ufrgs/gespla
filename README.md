# gespla

GESPLA Python Toolbox

This is the Python Library of GESPLA - Research Group on Water Resources Planning and Management 

The purpose of the tool is to provide a set of routines in the form of function modules in Python 3. The available routines are unitary operations that can be coupled in any python code. The modules are separated by functionality class:
* Download of hydrometeorological data.
* Import of hydrometeorological data.
* Time series processing.
* Time series analysis.
* Data visualization.
* Regression models.
* Hydroeconomic models.

Website: https://www.ufrgs.br/warp/

# License
This Repository is under `LICENSE`: GNU General Public License v3.0

Permissions:
* Commercial use
* Modification
* Distribution
* Patent use
* Private use 

Limitations:
* Liability
* Warranty

Conditions:
* License and copyright notice
* State changes
* Disclose source
* Same license 

# How to use on [google colab](colab.research.google.com/):

1) Donwload the repository and extract the folder to your workspace. 

2) On google colab, in the first cell, import and install the main packages as follows:
```python
import pandas as pd
import numpy as np  # if used
import matplotlib.pyplot as plt  # if used
import scipy  # if used

# for the download.py module:
!pip install hydrobr
```

2) In the side pannel, load the desired modules python files. 

3) Import the desired modules. Example:
```python
import load
```

# How to use on a machine

1) Install [Python 3](https://www.python.org/downloads/).

2) Install packages dependencies:
* [Numpy](https://numpy.org/)
* [Pandas](https://pandas.pydata.org/)
* [Matplotlib](https://matplotlib.org/)
* [Scipy](https://www.scipy.org/)
* [Statsmodels](https://www.statsmodels.org)
* [tdqm](https://github.com/tqdm/tqdm)
* [HydroBR](https://github.com/wallissoncarvalho/hydrobr)

3) Donwload the repository and extract the folder to your workspace. The name of the folder must be `gespla`.

4) From a python file, import the modules (ex: `download` module):
```python
from gespla import download  # this imports the download module

# call the module functions:
my_file = download.metadata_ana_flow(folder='C:/Datasets/ANA/')

```

# Modules

Modules are python files storing a collection of functions created by the `def` statement. If you are in a python file in the same directory of a module, you can import it by the `import` statement. Example:
```python
import download  # this imports the download module

# call the module functions:
my_file = download.metadata_ana_flow(folder='C:/Datasets/ANA/')
```
You may also just import the desired function:
```python
from download import metadata_ana_flow  # this imports the function called metadata_ana_flow

# call the module functions:
my_file = metadata_ana_flow(folder='C:/Datasets/ANA/')
```

## Modules are independent
The modules are independent from each other. This means that any subset of modules can be used without crash concerns.

## Functions may have local dependencies
Functions defined in a single module may depend on other local functions. Unless full code inspection, avoid to simply copying functions to your code. Instead, use the entire module as a dependency. 

## Functions documentation
Fuctions (and related returns and parameters) are fully documented by `docstrings`. A `docstring` is an extended comment in the heading of the function. You can read it by:
1) Simply finding the `docstring` in the code;
2) Call the buil-in function `help()` passing the function name to it. Example:
```python
from download import metadata_ana_flow  # this imports the function called metadata_ana_flow

# call the docstring by the help() function:
help(metadata_ana_flow)
```

3) Printing the `.__doc__` attribute on screen. Example:
```python
from download import metadata_ana_flow  # this imports the function called metadata_ana_flow

# print the function docstring:
print(metadata_ana_flow.__doc__)
```

## `download.py` - Download data and metadata to a CSV file
This module stores functions for download data and metadata. Files are saved in `.txt` format in a directory specifyed in the `folder=` parameter. If not passed, the default directory is the current folder of the python file calling the function. All functions return the string of the saved file path.

General dependencies:
* [Pandas](https://pandas.pydata.org/)
* [tdqm](https://github.com/tqdm/tqdm)
* [HydroBR](https://github.com/wallissoncarvalho/hydrobr)

List of current functions:
* `.metadata_ana_flow(folder)` - downloads metadata of all flow stations of ANA.
* `.metadata_ana_prec(folder)` - downloads metadata of all precipitation stations of ANA.
* `.metadata_inmet(folder, opt)` - downloads metadata of all climate stations of INMET, defined by type.
* `.metadata_ana_telemetry(folder)` - downloads metadata of all telemetric stations of ANA.
* `.metadata_ana_rhn_inventory(folder)` - downloads metadata of full inventory of stations of ANA.
* `.ana_flow(code, folder)` - downloads flow data from a single flow station of ANA. 
* `.ana_stage(code, folder)` - downloads stage data from a single flow station of ANA.
* `.ana_prec(code, folder)` - downloads precipitation data from a single precipitation station of ANA.
* `.inmet_daily(code, folder)` - downloads daily climate data from a single climate station of INMET. See docstring for variables.
* `.inmet_hourly(code, folder)` - downloads hourly climate data from a single climate station of INMET. See docstring for variables.

Example:

```python
from gespla import download  # this imports the download module

# call the module functions:
my_file = download.metadata_ana_flow(folder='C:/Datasets/ANA/')
# print on screen where the file was saved:
print('The file was saved in: {}'.format(my_file))
```

## `load.py` - Get a DataFrame to work with

This module stores functions for loading data stored on files created by the `download.py` module. Most of `download.py` has a counterpart in `load.py`.

The data in the files are loaded to a `DataFrame` object. This procedure allows the data processing using the `pandas` library.

General dependencies:
* [Pandas](https://pandas.pydata.org/)

List of current functions:
* `.ana_flow(file)` - loads to `DataFrame` the flow data from a single flow station of ANA. 
* `.ana_stage(file)` - loads to `DataFrame` the stage data from a single flow station of ANA.
* `.ana_prec(file)` - loads to `DataFrame` the precipitation data from a single precipitation station of ANA.
* `.inmet_daily(file)` - loads to `DataFrame` the daily climate data from a single climate station of INMET. See docstring for variables.
* `.inmet_hourly(file)` - loads to `DataFrame` the hourly climate data from a single climate station of INMET. See docstring for variables.
* `.metadata_ana_flow(file)` - loads to `DataFrame` the metadata of flow/stage stations of ANA. 
* `.metadata_ana_prec(file)` - loads to `DataFrame` the metadata of precipitations stations of ANA.
* `.metadata_inmet(file)` - loads to `DataFrame` the metadata of climate stations of INMET.

Example:

```python
[in:]
from gespla import load  # this imports the load module

# load to DataFrame the timeseries of flow data:
df = load.ana_flow(file='C:/Datasets/ANA/ANA-flow_11444900_2020-10-20.txt')
# print on screen the first 4 lines of the DataFrame:
print(df.head(4).to_string())

[out:]
        Date     Flow
0 1993-04-01  10126.0
1 1993-04-02  10154.0
2 1993-04-03  10210.0
3 1993-04-04  10266.0

```

## `resample.py` - Interpolate and Aggregate Time Variables

This module stores functions for resampling time series, such as from daily to monthly. The input data is passed as a `DataFrame` object so the functions process it and then returns a new `DataFrame` object containing the new time series. The returned aggregated variables are:

* Periods count  - number of periods aggregated
* Count  - number of valid records aggregated
* Sum
* Mean
* Min
* Min
* Max
* Q25 -  quantile 25
* Q50 - quantile 50 (median)
* Q75 - quantile 75

General dependencies:
* [Pandas](https://pandas.pydata.org/)
* [Numpy](https://numpy.org/)

List of current functions:
* `.d2m_prec(dataframe)` - resamples precipitation time series from daily to monthly. 
* `.d2m_flow(dataframe)` - resamples flow time series from daily to monthly.
* `.d2m_stage(dataframe)` - resamples stage time series from daily to monthly.
* `.d2m_clim(dataframe)` - resamples climate variable time series from daily to monthly.
* `.d2y_prec(dataframe)` - resamples precipitation time series from daily to yearly. 
* `.d2y_flow(dataframe)` - resamples flow time series from daily to yearly.
* `.d2y_stage(dataframe)` - resamples stage time series from daily to yearly.
* `.d2y_clim(dataframe)` - resamples climate variable time series from daily to yearly.
* `.insert_gaps(dataframe)` - insert null records in missing date gaps of time series.
* `.interpolate_gaps(dataframe)` - interpolates size-defined gaps of null records on a time series.
* `.clear_bad_months(dataframe)` - removes from the time series months that have null records (gaps).
* `.clear_bad_years(dataframe)` - removes from the time series years that have null records (gaps).
* `.resampler(dataframe)` - this function is the module utility resampler function.
* `.group_by_month(dataframe)` - this function groups a daily time series into 12 timeseries for each month in the year.


Example:
```python
[in:]
from gespla import load, resample  # this imports the load and resample modules

# load to DataFrame the timeseries of flow data:
df_d = load.ana_flow(file='C:/Datasets/ANA/ANA-flow_11444900_2020-10-20.txt')

f = 1 / 1000000  # this converts m3/month to hm3/month
# call the d2m_flow() function:
df_m = resample.d2m_flow(dataframe=df_d, factor=f)
# print on screen the first 4 lines of the DataFrame:
print(df_m.head(4).to_string())

[out:]
        Date  Period_Count  Count         Sum       Mean       Min        Max       Q25        Q50        Q75
0 1964-12-01             0      0         NaN        NaN       NaN        NaN       NaN        NaN        NaN
1 1965-01-01            31     31  193.509216   6.242233  3.556466  10.630310  4.321797   5.854058   7.606224
2 1965-02-01            28     28  241.351816   8.619708  4.596998  12.337056  5.997519   9.404813  10.810994
3 1965-03-01            31     31  332.557963  10.727676  5.712310  16.852925  9.092995  10.630310  12.254717

```

## `tsa.py` - Time Series Analysis

This module stores functions for time series analysis, such as frequency analysis and ETS decomposition. The input data is passed as a `DataFrame` object so the functions process it and then returns a new `DataFrame` object containing the new time series/objects.

General dependencies:
* [Pandas](https://pandas.pydata.org/)
* [Numpy](https://numpy.org/)
* [Statsmodels](https://www.statsmodels.org)

List of current functions:
* `.frequency(dataframe)` - Get a dataframe with Percentiles, Exceedance Probability, Frequency and Empirical Probability of a time series.
* `.sma(dataframe)` - Get the Simple Moving Average time series of a defined period window.
* `.ewma(dataframe)` - Get the Exponential Weighting Moving Average of a defined period window.
* `.hpfilter(dataframe)` - This functions performs the Hodrick-Prescott Filter on time series.
* `.ets_decomposition(dataframe)` - This functions performs the ETS (Error, Trend, Seasonality) Decomposition on time series.
* `.ses(dataframe)` - This function performs Simple Exponential Smoothing (Holt Linear) on a given time series.
* `.des(dataframe)` - This function performs Double Exponential Smoothing (Holt-Winters Second Order) on a given time series
* `.tes(dataframe)` - This function performs Triple Exponential Smoothing (Holt-Winters Third Order) on a given time series
* `.des_forecast(dataframe)` - This function performs Double Exponential Smoothing (Holt-Winters Second Order) fit and forecast on a given time series
* `.tes_forecast(dataframe)` - This function performs Triple Exponential Smoothing (Holt-Winters Third Order) fit and forecast on a given time series


Example:

```python
[ in:]
import load, tsa  # this imports the load and tsa modules

# load to DataFrame the timeseries of flow data:
df_d = load.ana_flow(file='samples/MISC/ANA-flow_20200000_1964-2020__by-2020-10-27.txt')

df_freq = tsa.frequency(df_d, 'Flow')
print(df_freq.head(4).to_string())

[out:]
Percentiles
Exeedance
Frequency
Probability
Values
0
0
100
1515
0.074900
1.3366
1
1
99
3531
0.174569
5.2180
2
2
98
2938
0.145251
6.5130
3
3
97
2193
0.108419
7.4166

```


## `visuals.py`

---- A module for standard data visualisation -  to be developed.


