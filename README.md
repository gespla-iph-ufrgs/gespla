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
The modules are independent from each other. This means that only one or module may be used. It does not going to crash.

## Functions may have local dependencies
Functions defined in a single module may depend on other local functions. Unless full code inspection, avoid to simply copying functions to your code. Instead, use the entire module as a dependency. 

## Functions documentation
Fuctions (and related returns and parameters) are fully documented by `docstrings`. A `docstring` is an extended comment in the heading of the function. You can read it by:
1) Simply finding the `docstring` in the code, or;
2) Printing the `.__doc__` attribute on screen. Example:
```python
from download import metadata_ana_flow  # this imports the function called metadata_ana_flow

# print the function docstring:
print(metadata_ana_flow.__doc__)
```

## `download.py`
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

## `load.py`

This module stores functions for loading data stored on files created by other modules, such as the `download.py` module. Each function of `download.py` has a counterpart in `load.py`.

The data in the files are loaded to a `DataFrame` object. This procedure allows the data processing using the `pandas` library.

General dependencies:
* [Pandas](https://pandas.pydata.org/)

List of current functions:
* `.ana_flow(file)` - loads to `DataFrame` the flow data from a single flow station of ANA. 
* `.ana_stage(file)` - loads to `DataFrame` the stage data from a single flow station of ANA.
* `.ana_prec(file)` - loads to `DataFrame` the precipitation data from a single precipitation station of ANA.
* `.inmet_daily(file)` - loads to `DataFrame` the daily climate data from a single climate station of INMET. See docstring for variables.
* `.inmet_hourly(file)` - loads to `DataFrame` the hourly climate data from a single climate station of INMET. See docstring for variables.

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

## `resample.py`

This module stores functions for resampling time series, such as from daily to monthly. The input data is passed as a `DataFrame` object so the functions process it and then returns a `DataFrame` object containing the new time series. The returned monthly variables are (when valid):

* Sum;
* Mean;
* Min;
* Max;
* Q25;
* Q50 (Median);
* Q75.

General dependencies:
* [Pandas](https://pandas.pydata.org/)
* [Numpy](https://numpy.org/)

List of current functions:
* `.d2m_prec(dataframe)` - resamples precipitation time series from daily to monthly. 
* `.d2m_flow(dataframe, factor)` - resamples flow time series from daily to monthly.
* `.d2m_stage(dataframe)` - resamples stage time series from daily to monthly.

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
print(df.head(4).to_string())

[out:]
        Date         Sum         Mean        Max       Min        Q25        Q50        Q75
0 1993-04-01  30424.8960  1014.163200  1125.9648  874.8864   929.9232  1024.8768  1104.7968
1 1993-05-01  33377.7024  1076.700077  1132.0128  996.3648  1026.7776  1095.7248  1127.7792
2 1993-06-01  27308.9664   910.298880   994.4640  805.9392   835.8768   936.5760   966.5136
3 1993-07-01  24485.7600   789.863226   831.3408  710.3808   766.6272   807.1488   819.2448

```



## `visuals.py`

---- A module for standard data visualisation


