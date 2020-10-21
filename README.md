# gespla

UNIVERSIDADE FEDERAL DO RIO GRANDE DO SUL

GESPLA IPH/UFGRS PYTHON LIBRARY

Website: https://www.ufrgs.br/warp/

This is the Python Library of GESPLA - Research Group on Water Resources Planning and Management 


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

1) In the first cell, import and install the main packages as follows:
```python
import pandas as pd
import numpy as np  # if used
import matplotlib.pyplot as plt  # if used
import scipy  # if used

# for the download.py module:
!pip install hydrobr
```

2) Copy and paste the desired code block in the code cells. 

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
This module stores functions for download data and metadata. Files are saved in `.txt` format in a directory specifyed in the `folder=` parameter. If not passed, the default directory is the current folder of the python file calling the function. 

General dependencies:
* [Pandas](https://pandas.pydata.org/)
* [tdqm](https://github.com/tqdm/tqdm)
* [HydroBR](https://github.com/wallissoncarvalho/hydrobr)

List of functions:
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

## `load.py`

This module stores functions for loading data stored on files created by other modules, such as the `download.py` module. Each function of `download.py` has a counterpart in `load.py`.

The data in the files are loaded to a `DataFrame` object. This procedure allows the data processing using the `pandas` library.

General dependencies:
* [Pandas](https://pandas.pydata.org/)

List of functions:
* `.ana_flow(file)` - loads to `DataFrame` the flow data from a single flow station of ANA. 
* `.ana_stage(file)` - loads to `DataFrame` the stage data from a single flow station of ANA.
* `.ana_prec(file)` - loads to `DataFrame` the precipitation data from a single precipitation station of ANA.
* `.inmet_daily(file)` - loads to `DataFrame` the daily climate data from a single climate station of INMET. See docstring for variables.
* `.inmet_hourly(file)` - loads to `DataFrame` the hourly climate data from a single climate station of INMET. See docstring for variables.

## `tseries.py`

---- A module for time series processing such as upscale (ex: daily to monthly precipitation)

## `visuals.py`

---- A module for standard data visualisation


