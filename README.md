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

# How to use

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
my_file = download.metadata_ana_flow('C:/Datasets/ANA/')

```

# Modules

## `download.py`
This module stores functions for download data and metadata. Files are saved in `.txt` format in a directory specifyed in the `folder=` parameter. If not passed, the default directory is the current folder of the python file calling the function. 

General dependencies:
* [Pandas](https://pandas.pydata.org/)
* [tdqm](https://github.com/tqdm/tqdm)
* [HydroBR](https://github.com/wallissoncarvalho/hydrobr)

## `load.py`

This module stores functions for loading the files created by the `download.py` module. Each function of `download.py` has a counterpart in `load.py`

The data in the files are loaded to a `DataFrame` object. This procedure allows the data processing using the `pandas` library.

General dependencies:
* [Pandas](https://pandas.pydata.org/)

## `tseries.py`

---- A module for time series processing such as upscale (ex: daily to monthly precipitation)

## `visuals.py`

---- A module for standard data visualisation


