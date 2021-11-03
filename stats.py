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
-- collection of model functions and convenience functions for statistical modelling

Authors:
Ipora Possantti: https://github.com/ipo-exe

First commit: 11 of November of 2021

"""

import numpy as np
import pandas as pd

def fit_cdf_gamma(obs_df, val_field='Values', p_field='Percentiles', iterations=10,
                  size=20, a_min=0.1, a_max=10, s_min=1, s_max=100):
    """

    Fit a CDF gamma distribution by incremental grid sampling

    :param obs_df: pandas dataframe
    :param val_field: string values field
    :param p_field: string percentiles field
    :param iterations: int number of iterations
    :param size: int size of grid
    :param a_min: float
    :param a_max: float
    :param s_min: float
    :param s_max: float
    :return: dict
    """
    from scipy.stats import gamma
    from analyst import rmse
    for t in range(iterations):
        if t == 0:
            _a = np.linspace(start=a_min, stop=a_max, num=size)
            _s = np.linspace(start=s_min, stop=s_max, num=size)
        else:
            _a = np.linspace(start=df_3['a'].min(), stop=df_3['a'].max(), num=size)
            _s = np.linspace(start=df_3['s'].min(), stop=df_3['s'].max(), num=size)
        # deploy lists
        _x = list()
        _y = list()
        _z = list()
        # loop in parameter space
        for i in range(len(_a)):
            for j in range(len(_s)):
                lcl_a = _a[i]
                lcl_s = _s[j]
                # compute the model simulation
                lcl_sim = gamma.cdf(obs_df[val_field], lcl_a, scale=lcl_s)
                # get model metric
                lcl_rmse = rmse(obs=obs_df[p_field].values, sim=lcl_sim)
                _x.append(lcl_a)
                _y.append(lcl_s)
                _z.append(lcl_rmse)
        # insert in dataframe
        df_2 = pd.DataFrame({'a': _x, 's': _y, 'RMSE': _z})
        # select top 10
        df_3 = df_2.sort_values(by='RMSE', ascending=True).head(10)
    return {'a': df_3['a'].values[0], 's': df_3['s'].values[0], 'RMSE': df_3['RMSE'].values[0]}