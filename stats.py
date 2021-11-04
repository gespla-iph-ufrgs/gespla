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
-- collection of model functions and convenience functions for statistical modelling

Authors:
Ipora Possantti: https://github.com/ipo-exe

First commit: 11 of November of 2021

'''

import numpy as np
import pandas as pd


def fit_cdf_gamma(obs_df,
                  val_field='Values',
                  p_field='Percentiles',
                  iterations=10,
                  gridsize=20,
                  shape_min=0.1,
                  shape_max=10,
                  scale_min=0.0,
                  scale_max=100):
    """

    Fit a CDF gamma distribution by incremental grid sampling

    :param obs_df: pandas dataframe
    :param val_field: string values field
    :param p_field: string percentiles field
    :param iterations: int number of iterations
    :param gridsize: int size of grid
    :param shape_min: float
    :param shape_max: float
    :param scale_min: float
    :param scale_max: float
    :return: dict
    """
    from scipy.stats import gamma
    from analyst import rmse
    #
    # sampling loop:
    for t in range(iterations):
        # starting bounding box
        if t == 0:
            _a = np.linspace(start=shape_min, stop=shape_max, num=gridsize)
            _s = np.linspace(start=scale_min, stop=scale_max, num=gridsize)
        # zooming bounding box
        else:
            _a = np.linspace(start=_df['shape'].min(), stop=_df['shape'].max(), num=gridsize)
            _s = np.linspace(start=_df['scale'].min(), stop=_df['scale'].max(), num=gridsize)
        #
        # deploy lists
        _x = list()
        _y = list()
        _z = list()
        #
        # loop in parameter space
        for i in range(len(_a)):
            for j in range(len(_s)):
                lcl_a = _a[i]
                lcl_s = _s[j]
                #
                # compute the model simulation
                lcl_sim = gamma.cdf(obs_df[val_field], lcl_a, scale=lcl_s)
                #
                # get model metric
                lcl_rmse = rmse(obs=obs_df[p_field].values, sim=lcl_sim)
                #
                # append
                _x.append(lcl_a)
                _y.append(lcl_s)
                _z.append(lcl_rmse)
        #
        # insert lists in dataframe
        _aux_df = pd.DataFrame({'shape': _x, 'scale': _y, 'RMSE': _z})
        #
        # select top 10
        _df = _aux_df.sort_values(by='RMSE', ascending=True).head(10)
    #
    # output dict
    out_dct = {'shape': _df['shape'].values[0],
               'scale': _df['scale'].values[0],
               'RMSE': _df['RMSE'].values[0]}
    return out_dct