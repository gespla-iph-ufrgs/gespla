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
-- collection of model functions and convenience functions for model analysis

Authors:
Ipora Possantti: https://github.com/ipo-exe

First commit: 11 of November of 2021

"""

import numpy as np


def error(obs, sim):
    """
    Error function
    :param obs: float, int or numpy array of Observerd data
    :param sim: float, int or numpy array of Simulated data
    :return: float, int, or numpy array of Error signal
    """
    return obs - sim


def sq_error(obs, sim):
    """
    Squared Error function
    :param obs: float, int or numpy array of Observerd data
    :param sim: float, int or numpy array of Simulated data
    :return: float, int, or numpy array of Squared Error signal
    """
    return np.square(error(obs=obs, sim=sim))


def mse(obs, sim):
    """
    Mean Squared Error (MSE) function
    :param obs: float, int or numpy array of Observerd data
    :param sim: float, int or numpy array of Simulated data
    :return: float value of MSE
    """
    return np.mean(sq_error(obs=obs, sim=sim))


def rmse(obs, sim):
    """
    Root of Mean Squared Error (RMSE) function
    :param obs: float, int or numpy array of Observerd data
    :param sim: float, int or numpy array of Simulated data
    :return: float value of RMSE
    """
    return np.sqrt(mse(obs=obs, sim=sim))


def nrmse(obs, sim):
    """
    Normalized RMSE by the mean observed value
    :param obs: float, int or numpy array of Observerd data
    :param sim: float, int or numpy array of Simulated data
    :return: float value of NRMSE
    """
    return rmse(obs=obs, sim=sim) / np.mean(obs)


def nse(obs, sim):
    """
    Nash-Sutcliffe Efficiency (NSE) coeficient
    :param obs: numpy array of Observerd data
    :param sim: numpy array of Simulated data
    :return: float of NSE
    """
    qmean = np.mean(obs)
    se_sim = sq_error(obs=obs, sim=sim)
    se_mean = sq_error(obs=obs, sim=qmean)
    return 1 - (np.sum(se_sim) / np.sum(se_mean))


def nnse(obs, sim):
    """
    Normalized NSE function (NSE re-scaled to [0,1])
    :param obs: numpy array of Observerd data
    :param sim: numpy array of Simulated data
    :return: float of NNSE
    """
    return 1 / (2 - nse(obs=obs, sim=sim))


def kge(obs, sim):
    """
    Kling-Gupta Efficiency (KGE) coeficient Gupta et al. (2009)

    KGE = 1 - sqroot( (r - 1)^2 + (sd_sim/sd_obs - 1)^2 + (m_sim/m_obs - 1)^2)

    - Correlation
    - Dispersion
    - Mean value

    :param obs: numpy array of Observerd data
    :param sim: numpy array of Simulated data
    :return: float of KGE
    """
    linmodel = linreg(obs=obs, sim=sim)
    r = linmodel['R']
    sd_obs = np.std(obs)
    sd_sim = np.std(sim)
    mean_obs = np.mean(obs)
    mean_sim = np.mean(sim)
    return 1 - np.sqrt(np.square(r - 1) + np.square((sd_sim/sd_obs) - 1) + np.square((mean_sim/mean_obs) - 1))


def pbias(obs, sim):
    """
    P-Bias function
    Negative P-Bias ->> Too much water! - ET have to work harder to remove water from the system
    Positive P-Bias ->> Too less water! -  ET is draining too much water
    :param obs: numpy array of Observerd data
    :param sim: numpy array of Simulated data
    :return: float of P-bias in % (0 to 100)
    """
    return 100 * (np.sum(error(obs=obs, sim=sim))/np.sum(obs))


def linreg(obs, sim):
    """
    Linear regression model function
    :param obs: 1-d numpy array of Observerd data
    :param sim: 1-d numpy array of Simulated data
    :return: dictionary object of linear model y=Ax+B parameters:
    A, B, R, P, SD
    """
    from scipy.stats import linregress
    a, b, r, p, sd = linregress(x=obs, y=sim)
    return {'A':a, 'B':b, 'R':r, 'P':p, 'SD':sd}