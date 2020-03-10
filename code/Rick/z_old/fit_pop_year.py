import os
import numpy as np
import pandas as pd
import scipy.optimize as opt
from scipy.ndimage.interpolation import shift
import math
from math import e
import matplotlib.pyplot as plt
import matplotlib.cm as mplcm
import matplotlib.colors as colors

cur_path = '/Volumes/GoogleDrive/My Drive/4th Year/Thesis/japan_olg_demographics'
os.chdir(cur_path + '/code')

import util

os.chdir(cur_path)

datadir = 'data/demographic/'
fert_dir = datadir + 'jpn_fertility.csv'
mort_dir = datadir + 'jpn_mortality.csv'
pop_dir = datadir + 'jpn_population.csv'

fert_data = util.get_fert_data(fert_dir)
mort_data, pop_data = util.get_mort_pop_data(mort_dir, pop_dir)
imm = util.calc_imm_resid(fert_data, mort_data, pop_data)
imm_rate = imm / pop_data

alphas = []
betas = []
ms = []
scales = []
start = 1970
end = 2014
smooth = 0
years = np.linspace(start, end, end - start + 1)
ages = np.linspace(0, 99, 100)
for year in range(start, end + 1):
    #Take 'smooth' years rolling average
    pop_yr = util.rolling_avg_year(pop_data, year, smooth)

    alpha, beta, m, scale = util.gen_gamma_est(pop_yr, year, smooth, datatype='population')

    alphas.append(alpha)
    betas.append(beta)
    ms.append(m)
    scales.append(scale)

alphas = np.array(alphas)
betas = np.array(betas)
ms = np.array(ms)
scales = np.array(scales)

util.plot_params(start, end, smooth, alphas, betas, ms, scales, datatype='population')

#########################################
#Fit betas to logistic function
L_0 = 0.55
k_0 = 1.5
x_0 = 1995
L_MLE_beta, k_MLE_beta, x_MLE_beta = util.logistic_est(betas, L_0, k_0, x_0, years, smooth, datatype='population', param='Beta')
beta_params = L_MLE_beta, k_MLE_beta, x_MLE_beta, np.min(betas)

#########################################
#Fit alphas to logistic function
L_0 = max(alphas)
k_0 = 1.5
x_0 = 1995
L_MLE_alpha, k_MLE_alpha, x_MLE_alpha = util.logistic_est(alphas, L_0, k_0, x_0, years, smooth, datatype='population', param='Alpha')
alpha_params = L_MLE_alpha, k_MLE_alpha, x_MLE_alpha, np.min(alphas)

#########################################
#Fit ms to logistic function
L_0 = 5#max(ms)
k_0 = 0.2#1e-50
x_0 = 1995
L_MLE_m, k_MLE_m, x_MLE_m = util.logistic_est(ms, L_0, k_0, x_0, years, smooth, datatype='population', param='M')
m_params = L_MLE_m, k_MLE_m, x_MLE_m, np.min(ms)

#########################################
#Fit scales to log function
a_0 = 2e6
b_0 = np.min(years) - 1
c_0 = 1.9
d_0 = np.min(scales)
e_0 = 1
a_MLE_scale, b_MLE_scale, c_MLE_scale, d_MLE_scale, e_MLE_scale = util.poly_est(scales, a_0, b_0, c_0, d_0, e_0, years, smooth, datatype='population', param='Scale', print_params=True)
scale_params = a_MLE_scale, b_MLE_scale, c_MLE_scale, d_MLE_scale, e_MLE_scale

#Transition graphs
util.plot_data_transition_gen_gamma_estimates(beta_params, alpha_params, m_params, scale_params, start, end, ages, smooth, datatype='population')
util.plot_data_transition(pop_data, start, end, ages, smooth, datatype='population')
util.plot_data_transition_gen_gamma_overlay_estimates(pop_data, beta_params, alpha_params, m_params, scale_params, start, end, ages, smooth, datatype='population')

#Graph comparison between 2014 and 2100
util.plot_2100(beta_params, alpha_params, m_params, scale_params, ages, smooth, datatype='population')
