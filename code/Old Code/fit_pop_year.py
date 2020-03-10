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

cur_path = '/Users/adamalexanderoppenheimer/Desktop/DynamicPop'
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

a_list = []
b_list = []
p_list = []
q_list = []
scales = []
start = 1970
end = 2014
smooth = 0
years = np.linspace(start, end, end - start + 1)
ages = np.linspace(0, 99, 100)
prev_estimates = False
for year in range(start, end + 1):
    #Take 'smooth' years rolling average
    pop_yr = util.rolling_avg_year(pop_data, year, smooth)

    prev_estimates = util.gen_beta2_est(pop_yr, year, smooth, datatype='population', prev_estimates=prev_estimates)

    a, b, p, q, scale = prev_estimates

    a_list.append(a)
    b_list.append(b)
    p_list.append(p)
    q_list.append(q)
    scales.append(scale)

a_list = np.array(a_list)
b_list = np.array(b_list)
p_list = np.array(p_list)
q_list = np.array(q_list)
scales = np.array(scales)

params_list = [('a', a_list), ('b', b_list), ('p', p_list), ('q', q_list), ('Scale', scales)]

util.plot_params(start, end, smooth, params_list, datatype='population')

#########################################
#Fit a_list to logistic function
L_0 = max(a_list)
k_0 = 1.5
x_0 = 1995
L_MLE_a, k_MLE_a, x_MLE_a = util.logistic_est(a_list, L_0, k_0, x_0, years, smooth, datatype='population', param='a', flip=True)
a_params = L_MLE_a, k_MLE_a, x_MLE_a, np.min(a_list)

#########################################
#Fit b_list to logistic function
L_0 = 0.55
k_0 = 1.5
x_0 = 1995
L_MLE_b, k_MLE_b, x_MLE_b = util.logistic_est(b_list, L_0, k_0, x_0, years, smooth, datatype='population', param='b')
b_params = L_MLE_b, k_MLE_b, x_MLE_b, np.min(b_list)

#########################################
#Fit p_list to logistic function
L_0 = 5#max(ms)
k_0 = 0.2#1e-50
x_0 = 1995
L_MLE_p, k_MLE_p, x_MLE_p = util.logistic_est(p_list, L_0, k_0, x_0, years, smooth, datatype='population', param='p', flip=True)
p_params = L_MLE_p, k_MLE_p, x_MLE_p, np.min(p_list)

#########################################
#Fit q_list to logistic function
L_0 = 5#max(ms)
k_0 = 0.2#1e-50
x_0 = 1995
L_MLE_q, k_MLE_q, x_MLE_q = util.logistic_est(q_list, L_0, k_0, x_0, years, smooth, datatype='population', param='q')
q_params = L_MLE_q, k_MLE_q, x_MLE_q, np.min(q_list)

#########################################
#Fit scales to exponential function
a_0 = max(scales)
b_0 = 0.07
c_0 = 155
a_MLE_scale, b_MLE_scale, c_MLE_scale = util.exp_est(scales, '', a_0, b_0, c_0, years, smooth, datatype='population', param='Scale')

scale_params = a_MLE_scale, b_MLE_scale, c_MLE_scale

# #Fit scales to logistic function
# L_0 = max(scales)
# k_0 = 2
# x_0 = 1965
# L_MLE_scale, k_MLE_scale, x_MLE_scale = util.logistic_est(scales, L_0, k_0, x_0, years, smooth, datatype='population', param='Scale')
# scale_params = L_MLE_scale, k_MLE_scale, x_MLE_scale, np.min(scales)

# #########################################
# #Fit betas to logistic function
# L_0 = 0.55
# k_0 = 1.5
# x_0 = 1995
# L_MLE_beta, k_MLE_beta, x_MLE_beta = util.logistic_est(betas, L_0, k_0, x_0, years, smooth, datatype='population', param='Beta')
# beta_params = L_MLE_beta, k_MLE_beta, x_MLE_beta, np.min(betas)

# #########################################
# #Fit alphas to logistic function
# L_0 = max(alphas)
# k_0 = 1.5
# x_0 = 1995
# L_MLE_alpha, k_MLE_alpha, x_MLE_alpha = util.logistic_est(alphas, L_0, k_0, x_0, years, smooth, datatype='population', param='Alpha')
# alpha_params = L_MLE_alpha, k_MLE_alpha, x_MLE_alpha, np.min(alphas)

# #########################################
# #Fit ms to logistic function
# L_0 = 5#max(ms)
# k_0 = 0.2#1e-50
# x_0 = 1995
# L_MLE_m, k_MLE_m, x_MLE_m = util.logistic_est(ms, L_0, k_0, x_0, years, smooth, datatype='population', param='M')
# m_params = L_MLE_m, k_MLE_m, x_MLE_m, np.min(ms)

# #########################################
# #Fit scales to log function
# a_0 = 2e6
# b_0 = np.min(years) - 1
# c_0 = 1.9
# d_0 = np.min(scales)
# e_0 = 1
# a_MLE_scale, b_MLE_scale, c_MLE_scale, d_MLE_scale, e_MLE_scale = util.poly_est(scales, a_0, b_0, c_0, d_0, e_0, years, smooth, datatype='population', param='Scale', print_params=True)
# scale_params = a_MLE_scale, b_MLE_scale, c_MLE_scale, d_MLE_scale, e_MLE_scale

#Transition graphs
util.plot_data_transition_gen_beta2_estimates(a_params, b_params, p_params, q_params, scale_params, start, end, ages, smooth, datatype='population')
util.plot_data_transition(pop_data, start, end, ages, smooth, datatype='population')
util.overlay_estimates(pop_data, a_params, b_params, p_params, q_params, scale_params, start, end, ages, smooth, datatype='population')

#Graph comparison between 2014 and 2100
util.plot_2100(a_params, b_params, p_params, q_params, scale_params, ages, smooth, datatype='population')
