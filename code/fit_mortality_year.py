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

infant_mort = mort_data.iloc[0]
infant_pop = pop_data.iloc[0].drop(2017)

smooth = 0

a_0 = 1
e_0 = 1.2
b_0 = e_0 * (min(infant_mort.index) - 1e-5)
c_0 = -1
d_0 = 0
years = np.array(infant_mort.index)

a_MLE, b_MLE, c_MLE, d_MLE, e_MLE = util.poly_est(infant_mort, a_0, b_0, c_0, d_0, e_0, years, smooth, datatype='mortality', param='Infant Mortality', pop=infant_pop)
print('a_MLE:', a_MLE, 'b_MLE:', b_MLE, 'c_MLE:', c_MLE, 'd_MLE:', d_MLE, 'e_MLE:', e_MLE)

non_infant_mort = mort_data.iloc[1:]

a_list = []
b_list = []
c_list = []
a_0 = 8e-7
b_0 = 1e-2
c_0 = -1
start = 1970
end = 2014
smooth = 1
years = np.linspace(start, end, end - start + 1)
for year in range(start, end + 1):
    #Take 'smooth' years rolling average
    mort_yr = util.rolling_avg_year(non_infant_mort, year, smooth)
    ages = np.array(mort_yr.index)
    pop_yr = util.rolling_avg_year(pop_data, year, smooth)[1:1 + len(mort_yr)] #1-99 year olds

    a, b, c = util.exp_est(mort_yr, year, a_0, b_0, c_0, ages, smooth, datatype='mortality', param='Non-Infant Mortality')#, pop=pop_yr)

    a_0 = a
    b_0 = b
    c_0 = c

    a_list.append(a)
    b_list.append(b)
    c_list.append(c)

params_list = [('a', a_list), ('b', b_list), ('c', c_list)]

util.plot_params(start, end, smooth, params_list, datatype='mortality')

#########################################
#Fit a_list to polynomial function
a_0 = 1e-4
e_0 = 0.025
b_0 = e_0 * (start - 1e-5)
c_0 = -1
d_0 = 0

a_MLE, b_MLE, c_MLE, d_MLE, e_MLE = util.poly_est(a_list, a_0, b_0, c_0, d_0, e_0, years, smooth, datatype='mortality', param='a')
a_params = a_MLE, b_MLE, c_MLE, d_MLE, e_MLE
print(a_params)

#########################################
#Fit b_list to logistic function
L_0 = max(b_list) # For smooth 1, set to 0.2
k_0 = 0.5#1.5 # For smooth 1, set to 1
x_0 = 1965#1995
L_MLE_b, k_MLE_b, x_MLE_b = util.logistic_est(b_list, L_0, k_0, x_0, years, smooth, datatype='mortality', param='b')
b_params = L_MLE_b, k_MLE_b, x_MLE_b, np.min(b_list)

#########################################
#Fit c_list to logistic function
L_0 = max(c_list)
k_0 = 1e-5#1.5#0.2#1e-50
x_0 = 1985#1995
L_MLE_c, k_MLE_c, x_MLE_c = util.logistic_est(c_list, L_0, k_0, x_0, years, smooth, datatype='mortality', param='c')
c_params = L_MLE_c, k_MLE_c, x_MLE_c, np.min(c_list)

ages = np.linspace(0, 99, 100)

#Transition graphs
util.plot_data_transition_exp_estimates(a_params, b_params, c_params, start, end, ages, smooth, datatype='mortality')
util.plot_data_transition(mort_data, start, end, ages, smooth, datatype='mortality')
util.overlay_estimates_mort(mort_data, a_params, b_params, c_params, start, end, ages, smooth, datatype='mortality')

#Graph comparison between 2014 and 2100
util.plot_2100_mort(a_params, b_params, c_params, ages, smooth, datatype='mortality')
