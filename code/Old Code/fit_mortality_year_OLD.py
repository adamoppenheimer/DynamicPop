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
a_0 = 5e-6#5e-5#8e-7
b_0 = 0.11#0.095#1e-2
c_0 = -1e-4#-0.002#-1
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
# #Fit b_list to logistic function
# L_0 = max(b_list) # For smooth 1, set to 0.2
# k_0 = 0.013#0.053#0.09#0.5#1.5 # For smooth 1, set to 1
# x_0 = 1975#1965#1995
# L_MLE_b, k_MLE_b, x_MLE_b = util.logistic_est(b_list, L_0, k_0, x_0, years, smooth, datatype='mortality', param='b')
# b_params = L_MLE_b, k_MLE_b, x_MLE_b, np.min(b_list)

#Fit b_list to polynomial function
a_0 = 0.1
e_0 = 0.35#Works:0.01
b_0 = e_0 * (start - 1e-5)
c_0 = 2.2#Works:1
d_0 = 0

a_MLE, b_MLE, c_MLE, d_MLE, e_MLE = util.poly_est(b_list, a_0, b_0, c_0, d_0, e_0, years, smooth, datatype='mortality', param='b')
b_params = a_MLE, b_MLE, c_MLE, d_MLE, e_MLE
print(b_params)


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

################################################################
################################################################
################################################################
################################################################
################################################################
################################################################
################################################################


a_list = []
b_list = []
p_list = []
q_list = []
scales = []
start = 1970
end = 2014
smooth = 0
years = np.linspace(start, end, end - start + 1)
ages = np.linspace(14, 50, 37)
for year in range(start, end + 1):
    #Take 'smooth' years rolling average
    mort_yr = util.rolling_avg_year(non_infant_mort, year, smooth)
    ages = np.array(mort_yr.index)
    pop_yr = util.rolling_avg_year(pop_data, year, smooth)[1:1 + len(mort_yr)] #1-99 year olds

    a, b, p, q, scale = util.gen_beta2_est(mort_yr, year, smooth, datatype='mortality_beta')#, pop=pop_yr)

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

util.plot_params(start, end, smooth, params_list, datatype='mortality_beta')

#########################################
#Fit a_list to logistic function
L_0 = max(a_list)
k_0 = 1.5
x_0 = 1995
L_MLE_a, k_MLE_a, x_MLE_a = util.logistic_est(a_list, L_0, k_0, x_0, years, smooth, datatype='fertility', param='a', flip=True)
a_params = L_MLE_a, k_MLE_a, x_MLE_a, np.min(a_list)

#########################################
#Fit b_list to logistic function
L_0 = 0.55
k_0 = 1.5
x_0 = 1995
L_MLE_b, k_MLE_b, x_MLE_b = util.logistic_est(b_list, L_0, k_0, x_0, years, smooth, datatype='fertility', param='b')
b_params = L_MLE_b, k_MLE_b, x_MLE_b, np.min(b_list)

#########################################
#Fit p_list to logistic function
L_0 = 5#max(ms)
k_0 = 0.2#1e-50
x_0 = 1995
L_MLE_p, k_MLE_p, x_MLE_p = util.logistic_est(p_list, L_0, k_0, x_0, years, smooth, datatype='fertility', param='p', flip=True)
p_params = L_MLE_p, k_MLE_p, x_MLE_p, np.min(p_list)

#########################################
#Fit q_list to logistic function
L_0 = 5#max(ms)
k_0 = 0.2#1e-50
x_0 = 1995
L_MLE_q, k_MLE_q, x_MLE_q = util.logistic_est(q_list, L_0, k_0, x_0, years, smooth, datatype='fertility', param='q')
q_params = L_MLE_q, k_MLE_q, x_MLE_q, np.min(q_list)

#########################################
#Fit scales to logistic function
L_0 = max(scales)
k_0 = 1
x_0 = 1995
L_MLE_scale, k_MLE_scale, x_MLE_scale = util.logistic_est(scales, L_0, k_0, x_0, years, smooth, datatype='fertility', param='Scale', flip=True)
scale_params = L_MLE_scale, k_MLE_scale, x_MLE_scale, np.min(scales)

#Transition graphs
util.plot_data_transition_gen_beta2_estimates(a_params, b_params, p_params, q_params, scale_params, start, end, ages, smooth, datatype='fertility')
util.plot_data_transition(fert_data, start, end, ages, smooth, datatype='fertility')
util.overlay_estimates(fert_data, a_params, b_params, p_params, q_params, scale_params, start, end, ages, smooth, datatype='fertility')

#Graph comparison between 2014 and 2100
util.plot_2100(a_params, b_params, p_params, q_params, scale_params, ages, smooth, datatype='fertility')

