import os
import numpy as np
import pandas as pd
import pickle

cur_path = '/Users/adamalexanderoppenheimer/Desktop/DynamicPop'
os.chdir(cur_path + '/code')

import util

os.chdir(cur_path)

pop_data, infant_pop, non_infant_mort, infant_mort = pickle.load(open('data/demographic/clean/mort.p', 'rb') )

################################
##### Fit Infant Mortality #####
################################

smooth = 0

a_0 = 1
e_0 = 1.2
b_0 = e_0 * (min(infant_mort.index) - 1e-5)
c_0 = -0.8
d_0 = 0
years = np.array(infant_mort.index)

infant_params = util.poly_est(infant_mort, a_0, b_0, c_0, d_0, e_0, years, smooth, datatype='mortality', param='Infant Mortality Rate', pop=infant_pop)

####################################
##### Fit Non-Infant Mortality #####
####################################

start = 1970
end = 2014
smooth = 0
years = np.linspace(start, end, end - start + 1).astype(int)

a_list = []
b_list = []
p_list = []
q_list = []
scales = []

prev_estimates = False
for year in years:
    #Take 'smooth' years rolling average
    print(year)
    mort_yr = util.rolling_avg_year(non_infant_mort, year, smooth)
    ages = np.array(mort_yr.index)
    #pop_yr = util.rolling_avg_year(pop_data, year, smooth)[1:1 + len(mort_yr)] #1-99 year olds

    a, b, p, q, scale = util.gen_beta2_est(mort_yr, year, smooth, datatype='mortality', prev_estimates=prev_estimates)#, pop=pop_yr)

    a_list.append(a)
    b_list.append(b)
    p_list.append(p)
    q_list.append(q)
    scales.append(scale)

    prev_a = sum(a_list) / len(a_list)
    prev_b = sum(b_list) / len(b_list)
    prev_p = sum(p_list) / len(p_list)
    prev_q = sum(q_list) / len(q_list)
    prev_scale = sum(scales) / len(scales)
    prev_estimates = prev_a, prev_b, prev_p, prev_q, prev_scale

a_list = np.array(a_list)
b_list = np.array(b_list)
p_list = np.array(p_list)
q_list = np.array(q_list)
scales = np.array(scales)

params_list = [('a', a_list), ('b', b_list), ('p', p_list), ('q', q_list), ('Scale', scales)]

util.plot_params(start, end, smooth, params_list, datatype='mortality')

#########################################
#Fit a_list to logistic function
L_0 = max(a_list)#0.55
k_0 = 1.5
x_0 = 1995
L_MLE_a, k_MLE_a, x_MLE_a = util.logistic_est(a_list, L_0, k_0, x_0, years, smooth, datatype='mortality', param='a')
a_params = L_MLE_a, k_MLE_a, x_MLE_a, np.min(a_list)

#########################################
#Fit b_list to logistic function
L_0 = 0.55
k_0 = 0.31
x_0 = 1995
L_MLE_b, k_MLE_b, x_MLE_b = util.logistic_est(b_list, L_0, k_0, x_0, years, smooth, datatype='mortality', param='b')
b_params = L_MLE_b, k_MLE_b, x_MLE_b, np.min(b_list)

#########################################
#Fit p_list to logistic function
L_0 = 0.55
k_0 = 0.31
x_0 = 1995
L_MLE_p, k_MLE_p, x_MLE_p = util.logistic_est(p_list, L_0, k_0, x_0, years, smooth, datatype='mortality', param='p')
p_params = L_MLE_p, k_MLE_p, x_MLE_p, np.min(p_list)

#########################################
#Fit q_list to logistic function
L_0 = 0.55
k_0 = 0.31
x_0 = 1995
L_MLE_q, k_MLE_q, x_MLE_q = util.logistic_est(q_list, L_0, k_0, x_0, years, smooth, datatype='mortality', param='q', flip=True)
q_params = L_MLE_q, k_MLE_q, x_MLE_q, np.min(q_list)

#########################################
#Fit scales to logistic function
L_0 = max(scales)
k_0 = 1.5
x_0 = 1995
L_MLE_scale, k_MLE_scale, x_MLE_scale = util.logistic_est(scales, L_0, k_0, x_0, years, smooth, datatype='mortality', param='Scale')
scale_params = L_MLE_scale, k_MLE_scale, x_MLE_scale, np.min(scales)

non_infant_params = a_params, b_params, p_params, q_params, scale_params

ages = np.linspace(0, 99, 100)
#Transition graphs
util.plot_forecast_transition( (non_infant_params, infant_params), ages, start, end, smooth, datatype='mortality')
util.plot_data_transition(non_infant_mort, ages, start, end, smooth, datatype='mortality')
util.overlay_estimates(non_infant_mort, (non_infant_params, infant_params), ages, start, end, smooth, datatype='mortality')

#Graph comparison between 2014 and 2100
util.plot_2100( (non_infant_params, infant_params), ages, smooth, datatype='mortality')

pickle.dump( (non_infant_params, infant_params), open('data/demographic/parameters/mort.p', 'wb') )
