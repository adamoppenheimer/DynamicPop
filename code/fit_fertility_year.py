import os
import numpy as np
import pandas as pd
import pickle

cur_path = '/Users/adamalexanderoppenheimer/Desktop/DynamicPop'
os.chdir(cur_path + '/code')

import util

os.chdir(cur_path)

pop_data, fert_data = pickle.load(open('data/demographic/clean/fert.p', 'rb') )

start = 1970
end = 2014
smooth = 0
years = np.linspace(start, end, end - start + 1).astype(int)
ages = np.linspace(14, 50, 37)

a_list = []
b_list = []
p_list = []
q_list = []
scales = []

for year in years:
    #Take 'smooth' years rolling average
    print(year)
    fert_yr = util.rolling_avg_year(fert_data, year, smooth)
    #pop_yr = util.rolling_avg_year(pop_data, year, smooth)[14:14 + len(fert_yr)] #14-50 year olds, or 14-49 for 1989

    a, b, p, q, scale = util.gen_beta2_est(fert_yr, year, smooth, datatype='fertility')#, pop=pop_yr)

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

util.plot_params(start, end, smooth, params_list, datatype='fertility')

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

params = a_params, b_params, p_params, q_params, scale_params

#Transition graphs
util.plot_forecast_transition(params, ages, start, end, smooth, datatype='fertility')
util.plot_data_transition(fert_data, ages, start, end, smooth, datatype='fertility')
util.overlay_estimates(fert_data, params, ages, start, end, smooth, datatype='fertility')

#Graph comparison between 2014 and 2100
util.plot_2100(params, ages, smooth, datatype='fertility')

pickle.dump( params, open('data/demographic/parameters/fert.p', 'wb') )
