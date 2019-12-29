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
smooth = 1
years = np.linspace(start, end, end - start + 1)
ages = np.linspace(14, 50, 37)
for year in range(start, end + 1):
    #Take 'smooth' years rolling average
    fert_yr = util.rolling_avg_year(fert_data, year, smooth)
    pop_yr = util.rolling_avg_year(pop_data, year, smooth)[14:14 + len(fert_yr)] #14-50 year olds, or 14-49 for 1989

    a, b, p, q, scale = util.gen_beta2_est(fert_yr, year, smooth, datatype='fertility', pop=pop_yr)

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

util.plot_params(start, end, smooth, a_list, b_list, p_list, q_list, scales, datatype='fertility')

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

##############################################

alphas = []
betas = []
ms = []
scales = []
for year in range(start, end + 1):
    #Take 'smooth' years rolling average
    mort_yr = util.rolling_avg_year(mort_data, year, smooth)

    alpha, beta, m, scale = util.gen_gamma_est(mort_yr, year, smooth, datatype='mortality')

    alphas.append(alpha)
    betas.append(beta)
    ms.append(m)
    scales.append(scale)

alphas = np.array(alphas)
betas = np.array(betas)
ms = np.array(ms)
scales = np.array(scales)

util.plot_params(start, end, smooth, alphas, betas, ms, scales, datatype='mortality')

##########################################################
########## Predict population data
#######################
prev_pop = pop_data[2014]
mort_fixed = mort_data[2014]
imm_fixed = imm_rate[2014]
NUM_COLORS = 2500 + 1 - 2015
cm = plt.get_cmap('Blues')
cNorm  = colors.Normalize(vmin=0, vmax=NUM_COLORS-1)
scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=cm)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_prop_cycle(color=[scalarMap.to_rgba(i) for i in range(NUM_COLORS)])

for year in range(2015, 2500):
    beta = util.logistic_function(year - 1, L_MLE_beta, k_MLE_beta, x_MLE_beta) + min(betas)
    m = util.logistic_function(year - 1, L_MLE_m, k_MLE_m, x_MLE_m) + min(ms)
    year_adj_alpha = - (year - 1 - x_MLE_alpha) + x_MLE_alpha
    alpha = util.logistic_function(year_adj_alpha, L_MLE_alpha, k_MLE_alpha, x_MLE_alpha) + min(alphas)
    year_adj_scale = - (year - 1 - x_MLE_scale) + x_MLE_scale
    scale = util.logistic_function(year_adj_scale, L_MLE_scale, k_MLE_scale, x_MLE_scale) + min(scales)
    ages = np.linspace(14, 50, 37)
    fert = util.gen_gamma_fun_pdf(ages, alpha, beta, m)

    fert = fert_data[2014]

    births = fert * prev_pop[14:51]
    births = np.sum(births)

    deaths = mort_fixed * prev_pop
    deaths = np.roll(deaths, 1)

    imms = imm_fixed * prev_pop
    imms = np.roll(imms, 1)

    pred_pop = np.roll(prev_pop, 1) + imms - deaths
    pred_pop[0] = births
    pred_pop += imms

    prev_pop = pred_pop

    ages = np.linspace(0,99,100)
    if year > 2450:
        plt.plot(ages, pd.DataFrame(pred_pop), label='Predicted population')
    #plt.plot(ages, pop_data[year], label='True population')
#plt.legend()
plt.show()
plt.close()
