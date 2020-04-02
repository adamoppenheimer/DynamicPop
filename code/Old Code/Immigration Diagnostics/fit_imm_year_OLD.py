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
imm_rate = imm_rate.drop([1947, 2016, 2017], axis=1)

# Split immigration dataset into regions

imm_birth = imm_rate.iloc[0]
imm_birth_1 = imm_rate.iloc[1]
imm_birth_2 = imm_rate.iloc[2]
imm_birth_3 = imm_rate.iloc[3]
imm_first = imm_rate.iloc[4:17]
imm_second = imm_rate.iloc[17:26]
imm_third = imm_rate.iloc[26:70]
imm_fourth = imm_rate.iloc[70:]

# Prepend extra years prior to first datapoint to help with fit
extra_years = 0#5
imm_first_slope = imm_first.iloc[1] - imm_first.iloc[0]
first_year = imm_first.iloc[0]
for i in range(extra_years):
    imm_prepend = first_year - (i + 1) * imm_first_slope
    imm_prepend = np.expand_dims(imm_prepend, 0)
    imm_first = np.concatenate([imm_prepend, imm_first], axis=0)
    imm_first = pd.DataFrame(imm_first)

imm_first = imm_first.rename(columns={i: imm_rate.columns[i] for i in range(len(imm_rate.columns))})
imm_first.index = imm_first.index - extra_years

sections = [('', imm_rate), ('first_', imm_first), ('second_', imm_second), ('third_', imm_third), ('fourth_', imm_fourth)]

#sections = [('first_', imm_first)]
#sections = [('second_', imm_second)]

start = 1970
end = 2014
smooth = 0
years = np.linspace(start, end, end - start + 1)
ages = np.linspace(-extra_years, 99, 100 + extra_years)
prev_estimates = False
datatype = 'immigration'

#################
##### Plots #####
#################

plt.plot(imm_birth)
plt.savefig('graphs/' + datatype + '/smooth_' + str(smooth) + '/_birth')
plt.close()
plt.plot(imm_birth_1)
plt.savefig('graphs/' + datatype + '/smooth_' + str(smooth) + '/_birth_1')
plt.close()
plt.plot(imm_birth_2)
plt.savefig('graphs/' + datatype + '/smooth_' + str(smooth) + '/_birth_2')
plt.close()
plt.plot(imm_birth_3)
plt.savefig('graphs/' + datatype + '/smooth_' + str(smooth) + '/_birth_3')
plt.close()

for year in range(start, end + 1):
    for i, section in enumerate(sections):
        #Take 'smooth' years rolling average
        imm_yr = section[1][year]#util.rolling_avg_year(imm_rate, year, smooth)
        plt.plot(imm_yr)
        plt.savefig('graphs/' + datatype + '/smooth_' + str(smooth) + '/' + section[0] + str(year))
        plt.close()

####################
##### Fit Data #####
####################

years = np.linspace(1948, 2015, 2015 - 1948 + 1)

#########################################
#Fit imm_birth to logistic function
L_0 = max(imm_birth)
k_0 = 1.5
x_0 = 1995
L_MLE, k_MLE, x_MLE = util.logistic_est(imm_birth, L_0, k_0, x_0, years, smooth, datatype='immigration', param='birth', flip=True)
birth_params = L_MLE, k_MLE, x_MLE, np.min(imm_birth)

#########################################
#Fit imm_birth_1 to logistic function
L_0 = max(imm_birth_1)
k_0 = 1
x_0 = 1995
L_MLE, k_MLE, x_MLE = util.logistic_est(imm_birth_1, L_0, k_0, x_0, years, smooth, datatype='immigration', param='birth_1', flip=True)
birth_1_params = L_MLE, k_MLE, x_MLE, np.min(imm_birth_1)

#########################################
#Fit imm_birth_2 and imm_birth_3 to mean

birth_2_params = np.mean(imm_birth_2)
birth_3_params = np.mean(imm_birth_3)

plt.plot(imm_birth_2)
plt.plot(imm_birth_2.index, [birth_2_params] * len(imm_birth_2))
plt.savefig('graphs/' + datatype + '/smooth_' + str(smooth) + '/_birth_2_predicted')
plt.close()
plt.plot(imm_birth_3)
plt.plot(imm_birth_3.index, [birth_3_params] * len(imm_birth_3))
plt.savefig('graphs/' + datatype + '/smooth_' + str(smooth) + '/_birth_3_predicted')
plt.close()

#########################################
#Fit imm_first to mean

section = sections[1]
imm_first_means = []
for year in range(start, end + 1):
    #Take 'smooth' years rolling average
    imm_yr = section[1][year]#util.rolling_avg_year(imm_rate, year, smooth)
    imm_first_means.append(np.mean(imm_yr))

imm_first_params = np.mean(imm_first_means[:3] + imm_first_means[4:]) # Outlier at 3

#########################################
#Fit imm_third to mean

section = sections[3]
imm_third_means = []
for year in range(start, end + 1):
    #Take 'smooth' years rolling average
    imm_yr = section[1][year]#util.rolling_avg_year(imm_rate, year, smooth)
    imm_third_means.append(np.mean(imm_yr))

imm_third_params = np.mean(imm_third_means[:3] + imm_third_means[4:]) # Outlier at 3

#########################################
#Fit imm_fourth to exponential

section = sections[4]
imm_fourth_as = []
imm_fourth_bs = []
imm_fourth_cs = []
for year in range(start, end + 1):
    #Take 'smooth' years rolling average
    imm_yr = section[1][year]#util.rolling_avg_year(imm_rate, year, smooth)

    #########################################
    #Fit to exponential function
    a_0 = 0.001
    b_0 = 0.05
    c_0 = -0.5
    a_MLE, b_MLE, c_MLE = util.exponential_est(imm_yr, year, a_0, b_0, c_0, imm_yr.index, smooth, datatype='immigration', param='imm_fourth_' + str(year))
    imm_fourth_as.append(a_MLE)
    imm_fourth_bs.append(b_MLE)
    imm_fourth_cs.append(c_MLE)

#########################################
#Fit imm_fourth_a, imm_fourth_b, and imm_fourth_c to mean

imm_fourth_a_params = np.mean(imm_fourth_as[:25]), np.mean(imm_fourth_as[25:])
imm_fourth_b_params = np.mean(imm_fourth_bs)
imm_fourth_c_params = np.mean(imm_fourth_cs)

plt.plot(range(start, end + 1), imm_fourth_as)
a_vals = [imm_fourth_a_params[0]] * 25 + [imm_fourth_a_params[1]] * (len(imm_fourth_as) - 25)
plt.plot(range(start, end + 1), a_vals)
plt.savefig('graphs/' + datatype + '/smooth_' + str(smooth) + '/_imm_fourth_as_predicted')
plt.close()

plt.plot(range(start, end + 1), imm_fourth_bs)
plt.plot(range(start, end + 1), [imm_fourth_b_params] * len(imm_fourth_bs))
plt.savefig('graphs/' + datatype + '/smooth_' + str(smooth) + '/_imm_fourth_bs_predicted')
plt.close()

plt.plot(range(start, end + 1), imm_fourth_cs)
plt.plot(range(start, end + 1), [imm_fourth_c_params] * len(imm_fourth_cs))
plt.savefig('graphs/' + datatype + '/smooth_' + str(smooth) + '/_imm_fourth_cs_predicted')
plt.close()

util.plot_data_transition(imm_rate, start, end, ages, smooth, datatype='immigration')
