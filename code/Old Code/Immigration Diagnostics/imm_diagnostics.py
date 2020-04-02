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

# Split immigration dataset into regions

imm_first = imm_rate.iloc[:15]
imm_second = imm_rate.iloc[15:30]
imm_third = imm_rate.iloc[30:70]
imm_fourth = imm_rate.iloc[70:]

# Prepend extra years prior to first datapoint to help with fit
extra_years = 5
imm_first_slope = imm_first.iloc[1] - imm_first.iloc[0]
first_year = imm_first.iloc[0]
for i in range(extra_years):
    imm_prepend = first_year - i * imm_first_slope
    imm_prepend = np.expand_dims(imm_prepend, 0)
    imm_first = np.concatenate([imm_prepend, imm_first], axis=0)
    imm_first = pd.DataFrame(imm_first)

imm_first = imm_first.rename(columns={i: imm_rate.columns[i] for i in range(len(imm_rate.columns))})
imm_first.index = imm_first.index - extra_years

#sections = [('', imm_rate), ('first_', imm_first), ('second_', imm_second), ('third_', imm_third)]

#sections = [('first_', imm_first)]
sections = [('second_', imm_second)]

start = 1970
end = 2014
smooth = 0
years = np.linspace(start, end, end - start + 1)
ages = np.linspace(-extra_years, 99, 100 + extra_years)
prev_estimates = False
datatype = 'immigration'

a_list = []
b_list = []
c_list = []
d_list = []
e_list = []
f_list = []

def estimate_breaks(imm_yr):
    # Return left, right boundaries of young outmigration
    min_val = min(imm_yr[10:40])
    min_index = list(imm_yr).index(min_val)
    
    # Get left start of decline
    i = min_index - 2
    #while (imm_yr[i - 1] > imm_yr[i]) or ( (imm_yr[i - 2] > imm_yr[i]) and (imm_yr[i - 3] > imm_yr[i]) ):
    #    i -= 1
    avg_diff = (imm_yr[i] - imm_yr[min_index]) / (min_index -  i)
    curr_diff_1 = imm_yr[i - 1] - imm_yr[i]
    curr_diff_2 = imm_yr[i - 2] - imm_yr[i - 1]
    while (curr_diff_1 > avg_diff) or (curr_diff_2 > avg_diff):
        i -= 1
        avg_diff = (imm_yr[i] - imm_yr[min_index]) / (min_index -  i)
        curr_diff_1 = imm_yr[i - 1] - imm_yr[i]
        curr_diff_2 = imm_yr[i - 2] - imm_yr[i - 1]
    
    # Get right start of decline
    j = min_index + 2
    while ( (imm_yr[j + 1] > imm_yr[j]) and imm_yr[j + 1] - imm_yr[j] >= 0.3 * (imm_yr[min_index + 2] - imm_yr[min_index]) / 2) \
             or ( (imm_yr[j + 2] > imm_yr[j]) and (imm_yr[j + 3] > imm_yr[j + 2]) and \
                    imm_yr[j + 3] - imm_yr[j + 2] >= 0.5 * (imm_yr[min_index + 2] - imm_yr[min_index]) / 2):
       j += 1
    # avg_diff = (imm_yr[j] - imm_yr[min_index]) / (j - min_index)
    # curr_diff_1 = imm_yr[j + 1] - imm_yr[j]
    # curr_diff_2 = imm_yr[j + 2] - imm_yr[j + 1]
    # while (curr_diff_1 > avg_diff) or (curr_diff_2 > avg_diff):
    #     j += 1
    #     avg_diff = (imm_yr[j] - imm_yr[min_index]) / (j - min_index)
    #     curr_diff_1 = imm_yr[j + 1] - imm_yr[j]
    #     curr_diff_2 = imm_yr[j + 2] - imm_yr[j + 1]
    
    return i, j

left_vals = []
right_vals = []

for year in range(start, end + 1):
    print(year)
    imm_yr = imm_rate[year]#util.rolling_avg_year(imm_rate, year, smooth)

    left, right = estimate_breaks(imm_yr)
    left_vals.append(left)
    right_vals.append(right)

    plt.axvline(x=left, ymin=0, ymax=1)
    plt.axvline(x=right, ymin=0, ymax=1)

    plt.plot(imm_yr)
    plt.savefig('graphs/' + datatype + '/smooth_' + str(smooth) + '/' + str(year))
    plt.close()

print('Average left boundary:')
print(np.mean(left_vals))
print('Average right boundary:')
print(np.mean(right_vals))

# Average left boundary:
# 17.022222222222222
# Average right boundary:
# 26.08888888888889

###############################
##### Age-by-age analysis #####
###############################

cycles = 8
colors = ['g', 'r', 'c', 'm', 'y', 'k', 'b'] # Put blue last, leave for actual data
for age in range(100):
    age_data = imm_rate.iloc[age].loc[start:end]
    plt.plot(age_data, label='Age ' + str(age))
    
    for cycle in range(cycles):
        x_max = 2013 - cycle
        while x_max >= start:
            plt.axvline(x=x_max, ymin=0, ymax=1, color=colors[cycle % len(colors)])
            x_max -= cycles
    plt.legend()
    plt.savefig('graphs/' + datatype + '/diagnostics/cycle_' + str(cycles) + '/' + str(age))
    plt.close()


# From 0-30 or so: 5 year pattern
# 31-80: just use last 3 years of data, no real pattern
# Starting 81-88: 5 year pattern
# 89-91: last 3 years of data, no real pattern
# 92: last 2 years of data
# 93: last year of data
# 94: skip last year, go one year back, then 3 years of data, no real pattern
# 95-96: last 3 years of data, no real pattern
# 97-99: 10 year pattern
