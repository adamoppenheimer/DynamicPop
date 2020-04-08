import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib import pyplot as plt
from math import e
import pickle

cur_path = '/Users/adamalexanderoppenheimer/Desktop/DynamicPop'
os.chdir(cur_path + '/code')

import util

os.chdir(cur_path)

imm_rate = pickle.load(open('data/demographic/clean/imm.p', 'rb') )

datatype = 'immigration'

start = 1997
end = 2015
ages = np.linspace(0, 99, 100).astype(int)

years = np.linspace(start, end, end - start + 1).astype(int)
years_OLS = sm.add_constant(years)
forecast_vals = np.array(range(2030 - end + 1))
forecast_years = np.linspace(end, 2030, 2030 - end + 1).astype(int)

age_params = []

for age in ages:
    imm_age = imm_rate.iloc[age]
    model = sm.OLS(endog=imm_age, exog=years_OLS)
    results = model.fit()
    constant, beta = results.params
    estline = years * beta + constant
    plt.plot(years, estline, label='OLS estimate')
    plt.plot(imm_age, label='Age ' + str(age))
    plt.legend()
    #plt.show()
    plt.savefig('graphs/' + datatype + '/age/' + str(age))
    plt.close()

    # Fit exponential with cubic polynomial to forecast
    last_val = estline[-1]
    g = last_val - 1 # Need positive value for logs
    c = np.log(last_val - g)
    b = beta / (e ** c)
    if beta > 0:
        if last_val > 0:
            f = 1.1 # Increase estimate by 10%
        elif last_val < 0:
            f = 0.9 # Decrease estimate by 10%
        else:
            f = 1 # Keep estimate constant
    elif beta < 0:
        if last_val > 0:
            f = 0.9 # Decrease estimate by 10%
        elif last_val < 0:
            f = 1.1 # Increase estimate by 10%
        else:
            f = 1 # Keep estimate constant
    else:
        f = 1 # Keep estimate constant
    a = 1 / 30 * (1e-20 / (f * last_val - g) - b )
    if age == -1:
        b = - b / 5
    age_params.append([constant, beta, a, b, c, g])
    forecast = e ** (a * forecast_vals ** 2 + b * forecast_vals + c) + g
    plt.plot(forecast_years, forecast, label='Forecast')
    plt.plot(years, estline, label='OLS Estimate')
    plt.plot(imm_age, label='Age ' + str(age))
    plt.xlabel(r'Year $t$')
    plt.ylabel(r'Immigration Rate $i_{s,t}$')
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    plt.tight_layout()
    plt.legend()
    plt.savefig('graphs/' + datatype + '/age_forecasts/' + str(age))
    plt.close()

# Transition graphs
util.plot_forecast_transition(age_params, ages, start, end, smooth=0, datatype='immigration', options={'transition_year': 2015})
util.plot_data_transition(imm_rate, ages, start, end, smooth=0, datatype='immigration')
util.overlay_estimates(imm_rate, age_params, ages, start, end, smooth=0, datatype='immigration', options={'transition_year': 2015})

#Graph comparison between 2014 and 2030
util.plot_2100(age_params, ages, smooth=0, datatype='immigration', options={'transition_year': 2015})

pickle.dump( age_params, open('data/demographic/parameters/imm.p', 'wb') )
