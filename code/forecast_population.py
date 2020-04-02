import os
import numpy as np
from matplotlib import pyplot as plt
import pickle

cur_path = '/Users/adamalexanderoppenheimer/Desktop/DynamicPop'
os.chdir(cur_path + '/code')

import util

os.chdir(cur_path)

fert_params = pickle.load(open('data/demographic/parameters/fert.p', 'rb') )
mort_params = pickle.load(open('data/demographic/parameters/mort.p', 'rb') )
imm_params = pickle.load(open('data/demographic/parameters/imm.p', 'rb') )

pop_data, fert_data, mort_data, imm_rate = pickle.load(open('data/demographic/clean/all.p', 'rb') )

datatype = 'population_forecasts'

start = 1971
end = 2017
ages = np.linspace(0, 99, 100).astype(int)
birth_ages = np.linspace(14, 50, 37)
years = np.linspace(start, end, end - start + 1).astype(int)

prev_pop = pop_data[start - 1]

for year in years:
    pred_fert = util.forecast(fert_params, year, birth_ages, datatype='fertility', options=False)
    pred_mort = util.forecast(mort_params, year, ages, datatype='mortality', options=False)
    pred_imm = util.forecast(imm_params, year, ages, datatype='immigration', options={'transition_year': 2015})
    prev_pop = util.predict_population(pred_fert, pred_mort, pred_imm, prev_pop)
    plt.plot(ages, prev_pop, label='Predicted')
    plt.plot(ages, pop_data[year], label='True ' + str(year))
    plt.legend()
    plt.savefig('graphs/' + datatype + '/' + 'start_' + str(start - 1) + '/' + str(year))
    plt.close()

future_start = 2018
future_end = 2500

future_years = np.linspace(future_start, future_end, future_end - future_start + 1).astype(int)

for i in range(2):
    prev_pop = pop_data[future_start - 1]

    for year in future_years:
        if i == 0:
            pred_fert = fert_data[2014]
            pred_mort = mort_data[2014]
            pred_imm = imm_rate[2014]

        elif i == 1:
            pred_fert = util.forecast(fert_params, year, birth_ages, datatype='fertility', options=False)
            pred_mort = util.forecast(mort_params, year, ages, datatype='mortality', options=False)
            pred_imm = util.forecast(imm_params, year, ages, datatype='immigration', options={'transition_year': 2015})

        prev_pop = util.predict_population(pred_fert, pred_mort, pred_imm, prev_pop)
        if year in (2020, 2050, 2070, 2100, 2250, 2500):
            plt.plot(ages, prev_pop, label=str(year))

    plt.legend()
    plt.xlabel('Age $s$')
    plt.ylabel('Population')
    plt.grid()
    axes = plt.gca()
    axes.set_xlim([0,100])
    if i == 0:
        plt.title('Basic population transition')
        plt.tight_layout()
        plt.savefig('graphs/' + datatype + '/' + 'start_' + str(future_start - 1) + '/predicted_basic')
    elif i == 1:
        plt.title('Parametric model population transition')
        plt.tight_layout()
        plt.savefig('graphs/' + datatype + '/' + 'start_' + str(future_start - 1) + '/predicted_parametric')
    plt.close()

    future_start = 2018
    future_end = 3000

    future_years = np.linspace(future_start, future_end, future_end - future_start + 1).astype(int)
    prev_pop = pop_data[future_start - 1]

    for year in future_years:
        if i == 0:
            pred_fert = fert_data[2014]
            pred_mort = mort_data[2014]
            pred_imm = imm_rate[2014]

        elif i == 1:
            pred_fert = util.forecast(fert_params, year, birth_ages, datatype='fertility', options=False)
            pred_mort = util.forecast(mort_params, year, ages, datatype='mortality', options=False)
            pred_imm = util.forecast(imm_params, year, ages, datatype='immigration', options={'transition_year': 2015})

        prev_pop = util.predict_population(pred_fert, pred_mort, pred_imm, prev_pop)
        if year in (2018, 2024, 2058, 2098, 3000):
            if year in (2018, 2024):
                lab = str(year) + ' pop.'
            elif year == 2058:
                lab = 'T=40 pop.'
            elif year == 2098:
                lab = 'T=80 pop.'
            elif year == 3000:
                lab = 'Adj. SS pop.'
            plt.plot(ages, prev_pop / np.sum(prev_pop), label=lab)

    plt.legend()
    plt.xlabel('Age $s$')
    plt.ylabel(r"Pop. dist'n $\omega_s$")
    plt.grid()
    axes = plt.gca()
    axes.set_xlim([0,100])
    if i == 0:
        plt.title('Basic population distribution transition')
        plt.tight_layout()
        plt.savefig('graphs/' + datatype + '/' + 'start_' + str(future_start - 1) + '/predicted_proportion_basic')
    elif i == 1:
        plt.title('Parametric model population distribution transition')
        plt.tight_layout()
        plt.savefig('graphs/' + datatype + '/' + 'start_' + str(future_start - 1) + '/predicted_proportion_parametric')
    plt.close()
