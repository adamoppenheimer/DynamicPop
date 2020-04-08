import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pickle

cur_path = '/Users/adamalexanderoppenheimer/Desktop/DynamicPop'
os.chdir(cur_path + '/code')

import util

os.chdir(cur_path)

def prep_demog_alternate():
    '''
    This function returns the alternate dynamic demographic forecasts

    Args:
        None

    Returns:
        pred_fert (Pandas dataframe): forecasted fertility
        pred_mort (Pandas dataframe): forecasted mortality
        pred_imm (Pandas dataframe): forecasted immigration
    '''
    fert_file = os.path.join(cur_path, 'data', 'demographic', 'r_forecasts', 'fert_pred.csv')
    mort_file = os.path.join(cur_path, 'data', 'demographic', 'r_forecasts', 'mort_pred.csv')
    imm_file = os.path.join(cur_path, 'data', 'demographic', 'r_forecasts', 'imm_pred.csv')

    pred_fert = pd.read_csv(fert_file)
    pred_mort = pd.read_csv(mort_file)
    pred_imm = pd.read_csv(imm_file)

    return pred_fert, pred_mort, pred_imm

fert_params = pickle.load(open('data/demographic/parameters/fert.p', 'rb') )
mort_params = pickle.load(open('data/demographic/parameters/mort.p', 'rb') )
imm_params = pickle.load(open('data/demographic/parameters/imm.p', 'rb') )

pop_data, fert_data, mort_data, imm_rate = pickle.load(open('data/demographic/clean/all.p', 'rb') )

fert_alt, mort_alt, imm_alt = prep_demog_alternate()

datatype = 'population_forecasts'

start = 1971
end = 2017
ages = np.linspace(0, 99, 100).astype(int)
birth_ages = np.linspace(14, 50, 37)
years = np.linspace(start, end, end - start + 1).astype(int)

prev_pop = pop_data[start - 1]

for year in years:
    pred_fert = util.forecast(fert_params, year - 1, birth_ages, datatype='fertility', options=False)
    pred_mort = util.forecast(mort_params, year - 1, ages, datatype='mortality', options=False)
    pred_imm = util.forecast(imm_params, year, ages, datatype='immigration', options={'transition_year': 2015})
    prev_pop = util.predict_population(pred_fert, pred_mort, pred_imm, prev_pop)
    plt.plot(ages, prev_pop, label='Predicted')
    plt.plot(ages, pop_data[year], label='True ' + str(year))
    plt.xlabel(r'Age $s$')
    plt.ylabel(r'Population $\omega_{s,t}$')
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    plt.tight_layout()
    plt.legend(loc='upper right')
    plt.savefig('graphs/' + datatype + '/' + 'start_' + str(start - 1) + '/' + str(year))
    plt.close()

future_start = 2018
future_end = 2500

future_years = np.linspace(future_start, future_end, future_end - future_start + 1).astype(int)

for i in range(3):
    prev_pop = pop_data[future_start - 1]

    for year in future_years:
        if i == 0:
            pred_fert = fert_data[2014]
            pred_mort = mort_data[2014]
            pred_imm = imm_rate[2015]

        elif i == 1:
            pred_fert = util.forecast(fert_params, year - 1, birth_ages, datatype='fertility', options=False)
            pred_mort = util.forecast(mort_params, year - 1, ages, datatype='mortality', options=False)
            pred_imm = util.forecast(imm_params, year, ages, datatype='immigration', options={'transition_year': 2015})

        elif i == 2:
            year_alt = min(year, 2030)
            pred_fert = fert_alt['rate.total.' + str(year_alt - 1)]
            pred_mort = mort_alt['rate.total.' + str(year_alt - 1)]
            pred_imm = imm_alt['rate.total.' + str(year_alt)]

        prev_pop = util.predict_population(pred_fert, pred_mort, pred_imm, prev_pop)
        if year in (2020, 2050, 2070, 2100, 2250, 2500):
            plt.plot(ages, prev_pop, label=str(year))

    plt.xlabel(r'Age $s$')
    plt.ylabel(r'Population $\omega_{s,t}$')
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    plt.tight_layout()
    plt.legend(loc='upper left')
    axes = plt.gca()
    axes.set_xlim([0,100])
    if i == 0:
        plt.title('Partial-Dynamic Population Transition')
        plt.tight_layout()
        plt.savefig('graphs/' + datatype + '/' + 'start_' + str(future_start - 1) + '/predicted_basic')
    elif i == 1:
        plt.title('Full-Dynamic Population Transition')
        plt.tight_layout()
        plt.savefig('graphs/' + datatype + '/' + 'start_' + str(future_start - 1) + '/predicted_parametric')
    elif i == 2:
        plt.title('Alternate Full-Dynamic Population Transition')
        plt.tight_layout()
        plt.savefig('graphs/' + datatype + '/' + 'start_' + str(future_start - 1) + '/predicted_alt')
    plt.close()

    future_start = 2018
    future_end = 3000

    future_years = np.linspace(future_start, future_end, future_end - future_start + 1).astype(int)
    prev_pop = pop_data[future_start - 1]

    for year in future_years:
        if i == 0:
            pred_fert = fert_data[2014]
            pred_mort = mort_data[2014]
            pred_imm = imm_rate[2015]

        elif i == 1:
            pred_fert = util.forecast(fert_params, year - 1, birth_ages, datatype='fertility', options=False)
            pred_mort = util.forecast(mort_params, year - 1, ages, datatype='mortality', options=False)
            pred_imm = util.forecast(imm_params, year, ages, datatype='immigration', options={'transition_year': 2015})

        elif i == 2:
            year_alt = min(year, 2030)
            pred_fert = fert_alt['rate.total.' + str(year_alt - 1)]
            pred_mort = mort_alt['rate.total.' + str(year_alt - 1)]
            pred_imm = imm_alt['rate.total.' + str(year_alt)]

        prev_pop = util.predict_population(pred_fert, pred_mort, pred_imm, prev_pop)
        if year in (2018, 2024, 2058, 2098, 3000):
            if year in (2018, 2024):
                lab = str(year) + ' Pop.'
            elif year == 2058:
                lab = 'T=40 Pop.'
            elif year == 2098:
                lab = 'T=80 Pop.'
            elif year == 3000:
                lab = 'SS Pop.'
            plt.plot(ages, prev_pop / np.sum(prev_pop), label=lab)

    plt.legend(loc='upper left')
    plt.xlabel(r'Age $s$')
    plt.ylabel(r"Pop. Dist'n $\omega_{s,t}$")
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    plt.tight_layout()
    axes = plt.gca()
    axes.set_xlim([0,100])
    if i == 0:
        plt.title('Partial-Dynamic Population Distribution Transition')
        plt.tight_layout()
        plt.savefig('graphs/' + datatype + '/' + 'start_' + str(future_start - 1) + '/predicted_proportion_basic')
    elif i == 1:
        plt.title('Full-Dynamic Population Distribution Transition')
        plt.tight_layout()
        plt.savefig('graphs/' + datatype + '/' + 'start_' + str(future_start - 1) + '/predicted_proportion_parametric')
    elif i == 2:
        plt.title('Alternate Full-Dynamic Population Distribution Transition')
        plt.tight_layout()
        plt.savefig('graphs/' + datatype + '/' + 'start_' + str(future_start - 1) + '/predicted_proportion_alt')
    plt.close()
