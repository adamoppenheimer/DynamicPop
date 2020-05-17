import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pickle

cur_path = '/Users/adamalexanderoppenheimer/Desktop/DynamicPop'
os.chdir(cur_path + '/code')

import util

os.chdir(cur_path)

def prep_demog_alternate(fert_data, mort_data, imm_data):
    '''
    This function returns the alternate dynamic demographic forecasts

    Args:
        fert_data (Numpy array): true fertility data
        mort_data (Numpy array): true mortality data
        imm_data (Numpy array): true immigration data

    Returns:
        fert_all (Pandas dataframe): forecasted and fitted (historical) fertility
        mort_all (Pandas dataframe): forecasted and fitted (historical) mortality
        imm_all (Pandas dataframe): forecasted and fitted (historical) immigration
    '''
    pred_fert_file = os.path.join(cur_path, 'data', 'demographic', 'r_forecasts', 'fert_pred.csv')
    pred_mort_file = os.path.join(cur_path, 'data', 'demographic', 'r_forecasts', 'mort_pred.csv')
    pred_imm_file = os.path.join(cur_path, 'data', 'demographic', 'r_forecasts', 'imm_pred.csv')

    pred_fert_cols = {'rate.total.' + str(i): i for i in range(2015, 3001)}
    pred_fert_drop = ['rate.upper.' + str(i) for i in range(2015, 3001)] + ['rate.lower.' + str(i) for i in range(2015, 3001)]
    pred_fert = pd.read_csv(pred_fert_file).drop(pred_fert_drop, axis=1).rename(pred_fert_cols, axis=1).set_index(fert_data.index) - 1
    pred_mort_cols = {'rate.total.' + str(i): i for i in range(2017, 3001)}
    pred_mort_drop = ['rate.upper.' + str(i) for i in range(2017, 3001)] + ['rate.lower.' + str(i) for i in range(2017, 3001)]
    pred_mort = pd.read_csv(pred_mort_file).drop(pred_mort_drop, axis=1).rename(pred_mort_cols, axis=1).set_index(mort_data.index) - 1
    pred_imm_cols = {'rate.total.' + str(i): i for i in range(2016, 3001)}
    pred_imm_drop = ['rate.upper.' + str(i) for i in range(2016, 3001)] + ['rate.lower.' + str(i) for i in range(2016, 3001)]
    pred_imm = pd.read_csv(pred_imm_file).drop(pred_imm_drop, axis=1).rename(pred_imm_cols, axis=1).set_index(imm_data.index) - 1

    fitted_fert_file = os.path.join(cur_path, 'data', 'demographic', 'r_forecasts', 'fert_fitted.csv')
    fitted_mort_file = os.path.join(cur_path, 'data', 'demographic', 'r_forecasts', 'mort_fitted.csv')
    fitted_imm_file = os.path.join(cur_path, 'data', 'demographic', 'r_forecasts', 'imm_fitted.csv')

    fitted_fert_cols = {'y.' + str(int(i)): i for i in fert_data.columns}
    fitted_fert = pd.read_csv(fitted_fert_file).rename(fitted_fert_cols, axis=1).set_index(fert_data.index)
    fitted_mort_cols = {'y.' + str(i): i for i in mort_data.columns}
    fitted_mort = pd.read_csv(fitted_mort_file).rename(fitted_mort_cols, axis=1).set_index(mort_data.index)
    fitted_imm_cols = {'y.' + str(i): i for i in imm_data.columns}
    fitted_imm = pd.read_csv(fitted_imm_file).rename(fitted_imm_cols, axis=1).set_index(imm_data.index)

    fert_all = pd.concat([fitted_fert, pred_fert], axis=1)
    mort_all = pd.concat([fitted_mort, pred_mort], axis=1)
    imm_all = pd.concat([fitted_imm, pred_imm], axis=1)

    fert_all[fert_all < 0] = 0
    mort_all[mort_all < 0] = 0
    mort_all[mort_all > 1] = 1
    imm_all[imm_all < -1] = -1

    return fert_all, mort_all, imm_all

fert_params = pickle.load(open('data/demographic/parameters/fert.p', 'rb') )
mort_params = pickle.load(open('data/demographic/parameters/mort.p', 'rb') )
imm_params = pickle.load(open('data/demographic/parameters/imm.p', 'rb') )

pop_data, fert_data, mort_data, imm_data = pickle.load(open('data/demographic/clean/all.p', 'rb') )

fert_alt, mort_alt, imm_alt = prep_demog_alternate(fert_data, mort_data, imm_data)

datatype = 'population_forecasts'



start = 1971
end = 2017
ages = np.linspace(0, 99, 100).astype(int)
birth_ages = np.linspace(14, 50, 37)
years = np.linspace(start, end, end - start + 1).astype(int)

# Plot true data transition
util.plot_data_transition(pop_data, ages, 1970, 2014, smooth=0, datatype='population')

# Start forecasts

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

for i in range(3):

    future_start = 2018
    future_end = 2500

    future_years = np.linspace(future_start, future_end, future_end - future_start + 1).astype(int)

    prev_pop = pop_data[future_start - 1]

    for year in future_years:
        if i == 0:
            pred_fert = fert_data[2014]
            pred_mort = mort_data[2014]
            pred_imm = imm_data[2015]

        elif i == 1:
            pred_fert = util.forecast(fert_params, year - 1, birth_ages, datatype='fertility', options=False)
            pred_mort = util.forecast(mort_params, year - 1, ages, datatype='mortality', options=False)
            pred_imm = util.forecast(imm_params, year, ages, datatype='immigration', options={'transition_year': 2015})

        elif i == 2:
            year_alt = min(year, 2100)
            pred_fert = fert_alt[year_alt - 1]
            pred_mort = mort_alt[year_alt - 1]
            pred_imm = imm_alt[year_alt]

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
            pred_imm = imm_data[2015]

        elif i == 1:
            pred_fert = util.forecast(fert_params, year - 1, birth_ages, datatype='fertility', options=False)
            pred_mort = util.forecast(mort_params, year - 1, ages, datatype='mortality', options=False)
            pred_imm = util.forecast(imm_params, year, ages, datatype='immigration', options={'transition_year': 2015})

        elif i == 2:
            year_alt = min(year, 2100)
            pred_fert = fert_alt[year_alt - 1]
            pred_mort = mort_alt[year_alt - 1]
            pred_imm = imm_alt[year_alt]

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

##############################################################
##### Forecast Alternate Fertility/Mortality/Immigration #####
##############################################################

# Fertility
birth_ages = np.linspace(15, 52, 37)

for year in (1990, 2000, 2014, 2050, 2100):
    plt.plot(birth_ages, fert_alt[year], linewidth=2, label=str(year))

plt.xlabel(r'Age $s$')
y = r'Fertility Rate $f_{s,t}$'
plt.ylabel(y)
plt.grid(b=True, which='major', color='0.65', linestyle='-')
plt.tight_layout()
plt.legend()

plt.savefig('graphs/fertility/alternate/_2100')
plt.close()

# Mortality

mort_ages = np.linspace(0, 99, 100)

for year in (1990, 2000, 2014, 2050, 2100):
    plt.plot(mort_ages, mort_alt[year], linewidth=2, label=str(year))

plt.xlabel(r'Age $s$')
y = r'Mortality Rate $\rho_{s,t}$'
plt.ylabel(y)
plt.grid(b=True, which='major', color='0.65', linestyle='-')
plt.tight_layout()
plt.legend()

plt.savefig('graphs/mortality/alternate/_2100')
plt.close()

# Immigration

imm_ages = np.linspace(0, 99, 100)

for year in (1997, 2000, 2014, 2050, 2100):
    plt.plot(imm_ages, imm_alt[year], linewidth=2, label=str(year))

plt.xlabel(r'Age $s$')
y = r'Immigration Rate $i_{s,t}$'
plt.ylabel(y)
plt.grid(b=True, which='major', color='0.65', linestyle='-')
plt.tight_layout()
plt.legend()

plt.savefig('graphs/immigration/alternate/_2100')
plt.close()
