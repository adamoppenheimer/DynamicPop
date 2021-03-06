'''
------------------------------------------------------------------------
Functions for generating demographic objects necessary for the OG-USA
model
------------------------------------------------------------------------
'''
# Import packages
import os
import numpy as np
import scipy.optimize as opt
import scipy.interpolate as si
import pandas as pd
import parameter_plots as pp
import pickle


# create output directory for figures
CUR_PATH = os.path.split(os.path.abspath(__file__))[0]
OUTPUT_DIR = os.path.join(CUR_PATH, 'OUTPUT', 'Demographics')
if os.access(OUTPUT_DIR, os.F_OK) is False:
    os.makedirs(OUTPUT_DIR)

#os.chdir('..')

import adam_util as util

os.chdir(CUR_PATH)

'''
------------------------------------------------------------------------
Define functions
------------------------------------------------------------------------
'''

def forecast_pop(country, min_age, max_age, demog_type, start_yr, end_yr=False, demog_files=[False, False, False]):
    '''
    This function uses forecasted fertility, mortality,
    and immigration rates to forecast population for
    a range of years

    Args:
        country: country that is source of data
        min_age (int): age in years at which agents are born, >= 0
        max_age (int): age in years at which agents die with certainty,
            >= 4
        demog_type (str): demography forecasting technique
        start_yr: first year to forecast
        end_yr: last year to forecast. If False, set equal to start_yr
        demog_files (Pandas dataframe): alternate demographic forecasts

    Returns:
        forecasted_pop (Pandas dataframe): population for each year
    '''
    if not end_yr:
        end_yr = start_yr

    fert_all, mort_all, imm_all = demog_files

    pop_file = os.path.join(CUR_PATH, 'data', 'demographic', country, 'clean', 'pop.p')

    pop_data = pickle.load(open(pop_file, 'rb') )

    # Access most recent year of population data
    starting_pop = pop_data[max(pop_data.columns)]
    prev_pop = starting_pop.copy()
    forecasted_pop = pd.DataFrame()

    for year in range(max(pop_data.columns) + 1, end_yr + 1):
        pred_fert = forecast_fert(country, year - 1, demog_type, fert_all)
        pred_mort = forecast_mort(country, year - 1, demog_type, mort_all)
        pred_imm = forecast_imm(country, year, demog_type, imm_all)
        new_pop = util.predict_population(pred_fert, pred_mort, pred_imm, prev_pop)
        if year >= start_yr:
            forecasted_pop[year] = new_pop[max(min_age, 0): min(max_age + 1, len(new_pop))]
        prev_pop = new_pop.copy()

    if start_yr == end_yr:
        return forecasted_pop[start_yr]

    return forecasted_pop

def forecast_fert(country, year, demog_type, fert_file=False):
    '''
    This function returns forecasted fertility for a particular year

    Args:
        country: country that is source of data
        year: year to forecast
        demog_type (str): demography forecasting technique
        fert_file (Pandas dataframe): alternate forecasting data

    Returns:
        pred_fert (Numpy array): forecasted fertility
    '''
    if demog_type == 'dynamic_full':
        fert_file = os.path.join(CUR_PATH, 'data', 'demographic', country, 'parameters', 'fert.p')

        fert_params = pickle.load(open(fert_file, 'rb') )

        birth_ages = np.linspace(14, 50, 37)

        pred_fert = util.forecast(fert_params, year, birth_ages, datatype='fertility', options=False)

    elif demog_type == 'dynamic_full_alternate':
        if year > 2050:
            year = 2050
        pred_fert = fert_file['rate.total.' + str(year)]

    return pred_fert

def forecast_mort(country, year, demog_type, mort_file=False):
    '''
    This function returns forecasted mortality for a particular year

    Args:
        country: country that is source of data
        year: year to forecast
        demog_type (str): demography forecasting technique
        mort_file (Pandas dataframe): alternate forecasting data

    Returns:
        pred_mort (Numpy array): forecasted mortality
    '''
    if demog_type == 'dynamic_full':
        mort_file = os.path.join(CUR_PATH, 'data', 'demographic', country, 'parameters', 'mort.p')

        mort_params = pickle.load(open(mort_file, 'rb') )

        ages = np.linspace(0, 99, 100).astype(int)

        pred_mort = util.forecast(mort_params, year, ages, datatype='mortality', options=False)

    elif demog_type == 'dynamic_full_alternate':
        if year > 2050:
            year = 2050
        pred_mort = mort_file['rate.total.' + str(year)]

    return pred_mort

def forecast_imm(country, year, demog_type, imm_file=False):
    '''
    This function returns forecasted immigration for a particular year

    Args:
        country: country that is source of data
        year: year to forecast
        demog_type (str): demography forecasting technique
        imm_file (Pandas dataframe): alternate forecasting data

    Returns:
        pred_imm (Numpy array): forecasted immigration
    '''
    if demog_type == 'dynamic_full':
        imm_file = os.path.join(CUR_PATH, 'data', 'demographic', country, 'parameters', 'imm.p')

        imm_params = pickle.load(open(imm_file, 'rb') )

        ages = np.linspace(0, 99, 100).astype(int)

        pred_imm = util.forecast(imm_params, year, ages, datatype='immigration', options={'transition_year': 2015})

    elif demog_type == 'dynamic_full_alternate':
        if year > 2050:
            year = 2050
        pred_imm = imm_file['rate.total.' + str(year)]

    return pred_imm

def prep_demog_alternate(country):
    '''
    This function returns the alternate dynamic demographic forecasts

    Args:
        country: country that is source of data

    Returns:
        pred_fert (Pandas dataframe): forecasted fertility
        pred_mort (Pandas dataframe): forecasted mortality
        pred_imm (Pandas dataframe): forecasted immigration
    '''
    fert_file = os.path.join(CUR_PATH, 'data', 'demographic', country, 'r_forecasts', 'fert_pred.csv')
    mort_file = os.path.join(CUR_PATH, 'data', 'demographic', country, 'r_forecasts', 'mort_pred.csv')
    imm_file = os.path.join(CUR_PATH, 'data', 'demographic', country, 'r_forecasts', 'imm_pred.csv')

    pred_fert = pd.read_csv(fert_file) - 1
    pred_mort = pd.read_csv(mort_file) - 1
    pred_imm = pd.read_csv(imm_file) - 1

    pred_fert[pred_fert < 0] = 0
    pred_mort[pred_mort < 0] = 0
    pred_mort[pred_mort > 1] = 1

    return pred_fert, pred_mort, pred_imm

def get_true_demog_data(country, min_age, max_age):
    '''
    This function returns the true demographic
    data for a country
    
    Args:
        country: country that is source of data
        min_age (int): age in years at which agents are born, >= 0
        max_age (int): age in years at which agents die with certainty,
            >= 4

    Returns:
        fert_data (Numpy array): fertility rates for each model period of life
        mort_data (Numpy array): mortality rates for each model period of life
        imm_data (Numpy array): immigration rates for each model period of life
        pop_data (Numpy array): population for each model period of life
    '''

    demog_file = os.path.join(CUR_PATH, 'data', 'demographic', country, 'clean', 'all.p')

    pop_data, fert_data, mort_data, imm_data = pickle.load(open(demog_file, 'rb') )

    pop_2014 = pop_data[2014][max(min_age, 0): min(max_age + 1, len(pop_data[2014]))]
    pop_2015 = pop_data[2015][max(min_age, 0): min(max_age + 1, len(pop_data[2015]))]

    return pop_2014, pop_2015, fert_data[2014], mort_data[2014], imm_data[2015]

def get_fert(totpers, min_age, max_age, year, country, demog_type, graph=False, demog_files=[False, False, False]):
    '''
    This function generates a vector of fertility rates by model period
    age that corresponds to the fertility rate data by age in years

    Args:
        totpers (int): total number of agent life periods (E+S), >= 3
        min_age (int): age in years at which agents are born, >= 0
        max_age (int): age in years at which agents die with certainty,
            >= 4
        year (int): year of fertility data
        country (str): country that is source of data
        demog_type (str): demography forecasting technique
        graph (bool): =True if want graphical output
        demog_files (Pandas dataframe): alternate demographic forecasts

    Returns:
        fert_rates (Numpy array): fertility rates for each model period of life

    '''
    fert_all, mort_all, imm_all = demog_files

    # Get data
    if demog_type in ['dynamic_full', 'dynamic_full_alternate']:
        curr_pop = forecast_pop(country, min_age, max_age, demog_type, year, demog_files=demog_files)
        curr_fert = forecast_fert(country, year, demog_type, fert_file=fert_all)
    elif demog_type in ['static', 'dynamic_partial']:
        curr_pop, _, curr_fert, _, _ = get_true_demog_data(country, min_age, max_age)
    # Birth ages
    birth_ages = np.linspace(14, 50, 37)
    # Population Distribution
    curr_pop_pct = curr_pop / curr_pop.sum()

    if (min_age == 0) and (max_age == 99) and (totpers == 100) and (not graph):
        fert_rates = np.zeros(totpers)
        # Births from 14-50, but age start at 0
        fert_rates[15:52] = curr_fert
        return fert_rates

    ### VARIABLE PREPARATION
    num_bins = max_age - min_age + 1
    binsize = num_bins / totpers
    num_sub_bins = float(10000)
    len_subbins = (np.float64(num_bins * num_sub_bins)) / totpers

    ### POPULATION CREATION
    ages = np.linspace(max(min_age, 0), min(max_age, 99), curr_pop_pct.shape[0])
    pop_func = si.splrep(ages, curr_pop_pct)
    new_bins = np.linspace(max(min_age, 0), min(max_age, 99), int(num_sub_bins * (num_bins - 1)), dtype=float)
    curr_pop_sub = si.splev(new_bins, pop_func)
    curr_pop_sub = curr_pop_sub / curr_pop_sub.sum()

    #### AGE BIN CREATION
    # Calculate implied fertility rates in sub-bins of curr_fert
    fert_func = si.splrep(birth_ages, curr_fert)
    fert_rates_sub = np.zeros(curr_pop_sub.shape)
    age_sub = (np.linspace(np.float64(binsize) / num_sub_bins + np.float64(min_age), np.float64(max_age), int(num_sub_bins * (num_bins - 1))) - 0.5 * np.float64(binsize) / num_sub_bins)
    # Fill in fertility rates
    pred_ind = (age_sub >= birth_ages[0]) * (age_sub <= birth_ages[-1]) # Makes sure it is inside valid range
    age_pred = age_sub[pred_ind] # Gets age_sub in the valid range by applying pred_ind
    fert_rates_sub[pred_ind] = np.float64(si.splev(age_pred, fert_func))
    fert_rates_sub[fert_rates_sub < 0] = 0
    fert_rates = np.zeros(totpers)

    for i in range(totpers):
        beg_sub_bin = int(np.rint(i * len_subbins))
        end_sub_bin = int(np.rint((i + 1) * len_subbins))
        if i == totpers - 1:
            end_sub_bin += 1
        fert_rates[i] = ((
            curr_pop_sub[beg_sub_bin:end_sub_bin] *
            fert_rates_sub[beg_sub_bin:end_sub_bin]).sum() /
            curr_pop_sub[beg_sub_bin:end_sub_bin].sum())
    fert_rates = np.nan_to_num(fert_rates)

    if graph:
        pp.plot_fert_rates(fert_func, birth_ages, totpers, min_age, max_age, curr_fert, fert_rates, output_dir=OUTPUT_DIR)

    if (min_age == 0) and (max_age == 99) and (totpers == 100):
        fert_rates = np.zeros(totpers)
        # Births from 14-50, but age start at 0
        fert_rates[15:52] = curr_fert

    return fert_rates


def get_mort(totpers, min_age, max_age, year, country, demog_type, graph=False, mort_file=False):
    '''
    This function generates a vector of mortality rates by model period
    age.

    Args:
        totpers (int): total number of agent life periods (E+S), >= 3
        min_age (int): age in years at which agents are born, >= 0
        max_age (int): age in years at which agents die with certainty,
            >= 4
        year (int): year of mortality data
        country (str): country that is source of data
        demog_type (str): demography forecasting technique
        graph (bool): =True if want graphical output
        demog_files (Pandas dataframe): alternate demographic forecasts

    Returns:
        mort_rates (Numpy array): mortality rates that correspond to each
            period of life
        infmort_rate (scalar): infant mortality rate

    '''

    # Get data
    if demog_type in ['dynamic_full', 'dynamic_full_alternate']:
        curr_mort = forecast_mort(country, year, demog_type, mort_file=mort_file)
    elif demog_type in ['static', 'dynamic_partial']:
        _, _, _, curr_mort, _ = get_true_demog_data(country, min_age, max_age)
    # Mortality ages
    mort_ages = np.linspace(0, 99, 100).astype(int)
    # Infant Mortality Rate
    infmort_rate = curr_mort[0]

    if (min_age == 0) and (max_age == 99) and (totpers == 100) and (not graph):
        return curr_mort, 0 # infmort_rate

    ### VARIABLE PREPARATION
    num_bins = max_age - min_age + 1
    binsize = num_bins / totpers
    num_sub_bins = int(100)
    len_subbins = ((np.float64(num_bins * num_sub_bins)) / totpers)

    #### AGE BIN CREATION
    # Calculate implied mortality rates in sub-bins of curr_mort
    mort_func = si.splrep(mort_ages, curr_mort)
    mort_sub = (np.linspace(np.float64(binsize) / num_sub_bins + np.float64(min_age), np.float64(max_age), int(num_sub_bins * (num_bins - 1))) - 0.5 * np.float64(binsize) / num_sub_bins) # CORRECT TO NUM_BINS NOT -1
    # Fill in mortality rates
    mort_rates_sub_orig = 1 - si.splev(mort_sub, mort_func)
    mort_rates_sub_orig[mort_rates_sub_orig > 1] = 1
    mort_rates_sub_orig[mort_rates_sub_orig < 0] = 0

    mort_rates_sub = np.zeros(mort_rates_sub_orig.shape, dtype=float)

    for i in range(totpers):
        beg_sub_bin = int(np.rint(i * num_sub_bins))
        end_sub_bin = int(np.rint((i + 1) * num_sub_bins))
        if i == totpers - 1:
            end_sub_bin += 1
        tot_period_surv = (np.log(mort_rates_sub_orig[beg_sub_bin:end_sub_bin]) ).sum()
        end_surv = np.log(1 - curr_mort[min_age:][i])
        if tot_period_surv != 0:
            power = end_surv / tot_period_surv
        else:
            power = 0
        mort_rates_sub[beg_sub_bin:end_sub_bin] = mort_rates_sub_orig[beg_sub_bin:end_sub_bin] ** power

    mort_rates = np.zeros(totpers)
    for i in range(totpers):
        beg_sub_bin = int(np.rint(i * len_subbins))
        end_sub_bin = int(np.rint((i + 1) * len_subbins))
        if i == totpers - 1:
            end_sub_bin += 1
        mort_rates[i] = 1 - mort_rates_sub[beg_sub_bin:end_sub_bin].prod()
    mort_rates[-1] = 1  # Mortality rate in last period is set to 1

    if graph:
        pp.plot_mort_rates_data(totpers, min_age, max_age, mort_ages[max(min_age, 0):min(max_age + 1, 100)],
                                curr_mort[max(min_age, 0):min(max_age + 1, 100)], infmort_rate,
                                mort_rates, output_dir=OUTPUT_DIR)

    if (min_age == 0) and (max_age == 99) and (totpers == 100):
        mort_rates = curr_mort

    return mort_rates, 0 # infmort_rate

def pop_rebin(curr_pop_dist, totpers_new):
    '''
    For cases in which totpers (E+S) is less than the number of periods
    in the population distribution data, this function calculates a new
    population distribution vector with totpers (E+S) elements.

    Args:
        curr_pop_dist (Numpy array): population distribution over N
            periods
        totpers_new (int): number of periods to which we are
            transforming the population distribution, >= 3

    Returns:
        curr_pop_new (Numpy array): new population distribution over
            totpers (E+S) periods that approximates curr_pop_dist

    '''
    # Number of periods in original data
    assert totpers_new >= 3
    totpers_orig = len(curr_pop_dist)
    if int(totpers_new) == totpers_orig:
        curr_pop_new = curr_pop_dist
    elif int(totpers_new) < totpers_orig:
        num_sub_bins = float(10000)
        ages = np.linspace(0, totpers_orig - 1, totpers_orig)
        pop_func = si.splrep(ages, curr_pop_dist)
        new_bins = np.linspace(0, totpers_orig - 1,\
                                int(num_sub_bins * totpers_orig))
        pop_ests = si.splev(new_bins, pop_func)
        len_subbins = ((np.float64(totpers_orig * num_sub_bins)) /
                       totpers_new)
        curr_pop_new = np.zeros(totpers_new, dtype=np.float64)
        for i in range(totpers_new):
            beg_sub_bin = int(np.rint(i * len_subbins))
            end_sub_bin = int(np.rint((i + 1) * len_subbins))
            curr_pop_new[i] = \
                np.average(pop_ests[beg_sub_bin:end_sub_bin])
        # Return curr_pop_new to single precision float (float32)
        # datatype
        curr_pop_new = np.float32(curr_pop_new) * np.sum(curr_pop_dist) / np.sum(curr_pop_new) # Adjust sum

    return curr_pop_new


def immsolve(imm_rates, *args):
    '''
    This function generates a vector of errors representing the
    difference in two consecutive periods stationary population
    distributions. This vector of differences is the zero-function
    objective used to solve for the immigration rates vector, similar to
    the original immigration rates vector from util.calc_imm_resid(), that
    sets the steady-state population distribution by age equal to the
    population distribution in period int(1.5*S)

    Args:
        imm_rates (Numpy array):immigration rates that correspond to
            each period of life, length E+S
        args (tuple): (fert_rates, mort_rates, infmort_rate, omega_cur,
            g_n_SS)

    Returns:
        omega_errs (Numpy array): difference between omega_new and
            omega_cur_pct, length E+S

    '''
    fert_rates, mort_rates, infmort_rate, omega_cur_lev, g_n_SS = args
    omega_cur_pct = omega_cur_lev / omega_cur_lev.sum()
    new_pop = util.predict_population(fert_rates, mort_rates, imm_rates, omega_cur_lev)
    omega_new = new_pop / new_pop.sum()
    omega_errs = omega_new - omega_cur_pct

    return omega_errs


def get_pop_objs_static(E, S, T, min_age, max_age, curr_year, country='Japan', GraphDiag=True):
    '''
    This function produces the demographics objects to be used in the
    OG-USA model package.

    Args:
        E (int): number of model periods in which agent is not
            economically active, >= 1
        S (int): number of model periods in which agent is economically
            active, >= 3
        T (int): number of periods to be simulated in TPI, > 2*S
        min_age (int): age in years at which agents are born, >= 0
        max_age (int): age in years at which agents die with certainty,
            >= 4
        curr_year (int): current year for which analysis will begin,
            >= 2020
        country (str): country that is source of data
        GraphDiag (bool): =True if want graphical output and printed
                diagnostics

    Returns:
        omega_path_S (Numpy array): time path of the population
            distribution from period when demographics stabilize to the steady-state,
            size T+S x S
        g_n_SS (scalar): steady-state population growth rate
        omega_SS (Numpy array): normalized steady-state population
            distribution, length S
        surv_rates (Numpy array): survival rates that correspond to
            each model period of life, length S
        mort_rates (Numpy array): mortality rates that correspond to
            each model period of life, length S
        g_n_path (Numpy array): population growth rates over the time
            path, length T + S
        imm_rates_mat (Numpy array): steady-state immigration rates by
                        age for economically active ages, length S

        (omega_path_S.T, g_n_SS, omega_SSfx[-S:] /
            omega_SSfx[-S:].sum(), 1 - mort_rates_S, mort_rates_S,
            g_n_path, imm_rates_mat.T, omega_S_preTP)

    '''
    # Prepare demographics with true data
    data_year = 2015
    pop_prev_data, pop_yr_data, _, _, _ = get_true_demog_data(country, min_age, max_age)
    pop_prev_rebin = pop_rebin(pop_prev_data, E + S)
    pop_yr_rebin = pop_rebin(pop_yr_data, E + S)
    fert_rates = get_fert(E + S, min_age, max_age, data_year - 1, country, demog_type='static', graph=True)
    mort_rates, infmort_rate = get_mort(E + S, min_age, max_age, data_year - 1, country, demog_type='static', graph=True)
    mort_rates_S = mort_rates[-S:]
    imm_rates_orig = util.calc_imm_resid(fert_rates, mort_rates, pop_prev_rebin, pop_yr_rebin)
    
    ages = np.arange(min_age, max_age + 1)

    # Generate time paths of:
    # Nonstationary population
    omega_path_lev = np.zeros((E + S, T + S))
    # Nonstationary population distribution of working age
    omega_path_S = np.zeros((S, T + S))
    omega_SS_orig = pop_yr_rebin / np.sum(pop_yr_rebin)
    pop_yr_S_pct = pop_yr_rebin[-S:] / pop_yr_rebin[-S:].sum()
    omega_S_preTP = np.expand_dims(pop_yr_S_pct, 1)
    # Immigration rates
    imm_rates_mat = np.zeros((S, T + S))
    for per in range(T + S):
        omega_path_lev[:, per] = pop_yr_rebin.copy()
        omega_path_S[:, per] = pop_yr_S_pct.copy()
        imm_rates_mat[:, per] = imm_rates_orig[E:].copy()

    # Population growth rate
    g_n_SS = 0
    g_n_path = np.zeros(T + S)

    # Generate time path of mortality rates
    rho_path_lev = np.zeros((S, T + S + S))
    for per in range(T + S + S):
        rho_path_lev[:, per] = mort_rates_S.copy()
    mort_rates_S = np.expand_dims(mort_rates_S, 1)

    if GraphDiag:
        # plots
        pp.plot_omega_fixed(ages, omega_SS_orig, omega_SS_orig, E,
                            S, output_dir=(OUTPUT_DIR, 'static'))
        pp.plot_imm_fixed(ages, imm_rates_orig, imm_rates_orig, E,
                          S, output_dir=(OUTPUT_DIR, 'static'))
        pp.plot_population_path(ages, omega_SS_orig,
                                omega_path_lev, omega_SS_orig, curr_year,
                                E, S, output_dir=(OUTPUT_DIR, 'static'))
        print('Static diagnostic graphs generated')

    # return omega_path_S, g_n_SS, omega_SSfx, survival rates,
    # mort_rates_S, and g_n_path
    print('------------------------------')
    print('E:', E)
    print('S:', S)
    print('T:', T)
    print('Omega_path_S.shape:', omega_path_S.shape) # Should be 80x240 (Sx3*S)
    # print(omega_path_S)
    print('g_n_SS:', g_n_SS) # Should be a scalar
    print('omega_S_preTP.shape:', omega_S_preTP.shape) # Should be 80x1 (Sx1)
    print('mort_rates_S.shape:', mort_rates_S.shape) # Should be size 80 (S)
    print('rho_path_lev.shape:', rho_path_lev.shape) # Should be 80x320
    print('g_n_path.shape:', g_n_path.shape) # Should be size 240 (S*3)
    print('imm_rates_mat.shape:', imm_rates_mat.shape) # Should be 80x240 (Sx3*S)

    return (omega_path_S.T, g_n_SS, omega_S_preTP, 1 - mort_rates_S, rho_path_lev.T,
            g_n_path, imm_rates_mat.T, omega_S_preTP)


def get_pop_objs_dynamic_partial(E, S, T, min_age, max_age, curr_year, country='Japan', GraphDiag=True):
    '''
    This function produces the demographics objects to be used in the
    OG-USA model package.

    Args:
        E (int): number of model periods in which agent is not
            economically active, >= 1
        S (int): number of model periods in which agent is economically
            active, >= 3
        T (int): number of periods to be simulated in TPI, > 2*S
        min_age (int): age in years at which agents are born, >= 0
        max_age (int): age in years at which agents die with certainty,
            >= 4
        curr_year (int): current year for which analysis will begin,
            >= 2020
        country (str): country that is source of data
        GraphDiag (bool): =True if want graphical output and printed
                diagnostics

    Returns:
        omega_path_S (Numpy array), time path of the population
            distribution from the current state to the steady-state,
            size T+S x S
        g_n_SS (scalar): steady-state population growth rate
        omega_SS (Numpy array): normalized steady-state population
            distribution, length S
        surv_rates (Numpy array): survival rates that correspond to
            each model period of life, lenght S
        mort_rates (Numpy array): mortality rates that correspond to
            each model period of life, length S
        g_n_path (Numpy array): population growth rates over the time
            path, length T + S

    '''
    # Prepare demographics for data_year
    data_year = 2015
    pop_prev_data, pop_yr_data, _, _, _ = get_true_demog_data(country, min_age, max_age)
    pop_prev_rebin = pop_rebin(pop_prev_data, E + S)
    pop_yr_rebin = pop_rebin(pop_yr_data, E + S)
    fert_rates = get_fert(E + S, min_age, max_age, data_year - 1, country, demog_type='dynamic_partial', graph=False)
    mort_rates, infmort_rate = get_mort(E + S, min_age, max_age, data_year - 1, country, demog_type='dynamic_partial', graph=False)
    mort_rates_S = mort_rates[-S:]
    mort_rates_S = np.expand_dims(mort_rates_S, 1)
    imm_rates_orig = util.calc_imm_resid(fert_rates, mort_rates, pop_prev_rebin, pop_yr_rebin)
    
    ages = np.arange(min_age, max_age + 1)
    pop_yr_pct = pop_yr_rebin / np.sum(pop_yr_rebin)

    OMEGA_orig = np.zeros((E + S, E + S))
    OMEGA_orig[0, :] = fert_rates / 2
    OMEGA_orig[1:, :-1] += np.diag(1 - mort_rates[:-1])
    OMEGA_orig += np.diag(imm_rates_orig)

    # Solve for steady-state population growth rate and steady-state
    # population distribution by age using eigenvalue and eigenvector
    # decomposition
    eigvalues, eigvectors = np.linalg.eig(OMEGA_orig)
    g_n_SS = (eigvalues[np.isreal(eigvalues)].real).max() - 1
    eigvec_raw = eigvectors[:, (eigvalues[np.isreal(eigvalues)].real).argmax()].real
    omega_SS_orig = eigvec_raw / eigvec_raw.sum()

    # Age most recent population data to the current year of analysis
    pop_past = pop_yr_rebin
    pop_curr = np.dot(OMEGA_orig, pop_yr_rebin)
    # Age the data to the current year
    for per in range(curr_year - data_year - 1):
        pop_past = pop_curr.copy()
        pop_curr = np.dot(OMEGA_orig, pop_curr)
    g_n_curr = ((pop_curr[-S:].sum() - pop_past[-S:].sum()) / pop_past[-S:].sum())
    omega_S_preTP = pop_curr[-S:] / pop_curr[-S:].sum()
    omega_S_preTP = np.expand_dims(omega_S_preTP, 1)

    # Generate time path of nonstationary population
    omega_path_lev = np.zeros((E + S, T + S))
    omega_path_lev[:, 0] = pop_curr.copy()
    for per in range(1, T + S):
        pop_curr = np.dot(OMEGA_orig, pop_curr)
        omega_path_lev[:, per] = pop_curr.copy()

    # Force the population distribution after 1.5*S periods to be the
    # steady-state distribution by adjusting immigration rates, holding
    # constant mortality, fertility, and SS growth rates
    imm_tol = 1e-14
    fixper = int(1.5 * S)
    omega_SSfx = (omega_path_lev[:, fixper] / omega_path_lev[:, fixper].sum())
    imm_objs = (fert_rates, mort_rates, infmort_rate, omega_path_lev[:, fixper], g_n_SS)
    imm_fulloutput = opt.fsolve(immsolve, imm_rates_orig, args=(imm_objs), full_output=True, xtol=imm_tol)
    imm_rates_adj = imm_fulloutput[0]
    imm_diagdict = imm_fulloutput[1]
    omega_path_S = (omega_path_lev[-S:, :] / np.tile(omega_path_lev[-S:, :].sum(axis=0), (S, 1)))
    omega_path_S[:, fixper:] = np.tile(omega_path_S[:, fixper].reshape((S, 1)), (1, T + S - fixper))

    # Population growth rate
    g_n_path = np.zeros(T + S)
    g_n_path[0] = g_n_curr.copy()
    g_n_path[1:] = ((omega_path_lev[-S:, 1:].sum(axis=0) - omega_path_lev[-S:, :-1].sum(axis=0)) / omega_path_lev[-S:, :-1].sum(axis=0))
    g_n_path[fixper + 1:] = g_n_SS

    # Generate time path of immigration rates
    imm_rates_mat = np.zeros((S, T + S))
    for per in range(T + S):
        if per <= fixper:
            imm_rates_mat[:, per] = imm_rates_orig[E:].copy()
        else:
            imm_rates_mat[:, per] = imm_rates_adj[E:].copy()

    # imm_rates_mat = np.hstack((np.tile(np.reshape(np.array(imm_rates_orig)[E:], (S, 1)), (1, fixper)), np.tile(np.reshape(imm_rates_adj[E:], (S, 1)), (1, T + S - fixper))))

    # Generate time path of mortality rates
    rho_path_lev = np.zeros((S, T + S + S))
    for per in range(T + S + S):
        rho_path_lev[:, per] = mort_rates[-S:].copy()

    if GraphDiag:
        # Check whether original SS population distribution is close to
        # the period-T population distribution
        omegaSSmaxdif = np.absolute(omega_SS_orig - (omega_path_lev[:, T] / omega_path_lev[:, T].sum())).max()
        if omegaSSmaxdif > 0.0003:
            print('POP. WARNING: Max. abs. dist. between original SS ' +
                  "pop. dist'n and period-T pop. dist'n is greater than" +
                  ' 0.0003. It is ' + str(omegaSSmaxdif) + '.')
        else:
            print('POP. SUCCESS: orig. SS pop. dist is very close to ' +
                  "period-T pop. dist'n. The maximum absolute " +
                  'difference is ' + str(omegaSSmaxdif) + '.')

        # Plot the adjusted steady-state population distribution versus
        # the original population distribution. The difference should be
        # small
        omegaSSvTmaxdiff = np.absolute(omega_SS_orig - omega_SSfx).max()
        if omegaSSvTmaxdiff > 0.0003:
            print('POP. WARNING: The maximimum absolute difference ' +
                  'between any two corresponding points in the ' +
                  'original and adjusted steady-state population ' +
                  'distributions is' + str(omegaSSvTmaxdiff) + ', ' +
                  'which is greater than 0.0003.')
        else:
            print('POP. SUCCESS: The maximum absolute difference ' +
                  'between any two corresponding points in the ' +
                  'original and adjusted steady-state population ' +
                  'distributions is ' + str(omegaSSvTmaxdiff))

        # Print whether or not the adjusted immigration rates solved the
        # zero condition
        immtol_solved = np.absolute(imm_diagdict['fvec'].max()) < imm_tol
        if immtol_solved:
            print('POP. SUCCESS: Adjusted immigration rates solved ' +
                  'with maximum absolute error of ' +
                  str(np.absolute(imm_diagdict['fvec'].max())) +
                  ', which is less than the tolerance of ' +
                  str(imm_tol))
        else:
            print('POP. WARNING: Adjusted immigration rates did not ' +
                  'solve. Maximum absolute error of ' +
                  str(np.absolute(imm_diagdict['fvec'].max())) +
                  ' is greater than the tolerance of ' + str(imm_tol))

        # Test whether the steady-state growth rates implied by the
        # adjusted OMEGA matrix equals the steady-state growth rate of
        # the original OMEGA matrix
        OMEGA2 = np.zeros((E + S, E + S))
        OMEGA2[0, :] = fert_rates / 2
        OMEGA2[1:, :-1] += np.diag(1 - mort_rates[:-1])
        OMEGA2 += np.diag(imm_rates_adj)
        eigvalues2, eigvectors2 = np.linalg.eig(OMEGA2)
        g_n_SS_adj = (eigvalues[np.isreal(eigvalues2)].real).max() - 1
        if np.max(np.absolute(g_n_SS_adj - g_n_SS)) > 10 ** (-8):
            print('FAILURE: The steady-state population growth rate' +
                  ' from adjusted OMEGA is different (diff is ' +
                  str(g_n_SS_adj - g_n_SS) + ') than the steady-' +
                  'state population growth rate from the original' +
                  ' OMEGA.')
        elif np.max(np.absolute(g_n_SS_adj - g_n_SS)) <= 10 ** (-8):
            print('SUCCESS: The steady-state population growth rate' +
                  ' from adjusted OMEGA is close to (diff is ' +
                  str(g_n_SS_adj - g_n_SS) + ') the steady-' +
                  'state population growth rate from the original' +
                  ' OMEGA.')

        # Do another test of the adjusted immigration rates. Create the
        # new OMEGA matrix implied by the new immigration rates. Plug in
        # the adjusted steady-state population distribution. Hit is with
        # the new OMEGA transition matrix and it should return the new
        # steady-state population distribution
        omega_new = np.dot(OMEGA2, omega_SSfx)
        omega_errs = np.absolute(omega_new - omega_SSfx)
        print('The maximum absolute difference between the adjusted ' +
              'steady-state population distribution and the ' +
              'distribution generated by hitting the adjusted OMEGA ' +
              'transition matrix is ' + str(omega_errs.max()))

        # Plot the original immigration rates versus the adjusted
        # immigration rates
        immratesmaxdiff = \
            np.absolute(imm_rates_orig - imm_rates_adj).max()
        print('The maximum absolute distance between any two points ' +
              'of the original immigration rates and adjusted ' +
              'immigration rates is ' + str(immratesmaxdiff))

        # plots
        pp.plot_omega_fixed(ages, omega_SS_orig, omega_SSfx, E,
                            S, output_dir=(OUTPUT_DIR, 'dynamic_partial'))
        pp.plot_imm_fixed(ages, imm_rates_orig, imm_rates_adj, E,
                          S, output_dir=(OUTPUT_DIR, 'dynamic_partial'))
        pp.plot_population_path(ages, pop_yr_pct,
                                omega_path_lev, omega_SSfx, curr_year,
                                E, S, output_dir=(OUTPUT_DIR, 'dynamic_partial'))

    return (omega_path_S.T, g_n_SS, omega_SSfx[-S:] /
            omega_SSfx[-S:].sum(), 1 - mort_rates_S, rho_path_lev.T,
            g_n_path, imm_rates_mat.T, omega_S_preTP)

def get_pop_objs_dynamic_full(E, S, T, min_age, max_age, curr_year, demog_type, country='Japan', GraphDiag=True):
    '''
    This function produces the demographics objects to be used in the
    OG-USA model package.

    Args:
        E (int): number of model periods in which agent is not
            economically active, >= 1
        S (int): number of model periods in which agent is economically
            active, >= 3
        T (int): number of periods to be simulated in TPI, > 2*S
        min_age (int): age in years at which agents are born, >= 0
        max_age (int): age in years at which agents die with certainty,
            >= 4
        curr_year (int): current year for which analysis will begin,
            >= 2020
        demog_type (str): demography forecasting technique
        country (str): country that is source of data
        GraphDiag (bool): =True if want graphical output and printed
                diagnostics

    Returns:
        omega_path_S (Numpy array): time path of the population
            distribution from period when demographics stabilize to the steady-state,
            size T+S x S
        g_n_SS (scalar): steady-state population growth rate
        omega_SS (Numpy array): normalized steady-state population
            distribution, length S
        surv_rates (Numpy array): survival rates that correspond to
            each model period of life, length S
        mort_rates (Numpy array): mortality rates that correspond to
            each model period of life, length S
        g_n_path (Numpy array): population growth rates over the time
            path, length T + S
        imm_rates_mat (Numpy array): steady-state immigration rates by
                        age for economically active ages, length S

        (omega_path_S.T, g_n_SS, omega_SSfx[-S:] /
            omega_SSfx[-S:].sum(), 1 - mort_rates_S, mort_rates_S,
            g_n_path, imm_rates_mat.T, omega_S_preTP)

    '''
    # If alternate, forecast all demographics data
    if demog_type == 'dynamic_full_alternate':
        fert_all, mort_all, imm_all = prep_demog_alternate(country)
    else:
        fert_all, mort_all, imm_all = False, False, False
    # Forecast all population data
    pop_data = forecast_pop(country, min_age, max_age, demog_type, curr_year - 1, curr_year + T + S - 1, demog_files=[fert_all, mort_all, imm_all]) # - 1 for Immigration Rates
    # Prepare demographics for stable_year
    stable_year = 2051
    pop_stable_prev_data = pop_data[stable_year - 1]
    pop_stable_prev_rebin = pop_rebin(pop_stable_prev_data, E + S)
    pop_stable_data = pop_data[stable_year]
    pop_stable_rebin = pop_rebin(pop_stable_data, E + S)
    fert_stable = get_fert(E + S, min_age, max_age, stable_year - 1, country, demog_type=demog_type, graph=False, demog_files=[fert_all, mort_all, imm_all])
    mort_stable, infmort_stable = get_mort(E + S, min_age, max_age, stable_year - 1, country, demog_type=demog_type, graph=False, mort_file=mort_all)
    mort_rates_S = mort_stable[-S:]
    imm_rates_orig = util.calc_imm_resid(fert_stable, mort_stable, pop_stable_prev_rebin, pop_stable_rebin)

    ages = np.arange(min_age, max_age + 1)

    OMEGA_orig = np.zeros((E + S, E + S))
    OMEGA_orig[0, :] = fert_stable / 2
    OMEGA_orig[1:, :-1] += np.diag(1 - mort_stable[:-1])
    OMEGA_orig += np.diag(imm_rates_orig)

    # Solve for steady-state population growth rate and steady-state
    # population distribution by age using eigenvalue and eigenvector
    # decomposition
    eigvalues, eigvectors = np.linalg.eig(OMEGA_orig)
    g_n_SS = (eigvalues[np.isreal(eigvalues)].real).max() - 1
    eigvec_raw = eigvectors[:, (eigvalues[np.isreal(eigvalues)].real).argmax()].real
    omega_SS_orig = eigvec_raw / eigvec_raw.sum()

    # Generate time path of nonstationary population
    omega_path_lev = np.zeros((E + S, T + S))
    for per in range(T + S):
        pop_per_data = pop_data[curr_year + per]
        pop_per_rebin = pop_rebin(pop_per_data, E + S)
        omega_path_lev[:, per] = pop_per_rebin.copy()

    # Force the population distribution after 1.5*S periods from start to be the
    # steady-state distribution by adjusting immigration rates, holding
    # constant mortality, fertility, and SS growth rates
    imm_tol = 1e-14
    fixper = int(1.5 * S)
    omega_SSfx = (omega_path_lev[:, fixper] / omega_path_lev[:, fixper].sum())
    imm_objs = (fert_stable, mort_stable, infmort_stable, omega_path_lev[:, fixper], g_n_SS)
    imm_fulloutput = opt.fsolve(immsolve, imm_rates_orig, args=(imm_objs), full_output=True, xtol=imm_tol)
    imm_rates_adj = imm_fulloutput[0]
    imm_diagdict = imm_fulloutput[1]
    omega_path_S = (omega_path_lev[-S:, :] / np.tile(omega_path_lev[-S:, :].sum(axis=0), (S, 1)))
    omega_path_S[:, fixper:] = np.tile(omega_path_S[:, fixper].reshape((S, 1)), (1, T + S - fixper))

    # Population growth rate
    pop_prev_data = pop_data[curr_year - 1]
    pop_prev_rebin = pop_rebin(pop_prev_data, E + S)
    pop_curr_data = pop_data[curr_year]
    pop_curr_rebin = pop_rebin(pop_curr_data, E + S)
    g_n_curr = ((pop_curr_rebin[-S:].sum() - pop_prev_rebin[-S:].sum()) / pop_prev_rebin[-S:].sum())
    omega_S_preTP = pop_curr_rebin[-S:] / pop_curr_rebin[-S:].sum()
    omega_S_preTP = np.expand_dims(omega_S_preTP, 1)
    pop_curr_pct = pop_curr_rebin / np.sum(pop_curr_rebin)
    
    g_n_path = np.zeros(T + S)
    g_n_path[0] = g_n_curr.copy()
    g_n_path[1:] = ((omega_path_lev[-S:, 1:].sum(axis=0) - omega_path_lev[-S:, :-1].sum(axis=0)) / omega_path_lev[-S:, :-1].sum(axis=0))
    g_n_path[fixper + 1:] = g_n_SS

    # Generate time path of immigration rates
    imm_rates_mat = np.zeros((S, T + S))
    for per in range(T + S):
        if per <= fixper:
            pop_prev_per_data = pop_data[curr_year + per - 1]
            pop_prev_per_rebin = pop_rebin(pop_prev_per_data, E + S)
            pop_per_data = pop_data[curr_year + per]
            pop_per_rebin = pop_rebin(pop_per_data, E + S)
            fert_rates_per = get_fert(E + S, min_age, max_age, curr_year + per - 1, country, demog_type=demog_type, graph=False, demog_files=[fert_all, mort_all, imm_all])
            mort_rates_per, infmort_rate = get_mort(E + S, min_age, max_age, curr_year + per - 1, country, demog_type=demog_type, graph=False, mort_file=mort_all)
            imm_rates_per = util.calc_imm_resid(fert_rates_per, mort_rates_per, pop_prev_per_rebin, pop_per_rebin)
            imm_rates_mat[:, per] = imm_rates_per[E:].copy()
        else:
            imm_rates_mat[:, per] = imm_rates_adj[E:].copy()

    # Generate time path of mortality rates
    rho_path_lev = np.zeros((S, T + S + S))
    for per in range(T + S + S):
        mort_per, inf_mort_per = get_mort(E + S, min_age, max_age, curr_year + per, country, demog_type=demog_type, graph=False, mort_file=mort_all)
        rho_path_lev[:, per] = mort_per[E:].copy()

    if GraphDiag:
        # Check whether original SS population distribution is close to
        # the period-T population distribution
        omegaSSmaxdif = np.absolute(omega_SS_orig - (omega_path_lev[:, T] / omega_path_lev[:, T].sum())).max()
        if omegaSSmaxdif > 0.0003:
            print('POP. WARNING: Max. abs. dist. between original SS ' +
                  "pop. dist'n and period-T pop. dist'n is greater than" +
                  ' 0.0003. It is ' + str(omegaSSmaxdif) + '.')
        else:
            print('POP. SUCCESS: orig. SS pop. dist is very close to ' +
                  "period-T pop. dist'n. The maximum absolute " +
                  'difference is ' + str(omegaSSmaxdif) + '.')

        # Plot the adjusted steady-state population distribution versus
        # the original population distribution. The difference should be
        # small
        omegaSSvTmaxdiff = np.absolute(omega_SS_orig - omega_SSfx).max()
        if omegaSSvTmaxdiff > 0.0003:
            print('POP. WARNING: The maximimum absolute difference ' +
                  'between any two corresponding points in the ' +
                  'original and adjusted steady-state population ' +
                  'distributions is' + str(omegaSSvTmaxdiff) + ', ' +
                  'which is greater than 0.0003.')
        else:
            print('POP. SUCCESS: The maximum absolute difference ' +
                  'between any two corresponding points in the ' +
                  'original and adjusted steady-state population ' +
                  'distributions is ' + str(omegaSSvTmaxdiff))

        # Print whether or not the adjusted immigration rates solved the
        # zero condition
        immtol_solved = np.absolute(imm_diagdict['fvec'].max()) < imm_tol
        if immtol_solved:
            print('POP. SUCCESS: Adjusted immigration rates solved ' +
                  'with maximum absolute error of ' +
                  str(np.absolute(imm_diagdict['fvec'].max())) +
                  ', which is less than the tolerance of ' +
                  str(imm_tol))
        else:
            print('POP. WARNING: Adjusted immigration rates did not ' +
                  'solve. Maximum absolute error of ' +
                  str(np.absolute(imm_diagdict['fvec'].max())) +
                  ' is greater than the tolerance of ' + str(imm_tol))

        # Test whether the steady-state growth rates implied by the
        # adjusted OMEGA matrix equals the steady-state growth rate of
        # the original OMEGA matrix
        OMEGA2 = np.zeros((E + S, E + S))
        OMEGA2[0, :] = fert_stable / 2
        OMEGA2[1:, :-1] += np.diag(1 - mort_stable[:-1])
        OMEGA2 += np.diag(imm_rates_adj)
        eigvalues2, eigvectors2 = np.linalg.eig(OMEGA2)
        g_n_SS_adj = (eigvalues[np.isreal(eigvalues2)].real).max() - 1
        if np.max(np.absolute(g_n_SS_adj - g_n_SS)) > 10 ** (-8):
            print('FAILURE: The steady-state population growth rate' +
                  ' from adjusted OMEGA is different (diff is ' +
                  str(g_n_SS_adj - g_n_SS) + ') than the steady-' +
                  'state population growth rate from the original' +
                  ' OMEGA.')
        elif np.max(np.absolute(g_n_SS_adj - g_n_SS)) <= 10 ** (-8):
            print('SUCCESS: The steady-state population growth rate' +
                  ' from adjusted OMEGA is close to (diff is ' +
                  str(g_n_SS_adj - g_n_SS) + ') the steady-' +
                  'state population growth rate from the original' +
                  ' OMEGA.')

        # Do another test of the adjusted immigration rates. Create the
        # new OMEGA matrix implied by the new immigration rates. Plug in
        # the adjusted steady-state population distribution. Hit is with
        # the new OMEGA transition matrix and it should return the new
        # steady-state population distribution
        omega_new = np.dot(OMEGA2, omega_SSfx)
        omega_errs = np.absolute(omega_new - omega_SSfx)
        print('The maximum absolute difference between the adjusted ' +
              'steady-state population distribution and the ' +
              'distribution generated by hitting the adjusted OMEGA ' +
              'transition matrix is ' + str(omega_errs.max()))

        # Plot the original immigration rates versus the adjusted
        # immigration rates
        immratesmaxdiff = \
            np.absolute(imm_rates_orig - imm_rates_adj).max()
        print('The maximum absolute distance between any two points ' +
              'of the original immigration rates and adjusted ' +
              'immigration rates is ' + str(immratesmaxdiff))

        # plots
        pp.plot_omega_fixed(ages, omega_SS_orig, omega_SSfx, E,
                            S, output_dir=(OUTPUT_DIR, demog_type))
        pp.plot_imm_fixed(ages, imm_rates_orig, imm_rates_adj, E,
                          S, output_dir=(OUTPUT_DIR, demog_type))
        pp.plot_population_path(ages, pop_curr_pct,
                                omega_path_lev, omega_SSfx, curr_year,
                                E, S, output_dir=(OUTPUT_DIR, demog_type))

    # return omega_path_S, g_n_SS, omega_SSfx, survival rates,
    # mort_rates_S, and g_n_path
    print('------------------------------')
    print('E:', E)
    print('S:', S)
    print('T:', T)
    print('Omega_path_S.shape:', omega_path_S.shape) # Should be 80x240 (Sx3*S)
    print(omega_path_S)
    print('g_n_SS:', g_n_SS) # Should be a scalar
    print('omega_SSfx.shape:', omega_SSfx.shape) # Should be size 100 (E + S)
    print('mort_rates_S.shape:', mort_rates_S.shape) # Should be size 80 (S)
    print('g_n_path.shape:', g_n_path.shape) # Should be size 240 (S*3)
    print('imm_rates_mat.shape:', imm_rates_mat.shape) # Should be 80x240 (Sx3*S)
    print('omega_S_preTP.shape:', omega_S_preTP.shape) # Should be 80x1 (Sx1)

    print('rho_path_lev.shape:', rho_path_lev.shape) # Should be 80x320

    return (omega_path_S.T, g_n_SS, omega_SSfx[-S:] /
            omega_SSfx[-S:].sum(), 1 - mort_rates_S, rho_path_lev.T,
            g_n_path, imm_rates_mat.T, omega_S_preTP)
