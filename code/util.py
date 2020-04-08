'''
Util file for Demographics thesis project
'''
import numpy as np
import pandas as pd
import scipy
import scipy.optimize as opt
import math
from math import e
import matplotlib.pyplot as plt
import matplotlib.cm as mplcm
import matplotlib.colors as colors

import warnings

#########################################
## Fertility/Mortality/Population data ##
#########################################
def select_cohort(data):
    '''
    Adjust data to be at the cohort-level
    '''
    data_adj = data.copy()
    data_adj['Cohort'] = data_adj['Year'] - data_adj['Age']
    data_adj.drop('Year', axis=1, inplace=True)
    data_adj = data_adj.pivot(index='Age', columns='Cohort', values='Values')
    return data_adj

def select_year(data):
    '''
    Adjust data to be at the year-level
    '''
    data_adj = data.copy()
    data_adj = data_adj.pivot(index='Age', columns='Year', values='Values')
    return data_adj

def select_fert_data(fert, set_zeroes=False):
    new_fert = fert[fert['AgeDef'] == 'ARDY']
    new_fert = new_fert[new_fert['Collection'] == 'HFD']
    new_fert = new_fert[(new_fert['RefCode'] == 'JPN_11')]
    new_fert.drop(['AgeDef', 'Collection', 'RefCode'], axis=1, inplace=True)
    new_fert.columns = ['Year', 'Age', 'Values']
    if set_zeroes:
        new_fert['Values'][new_fert['Age'] == 14] = 0
        new_fert['Values'][new_fert['Age'] == 15] = 0
        new_fert['Values'][new_fert['Age'] == 49] = 0
        new_fert['Values'][new_fert['Age'] == 50] = 0
    return new_fert.astype(float)

def get_fert_data(fert_dir):
    fert_data = pd.read_csv(fert_dir, sep=r',\s*',\
        usecols=['Year1', 'Age', 'ASFR', 'AgeDef',\
                    'Collection', 'RefCode'], engine='python')
    fert_data = select_fert_data(fert_data)
    fert_data = select_year(fert_data)
    return fert_data

def select_mort_pop_data(data):
    data_adj = data[data["Age"] != "110+"].astype(int)
    #data_adj = data_adj[data_adj["Age"] <= 100]
    #data_adj = data_adj[data_adj["Year"] == 2014]
    return data_adj

def get_mort_pop_data(mort_dir, pop_dir):
    #infmort_rate = 1.9 / 1000  # https://knoema.com/atlas/Japan/Infant-mortality-rate
    mort_data = pd.read_csv(mort_dir, sep=r'\s+', usecols=['Year', 'Age', 'Total'])
    mort_data = select_mort_pop_data(mort_data)
    pop_data = pd.read_csv(pop_dir, sep=r'\s+', usecols=['Year', 'Age', 'Total'])
    pop_data = select_mort_pop_data(pop_data)

    mort_data['Values'] = mort_data['Total'] / pop_data['Total']
    mort_data = mort_data.drop('Total', axis=1)
    mort_data = select_year(mort_data)
    mort_data = mort_data[mort_data.index < 100] # only concerned with ages 0-99

    #mort_rates[-1] = 1  # Mortality rate in last period is set to 1

    pop_data['Values'] = pop_data['Total']
    pop_data = pop_data.drop('Total', axis=1)
    pop_data = select_year(pop_data)
    pop_data = pop_data[pop_data.index < 100]

    return mort_data, pop_data

def predict_population(fert_rate, mort_rate, imm_rate, pop_data):
    '''
    Predict population as pop_{s+1,t+1} = pop_{s,t}(1 - mort_{t-1})) + pop_{s+1,t}(imm_{t-1}),
    and setting pop_{0,t+1} = pop_t * fert_t
    '''
    # First, calculate births
    if len(fert_rate) == 100:
        births = fert_rate * pop_data / 2
    else:
        # Births from 14-50, but age start at 0
        births = fert_rate * pop_data[15:52] / 2
    births = np.sum(births)
    # Second, calculate survivors
    survivors = (1 - mort_rate) * pop_data
    survivors = np.roll(survivors, 1)
    # Third, correct births
    survivors[0] = births
    # Third, calculate immigration
    imm = imm_rate * pop_data
    # Fourth, calculate predicted population
    pred_pop = survivors + imm

    return pred_pop

def calc_imm_resid(fert_t_minus_1, mort_t_minus_1, pop_t_minus_1, pop_t):
    '''
    Calculate immigration rate in year t
    as (pop_t - pop_{t-1}(1 - mort_{t-1})) / (pop_{t-1}),
    and setting pop_t_0 = pop_{t-1} * fert_{t-1}
    '''
    # First, calculate births
    if len(fert_t_minus_1) == 100:
        births = fert_t_minus_1 * pop_t_minus_1 / 2
    else:
        # Births from 14-50, but age start at 0
        births = fert_t_minus_1 * pop_t_minus_1[15:52] / 2
    births = np.sum(births)
    # Second, calculate deaths
    deaths = mort_t_minus_1 * pop_t_minus_1
    deaths = np.roll(deaths, 1)
    # Third, calculate predicted population
    pred_pop = np.roll(pop_t_minus_1, 1) - deaths
    pred_pop[0] = births
    # Fourth, calculate immigration rate
    imm = (pop_t - pred_pop) / (pop_t_minus_1)
    return imm

def calc_imm_resids(fert_data, mort_data, pop_data):
    '''
    Calculate immigration rates over all data
    '''
    imm = pd.DataFrame()
    for year in range(1948, 2016):
        fert_t_minus_1 = fert_data[year - 1]
        mort_t_minus_1 = mort_data[year - 1]
        pop_t_minus_1 = pop_data[year - 1]
        pop_t = pop_data[year]
        imm[year] = calc_imm_resid(fert_t_minus_1, mort_t_minus_1, pop_t_minus_1, pop_t)
    return imm

def array_add(array_list):
    #Array non na keeps track of how many non-na values
    #there are for each index, so you divide correctly
    #at the end
    array_non_na = array_list[0].copy()
    array_non_na[array_non_na != 0] = 0
    if len(array_list) == 1:
        return array_list[0]
    array_sum = array_list[0].copy()
    array_sum[array_sum > 0] = 0

    for i in range(len(array_list)):
        #Figure out which ages in the cohort are na
        array_na = array_list[i].copy()
        array_na[array_na.isna()] = 0
        array_na[array_na != 0] = 1
        array_non_na += array_na

        array_sum += array_list[i].fillna(0)
    #Divide each age by number of non-na years
    return array_sum / array_non_na

def rolling_avg_year(data, year, roll):
    '''
    Smooth data by taking it as average over
    'year' range (take 'year' years prior and
    'year' years post, then average)
    '''
    years = []
    for yr in range(year - roll, year + roll + 1):
        try:
            years.append(data[yr])
        except:
            pass
    avg = array_add(years)
    return avg.dropna()

##########################
## PDFs/Log likelihoods ##
##########################
def gamma_fun_pdf(xvals, alpha, beta):
    '''
    pdf for gamma function
    '''
    # Ignore warning for this line - error values become 0 anyway
    pdf_vals = ( xvals ** (alpha - 1) * e ** ( - xvals / beta ) )/\
                   ( beta ** alpha * math.gamma(alpha) )
    return pdf_vals

def gen_gamma_fun_pdf(xvals, alpha, beta, m):
    '''
    pdf for generalized gamma function
    '''
    pdf_vals = (m * e ** ( - ( xvals / beta ) ** m ))/\
            (xvals * math.gamma(alpha / m)) *\
            (xvals / beta) ** alpha
    if np.isnan(pdf_vals).any():
        print('gen_gamma_fun_pdf has NaN values')
        pdf_vals = np.nan_to_num(pdf_vals) # replace NaN with 0
    return pdf_vals

def gen_gamma_fun_log(xvals, alpha, beta, m):
    '''
    log sum for generalized gamma function
    '''
    log_vals = np.log(m) + (alpha - 1) * np.log(xvals) -\
            (xvals / beta) ** m - alpha * np.log(beta) -\
            np.log(math.gamma(alpha / m))
    return log_vals.sum()

def gb2_beta(p, q):
    '''
    beta function for use in generalized beta^2 distribution
    '''
    betainc = scipy.special.betainc(p, q, 1)
    return betainc * scipy.special.beta(p, q)

def gen_beta2_fun_pdf(xvals, a, b, p, q):
    '''
    pdf for generalized beta^2 function
    '''
    pdf_vals = (xvals / b) ** (a * p) * a / (xvals * gb2_beta(p, q) *\
                (1 + (xvals / b) ** a) ** (p + q) )
    return pdf_vals

def gen_beta2_fun_log(xvals, a, b, p, q):
    '''
    log sum for generalized beta^2 function
    '''
    log_vals = (a * p - 1) * np.log(xvals) + np.log(a) -\
                a * p * np.log(b) - np.log(gb2_beta(p, q))\
                - (p + q) * np.log(1 + (xvals / b) ** a)
    return log_vals.sum()

def logistic_function(xvals, L, k, x):
    '''
    Logistic function
    '''
    log_vals = L / (1 + e ** (-k * (xvals - x) ) )
    if np.isnan(log_vals).any():
        print('log_pdf has NaN values')
        log_vals = np.nan_to_num(log_vals) # replace NaN with 0
    return log_vals

def polynomial_fn(xvals, a, b, c, d, e):
    '''
    Polynomial of the form:
        y = a * (e * x - b) ** (1 / c) + d
    '''
    poly_vals = a * ( (e * xvals - b) ** (1 / c) ) + d
    return poly_vals

def exp_fn(xvals, a, b, c):
    '''
    Exponential of the form:
        y = a - e ** ( - (b * x - c) )
    '''
    exp_vals = a - e ** ( - (b * xvals - c) )
    return exp_vals

def exponential_fn(xvals, a, b, c):
    '''
    Exponential of the form:
        y = a * e ** (b * x - c)
    '''
    exp_vals = a * e ** (b * xvals - c)
    return exp_vals

####################
## Crit functions ##
####################
def crit_gamma(params, *args):
    alpha, beta = params
    xvals, ages, pop = args
    guess = gamma_fun_pdf(ages, alpha, beta)
    if guess.sum() != 0:
        if isinstance(pop, bool):
            guess_compare = guess * ( xvals.sum() / guess.sum() ) #Restandardize data
        else:
            if np.sum(guess * pop) != 0:
                guess_compare = guess * ( np.sum(xvals * pop) / np.sum(guess * pop) )
            elif np.sum(guess * pop) == 0:
                guess_compare = guess
    else:
        guess_compare = guess
    diff = np.sum((xvals - guess_compare) ** 2)
    return diff

def crit_gen_gamma(params, *args):
    alpha, beta, m = params
    xvals, ages, pop = args
    guess = gen_gamma_fun_pdf(ages, alpha, beta, m)
    if guess.sum() != 0:
        if isinstance(pop, bool):
            guess_compare = guess * ( xvals.sum() / guess.sum() ) #Restandardize data
        else:
            guess_compare = guess * ( np.sum(xvals * pop) / np.sum(guess * pop) )
    else:
        guess_compare = 0
    diff = np.sum((xvals - guess_compare) ** 2)
    return diff

def crit_gen_beta2(params, *args):
    a, b, p, q = params
    xvals, ages, pop = args
    guess = gen_beta2_fun_pdf(ages, a, b, p, q)
    if guess.sum() != 0:
        if isinstance(pop, bool):
            guess_compare = guess * ( xvals.sum() / guess.sum() ) #Restandardize data
        else:
            guess_compare = guess * ( np.sum(xvals * pop) / np.sum(guess * pop) )
    else:
        guess_compare = 0
    diff = np.sum((xvals - guess_compare) ** 2)
    return diff

def crit_logistic(params, *args):
    L, k, x = params
    data, years = args
    guess = logistic_function(years, L, k, x)
    diff = np.sum((data - guess) ** 2)
    return diff

def crit_logistic_flip(params, *args):
    L, k, x = params
    data, years = args
    guess = logistic_function( - (years - x) + x, L, k, x)
    diff = np.sum((data - guess) ** 2)
    return diff

def crit_log(params, *args):
    a, b, x = params
    data, years = args
    guess = a * np.log(years - x) + b
    diff = np.sum((data - guess) ** 2)
    return diff

def crit_polyvals(params, *args):
    a_0, b_0, c_0, d_0, e_0 = params
    xvals, years, datatype, param = args
    guess = polynomial_fn(years, a_0, b_0, c_0, d_0, e_0)

    # Penalize guesses < 0
    guess[guess < 0] = -5

    if datatype == 'mortality' and param == 'a':
        # Weight most recently data more heavily
        diff = np.sum( (xvals[:-10] - guess[:-10]) ** 2) \
                + np.sum( (1 + (xvals[-10:] - guess[-10:]) ** 2) ** 10)
    elif datatype == 'mortality' and param == 'Infant Mortality Rate':
        # Weight most recent data more heavily
        impt_ages = 15
        diff = np.sum( (xvals[:-impt_ages] - guess[:-impt_ages]) ** 2) ** 0.1 \
                + np.sum( (1 + (xvals[-impt_ages:] - guess[-impt_ages:]) ** 2) ** 40)
        # Add in penalty if predictions become negative
        future_val = polynomial_fn(2100, a_0, b_0, c_0, d_0, e_0)
        if future_val < 0:
            diff += (1 + abs(future_val) ) ** 100
    
    else:
        diff = np.sum((xvals - guess) ** 2)
    return diff

def crit_exp(params, *args):
    a_0, b_0, c_0 = params
    xvals, years, pop = args
    guess = exp_fn(years, a_0, b_0, c_0)
    if isinstance(pop, bool):
        xvals_compare = xvals
        guess_compare = guess
    else:
        # Weight by population
        xvals_compare = xvals * pop
        guess_compare = (guess * pop)
    diff = np.sum( (xvals_compare - guess_compare) ** 2)

    return diff

def crit_exponential(params, *args):
    a_0, b_0, c_0 = params
    xvals, ages, pop = args
    guess = exponential_fn(ages, a_0, b_0, c_0)
    if isinstance(pop, bool):
        xvals_compare = xvals
        guess_compare = guess
    else:
        # Weight by population
        xvals_compare = xvals * pop
        guess_compare = (guess * pop)
    diff = np.sum( (xvals_compare - guess_compare) ** 2)

    return diff

###################
## MLE functions ##
###################
def gamma_est(data, year, smooth, datatype, print_params=False, pop=False):
    '''
    Estimate parameters for gamma distribution
    '''
    count = data.shape[0]
    mean = np.mean(data.index)
    var = np.var(data.index)

    if datatype == 'fertility':
        ages = np.linspace(14, 14 + count - 1, count)
    elif datatype in ('mortality', 'mortality_beta', 'population'):
        ages = np.linspace(1e-2, 1e-2 + count - 1, count)

    beta_0 = var/mean
    alpha_0 = mean/beta_0
    params_init = np.array([alpha_0, beta_0])

    # begin gamma estimation
    results_cstr = opt.minimize(crit_gamma, params_init,\
                    args=(np.array(data), ages, pop), method="L-BFGS-B",\
                    bounds=((1e-10, None), (1e-10, None)))
    alpha_MLE_b, beta_MLE_b = results_cstr.x

    if print_params:
        print("alpha_MLE_b=", alpha_MLE_b, " beta_MLE_b=", beta_MLE_b)

    gamma = gamma_fun_pdf(ages, alpha_MLE_b, beta_MLE_b)
    
    if isinstance(pop, bool):
        scale = np.sum(data) / np.sum(gamma)
    else:
        scale = np.sum(data * pop) / np.sum(gamma * pop)

    return alpha_MLE_b, beta_MLE_b, scale

def gen_gamma_est(data, year, smooth, datatype, print_params=False, pop=False):
    '''
    Estimate parameters for generalized gamma distribution
    '''
    count = data.shape[0]

    if datatype == 'fertility':
        ages = np.linspace(14, 14 + count - 1, count)
    elif datatype in ('mortality', 'mortality_beta', 'population'):
        ages = np.linspace(1e-2, 1e-2 + count - 1, count)

    # start with gamma distribution to get starting
    # values for generalized gamma distribution
    alpha_MLE_b, beta_MLE_b, scale = gamma_est(data, year, smooth, datatype,\
                                                print_params=print_params, pop=pop)

    # begin generalized gamma estimation
    alpha_0 = alpha_MLE_b
    beta_0 = beta_MLE_b
    m_0 = 1
    params_init = np.array([alpha_0, beta_0, m_0])

    try:
        results_cstr = opt.minimize(crit_gen_gamma, params_init,\
                                    args=(data, ages, pop), method='L-BFGS-B',\
                                    bounds=((1e-10, None), (1e-10, None),\
                                    (1e-10, None)))
        alpha_MLE_c, beta_MLE_c, m_MLE_c = results_cstr.x
    except:
        # if the optimization fails, assign values
        # from first step of optimizations, and assume
        # m=1
        print('Generalized gamma failed, reverting to Gamma parameter estimates')
        alpha_MLE_c = alpha_MLE_b
        beta_MLE_c = beta_MLE_b
        try:
            m_MLE_c = m_MLE_c
        except:
            m_MLE_c = m_0

    if print_params:
        print('alpha_MLE_c=', alpha_MLE_c, ' beta_MLE_c=', beta_MLE_c, ' m_MLE_c=', m_MLE_c)

    gen_gamma = gen_gamma_fun_pdf(ages, alpha_MLE_c, beta_MLE_c, m_MLE_c)
    
    if isinstance(pop, bool):
        scale = np.sum(data) / np.sum(gen_gamma)
    else:
        scale = np.sum(data * pop) / np.sum(gen_gamma * pop)

    return alpha_MLE_c, beta_MLE_c, m_MLE_c, scale

def gen_beta2_est(data, year, smooth, datatype, print_params=False, pop=False, prev_estimates=False):
    '''
    Estimate parameters for generalized beta^2 distribution
    '''
    count = data.shape[0]

    if datatype == 'fertility':
        ages = np.linspace(14, 14 + count - 1, count)
    elif datatype in ('mortality', 'mortality_beta', 'population'):
        ages = np.linspace(1e-2, 1e-2 + count - 1, count)

    if prev_estimates:
        a_0, b_0, p_0, q_0, scale = prev_estimates

        params_init = np.array([a_0, b_0, p_0, q_0])

        try:
            results_cstr = opt.minimize(crit_gen_beta2, params_init,\
                                        args=(data, ages, pop), method='L-BFGS-B',\
                                        bounds=((1e-10, None), (1e-10, None),\
                                        (1e-10, None), (1e-10, None)))
            a_MLE_d, b_MLE_d, p_MLE_d, q_MLE_d = results_cstr.x
        
        except:
            # If fails with previous estimates, try again using guesses
            return gen_beta2_est(data, year, smooth, datatype, print_params, pop, prev_estimates=False)

    else:
        # start with generalized gamma distribution to get starting
        # values for generalized beta^2 distribution
        alpha_MLE_c, beta_MLE_c, m_MLE_c, scale = gen_gamma_est(data, year, smooth, datatype,\
                                                print_params=print_params, pop=pop)

        q_guesses = [50] #np.linspace(1, 100, 100)#[50, 738]
        param_guesses = []
        crit_vals = []
        for i, q_0 in enumerate(q_guesses):
            #print('q guess:', i)
            # begin generalized beta^2 estimation
            #q_0 = 50
            a_0 = m_MLE_c
            b_0 = q_0 ** (1 / a_0) * beta_MLE_c
            p_0 = alpha_MLE_c / m_MLE_c
            params_init = np.array([a_0, b_0, p_0, q_0])

            try:
                results_cstr = opt.minimize(crit_gen_beta2, params_init,\
                                            args=(data, ages, pop), method='L-BFGS-B',\
                                            bounds=((1e-10, None), (1e-10, None),\
                                            (1e-10, None), (1e-10, None)))
                estimates = results_cstr.x
                crit_val = results_cstr.fun
                a_MLE_d, b_MLE_d, p_MLE_d, q_MLE_d = estimates
                param_guesses.append(estimates)
                crit_vals.append(crit_val)
            except:
                # if the optimization fails, assign values
                # from first step of optimizations, and assume
                # q=10000
                print('Generalized beta^2 failed, reverting to Generalized Gamma parameter estimates')
                q_MLE_d = 10000
                a_MLE_d = m_MLE_c
                b_MLE_d = q_MLE_d ** (1 / a_MLE_d) * beta_MLE_c
                p_MLE_d = alpha_MLE_c / m_MLE_c

                estimates = a_MLE_d, b_MLE_d, p_MLE_d, q_MLE_d
                crit_val = crit_gen_beta2(estimates, (data, ages, pop))
                param_guesses.append(estimates)
                crit_vals.append(crit_val)
        crit_array = np.array(crit_vals)
        best_index = crit_vals.index(np.min(crit_array[np.nonzero(crit_array)]))
        best_params = param_guesses[best_index] # Take care not to count crit_val = 0
        a_MLE_d, b_MLE_d, p_MLE_d, q_MLE_d = best_params

    if print_params:
        print('a_MLE_d=', a_MLE_d, ' b_MLE_d=', b_MLE_d, ' p_MLE_d=', p_MLE_d, ' q_MLE_d=', q_MLE_d)

    gen_beta2 = gen_beta2_fun_pdf(ages, a_MLE_d, b_MLE_d, p_MLE_d, q_MLE_d)

    if isinstance(pop, bool):
        scale = np.sum(data) / np.sum(gen_beta2)
    else:
        scale = np.sum(data * pop) / np.sum(gen_beta2 * pop)

    plt.plot(ages, gen_beta2 * scale,\
            linewidth=2, label='Generalized Beta 2 Estimate', color='r')
    plt.plot(ages, data, label=datatype.capitalize() + ' ' + str(year))
    plt.xlabel(r'Age $s$')
    plt.ylabel(r'Fertility Rate $f_{s,t}$')
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    plt.tight_layout()
    plt.legend()
    plt.savefig('graphs/' + datatype + '/smooth_' + str(smooth) + '/' + str(year))
    plt.close()

    return a_MLE_d, b_MLE_d, p_MLE_d, q_MLE_d, scale

def logistic_est(data, L_0, k_0, x_0, years, smooth, datatype, param='', flip=False, print_params=False, show_plot=False):
    '''
    Estimate parameters for logistic function
    Requires flip=True if data follows mirror of logistic function
    '''
    params_init = np.array([L_0, k_0, x_0])

    if flip:
        results_cstr = opt.minimize(crit_logistic_flip, params_init,\
                args=(np.array(data - np.min(data)), years), method="L-BFGS-B")
    else:
        results_cstr = opt.minimize(crit_logistic, params_init,\
                        args=(np.array(data - np.min(data)), years), method="L-BFGS-B")
    L_MLE, k_MLE, x_MLE = results_cstr.x

    if print_params:
        print('L:', L_MLE, 'k:', k_MLE, 'x:', x_MLE)

    plots = []
    plots.append(plt.plot(years, data))
    if flip:
        years_adj = - (years - x_MLE) + x_MLE
        plots.append(plt.plot(years, logistic_function(years_adj, L_MLE, k_MLE, x_MLE) + np.min(data)))
    else:
        plots.append(plt.plot(years, logistic_function(years, L_MLE, k_MLE, x_MLE) + np.min(data)))
    if param != 'Scale':
        lab = r'$' + param + r'$'
    else:
        lab = param
    plots[0][0].set_label(lab)
    plots[1][0].set_label(lab + ' Estimate')
    plt.xlabel(r'Year $t$')
    plt.ylabel(lab)
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    plt.tight_layout()
    plt.legend()
    plt.savefig('graphs/' + datatype + '/smooth_' + str(smooth) + '/_' + param.lower() + '_predicted')
    if show_plot:
        plt.show()
    plt.close()

    return L_MLE, k_MLE, x_MLE

def log_est(data, a_0, b_0, x_0, years, smooth, datatype, param='', print_params=False, show_plot=False):
    '''
    Estimate parameters for log
    Requires flip=True if data follows mirror of log
    '''
    params_init = np.array([a_0, b_0, x_0])

    results_cstr = opt.minimize(crit_log, params_init,\
                    args=(np.array(data - np.min(data)), years), method="L-BFGS-B",\
                    bounds=((None, None), (1e-10, None), (None, min(years) - 1e-10)))
    a_MLE, b_MLE, x_MLE = results_cstr.x

    if print_params:
        print('a:', a_MLE, 'b:', b_MLE, 'x:', x_MLE)

    plots = []
    plots.append(plt.plot(years, data))
    plots.append(plt.plot(years, a_MLE * np.log( b_MLE * (years - x_MLE) ) + np.min(data)))

    plots[0][0].set_label(param)
    plots[1][0].set_label(param + ' estimate')
    plt.legend()
    plt.savefig('graphs/' + datatype + '/smooth_' + str(smooth) + '/_' + param.lower() + '_predicted')
    if show_plot:
        plt.show()
    plt.close()

    return a_MLE, b_MLE, x_MLE

def poly_est(data, a_0, b_0, c_0, d_0, e_0, years, smooth, datatype, param='', print_params=False, show_plot=False, pop=False):
    '''
    Estimate parameters for polynomial function
    '''
    params_init = np.array([a_0, b_0, c_0, d_0, e_0])

    results_cstr = opt.minimize(crit_polyvals, params_init,\
                    args=(np.array(data), years, datatype, param), method="L-BFGS-B")
    a_MLE, b_MLE, c_MLE, d_MLE, e_MLE = results_cstr.x

    if print_params:
        print('a_MLE=', a_MLE, 'b_MLE=', b_MLE, 'c_MLE=', c_MLE, 'd_MLE=', d_MLE, 'e_MLE=', e_MLE)
    
    gen_poly = polynomial_fn(years, a_MLE, b_MLE, c_MLE, d_MLE, e_MLE)

    plots = []
    plots.append(plt.plot(years, data))
    plots.append(plt.plot(years, gen_poly))

    plots[0][0].set_label(param)
    plots[1][0].set_label(param + ' Estimate')
    plt.xlabel(r'Year $t$')
    plt.ylabel(param)
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    plt.tight_layout()
    plt.legend()
    plt.savefig('graphs/' + datatype + '/smooth_' + str(smooth) + '/_' + param.lower().replace(' ', '_') + '_predicted')
    if show_plot:
        plt.show()
    plt.close()

    return a_MLE, b_MLE, c_MLE, d_MLE, e_MLE

def exp_est(data, year, a_0, b_0, c_0, ages, smooth, datatype, param='', print_params=False, show_plot=False, pop=False):
    '''
    Estimate parameters for polynomial function
    '''
    params_init = np.array([a_0, b_0, c_0])

    results_cstr = opt.minimize(crit_exp, params_init,\
                    args=(np.array(data), ages, pop), method="L-BFGS-B")
    a_MLE, b_MLE, c_MLE = results_cstr.x

    if print_params:
        print('a_MLE=', a_MLE, 'b_MLE=', b_MLE, 'c_MLE=', c_MLE)

    exp_vals = exp_fn(ages, a_MLE, b_MLE, c_MLE)

    plots = []
    plots.append(plt.plot(ages, data))
    plots.append(plt.plot(ages, exp_vals))

    plots[0][0].set_label(param)
    plots[1][0].set_label(param + ' estimate')
    plt.legend()
    if param == '':
        plt.savefig('graphs/' + datatype + '/smooth_' + str(smooth) + '/' + str(year))
    else:
        plt.savefig('graphs/' + datatype + '/smooth_' + str(smooth) + '/_' + param + '_predicted')
    if show_plot:
        plt.show()
    plt.close()

    return a_MLE, b_MLE, c_MLE

def exponential_est(data, year, a_0, b_0, c_0, ages, smooth, datatype, param='', print_params=False, show_plot=False, pop=False):
    '''
    Estimate parameters for polynomial function
    '''
    params_init = np.array([a_0, b_0, c_0])

    results_cstr = opt.minimize(crit_exponential, params_init,\
                    args=(np.array(data), ages, pop), method="L-BFGS-B")
    a_MLE, b_MLE, c_MLE = results_cstr.x

    if print_params:
        print('a_MLE=', a_MLE, 'b_MLE=', b_MLE, 'c_MLE=', c_MLE)

    exp_vals = exponential_fn(ages, a_MLE, b_MLE, c_MLE)

    plots = []
    plots.append(plt.plot(ages, data))
    plots.append(plt.plot(ages, exp_vals))

    plots[0][0].set_label(param)
    plots[1][0].set_label(param + ' estimate')
    plt.legend()
    if param == '':
        plt.savefig('graphs/' + datatype + '/smooth_' + str(smooth) + '/' + str(year))
    else:
        plt.savefig('graphs/' + datatype + '/smooth_' + str(smooth) + '/_' + param + '_predicted')
    if show_plot:
        plt.show()
    plt.close()

    return a_MLE, b_MLE, c_MLE

################################################
## Fertility/Mortality/Immigration parameters ##
################################################

def forecast_fertility(a_params, b_params, p_params, q_params, scale_params, year, ages):
    '''
    Using fertility parameter estimates, forecast
    fertility rates for a set of ages in a particular year
    '''
    L_MLE_a, k_MLE_a, x_MLE_a, min_a = a_params
    L_MLE_b, k_MLE_b, x_MLE_b, min_b = b_params
    L_MLE_p, k_MLE_p, x_MLE_p, min_p = p_params
    L_MLE_q, k_MLE_q, x_MLE_q, min_q = q_params
    L_MLE_scale, k_MLE_scale, x_MLE_scale, min_scale = scale_params

    year_adj_a = - (year - x_MLE_a) + x_MLE_a
    a = logistic_function(year_adj_a, L_MLE_a, k_MLE_a, x_MLE_a) + min_a

    b = logistic_function(year, L_MLE_b, k_MLE_b, x_MLE_b) + min_b

    year_adj_p = - (year - x_MLE_p) + x_MLE_p
    p = logistic_function(year_adj_p, L_MLE_p, k_MLE_p, x_MLE_p) + min_p

    q = logistic_function(year, L_MLE_q, k_MLE_q, x_MLE_q) + min_q

    year_adj_scale = - (year - x_MLE_scale) + x_MLE_scale
    scale = logistic_function(year_adj_scale, L_MLE_scale, k_MLE_scale, x_MLE_scale) + min_scale

    unscaled_forecast = gen_beta2_fun_pdf(ages, a, b, p, q)

    scaled_forecast = unscaled_forecast * scale

    return scaled_forecast

def forecast_mortality(non_infant_params, infant_params, year, ages):
    '''
    Using mortality parameter estimates, forecast
    mortality rates for a set of ages in a particular year
    '''
    a_params, b_params, p_params, q_params, scale_params = non_infant_params
    L_MLE_a, k_MLE_a, x_MLE_a, min_a = a_params
    L_MLE_b, k_MLE_b, x_MLE_b, min_b = b_params
    L_MLE_p, k_MLE_p, x_MLE_p, min_p = p_params
    L_MLE_q, k_MLE_q, x_MLE_q, min_q = q_params

    L_MLE_scale, k_MLE_scale, x_MLE_scale, min_scale = scale_params

    a = logistic_function(year, L_MLE_a, k_MLE_a, x_MLE_a) + min_a

    b = logistic_function(year, L_MLE_b, k_MLE_b, x_MLE_b) + min_b

    p = logistic_function(year, L_MLE_p, k_MLE_p, x_MLE_p) + min_p

    year_adj_q = - (year - x_MLE_q) + x_MLE_q
    q = logistic_function(year_adj_q, L_MLE_q, k_MLE_q, x_MLE_q) + min_q
    
    scale = logistic_function(year, L_MLE_scale, k_MLE_scale, x_MLE_scale) + min_scale

    if np.isin(0, ages):
        index_of_zero = list(ages).index(0)
        ages[index_of_zero] = 1 # Set age to 1 to prevent errors

        unscaled_forecast = gen_beta2_fun_pdf(ages, a, b, p, q)

        scaled_forecast = unscaled_forecast * scale

        a, b, c, d, e = infant_params
        infant_forecast = polynomial_fn(year, a, b, c, d, e)
        if infant_forecast < 0:
            print('NEGATIVE INFANT MORTALITY RATE')
        scaled_forecast[0] = infant_forecast

        ages[index_of_zero] = 0 # Set age back to 0

    else:
        unscaled_forecast = gen_beta2_fun_pdf(ages, a, b, p, q)

        scaled_forecast = unscaled_forecast * scale

    return scaled_forecast

def forecast_immigration(age_params, year, ages, transition_year):
    '''
    Using immigration parameter estimates, forecast
    immigration rates for a set of ages in a particular year
    '''
    year_estimates = []
    for age in ages:
        constant, beta, a, b, c, g = age_params[age]
        if year <= transition_year:
            age_estimate = constant + beta * year
        elif year <= transition_year + 15:
            year_adj = year - transition_year
            age_estimate = e ** (a * year_adj ** 2 + b * year_adj + c) + g
        else:
            year_adj = 15
            age_estimate = e ** (a * year_adj ** 2 + b * year_adj + c) + g
        year_estimates.append(age_estimate)
    forecast = np.array(year_estimates)

    return forecast

def forecast(params, year, ages, datatype, options):
    '''
    Use forecast functions above to get forecasts
    for a particular datatype
    '''
    # All forecasts converge by about 2050
    if year > 2050:
        year = 2050
    if datatype == 'fertility':
        a_params, b_params, p_params, q_params, scale_params = params
        return forecast_fertility(a_params, b_params, p_params, q_params, scale_params, year, ages)

    elif datatype == 'mortality':
        non_infant_params, infant_params = params
        return forecast_mortality(non_infant_params, infant_params, year, ages)

    elif datatype == 'immigration':
        transition_year = options['transition_year']
        return forecast_immigration(params, year, ages, transition_year)

########################
## Plotting functions ##
########################
def plot_params(start, end, smooth, params_list, datatype):
    '''
    Plot parameter estimates over time
    '''
    years = np.linspace(start, end, end - start + 1)
    lines = []
    for i, param_list in enumerate(params_list):
        lines.append(plt.plot(years, param_list[1]))
        lines[i][0].set_label(param_list[0])

    plt.legend()
    plt.savefig('graphs/' + datatype + '/smooth_' + str(smooth) + '/_parameters')
    plt.close()

    for i, param_list in enumerate(params_list):
        plt.plot(years, param_list[1], label=param_list[0])
        plt.legend()
        plt.savefig('graphs/' + datatype + '/smooth_' + str(smooth) + '/_' + param_list[0])
        plt.close()

def plot_data_transition(data, ages, start, end, smooth, datatype):
    '''
    Plot true data transition over time
    '''
    NUM_COLORS = end + 1 - start

    cm = plt.get_cmap('Blues')
    cNorm  = colors.Normalize(vmin=0, vmax=NUM_COLORS-1)
    scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=cm)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_prop_cycle(color=[scalarMap.to_rgba(i) for i in range(NUM_COLORS)])

    for year in range(start, end + 1):
        data_yr = rolling_avg_year(data, year, smooth)
        ax.plot(ages[:len(data_yr)], data_yr, linewidth=2)

    plt.xlabel(r'Age $s$')
    if datatype == 'fertility':
        y = r'Fertility Rate $f_{s,t}$'
    elif datatype == 'mortality':
        y = r'Mortality Rate $\rho_{s,t}$'
    elif datatype == 'immigration':
        y = r'Immigration Rate $i_{s,t}$'
    plt.ylabel(y)
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    plt.tight_layout()

    plt.savefig('graphs/' + datatype + '/smooth_' + str(smooth) + '/_aggregate_true')
    plt.close()

def plot_forecast_transition(params, ages, start, end, smooth, datatype, options=False):
    '''
    Plot forecast transition over time
    '''
    NUM_COLORS = end + 1 - start

    cm = plt.get_cmap('Blues')
    cNorm  = colors.Normalize(vmin=0, vmax=NUM_COLORS-1)
    scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=cm)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_prop_cycle(color=[scalarMap.to_rgba(i) for i in range(NUM_COLORS)])

    for year in range(start, end + 1):
        forecast_estimates = forecast(params, year, ages, datatype, options)
        plt.plot(ages, forecast_estimates, linewidth=2)

    plt.xlabel(r'Age $s$')
    if datatype == 'fertility':
        y = r'Fertility Rate $f_{s,t}$'
    elif datatype == 'mortality':
        y = r'Mortality Rate $\rho_{s,t}$'
    elif datatype == 'immigration':
        y = r'Immigration Rate $i_{s,t}$'
    plt.ylabel(y)
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    plt.tight_layout()

    plt.savefig('graphs/' + datatype + '/smooth_' + str(smooth) + '/_aggregate_predicted')
    plt.close()

def overlay_estimates(data, params, ages, start, end, smooth, datatype, options=False):
    '''
    Compare forecast and true data transitions over time
    '''
    NUM_COLORS = end + 1 - start

    cm1 = plt.get_cmap('Blues')
    cNorm  = colors.Normalize(vmin=0, vmax=NUM_COLORS-1)
    scalarMap1 = mplcm.ScalarMappable(norm=cNorm, cmap=cm1)
    cm2 = plt.get_cmap('Reds')
    scalarMap2 = mplcm.ScalarMappable(norm=cNorm, cmap=cm2)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_prop_cycle(color=[scalarMap1.to_rgba(i) for i in range(NUM_COLORS)] + [scalarMap2.to_rgba(i) for i in range(NUM_COLORS)])

    for year in range(start, end + 1):
        forecast_estimates = forecast(params, year, ages, datatype, options)
        plt.plot(ages, forecast_estimates, linewidth=2)

    for year in range(start, end + 1):
        data_yr = rolling_avg_year(data, year, smooth)
        ax.plot(ages[:len(data_yr)], data_yr, linewidth=2)

    plt.xlabel(r'Age $s$')
    if datatype == 'fertility':
        y = r'Fertility Rate $f_{s,t}$'
    elif datatype == 'mortality':
        y = r'Mortality Rate $\rho_{s,t}$'
    elif datatype == 'immigration':
        y = r'Immigration Rate $i_{s,t}$'
    plt.ylabel(y)
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    plt.tight_layout()

    plt.savefig('graphs/' + datatype + '/smooth_' + str(smooth) + '/_aggregate_overlay_predicted')
    plt.close()

def plot_2100(params, ages, smooth, datatype, options=False):
    '''
    Plot 2014 vs 2100 using forecasts
    '''
    for year in (1990, 2000, 2014, 2050, 2100):
        forecast_estimates = forecast(params, year, ages, datatype, options)
        plt.plot(ages, forecast_estimates, linewidth=2, label=str(year))

    plt.xlabel(r'Age $s$')
    if datatype == 'fertility':
        y = r'Fertility Rate $f_{s,t}$'
    elif datatype == 'mortality':
        y = r'Mortality Rate $\rho_{s,t}$'
    elif datatype == 'immigration':
        y = r'Immigration Rate $i_{s,t}$'
    plt.ylabel(y)
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    plt.tight_layout()
    plt.legend()

    plt.savefig('graphs/' + datatype + '/smooth_' + str(smooth) + '/_2100')
    plt.close()
