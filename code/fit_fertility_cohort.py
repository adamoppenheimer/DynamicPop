import os
import numpy as np
import pandas as pd
import scipy.optimize as opt
from scipy.ndimage.interpolation import shift
import math
from math import e
import matplotlib.pyplot as plt

cur_path = '/Volumes/GoogleDrive/My Drive/4th Year/Thesis/japan_olg_demographics'
os.chdir(cur_path)
datadir = 'data/demographic/'
fert_dir = datadir + 'jpn_fertility.csv'
mort_dir = datadir + 'jpn_mortality.csv'
pop_dir = datadir + 'jpn_population.csv'

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

fert_data = pd.read_csv(fert_dir, sep=r',\s*',\
    usecols=['Year1', 'Age', 'ASFR', 'AgeDef',\
                    'Collection', 'RefCode'])
fert_data = select_fert_data(fert_data)

# Try this commented out section to switch back to yearly rather than cohort data
# fert_new = {}
# for i in range(1947, 2015):
#     lst = []
#     for j in range(37): #From 14 to 50
#         val = fert_data[fert_data['Year'] == i][fert_data['Age'] == j]['Values'].values
#         try:
#             lst.append(val[0])
#         except:
#             lst.append(np.nan)
#     fert_new[i] = lst
# fert_data = fert_new

fert_data['Cohort'] = fert_data['Year'] - fert_data['Age']
fert_data.drop('Year', axis=1, inplace=True)
fert_data = fert_data.pivot(index='Age', columns='Cohort', values='Values')

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

def rolling_avg_year(year, roll):
    years = []
    for yr in range(year - roll, year + roll + 1):
        try:
            years.append(fert_data[yr])
        except:
            pass
    avg = array_add(years)
    return avg.dropna()

def gamma_fun_pdf(xvals, alpha, beta):
    pdf_vals = ( xvals ** (alpha - 1) * e ** ( - xvals / beta ) )/\
        ( beta ** alpha * math.gamma(alpha) )
    return pdf_vals

def crit_b(params, *args):
    alpha, beta = params
    xvals, dist_pts = args
    guess = gamma_fun_pdf(dist_pts, alpha, beta)
    if guess.sum() != 0:
        guess_compare = guess * ( xvals.sum() / guess.sum() ) #Restandardize data
    else:
        guess_compare = 0
    diff = np.sum((xvals - guess_compare) ** 2)
    return diff

def gen_gamma_fun_pdf(xvals, alpha, beta, m):
    pdf_vals = (m * e ** ( - ( xvals / beta ) ** m ))/\
            (xvals * math.gamma(alpha / m)) *\
            (xvals / beta) ** alpha
    return pdf_vals

def log_sum_c(xvals, alpha, beta, m):
    log_vals = np.log(m) + (alpha - 1) * np.log(xvals) -\
            (xvals / beta) ** m - alpha * np.log(beta) -\
            np.log(math.gamma(alpha / m))
    return log_vals.sum()

def crit_c(params, *args):
    alpha, beta, m = params
    xvals, dist_pts = args
    guess = gen_gamma_fun_pdf(dist_pts, alpha, beta, m)
    if guess.sum() != 0:
        guess_compare = guess * ( xvals.sum() / guess.sum() ) #Restandardize data
    else:
        guess_compare = 0
    diff = np.sum((xvals - guess_compare) ** 2)
    return diff

alphas = []
betas = []
ms = []
scales = []
for year in range(1975, 1991):#range(1975, 2001):
    #Fit data for 1980 cohort
    #fert_1980 = fert_data[year].dropna()

    #Take 5 years rolling average
    fert_1980 = rolling_avg_year(year, 4)

    #fert_1980 = fert_1980 / np.sum(fert_1980)
    count = fert_1980.shape[0]
    mean = 30#fert_1980.mean()
    median = 30#np.median(fert_1980)
    std = 5#fert_1980.std()
    var = 25#fert_1980.var()

    if year == 1975:
        beta_0 = var/mean
        alpha_0 = mean/beta_0
    if beta_0 < 0.3:
        beta_0 = (beta_0 + 0.3) / 2
    params_init = np.array([alpha_0, beta_0])
    dist_pts = np.linspace(14, 14 + count - 1, count)

    results_cstr = opt.minimize(crit_b, params_init,\
                    args=(np.array(fert_1980), dist_pts), method="L-BFGS-B",\
                    bounds=((1e-10, None), (1e-10, None)), options={'eps':1})
    alpha_MLE_b, beta_MLE_b = results_cstr.x

    print("alpha_MLE_b=", alpha_MLE_b, " beta_MLE_b=", beta_MLE_b)

    # plt.plot(dist_pts, gamma_fun_pdf(dist_pts, alpha_MLE_b, beta_MLE_b) / gamma_fun_pdf(dist_pts, alpha_MLE_b, beta_MLE_b).sum(),\
    #         linewidth=2, label="Gamma", color="r")
    # plt.plot(fert_1980 / fert_1980.sum())
    # plt.legend()
    # plt.show()

    ####################################################################################

    alpha_0 = alpha_MLE_b
    beta_0 = beta_MLE_b
    if beta_0 < 0.3:
        beta_0 = (beta_0 + 0.3) / 2
    if alpha_0 > 40:
        alpha_0 = 40
    if year == 1975:
        m_0 = 1
    params_init = np.array([alpha_0, beta_0, m_0])

    results_cstr = opt.minimize(crit_c, params_init,\
                                args=(fert_1980, dist_pts), method='L-BFGS-B',\
                                bounds=((1e-10, 40), (0.1, None),\
                                (1e-10, None)))#, tol=1e-100, options={'eps':9})
    alpha_MLE_c, beta_MLE_c, m_MLE_c = results_cstr.x

    #Use previous year's results as start values for next year
    alpha_0 = alpha_MLE_c
    beta_0 = beta_MLE_c
    m_0 = m_MLE_c

    print('alpha_MLE_c=', alpha_MLE_c, ' beta_MLE_c=', beta_MLE_c, ' m_MLE_c=', m_MLE_c)

    gen_gamma = gen_gamma_fun_pdf(dist_pts, alpha_MLE_c, beta_MLE_c, m_MLE_c)
    scales.append(np.sum(fert_1980) / np.sum(gen_gamma))

    plt.plot(dist_pts, gen_gamma * scales[len(scales) - 1],\
            linewidth=2, label='Generalized Gamma', color='r')
    plt.plot(dist_pts, fert_1980, label='True Fertility ' + str(year))
    plt.legend()
    plt.savefig('graphs/yearly/smooth_4/' + str(year))
    plt.close()

    alphas.append(alpha_MLE_c)
    betas.append(beta_MLE_c)
    ms.append(m_MLE_c)

alphas = np.array(alphas)
betas = np.array(betas)
ms = np.array(ms)
scales = np.array(scales)

years = np.linspace(1975, 1990, 16)
lines = []
lines.append(plt.plot(years, alphas))
lines.append(plt.plot(years, betas))
lines.append(plt.plot(years, ms))
lines.append(plt.plot(years, scales))

lines[0][0].set_label('Alpha')
lines[1][0].set_label('Beta')
lines[2][0].set_label('M')
lines[3][0].set_label('Scale')

plt.legend()
plt.savefig('graphs/yearly/smooth_4/_parameters')
plt.close()

plt.plot(years, alphas, label='Alpha')
plt.legend()
plt.savefig('graphs/yearly/smooth_4/_alpha')
plt.close()
plt.plot(years, betas, label='Beta')
plt.legend()
plt.savefig('graphs/yearly/smooth_4/_beta')
plt.close()
plt.plot(years, ms, label='M')
plt.legend()
plt.savefig('graphs/yearly/smooth_4/_m')
plt.close()
plt.plot(years, scales, label='Scale')
plt.legend()
plt.savefig('graphs/yearly/smooth_4/_scale')
plt.close()
