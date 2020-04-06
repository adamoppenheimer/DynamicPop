'''
------------------------------------------------------------------------
This module contains the functions used to solve the steady state of
the model with S-period lived agents, endogenous labor supply,
non-constant demographics, bequests, and productivity growth.

This Python module imports the following module(s):
    households.py
    firms.py
    aggregates.py
    utilities.py

This Python module defines the following function(s):
    rBQ_errors()
    get_SS()
    ss_graphs()
------------------------------------------------------------------------
'''
# Import packages
import time
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import os
import households as hh
import firms
import aggregates as aggr
import utilities as utils

'''
------------------------------------------------------------------------
    Functions
------------------------------------------------------------------------
'''


def rBQ_errors(rBQ_vals, *args):
    '''
    --------------------------------------------------------------------
    Generate interest rate and wage errors from labor market clearing
    and capital market clearing conditions
    --------------------------------------------------------------------
    INPUTS:
    rBQ_vals = (2,) vector, steady-state r value and BQ value
    args     = length 2 tuple, (nb_guess, p)

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        firms.get_wt()
        hh.get_cnb_vecs()
        aggr.get_Lt()
        aggr.get_Kt()
        firms.get_rt()
        aggr.get_BQt()

    OBJECTS CREATED WITHIN FUNCTION:
    nb_buess   = (2S,) vector, initial guesses for n_s and b_sp1
    p          = parameters class object
    r_init     = scalar, initial guess for steady-state interest rate
    BQ_init    = scalar, initial guess for steady-state total bequests
    w          = scalar, steady-state wage implied by r_init
    b_init     = scalar = 0, initial wealth of initial age individuals
    r_path     = (S,) vector, constant interest rate time path over
                 lifetime of individual
    w_path     = (S,) vector, constant wage time path over lifetime of
                 individual
    BQ_path    = (S,) vector, constant total bequests time path over
                 lifetime of individual
    c_s        = (S,) vector, steady-state consumption by age
    n_s        = (S,) vector, steady-state labor supply by age
    b_s        = (S+1,) vector, steady-state wealth or savings by age
    n_errors   = (S,) vector, errors associated with optimal labor
                 supply solution
    b_errors   = (S,) vector, errors associated with optimal savings
                 solution
    L          = scalar, aggregate labor implied by household and firm
                 optimization
    K          = scalar, aggregate capital implied by household and firm
                 optimization
    r_new      = scalar, new value of r implied by household and firm
                 optimization
    BQ_new     = scalar, new value of BQ implied by household and firm
                 optimization
    r_error    = scalar, difference between r_new and r_init
    BQ_error   = scalar, difference between BQ_new and BQ_init
    rBQ_errors = (2,) vector, r_error and BQ_error

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: rBQ_errors
    --------------------------------------------------------------------
    '''
    (nb_guess, p) = args
    r_init, BQ_init = rBQ_vals
    # Solve for steady-state wage w implied by r
    w = firms.get_wt(r_init, p)
    # Solve for household steady-state decisions c_s, n_s, b_{s+1} given
    # r, w, and BQ
    b_init = 0.0
    r_path = r_init * np.ones(p.S)
    w_path = w * np.ones(p.S)
    BQ_path = BQ_init * np.ones(p.S)
    c_s, n_s, b_s, n_errors, b_errors = \
        hh.get_cnb_vecs(nb_guess, b_init, r_path, w_path, BQ_path,
                        p.rho_ss, p.SS_EulDif, p, p.SS_EulTol)
    # Solve for aggregate labor and aggregate capital
    L = aggr.get_Lt(n_s, p)
    K = aggr.get_Kt(b_s[1:], p)
    # Solve for updated values of the interest rate r and total bequests
    # BQ
    r_new = firms.get_rt(K, L, p)
    BQ_new = aggr.get_BQt(b_s[1:], r_init, p)
    # solve for errors in interest rate and total bequests guesses
    r_error = r_new - r_init
    BQ_error = BQ_new - BQ_init
    rBQ_errors = np.array([r_error, BQ_error])

    return rBQ_errors


def get_SS(rBQ_init, p, graphs=False):
    '''
    --------------------------------------------------------------------
    Solve for the steady-state solution of the S-period-lived agent OG
    model with endogenous labor supply and multiple industries using the
    root finder method in r and w for the outer loop
    --------------------------------------------------------------------
    INPUTS:
    rBQ_init = (2,) vector, initial guesses for (rss_init, BQss_init)
    p        = parameters class object
    graphs   = boolean, =True if output steady-state graphs

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        rBQ_errors()
        firms.get_wt()
        hh.get_cnb_vecs()
        aggr.get_Lt()
        aggr.get_Kt()
        firms.get_Yt()
        C_ss = aggr.get_Ct()
        I_ss = aggr.get_It()
        NX_ss = aggr.get_NXt()
        utils.print_time()
        ss_graphs()

    OBJECTS CREATED WITHIN FUNCTION:
    start_time  = scalar > 0, clock time at beginning of program
    nvec_guess  = (S,) vector, initial guess for optimal household labor
                  supply n_s
    bvec_guess  = (S,) vector, initial guess for optimal household
                  savings b_sp1
    nb_guess    = (2S,) vector, initial guesses for optimal household
                  labor supply and savings (n_s, b_sp1)
    rBQ_args    = length 2 tuple, (nb_guess, p)
    results_rBQ = root results object
    err_msg     = string, error message text string
    r_ss        = scalar > -delta, steady-state interest rate
    BQ_ss       = scalar > 0, steady-state total bequests
    r_err_ss    = scalar, error in steady-state optimal solution firm
                  first order condition for r_ss
    BQ_err_ss   = scalar, error in steady-state optimal solution
                  bequests law of motion for BQ_ss
    w_ss        = scalar > 0, steady-state wage
    b_init      = scalar = 0, initial wealth of initial age individuals
    r_path      = (S,) vector, constant interest rate time path over
                  lifetime of individual
    w_path      = (S,) vector, constant wage time path over lifetime of
                  individual
    BQ_path     = (S,) vector, constant total bequests time path over
                  lifetime of individual
    c_ss        = (S,) vector, steady-state consumption by age
    n_ss        = (S,) vector, steady-state labor supply by age
    b_ss        = (S+1,) vector, steady-state wealth or savings by age
    n_err_ss    = (S,) vector, errors associated with optimal labor
                  supply solution
    b_err_ss    = (S,) vector, errors associated with optimal savings
                  solution
    L_ss        = scalar > 0, steady-state aggregate labor
    K_ss        = scalar > 0, steady-state aggregate capital stock
    Y_ss        = scalar > 0, steady-state aggregate output
    C_ss        = scalar > 0, steady-state aggregate consumption
    I_ss        = scalar, steady-state aggregate investment
    NX_ss       = scalar, steady-state net exports
    RCerr_ss    = scalar, steady-state resource constraint (goods market
                  clearing) error
    ss_time     = scalar, seconds elapsed for steady-state computation
    ss_output   = length 18 dictionary, steady-state output {c_ss, n_ss,
                  b_ss, n_err_ss, b_err_ss, r_ss, w_ss, BQ_ss, r_err_ss,
                  BQ_err_ss, L_ss, K_ss, Y_ss, C_ss, I_ss, NX_ss,
                  RCerr_ss, ss_time}

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: ss_output
    --------------------------------------------------------------------
    '''
    start_time = time.clock()
    nvec_guess = 0.4 * p.l_tilde * np.ones(p.S)
    bvec_guess = 0.1 * np.ones(p.S)
    nb_guess = np.append(nvec_guess, bvec_guess)
    rBQ_args = (nb_guess, p)
    results_rBQ = opt.root(rBQ_errors, rBQ_init, args=rBQ_args,
                           tol=p.SS_OutTol)
    if not results_rBQ.success:
        err_msg = ('SS Error: Steady-state root finder did not ' +
                   'solve. results_rBQ.success=False')
        raise ValueError(err_msg)
    else:
        print('SS SUCCESS: steady-state solution converged.')

    # print(results_rw)
    r_ss, BQ_ss = results_rBQ.x
    r_err_ss, BQ_err_ss = results_rBQ.fun
    # Solve for steady-state wage w_ss implied by r_ss
    w_ss = firms.get_wt(r_ss, p)
    # Solve for household steady-state decisions c_s, n_s, b_{s+1} given
    # r, w, and BQ
    b_init = 0.0
    r_path = r_ss * np.ones(p.S)
    w_path = w_ss * np.ones(p.S)
    BQ_path = BQ_ss * np.ones(p.S)
    c_ss, n_ss, b_ss, n_err_ss, b_err_ss = \
        hh.get_cnb_vecs(nb_guess, b_init, r_path, w_path, BQ_path,
                        p.rho_ss, p.SS_EulDif, p, p.SS_EulTol)
    # Solve for steady-state aggregate labor and aggregate capital
    L_ss = aggr.get_Lt(n_ss, p)
    K_ss = aggr.get_Kt(b_ss[1:], p)
    # Solve for steady-state aggregate output Y, consumption C,
    # investment I, and net exports
    Y_ss = firms.get_Yt(K_ss, L_ss, p)
    C_ss = aggr.get_Ct(c_ss, p)
    I_ss = aggr.get_It(K_ss, p)
    NX_ss = aggr.get_NXt(b_ss[1:], p)
    # Solve for steady-state resource constraint error
    RCerr_ss = Y_ss - C_ss - I_ss - NX_ss

    ss_time = time.clock() - start_time

    ss_output = {
        'c_ss': c_ss, 'n_ss': n_ss, 'b_ss': b_ss, 'n_err_ss': n_err_ss,
        'b_err_ss': b_err_ss, 'r_ss': r_ss, 'w_ss': w_ss,
        'BQ_ss': BQ_ss, 'r_err_ss': r_err_ss, 'BQ_err_ss': BQ_err_ss,
        'L_ss': L_ss, 'K_ss': K_ss, 'Y_ss': Y_ss, 'C_ss': C_ss,
        'I_ss': I_ss, 'NX_ss': NX_ss, 'RCerr_ss': RCerr_ss,
        'ss_time': ss_time}

    print('n_ss=', n_ss)
    print('b_ss=', b_ss)
    print('K_ss=', K_ss)
    print('L_ss=', L_ss)
    print('r_ss=', r_ss, ', w_ss=', w_ss, ', BQ_ss=', BQ_ss)
    print('Maximum abs. labor supply Euler error is: ',
          np.absolute(n_err_ss).max())
    print('Maximum abs. savings Euler error is: ',
          np.absolute(b_err_ss).max())
    print('RC error is: ', RCerr_ss)

    # Print SS computation time
    utils.print_time(ss_time, 'SS')

    if graphs:
        ss_graphs(c_ss, n_ss, b_ss, p)

    return ss_output


def ss_graphs(c_ss, n_ss, b_ss, p):
    '''
    --------------------------------------------------------------------
    Plot steady-state equilibrium results
    --------------------------------------------------------------------
    INPUTS:
    c_ss = (S,) vector, steady-state lifetime consumption
    n_ss = (S,) vector, steady-state lifetime labor supply
    b_ss = (S+1,) vector, steady-state lifetime savings (b1, b2,...bSp1)
            where b1 = 0

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:

    OBJECTS CREATED WITHIN FUNCTION:
    cur_path    = string, path name of current directory
    image_fldr  = string, folder in current path to save files
    image_dir   = string, total path of images folder
    output_path = string, path of file name of figure to be saved
    age_pers_c  = (S,) vector, vector of age periods 21 to 100
    age_pers_b  = (S+1,) vector, vector of age periods including period
                  after death 21 to 101

    FILES CREATED BY THIS FUNCTION:
        SS_bc.png
        SS_n.png

    RETURNS: None
    --------------------------------------------------------------------
    '''
    # Create directory if images directory does not already exist
    cur_path = os.path.split(os.path.abspath(__file__))[0]
    image_fldr = 'OUTPUT/SS/' + p.demog_type + '/images'
    image_dir = os.path.join(cur_path, image_fldr)
    if not os.access(image_dir, os.F_OK):
        os.makedirs(image_dir)

    # Plot steady-state consumption and savings distributions
    age_pers_c = np.arange(p.E + 1, p.E + p.S + 1)
    age_pers_b = np.arange(p.E + 1, p.E + p.S + 2)
    fig, ax = plt.subplots()
    plt.plot(age_pers_c, c_ss, marker='D', label='Consumption')
    plt.plot(age_pers_b, b_ss, marker='D', label='Savings')
    # for the minor ticks, use no labels; default NullFormatter
    minorLocator = MultipleLocator(1)
    ax.xaxis.set_minor_locator(minorLocator)
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    # plt.title('Steady-state consumption and savings', fontsize=20)
    plt.xlabel(r'Age $s$')
    plt.ylabel(r'Units of Consumption')
    plt.xlim((p.E, p.E + p.S + 2))
    # plt.ylim((-1.0, 1.15 * (b_ss.max())))
    plt.legend(loc='upper left')
    plt.tight_layout()
    output_path = os.path.join(image_dir, 'SS_bc')
    plt.savefig(output_path)
    # plt.show()
    plt.close()

    # Plot steady-state labor supply distributions
    fig, ax = plt.subplots()
    plt.plot(age_pers_c, n_ss, marker='D', label='Labor Supply')
    # for the minor ticks, use no labels; default NullFormatter
    minorLocator = MultipleLocator(1)
    ax.xaxis.set_minor_locator(minorLocator)
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    # plt.title('Steady-state labor supply', fontsize=20)
    plt.xlabel(r'Age $s$')
    plt.ylabel(r'Labor supply')
    plt.xlim((p.E, p.E + p.S + 1))
    # plt.ylim((-0.1, 1.15 * (n_ss.max())))
    plt.legend(loc='upper right')
    plt.tight_layout()
    output_path = os.path.join(image_dir, 'SS_n')
    plt.savefig(output_path)
    # plt.show()
    plt.close()
