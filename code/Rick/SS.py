'''
------------------------------------------------------------------------
This module contains the functions used to solve the steady state of
the model with S-period lived agents, endogenous labor supply, and
multiple static industries from Chapter 17 of the OG textbook.

This Python module imports the following module(s):
    households.py
    industries.py
    aggregates.py
    utilities.py

This Python module defines the following function(s):
    rw_errors()
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
        asdf

    OBJECTS CREATED WITHIN FUNCTION:

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: rw_errors
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
    rw_init = (2,) vector, initial guesses for (rss_init, wss_init)
    p       = parameters class object
    graphs  = boolean, =True if output steady-state graphs

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        hh.get_cnb_vecs()
        hh.get_cms()
        indust.get_KLrat()
        indust.get_pm()
        indust.get_pM
        indust.get_r()
        indust.get_w()
        aggr.get_Cm()
        aggr.get_Lm_ss()
        aggr.get_Ym()
        utils.print_time()
        ss_graphs()

    OBJECTS CREATED WITHIN FUNCTION:
    start_time = scalar > 0, clock time at beginning of program
    r_init     = scalar > -delta, initial guess for steady-state
                 interest rate
    c1_init    = scalar > 0, initial guess for first period consumpt'n
    S          = integer in [3, 80], number of periods an individual
                 lives
    beta       = scalar in (0,1), discount factor for each model per
    sigma      = scalar > 0, coefficient of relative risk aversion
    l_tilde    = scalar > 0, time endowment for each agent each period
    b_ellip    = scalar > 0, fitted value of b for elliptical disutility
                 of labor
    upsilon    = scalar > 1, fitted value of upsilon for elliptical
                 disutility of labor
    chi_n_vec  = (S,) vector, values for chi^n_s
    A          = scalar > 0, total factor productivity parameter in
                 firms' production function
    alpha      = scalar in (0,1), capital share of income
    delta      = scalar in [0,1], model-period depreciation rate of
                 capital
    Bsct_Tol   = scalar > 0, tolderance level for outer-loop bisection
                 method
    Eul_Tol    = scalar > 0, tolerance level for inner-loop root finder
    EulDiff    = Boolean, =True if want difference version of Euler
                 errors beta*(1+r)*u'(c2) - u'(c1), =False if want
                 ratio version [beta*(1+r)*u'(c2)]/[u'(c1)] - 1
    xi         = scalar in (0, 1], SS updating parameter in outer-loop
                 bisection method
    maxiter    = integer >= 1, maximum number of iterations in outer
                 loop bisection method
    iter_SS    = integer >= 0, index of iteration number
    dist       = scalar > 0, distance metric for current iteration
    rw_params  = length 3 tuple, (A, alpha, delta) args to pass into
                 firms.get_r() and firms.get_w()
    w_init     = scalar, initial value for wage
    inner_args = length 14 tuple, args to pass into inner_loop()
    K_new      = scalar > 0, updated K given r_init, w_init, and bvec
    L_new      = scalar > 0, updated L given r_init, w_init, and nvec
    cvec       = (S,) vector, updated values for lifetime consumption
    nvec       = (S,) vector, updated values for lifetime labor supply
    bvec       = (S,) vector, updated values for lifetime savings
                 (b1, b2,...bS)
    b_Sp1      = scalar, updated value for savings in last period,
                 should be arbitrarily close to zero
    r_new      = scalar > 0, updated interest rate given bvec and nvec
    w_new      = scalar > 0, updated wage given bvec and nvec
    n_errors   = (S,) vector, labor supply Euler errors given r_init
                 and w_init
    b_errors   = (S-1,) vector, savings Euler errors given r_init and
                 w_init
    all_errors = (2S,) vector, (n_errors, b_errors, b_Sp1)
    c_ss       = (S,) vector, steady-state lifetime consumption
    n_ss       = (S,) vector, steady-state lifetime labor supply
    b_ss       = (S,) vector, steady-state wealth enter period with
                 (b1, b2, ...bS)
    b_Sp1_ss   = scalar, steady-state savings for period after last
                 period of life. b_Sp1_ss approx. 0 in equilibrium
    n_err_ss   = (S,) vector, lifetime labor supply Euler errors
    b_err_ss   = (S-1) vector, lifetime savings Euler errors
    r_ss       = scalar > 0, steady-state interest rate
    w_ss       = scalar > 0, steady-state wage
    K_ss       = scalar > 0, steady-state aggregate capital stock
    L_ss       = scalar > 0, steady-state aggregate labor
    Y_params   = length 2 tuple, (A, alpha)
    Y_ss       = scalar > 0, steady-state aggregate output (GDP)
    C_ss       = scalar > 0, steady-state aggregate consumption
    RCerr_ss   = scalar, resource constraint error
    ss_time    = scalar, seconds elapsed for steady-state computation
    ss_output  = length 14 dict, steady-state objects {n_ss, b_ss,
                 c_ss, b_Sp1_ss, w_ss, r_ss, K_ss, L_ss, Y_ss, C_ss,
                 n_err_ss, b_err_ss, RCerr_ss, ss_time}

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
        print('SS SUCESSS: steady-state solution converged.')

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
    b_ss = (S+1,) vector, steady-state lifetime savings (b1, b2, ...bS)
            where b1 = 0

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:

    OBJECTS CREATED WITHIN FUNCTION:
    cur_path    = string, path name of current directory
    output_fldr = string, folder in current path to save files
    output_dir  = string, total path of images folder
    output_path = string, path of file name of figure to be saved
    S           = integer in [3, 80], number of periods an individual
                  lives
    b_ss_full   = (S+1,) vector, b_ss with zero appended on end.
                  (b1, b2, ...bS, bSp1) where b1, bSp1 = 0
    age_pers_c  = (S,) vector, ages from 1 to S
    age_pers_b  = (S+1,) vector, ages from 1 to S+1

    FILES CREATED BY THIS FUNCTION:
        SS_bc.png
        SS_cm.png
        SS_n.png

    RETURNS: None
    --------------------------------------------------------------------
    '''
    # Create directory if images directory does not already exist
    cur_path = os.path.split(os.path.abspath(__file__))[0]
    image_fldr = 'OUTPUT/SS/images'
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
    plt.ylabel(r'Units of consumption')
    plt.xlim((p.E, p.E + p.S + 2))
    # plt.ylim((-1.0, 1.15 * (b_ss.max())))
    plt.legend(loc='upper left')
    output_path = os.path.join(image_dir, 'SS_bc')
    plt.savefig(output_path)
    # plt.show()
    plt.close()

    # Plot steady-state labor supply distributions
    fig, ax = plt.subplots()
    plt.plot(age_pers_c, n_ss, marker='D', label='Labor supply')
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
    output_path = os.path.join(image_dir, 'SS_n')
    plt.savefig(output_path)
    # plt.show()
    plt.close()
