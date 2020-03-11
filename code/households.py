'''
------------------------------------------------------------------------
This module contains the functions that generate the variables
associated with households' optimization in the steady-state or in the
transition path of the overlapping generations model with S-period lived
agents, endogenous labor supply, non-constant demographics, and
productivity growth.

This Python module imports the following module(s): None

This Python module defines the following function(s):
    get_cons()
    MU_c_stitch()
    MDU_n_stitch()
    get_n_errors()
    get_b_errors()
    get_cnb_vecs()
    c1_bSp1err()
------------------------------------------------------------------------
'''
# Import packages
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import os

'''
------------------------------------------------------------------------
    Functions
------------------------------------------------------------------------
'''


def get_cs_vec(b_s, b_sp1, n_s, r, w, BQ, p):
    '''
    --------------------------------------------------------------------
    Calculate household consumption given prices, labor supply, current
    wealth, and savings
    --------------------------------------------------------------------
    INPUTS:
    b_s   = (rp,) vector, current period wealth or time path of current-
            period wealths over remaining life periods
    b_sp1 = (rp,) vector, next-period savings chosen this period over
            remaining life periods
    n_s   = (rp,) vector, labor supply over remaining life periods
    r     = scalar > -delta or (rp,) vector, steady-state interest rate
            or time path of interest rate over remaining life periods
    w     = scalar > 0 or (rp,) vector, steady-state wage or time path
            of wage over remaining life periods
    BQ    = scalar > 0 or (rp,) vector, steady-state total bequests or
            time path of total bequests over remaining life periods
    p     = parameters class object

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    c_s = (rp,) vector, age-s consumption over remaining life periods

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: c_s
    --------------------------------------------------------------------
    '''
    c_s = (1 + r) * b_s + w * n_s + BQ - np.exp(p.g_y) * b_sp1

    return c_s


def MU_c_stitch(cvec, sigma, graph=False):
    '''
    --------------------------------------------------------------------
    Generate marginal utility(ies) of consumption with CRRA consumption
    utility and stitched function at lower bound such that the new
    hybrid function is defined over all consumption on the real
    line but the function has similar properties to the Inada condition.

    u'(c) = c ** (-sigma) if c >= epsilon
          = g'(c) = 2 * b2 * c + b1 if c < epsilon

        such that g'(epsilon) = u'(epsilon)
        and g''(epsilon) = u''(epsilon)

        u(c) = (c ** (1 - sigma) - 1) / (1 - sigma)
        g(c) = b2 * (c ** 2) + b1 * c + b0
    --------------------------------------------------------------------
    INPUTS:
    cvec  = scalar or (p,) vector, individual consumption value or
            lifetime consumption over p consecutive periods
    sigma = scalar >= 1, coefficient of relative risk aversion for CRRA
            utility function: (c**(1-sigma) - 1) / (1 - sigma)
    graph = boolean, =True if want plot of stitched marginal utility of
            consumption function

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    epsilon    = scalar > 0, positive value close to zero
    c_s        = scalar, individual consumption
    c_s_cnstr  = boolean, =True if c_s < epsilon
    b1         = scalar, intercept value in linear marginal utility
    b2         = scalar, slope coefficient in linear marginal utility
    MU_c       = scalar or (p,) vector, marginal utility of consumption
                 or vector of marginal utilities of consumption
    p          = integer >= 1, number of periods remaining in lifetime
    cvec_cnstr = (p,) boolean vector, =True for values of cvec < epsilon

    FILES CREATED BY THIS FUNCTION:
        MU_c_stitched.png

    RETURNS: MU_c
    --------------------------------------------------------------------
    '''
    epsilon = 0.003
    if np.ndim(cvec) == 0:
        c_s = cvec
        c_s_cnstr = c_s < epsilon
        if c_s_cnstr:
            b2 = (-sigma * (epsilon ** (-sigma - 1))) / 2
            b1 = (epsilon ** (-sigma)) - 2 * b2 * epsilon
            MU_c = 2 * b2 * c_s + b1
        else:
            MU_c = c_s ** (-sigma)
    elif np.ndim(cvec) == 1:
        p = cvec.shape[0]
        cvec_cnstr = cvec < epsilon
        MU_c = np.zeros(p)
        MU_c[~cvec_cnstr] = cvec[~cvec_cnstr] ** (-sigma)
        b2 = (-sigma * (epsilon ** (-sigma - 1))) / 2
        b1 = (epsilon ** (-sigma)) - 2 * b2 * epsilon
        MU_c[cvec_cnstr] = 2 * b2 * cvec[cvec_cnstr] + b1

    if graph:
        '''
        ----------------------------------------------------------------
        cur_path    = string, path name of current directory
        output_fldr = string, folder in current path to save files
        output_dir  = string, total path of images folder
        output_path = string, path of file name of figure to be saved
        cvec_CRRA   = (1000,) vector, support of c including values
                      between 0 and epsilon
        MU_CRRA     = (1000,) vector, CRRA marginal utility of
                      consumption
        cvec_stitch = (500,) vector, stitched support of consumption
                      including negative values up to epsilon
        MU_stitch   = (500,) vector, stitched marginal utility of
                      consumption
        ----------------------------------------------------------------
        '''
        # Create directory if images directory does not already exist
        cur_path = os.path.split(os.path.abspath(__file__))[0]
        output_fldr = "images"
        output_dir = os.path.join(cur_path, output_fldr)
        if not os.access(output_dir, os.F_OK):
            os.makedirs(output_dir)

        # Plot steady-state consumption and savings distributions
        cvec_CRRA = np.linspace(epsilon / 2, epsilon * 3, 1000)
        MU_CRRA = cvec_CRRA ** (-sigma)
        cvec_stitch = np.linspace(-0.00005, epsilon, 500)
        MU_stitch = 2 * b2 * cvec_stitch + b1
        fig, ax = plt.subplots()
        plt.plot(cvec_CRRA, MU_CRRA, ls='solid', label='$u\'(c)$: CRRA')
        plt.plot(cvec_stitch, MU_stitch, ls='dashed', color='red',
                 label='$g\'(c)$: stitched')
        # for the minor ticks, use no labels; default NullFormatter
        minorLocator = MultipleLocator(1)
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65', linestyle='-')
        plt.title('Marginal utility of consumption with stitched ' +
                  'function', fontsize=14)
        plt.xlabel(r'Consumption $c$')
        plt.ylabel(r'Marginal utility $u\'(c)$')
        plt.xlim((-0.00005, epsilon * 3))
        # plt.ylim((-1.0, 1.15 * (b_ss.max())))
        plt.legend(loc='upper right')
        output_path = os.path.join(output_dir, "MU_c_stitched")
        plt.savefig(output_path)
        # plt.show()

    return MU_c


def MDU_n_stitch(ns_vec, p, graph=False):
    '''
    --------------------------------------------------------------------
    Generate marginal disutility(ies) of labor with elliptical
    disutility of labor function and stitched functions at lower bound
    and upper bound of labor supply such that the new hybrid function is
    defined over all labor supply on the real line but the function has
    similar properties to the Inada conditions at the upper and lower
    bounds.

    v'(n) = (b / l_tilde) * ((n / l_tilde) ** (upsilon - 1)) *
            ((1 - ((n / l_tilde) ** upsilon)) ** ((1-upsilon)/upsilon))
            if n >= eps_low <= n <= eps_high
          = g_low'(n)  = 2 * b2 * n + b1 if n < eps_low
          = g_high'(n) = 2 * d2 * n + d1 if n > eps_high

        such that g_low'(eps_low) = u'(eps_low)
        and g_low''(eps_low) = u''(eps_low)
        and g_high'(eps_high) = u'(eps_high)
        and g_high''(eps_high) = u''(eps_high)

        v(n) = -b *(1 - ((n/l_tilde) ** upsilon)) ** (1/upsilon)
        g_low(n)  = b2 * (n ** 2) + b1 * n + b0
        g_high(n) = d2 * (n ** 2) + d1 * n + d0
    --------------------------------------------------------------------
    INPUTS:
    nvec  = scalar or (p,) vector, labor supply value or labor supply
            values over remaining periods of lifetime
    p     = parameters class object
    graph = Boolean, =True if want plot of stitched marginal disutility
             of labor function

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    l_tilde       = scalar > 0, time endowment for each agent each per
    b_ellip       = scalar > 0, scale parameter for elliptical utility
                    of leisure function
    upsilon       = scalar > 1, shape parameter for elliptical utility
                    of leisure function
    eps_low       = scalar > 0, positive value close to zero
    eps_high      = scalar > 0, positive value just less than l_tilde
    n_s           = scalar, individual labor supply
    n_s_low       = boolean, =True for n_s < eps_low
    n_s_high      = boolean, =True for n_s > eps_high
    n_s_uncstr    = boolean, =True for n_s >= eps_low and
                    n_s <= eps_high
    MDU_n         = scalar or (p,) vector, marginal disutility or
                    marginal utilities of labor supply
    b1            = scalar, intercept value in linear marginal
                    disutility of labor at lower bound
    b2            = scalar, slope coefficient in linear marginal
                    disutility of labor at lower bound
    d1            = scalar, intercept value in linear marginal
                    disutility of labor at upper bound
    d2            = scalar, slope coefficient in linear marginal
                    disutility of labor at upper bound
    p             = integer >= 1, number of periods remaining in life
    nvec_s_low    = boolean, =True for n_s < eps_low
    nvec_s_high   = boolean, =True for n_s > eps_high
    nvec_s_uncstr = boolean, =True for n_s >= eps_low and
                    n_s <= eps_high

    FILES CREATED BY THIS FUNCTION:
        MDU_n_stitched.png

    RETURNS: MDU_n
    --------------------------------------------------------------------
    '''
    eps_low = 0.001
    eps_high = p.l_tilde - 0.001
    # This if is for when ns_vec is a scalar
    if np.ndim(ns_vec) == 0:
        rp = 1
        n_s = ns_vec
        n_s_low = n_s < eps_low
        n_s_high = n_s > eps_high
        n_s_uncstr = (n_s >= eps_low) and (n_s <= eps_high)
        if n_s_uncstr:
            MDU_n = \
                ((p.b_ellip / p.l_tilde) * ((n_s / p.l_tilde) **
                 (p.upsilon - 1)) * ((1 - ((n_s / p.l_tilde) **
                                           p.upsilon)) **
                 ((1 - p.upsilon) / p.upsilon)))
        elif n_s_low:
            b2 = (0.5 * p.b_ellip * (p.l_tilde ** (-p.upsilon)) *
                  (p.upsilon - 1) * (eps_low ** (p.upsilon - 2)) *
                  ((1 - ((eps_low / p.l_tilde) ** p.upsilon)) **
                  ((1 - p.upsilon) / p.upsilon)) *
                  (1 + ((eps_low / p.l_tilde) ** p.upsilon) *
                  ((1 - ((eps_low / p.l_tilde) ** p.upsilon)) ** (-1))))
            b1 = ((p.b_ellip / p.l_tilde) * ((eps_low / p.l_tilde) **
                  (p.upsilon - 1)) *
                  ((1 - ((eps_low / p.l_tilde) ** p.upsilon)) **
                  ((1 - p.upsilon) / p.upsilon)) - (2 * b2 * eps_low))
            MDU_n = 2 * b2 * n_s + b1
        elif n_s_high:
            d2 = (0.5 * p.b_ellip * (p.l_tilde ** (-p.upsilon)) *
                  (p.upsilon - 1) * (eps_high ** (p.upsilon - 2)) *
                  ((1 - ((eps_high / p.l_tilde) ** p.upsilon)) **
                  ((1 - p.upsilon) / p.upsilon)) *
                  (1 + ((eps_high / p.l_tilde) ** p.upsilon) *
                  ((1 - ((eps_high / p.l_tilde) ** p.upsilon)) **
                   (-1))))
            d1 = ((p.b_ellip / p.l_tilde) * ((eps_high / p.l_tilde) **
                  (p.upsilon - 1)) *
                  ((1 - ((eps_high / p.l_tilde) ** p.upsilon)) **
                  ((1 - p.upsilon) / p.upsilon)) - (2 * d2 * eps_high))
            MDU_n = 2 * d2 * n_s + d1
    # This if is for when ns_vec is a one-dimensional vector
    elif np.ndim(ns_vec) == 1:
        rp = ns_vec.shape[0]
        nvec_low = ns_vec < eps_low
        nvec_high = ns_vec > eps_high
        nvec_uncstr = np.logical_and(~nvec_low, ~nvec_high)
        MDU_n = np.zeros(rp)
        MDU_n[nvec_uncstr] = (
            (p.b_ellip / p.l_tilde) *
            ((ns_vec[nvec_uncstr] / p.l_tilde) ** (p.upsilon - 1)) *
            ((1 - ((ns_vec[nvec_uncstr] / p.l_tilde) ** p.upsilon)) **
             ((1 - p.upsilon) / p.upsilon)))
        b2 = (0.5 * p.b_ellip * (p.l_tilde ** (-p.upsilon)) *
              (p.upsilon - 1) * (eps_low ** (p.upsilon - 2)) *
              ((1 - ((eps_low / p.l_tilde) ** p.upsilon)) **
              ((1 - p.upsilon) / p.upsilon)) *
              (1 + ((eps_low / p.l_tilde) ** p.upsilon) *
              ((1 - ((eps_low / p.l_tilde) ** p.upsilon)) ** (-1))))
        b1 = ((p.b_ellip / p.l_tilde) * ((eps_low / p.l_tilde) **
              (p.upsilon - 1)) *
              ((1 - ((eps_low / p.l_tilde) ** p.upsilon)) **
              ((1 - p.upsilon) / p.upsilon)) - (2 * b2 * eps_low))
        MDU_n[nvec_low] = 2 * b2 * ns_vec[nvec_low] + b1
        d2 = (0.5 * p.b_ellip * (p.l_tilde ** (-p.upsilon)) *
              (p.upsilon - 1) * (eps_high ** (p.upsilon - 2)) *
              ((1 - ((eps_high / p.l_tilde) ** p.upsilon)) **
              ((1 - p.upsilon) / p.upsilon)) *
              (1 + ((eps_high / p.l_tilde) ** p.upsilon) *
              ((1 - ((eps_high / p.l_tilde) ** p.upsilon)) ** (-1))))
        d1 = ((p.b_ellip / p.l_tilde) * ((eps_high / p.l_tilde) **
              (p.upsilon - 1)) *
              ((1 - ((eps_high / p.l_tilde) ** p.upsilon)) **
              ((1 - p.upsilon) / p.upsilon)) - (2 * d2 * eps_high))
        MDU_n[nvec_high] = 2 * d2 * ns_vec[nvec_high] + d1

    if graph:
        '''
        ----------------------------------------------------------------
        cur_path    = string, path name of current directory
        output_fldr = string, folder in current path to save files
        output_dir  = string, total path of images folder
        output_path = string, path of file name of figure to be saved
        nvec_ellip  = (1000,) vector, support of n including values
                      between 0 and eps_low and between eps_high and
                      l_tilde
        MU_CRRA     = (1000,) vector, CRRA marginal utility of
                      consumption
        cvec_stitch = (500,) vector, stitched support of consumption
                      including negative values up to epsilon
        MU_stitch   = (500,) vector, stitched marginal utility of
                      consumption
        ----------------------------------------------------------------
        '''
        # Create directory if images directory does not already exist
        cur_path = os.path.split(os.path.abspath(__file__))[0]
        output_fldr = "images"
        output_dir = os.path.join(cur_path, output_fldr)
        if not os.access(output_dir, os.F_OK):
            os.makedirs(output_dir)

        # Plot steady-state consumption and savings distributions
        nvec_ellip = np.linspace(eps_low / 2, eps_high +
                                 ((p.l_tilde - eps_high) / 5), 1000)
        MDU_ellip = (
            (p.b_ellip / p.l_tilde) *
            ((nvec_ellip / p.l_tilde) ** (p.upsilon - 1)) *
            ((1 - ((nvec_ellip / p.l_tilde) ** p.upsilon)) **
             ((1 - p.upsilon) / p.upsilon)))
        n_stitch_low = np.linspace(-0.05, eps_low, 500)
        MDU_stitch_low = 2 * b2 * n_stitch_low + b1
        n_stitch_high = np.linspace(eps_high, p.l_tilde + 0.000005, 500)
        MDU_stitch_high = 2 * d2 * n_stitch_high + d1
        fig, ax = plt.subplots()
        plt.plot(nvec_ellip, MDU_ellip, ls='solid', color='black',
                 label='$v\'(n)$: Elliptical')
        plt.plot(n_stitch_low, MDU_stitch_low, ls='dashed', color='red',
                 label='$g\'(n)$: low stitched')
        plt.plot(n_stitch_high, MDU_stitch_high, ls='dotted',
                 color='blue', label='$g\'(n)$: high stitched')
        # for the minor ticks, use no labels; default NullFormatter
        minorLocator = MultipleLocator(1)
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65', linestyle='-')
        plt.title('Marginal utility of consumption with stitched ' +
                  'function', fontsize=14)
        plt.xlabel(r'Labor $n$')
        plt.ylabel(r'Marginal disutility $v\'(n)$')
        plt.xlim((-0.05, p.l_tilde + 0.01))
        # plt.ylim((-1.0, 1.15 * (b_ss.max())))
        plt.legend(loc='upper left')
        output_path = os.path.join(output_dir, "MDU_n_stitched")
        plt.savefig(output_path)
        # plt.show()

    return MDU_n


def get_n_errors(n_s, c_s, args):
    '''
    --------------------------------------------------------------------
    Generates vector of static Euler errors that characterize the
    optimal lifetime labor supply decision in both the steady state and
    transition path.
    --------------------------------------------------------------------
    INPUTS:
    n_s  = scalar or (rp,) vector, labor supply over remaining lifetime
           periods
    c_s  = scalar or (rp,) vector, consumption over remaining lifetime
           periods
    args = length 3 tuple, (w, diff, p)

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        MU_c_stitch()
        MDU_n_stitch()

    OBJECTS CREATED WITHIN FUNCTION:
    w        = scalar or (rp,) vector, steady-state wage or time path of
               wages over remaining life periods
    diff     = boolean, =True if simple difference Euler equations.
               Otherwise use ratio Euler equation
    p        = parameters class object
    rp       = integer >= 1, periods remaining in individual's life
    MU_c     = scalar or (rp,) vector, marginal utility of current
               period consumption over remaining life periods
    MDU_n    = scalar or (rp,) vector, marginal disutility of labor
               supply over remaining life periods
    n_errors = scalar or (rp,) vector, labor supply Euler errors over
               remaining life periods

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: n_errors
    --------------------------------------------------------------------
    '''
    (w, diff, p) = args
    MU_c = MU_c_stitch(c_s, p.sigma)
    MDU_n = MDU_n_stitch(n_s, p)
    if np.isscalar(n_s):
        rp = 1
    else:
        rp = len(n_s)
    if diff:
        n_errors = (w * MU_c) - p.chi_n_vec[-rp:] * MDU_n
    else:
        n_errors = ((w * MU_c) / (p.chi_n_vec[-rp:] * MDU_n)) - 1

    return n_errors


def get_b_errors(c_s, b_sp1, args):
    '''
    --------------------------------------------------------------------
    Generates vector of dynamic Euler errors that characterize the
    optimal lifetime savings decision in both the steady-state and
    transition path
    --------------------------------------------------------------------
    INPUTS:
    c_s   = scalar or (rp,) vector, consumption over remaining life
            periods
    b_sp1 = scalar or (rp,) vector, savings over remaining life periods
    args  = length 4 tuple, (r, rho_st, diff, p)

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        MU_c_stitch()

    OBJECTS CREATED WITHIN FUNCTION:
    r          = (rp,) vector, interest rate over remaining life periods
    rho_st     = (rp,) vector, mortality rates over remaining life
                 periods
    diff       = boolean, =True if simple difference Euler equations.
                 Otherwise use ratio Euler equation
    p          = parameters class object
    MU_cs      = (rp-1,) vector, marginal utility of current consumption
    MU_csp1    = (rp-1,) vector, marginal utility of next period
                 consumption
    MU_bsp1    = (rp-1,) vector, marginal utility of savings for next
                 period
    MU_cS      = scalar, marginal utility of final age consumption
    MU_bSp1    = scalar, marginal utility of final age savings
    b_errors_s = (rp-1,) vector, consumption Euler errors for first rp-1
                 ages
    b_error_S  = scalar, final age consumption Euler error
    b_errors   = scalar or (rp,) vector, consumption Euler errors

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: b_errors
    --------------------------------------------------------------------
    '''
    (r, rho_st, diff, p) = args
    if np.isscalar(c_s):  # Transition path initial period old (rp = 1)
        MU_cS = MU_c_stitch(c_s, p.sigma)
        MU_bSp1 = MU_c_stitch(b_sp1, p.sigma)
        if diff:
            b_errors = (np.exp(-p.sigma * p.g_y) * p.chi_b * MU_bSp1 -
                        MU_cS)
        else:
            b_errors = (np.exp(-p.sigma * p.g_y) * p.chi_b * MU_bSp1 /
                        MU_cS) - 1
    else:  # Trans path and steady-state (rp > 1)
        MU_cs = MU_c_stitch(c_s[:-1], p.sigma)
        MU_csp1 = MU_c_stitch(c_s[1:], p.sigma)
        MU_bsp1 = MU_c_stitch(b_sp1[:-1], p.sigma)
        MU_cS = MU_c_stitch(c_s[-1], p.sigma)
        MU_bSp1 = MU_c_stitch(b_sp1[-1], p.sigma)

        if diff:
            b_errors_s = \
                ((np.exp(-p.sigma * p.g_y) *
                  ((rho_st[:-1] * p.chi_b * MU_bsp1) +
                   (p.beta * (1 + r[1:]) * (1 - rho_st[:-1]) *
                    MU_csp1))) - MU_cs)
            b_errors_S = (np.exp(-p.sigma * p.g_y) * p.chi_b * MU_bSp1 -
                          MU_cS)
        else:
            b_errors_s = \
                ((np.exp(-p.sigma * p.g_y) *
                  ((rho_st[:-1] * p.chi_b * MU_bsp1) +
                   (p.beta * (1 + r[1:]) * (1 - rho_st[:-1]) *
                    MU_csp1))) / MU_cs) - 1
            b_errors_S = (np.exp(-p.sigma * p.g_y) * p.chi_b * MU_bSp1 /
                          MU_cS) - 1
        b_errors = np.append(b_errors_s, b_errors_S)

    return b_errors


def get_nb_errors(nb_vec, *args):
    '''
    --------------------------------------------------------------------
    Computes labor supply and savings Euler errors for given n_s and
    b_{s+1} vectors and given the path of interest rates, wages, and
    industry-specific goods prices
    --------------------------------------------------------------------
    INPUTS:
    nb_vec = (2*rp,) vector, values for remaining life n_s and b_{s+1}
    args   = length 7 tuple, (b_init, r, w, BQ, rho_st, diff, p)

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        get_cs_vec()
        get_n_errors()
        get_b_errors()

    OBJECTS CREATED WITHIN FUNCTION:
    b_init    = scalar, initial wealth
    r         = scalar or (rp,) vector, time path of interest rates over
                remaining lifetime periods
    w         = scalar or (rp,) vector, time path of wages over
                remaining lifetime periods
    BQ        = scalar or (rp,) vector, time path of total bequests over
                remaining lifetime periods
    rho_st    = scalar or (rp,) vector, time path of mortality rates
                over remaining lifetime periods
    diff      = boolean, =True if simple difference Euler equations.
                Otherwise use ratio Euler equation
    p         = parameters class object
    n_S_0     = scalar, labor supply in last period of life
    b_Sp1_0   = scalar, savings in last period of life
    n_args    = length 3 tuple, (w, diff, p)
    n_errors  = scalar or (rp,) vector, labor supply Euler errors over
                remaining life periods
    b_args    = length 4 tuple, (r, rho_st, diff, p)
    b_errors  = scalar or (rp,) vector, savings Euler errors ove
                remaining life periods
    rp        = scalar >= 1, remaining periods in individual lifetime
    n_s       = (rp,) vector, labor supply over remaining life periods
    b_sp1     = (rp,) vector, savings over remaining life periods
    b_s       = (rp,) vector, wealth over remaining life periods
    c_s       = (rp,) vector, consumption over remaining life periods
    nb_errors = (2*rp,) vector, labor supply and savings Euler errors
                over remaining life periods

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: nb_errors
    --------------------------------------------------------------------
    '''
    (b_init, r, w, BQ, rho_st, diff, p) = args
    if nb_vec.shape[0] == 2:  # Age S individual in period 0
        n_S_0, b_Sp1_0 = nb_vec
        c_S_0 = get_cs_vec(b_init, b_Sp1_0, n_S_0, r, w, BQ, p)
        n_args = (w, diff, p)
        n_errors = get_n_errors(n_S_0, c_S_0, n_args)
        b_args = (r, rho_st, diff, p)
        b_errors = get_b_errors(c_S_0, b_Sp1_0, b_args)
    else:
        rp = int(nb_vec.shape[0] / 2)
        n_s = nb_vec[:rp]
        b_sp1 = nb_vec[rp:]
        b_s = np.append(b_init, b_sp1[:-1])
        c_s = get_cs_vec(b_s, b_sp1, n_s, r, w, BQ, p)
        n_args = (w, diff, p)
        n_errors = get_n_errors(n_s, c_s, n_args)
        b_args = (r, rho_st, diff, p)
        b_errors = get_b_errors(c_s, b_sp1, b_args)
    nb_errors = np.append(n_errors, b_errors)

    return nb_errors


def get_cnb_vecs(nb_guess, b_init, r, w, BQ, rho_st, diff, p, toler):
    '''
    --------------------------------------------------------------------
    Solve for lifetime consumption c_{s,t}, labor supply n_{s,t}, and
    savings b_{s+1,t+1} vectors for individual given prices r_t, w_t,
    BQ_t, and rho_{s,t} over the agent's lifetime in either the steady
    state or in the transition path equilibrium.
    --------------------------------------------------------------------
    INPUTS:
    nb_guess = (2*rp,) vector, initial guesses for n_s and b_sp1 vectors
    b_init   = scalar, wealth in initial period of life. Can be non-zero
               for incomplete lifetimes at beginning of transition path
    r        = (rp,) vector, time path of interest rates over remaining
               periods rp of agent's life
    w        = (rp,) vector, time path of wages over remaining periods
               rp of agent's life
    BQ       = (rp,) vector, time path of total bequests over remaining
               life periods
    rho_st   = (rp,) vetor, time path of mortality rates over remaining
               life periods
    diff     = boolean, =True if simple difference Euler equations.
               Otherwise use ratio Euler equation
    p        = parameters class object
    toler    = scalar > 0, tolerance value for root finder

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        get_nb_errors()
        get_cs_vec()

    OBJECTS CREATED WITHIN FUNCTION:]
    nb_args    = length 7 tuple, (b_init, r, w, BQ, rho_st, diff, p)
    results_nb = results object
    err_msg    = string, error message if root finder did not converge
    ns_vec     = scalar or (rp,) vector, optimal labor supply over
                 remaining life periods
    bsp1_vec   = scalar or (rp,) vector, optimal savings over remaining
                 life periods
    bs_vec     = (rp+1,) vector, optimal savings over remaining life
                 periods with initial wealth appended at the beginning
    n_errors   = (rp,) vector, labor supply Euler errors at optimum over
                 remaining life periods
    b_errors   = (rp,) vector, savings Euler errors at optimum over
                 remaining life periods
    cs_vec     = scalar or (rp,) vector, optimal consumption over
                 remaining life periods

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: cs_vec, ns_vec, bs_vec, n_errors, b_errors
    --------------------------------------------------------------------
    '''
    rp = r.shape[0]
    nb_args = (b_init, r, w, BQ, rho_st, diff, p)
    results_nb = opt.root(get_nb_errors, nb_guess, args=nb_args,
                          method='lm', tol=toler)
    # print(results_nb)
    if not results_nb.success:
        print(nb_guess)
        print(results_nb)
        print('r=', r)
        print('w=', w)
        print('BQ=', BQ)
        err_msg = ('ERROR in hh.get_cnb_vecs: Root finder did not ' +
                   'successfully find root in nb computation for ' +
                   'rp = ' + str(rp) + '.')
        raise ValueError(err_msg)
    ns_vec = results_nb.x[:rp]
    bsp1_vec = results_nb.x[rp:]
    bs_vec = np.append(b_init, bsp1_vec)
    n_errors = results_nb.fun[:rp]
    b_errors = results_nb.fun[rp:]
    cs_vec = get_cs_vec(bs_vec[:-1], bsp1_vec, ns_vec, r, w, BQ, p)

    return cs_vec, ns_vec, bs_vec, n_errors, b_errors


def get_cnb_paths(r_path, w_path, BQ_path, cnb_args):
    '''
    --------------------------------------------------------------------
    Given time series for r_t, w_t, and p_{m,t} solve for time series of
    all household decisions c_{s,t}, n_{s,t}, b_{s+1,t+1}, and c_{m,s,t}
    --------------------------------------------------------------------
    INPUTS:
    r_path   = (T2 + S,) vector, time series for interest rate r_t
    w_path   = (T2 + S,) vector, time series for wage w_t
    BQ_path  = (T2 + S,) matrix, time series for industry-specific
               prices p_{m,t}
    cnb_args = length 2 tuple, (ss_output, p)

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        get_n_errors()

    OBJECTS CREATED WITHIN FUNCTION:
    ss_output    =
    p            = parameters class object
    n_ss         = (S,) vector, steady-state individual labor supply
    c_s_path     = (S, T2 + S) matrix, ?
    n_s_path     = (S, T2 + S) matrix, ?
    b_s_path     = (S, T2 + S) matrix, ?
    c_ms_path    = (M, S, T2 + S) array, ?
    n_s_err_path = (S, T2 + S) matrix, ?
    b_s_err_path = (S, T2 + S) matrix, ?
    rp           = integer in [1, S-1], remaining periods in incomplete
                   lifetime individual's life

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: c_s_path, n_s_path, b_s_path, c_ms_path, n_s_err_path,
             b_s_err_path
    --------------------------------------------------------------------
    '''
    ss_output, p = cnb_args
    n_ss = ss_output['n_ss']
    b_ss = ss_output['b_ss']
    c_s_path = np.zeros((p.S, p.T2 + p.S))
    n_s_path = np.zeros((p.S, p.T2 + p.S))
    b_s_path = np.zeros((p.S + 1, p.T2 + p.S + 1))
    b_s_path[:, 0] = p.b_s0_vec
    n_s_err_path = np.zeros((p.S, p.T2 + p.S))
    b_s_err_path = np.zeros((p.S + 1, p.T2 + p.S + 1))

    # Solve the incomplete remaining lifetime decisions of agents alive
    # in period t=0 but not born in period t=0
    for rp in range(1, p.S):
        if rp == 1:
            # rp=1 individual only has an s=S labor supply decision n_S
            n_S0_guess = n_ss[-rp]
            b_Sp10_guess = b_ss[-rp]
            nb_guess = np.append(n_S0_guess, b_Sp10_guess)
            b_S0 = b_s_path[-(rp + 1), 0]
            r0 = r_path[0]
            w0 = w_path[0]
            BQ0 = BQ_path[0]
            nbS0_args = (b_S0, r0, w0, BQ0, p.rho_st[-1, 0],
                         p.TP_EulDif, p)
            results_nbS0 = opt.root(get_nb_errors, nb_guess,
                                    args=(nbS0_args), tol=p.TP_EulTol)
            if not results_nbS0.success:
                print(results_nbS0)
                err_msg = ('ERROR in hh.get_cnb_paths: Root finder ' +
                           'did not successfully find root in n_S0 ' +
                           'computation.')
                raise ValueError(err_msg)
            n_S0, b_Sp10 = results_nbS0.x
            n_s_path[-1, 0] = n_S0
            b_s_path[-1, 1] = b_Sp10
            nS0_err, bSp10_err = results_nbS0.fun
            n_s_err_path[-1, 0] = nS0_err
            b_s_err_path[-1, 1]
            c_S0 = get_cs_vec(b_S0, b_Sp10, n_S0, r0, w0, BQ0, p)
            c_s_path[-1, 0] = c_S0
            print('Solved incomplete lifetime rp=', rp)
        else:
            # 1<rp<S chooses b_{s+1} and n_s and has incomplete lives
            DiagMask = np.eye(rp, dtype=bool)
            n_s_guess = \
                np.hstack((n_ss[-rp], np.diag(n_s_path[-(rp - 1):,
                                                       :rp - 1])))
            b_sp1_guess = \
                np.hstack((b_ss[-rp], np.diag(b_s_path[-(rp - 1):,
                                                       1:rp])))
            nb_guess = np.hstack((n_s_guess, b_sp1_guess))
            b_1 = b_s_path[-(rp + 1), 0]
            rho_st = np.diag(p.rho_st[-rp:, :rp])
            c_s, n_s, b_s, n_errors, b_errors = \
                get_cnb_vecs(nb_guess, b_1, r_path[:rp], w_path[:rp],
                             BQ_path[:rp], rho_st, p.TP_EulDif, p,
                             p.TP_EulTol)
            n_s_path[-rp:, :rp] += DiagMask * n_s
            b_s_path[-rp:, 1:rp + 1] += DiagMask * b_s[1:]
            n_s_err_path[-rp:, :rp] += DiagMask * n_errors
            b_s_err_path[-rp:, 1:rp + 1] += DiagMask * b_errors
            c_s_path[-rp:, :rp] += DiagMask * c_s
            print('Solved incomplete lifetime rp=', rp)

    # Solve the complete remaining lifetime decisions of agents born
    # between period t=0 and t=T2
    # for t in range(p.T2 + 1):
    #     DiagMask = np.eye(p.S, dtype=bool)
    #     if t == 0:
    #         n_s_guess = np.hstack((n_ss[0],
    #                                np.diag(n_s_path[1:,
    #                                                 t:t + p.S - 1])))
    #     else:
    #         n_s_guess = np.diag(n_s_path[:, t - 1:t + p.S - 1])
    #     b_sp1_guess = np.diag(b_s_path[1:, t:t + p.S - 1])
    #     nb_guess = np.hstack((n_s_guess, b_sp1_guess))
    #     b_1 = 0.0
    #     c_s, n_s, b_sp1, n_errors, b_errors, c_ms = \
    #         get_cnb_vecs(nb_guess, b_1, r_path[t:t + p.S],
    #                      w_path[t:t + p.S], pm_path[:, t:t + p.S],
    #                      p.TP_EulDif, p, p.TP_EulTol)
    #     n_s_path[:, t:t + p.S] = \
    #         DiagMaskn * n_s + n_s_path[:, t:t + p.S]
    #     b_s_path[1:, t + 1:t + p.S] = (DiagMaskb * b_sp1 +
    #                                    b_s_path[1:, t + 1:t + p.S])
    #     n_s_err_path[:, t:t + p.S] = (DiagMaskn * n_errors +
    #                                   n_s_err_path[:, t:t + p.S])
    #     b_s_err_path[1:, t + 1:t + p.S] = (DiagMaskb * b_errors +
    #                                        b_s_err_path[1:,
    #                                                     t + 1:t + p.S])
    #     c_s_path[:, t:t + p.S] = DiagMaskn * c_s + c_s_path[:,
    #                                                         t:t + p.S]
    #     for m in range(p.M):
    #         c_ms_path[m, :, t:t + p.S] = (DiagMaskn * c_ms[m, :] +
    #                                       c_ms_path[m, :, t:t + p.S])
    #     # print('Solved complete lifetime t=', t)

    # return (c_s_path, n_s_path, b_s_path, c_ms_path, n_s_err_path,
    #         b_s_err_path)
