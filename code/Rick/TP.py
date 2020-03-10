'''
------------------------------------------------------------------------
This module contains the functions used to solve the transition path
equilibrium using time path iteration (TPI) for the model with S-period
lived agents and endogenous labor supply from Chapter 4 of the OG
textbook.

This Python module imports the following module(s):
    aggregates.py
    firms.py
    households.py
    utilities.py

This Python module defines the following function(s):
    get_path()
    inner_loop()
    get_TPI()
    create_graphs()
------------------------------------------------------------------------
'''
# Import Packages
import time
import numpy as np
import aggregates as aggr
import households as hh
import firms
import utilities as utils
import scipy.optimize as opt
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import os

'''
------------------------------------------------------------------------
    Functions
------------------------------------------------------------------------
'''


def get_path(x1, xT, T, spec):
    '''
    --------------------------------------------------------------------
    This function generates a path from point x1 to point xT such that
    that the path x is a linear or quadratic function of time t.

        linear:    x = d*t + e
        quadratic: x = a*t^2 + b*t + c

    The identifying assumptions for quadratic are the following:

        (1) x1 is the value at time t=0: x1 = c
        (2) xT is the value at time t=T-1: xT = a*(T-1)^2 + b*(T-1) + c
        (3) the slope of the path at t=T-1 is 0: 0 = 2*a*(T-1) + b
    --------------------------------------------------------------------
    INPUTS:
    x1 = scalar, initial value of the function x(t) at t=0
    xT = scalar, value of the function x(t) at t=T-1
    T  = integer >= 3, number of periods of the path
    spec = string, "linear" or "quadratic"

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    cc    = scalar, constant coefficient in quadratic function
    bb    = scalar, coefficient on t in quadratic function
    aa    = scalar, coefficient on t^2 in quadratic function
    xpath = (T,) vector, parabolic xpath from x1 to xT

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: xpath
    --------------------------------------------------------------------
    '''
    if spec == "linear":
        xpath = np.linspace(x1, xT, T)
    elif spec == "quadratic":
        cc = x1
        bb = 2 * (xT - x1) / (T - 1)
        aa = (x1 - xT) / ((T - 1) ** 2)
        xpath = (aa * (np.arange(0, T) ** 2) + (bb * np.arange(0, T)) +
                 cc)

    return xpath


def get_TP(p, ss_output, graphs):
    '''
    --------------------------------------------------------------------
    Solves for transition path equilibrium using time path iteration
    (TPI)
    --------------------------------------------------------------------
    INPUTS:
    p         = parameters class object
    ss_output = dict, steady-state output
    graphs    = boolean, =True if generate transition path equilibrium
                graphs

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        aggr.get_K()
        get_path()
        firms.get_r()
        firms.get_w()
        inner_loop()
        solve_bn_path()
        aggr.get_L()
        aggr.get_Y()
        aggr.get_C()
        utils.print_time()

    OBJECTS CREATED WITHIN FUNCTION:
    start_time        = scalar, current processor time in seconds (float)
    r_ss              = scalar > 0, steady-state interest rate
    w_ss              = scalar > 0, steady-state wage
    Km_ss             = (M,) vector, steady-state capital across
                        industries
    rpath_init        = (T2+S,) vector, initial guess of time path of
                        the interest rate
    wpath_init        = (T2+S,) vector, initial guess of time path of
                        the wage
    r0                = scalar > 0,
    w0                = scalar > 0,
    rwpath_init       = (2 * (T2 + S),) vector,
    iter_TPI          = integer >= 0,
    dist              =
    cnb_args          = length 2 tuple,
    KLrat_path        = (M, T2+S) matrix,
    pm_path           = (M, T2+S) matrix,
    c_s_path          = (S, T2+S) matrix,
    n_s_path          = (S, T2+S) matrix,
    b_s_path          = (S, T2+S) matrix,
    c_ms_path         = (M, S, T2+S) array,
    n_s_err_path      = (S, T2+S) matrix,
    b_s_err_path      = (S, T2+S) matrix,
    C_m_path          = (M, T2+1) matrix,
    K_m_path          = (M, T2+1) matrix,
    L_m_path          = (M, T2+1) matrix,
    t_ind             = integer >= 0,
    C_mtm1            = (M,) vector,
    K_mtm1            = (M,) vector,
    L_mtm1            = (M,) vector,
    K_dem_path        = (T2+1,) vector,
    K_sup_path        = (T2+1,) vector,
    L_dem_path        = (T2+1,) vector,
    L_sup_path        = (T2+1,) vector,
    KL_dem_path       = (2 * (T2 + 1),) vector,
    KL_sup_path       = (2 * (T2 + 1),) vector,
    K_dem_path_pctdif = (T2+1,) vector,
    L_dem_path_pctdif = (T2+1,) vector,
    rpath_new         =
    wpath_new         =

    S             = integer in [3,80], number of periods an individual
                    lives
    T1            = integer > S, number of time periods until steady
                    state is assumed to be reached
    T2            = integer > T1, number of time periods after which
                    steady-state is forced in TPI
    beta          = scalar in (0,1), discount factor for model period
    sigma         = scalar > 0, coefficient of relative risk aversion
    l_tilde       = scalar > 0, time endowment for each agent each
                    period
    b_ellip       = scalar > 0, fitted value of b for elliptical
                    disutility of labor
    upsilon       = scalar > 1, fitted value of upsilon for elliptical
                    disutility of labor
    chi_n_vec     = (S,) vector, values for chi^n_s
    A             = scalar > 0, total factor productivity parameter in
                    firms' production function
    alpha         = scalar in (0,1), capital share of income
    delta         = scalar in [0,1], per-period capital depreciation rt
    r_ss          = scalar > 0, steady-state aggregate interest rate
    K_ss          = scalar > 0, steady-state aggregate capital stock
    L_ss          = scalar > 0, steady-state aggregate labor
    C_ss          = scalar > 0, steady-state aggregate consumption
    b_ss          = (S,) vector, steady-state savings distribution
                    (b1, b2,... bS)
    n_ss          = (S,) vector, steady-state labor supply distribution
                    (n1, n2,... nS)
    maxiter       = integer >= 1, Maximum number of iterations for TPI
    mindist       = scalar > 0, convergence criterion for TPI
    TPI_tol       = scalar > 0, tolerance level for TPI root finders
    xi            = scalar in (0,1], TPI path updating parameter
    diff          = Boolean, =True if want difference version of Euler
                    errors beta*(1+r)*u'(c2) - u'(c1), =False if want
                    ratio version [beta*(1+r)*u'(c2)]/[u'(c1)] - 1
    K1            = scalar > 0, initial aggregate capital stock
    K1_cnstr      = Boolean, =True if K1 <= 0
    rpath_init    = (T2+S-1,) vector, initial guess for the time path of
                    interest rates
    iter_TPI      = integer >= 0, current iteration of TPI
    dist          = scalar >= 0, distance measure between initial and
                    new paths
    rw_params     = length 3 tuple, (A, alpha, delta)
    Y_params      = length 2 tuple, (A, alpha)
    cnb_params    = length 11 tuple, args to pass into inner_loop()
    rpath         = (T2+S-1,) vector, time path of the interest rates
    wpath         = (T2+S-1,) vector, time path of the wages
    ind           = (S,) vector, integers from 0 to S
    bn_args       = length 14 tuple, arguments to be passed to
                    solve_bn_path()
    cpath         = (S, T2+S-1) matrix, time path of distribution of
                    individual consumption c_{s,t}
    npath         = (S, T2+S-1) matrix, time path of distribution of
                    individual labor supply n_{s,t}
    bpath         = (S, T2+S-1) matrix, time path of distribution of
                    individual savings b_{s,t}
    n_err_path    = (S, T2+S-1) matrix, time path of distribution of
                    individual labor supply Euler errors
    b_err_path    = (S, T2+S-1) matrix, time path of distribution of
                    individual savings Euler errors. First column and
                    first row are identically zero
    bSp1_err_path = (S, T2) matrix, residual last period savings, which
                    should be close to zero in equilibrium. Nonzero
                    elements of matrix should only be in first column
                    and first row
    Kpath_new     = (T2+S-1,) vector, new path of the aggregate capital
                    stock implied by household and firm optimization
    Kpath_cnstr   = (T2+S-1,) Boolean vector, =True if K_t<=0
    Lpath_new     = (T2+S-1,) vector, new path of the aggregate labor
    rpath_new     = (T2+S-1,) vector, updated time path of interest rate
    wpath_new     = (T2+S-1,) vector, updated time path of the wages
    Ypath         = (T2+S-1,) vector, equilibrium time path of aggregate
                    output (GDP) Y_t
    Cpath         = (T2+S-1,) vector, equilibrium time path of aggregate
                    consumption C_t
    RCerrPath     = (T2+S-2,) vector, equilibrium time path of the
                    resource constraint error:
                    Y_t - C_t - K_{t+1} + (1-delta)*K_t
    Kpath         = (T2+S-1,) vector, equilibrium time path of aggregate
                    capital stock K_t
    Lpath         = (T2+S-1,) vector, equilibrium time path of aggregate
                    labor L_t
    tpi_time      = scalar, time to compute TPI solution (seconds)
    tpi_output    = length 14 dictionary, {cpath, npath, bpath, wpath,
                    rpath, Kpath, Lpath, Ypath, Cpath, bSp1_err_path,
                    n_err_path, b_err_path, RCerrPath, tpi_time}

    FILES CREATED BY THIS FUNCTION:
        Kpath.png
        Lpath.png
        Ypath.png
        C_aggr_path.png
        wpath.png
        rpath.png
        cpath.png
        npath.png
        bpath.png

    RETURNS: tpi_output
    --------------------------------------------------------------------
    '''
    # Create directory if images directory does not already exist
    cur_path = os.path.split(os.path.abspath(__file__))[0]
    output_fldr = 'OUTPUT'
    output_dir = os.path.join(cur_path, output_fldr)
    if not os.access(output_dir, os.F_OK):
        os.makedirs(output_dir)
    # Excel_header_str = \
    #     ('Variables are,r_path_nn_init,w_path_nn_init,Y1path_init,' +
    #      'Y2path_init,Y3path_init,KLrat1_path,KLrat2_path,' +
    #      'KLrat3_path,K1_path,K2_path,K3_path,L1_path,L2_path,' +
    #      'L3_path,p1_path_nn,p2_path_nn,p3_path_nn,p_path_nn,' +
    #      'p1_path_norm,p2_path_norm,p3_path_norm,rpath_norm,' +
    #      'wpath_norm,C1_path,C2_path,C3_path,K_dem_path,K_sup_path,' +
    #      'eps_r_path,L_dem_path,L_sup_path,eps_w_path,Y1_dem_path,' +
    #      'Y2_dem_path,Y3_dem_path,Y1_sup_path,Y2_sup_path,' +
    #      'Y3_sup_path,eps_Y1_path,eps_Y2_path,eps_Y3_path,dist,' +
    #      'r_path_nn_new,w_path_nn_new,Y1path_new,Y2path_new,Y3path_new')

    start_time = time.clock()

    # Unpack steady-state objects to be used in this algorithm
    r_ss = ss_output['r_ss']
    BQ_ss = ss_output['BQ_ss']

    # Create initial time paths for r, w
    r_path_init = np.zeros(p.T2 + p.S)
    BQ_path_init = np.zeros(p.T2 + p.S)
    # pct_r = 1 + p.xi_TP * ((Km_ss.sum() - p.b_s0_vec.sum()) /
    #                        p.b_s0_vec.sum())
    r_0 = r_ss  # pct_r * r_ss
    BQ_0 = BQ_ss  # (2 - pct_r) * w_ss
    r_path_init[:p.T2 + 1] = get_path(r_0, r_ss, p.T2 + 1,
                                      'quadratic')
    r_path_init[p.T2 + 1:] = r_ss
    BQ_path_init[:p.T2 + 1] = get_path(BQ_0, BQ_ss, p.T2 + 1,
                                       'quadratic')
    BQ_path_init[p.T2 + 1:] = BQ_ss
    # print('rpath_init=', rpath_init)
    # print('BQpath_init=', BQpath_init)
    rBQpath_init = np.zeros((2, p.T2 + p.S))
    rBQpath_init[0, :] = r_path_init
    rBQpath_init[1, :] = BQ_path_init
    # raise ValueError('Pause program')
    iter_TPI = int(0)
    dist = 10.0
    cnb_args = (ss_output, p)
    while (iter_TPI < p.maxiter_TP) and (dist > p.TP_OutTol):
        iter_TPI += 1

        r_path_init = rBQpath_init[0, :]
        BQ_path_init = rBQpath_init[1, :]
        # Solve for time path of w_t given r_t
        w_path = firms.get_wt(r_path_init, p)
        # Solve for time path of household optimal decisions n_{s,t}
        # and b_{s+1,t+1} given prices r_t and w_t and total bequests
        # BQ_t
        (cs_path, ns_path, bs_path, ns_err_path, bs_err_path) = \
            hh.get_cnb_paths(r_path_init, w_path, BQ_path_init,
                             cnb_args)
        # Solve for time paths of aggregate capital K_t and aggregate
        # labor L_t
        L_path = aggr.get_Lt(ns_path[:, :p.T2 + 1], p)
        K_path = aggr.get_Kt(bs_path[1:, :p.T2 + 1], p)
        # Solve for new time paths of r_t^{i+1} and total bequests
        # BQ_t^{i+1}
        r_path_new = np.zeros(p.T2 + p.S)
        r_path_new[:p.T2 + 1] = firms.get_rt(K_path, L_path, p)
        r_path_new[p.T2 + 1:] = r_path_init[p.T2 + 1:]
        BQ_path_new = np.zeros(p.T2 + p.S)
        BQ_path_new[:p.T2 + 1] = aggr.get_BQt(bs_path[1:, :p.T2 + 1],
                                              r_path_init[:p.T2 + 1], p)
        BQ_path_new[p.T2 + 1:] = BQ_path_init[p.T2 + 1:]
        # Calculate distance measure between (r^{i+1},BQ^{i+1}) and
        # (r^i,BQ^i)
        rBQpath_new = np.vstack((r_path_new.reshape((1, p.T2 + p.S)),
                                 BQ_path_new.reshape((1, p.T2 + p.S))))
        dist = np.absolute(rBQpath_new - rBQpath_init).max()
        print(
            'TPI iter: ', iter_TPI, ', dist: ', "%10.4e" % (dist),
            ', max abs all errs: ', "%10.4e" %
            (np.absolute(np.hstack((bs_err_path.max(axis=0),
             ns_err_path.max(axis=0)))).max()))
        if dist > p.TP_OutTol:
            rBQpath_init = rBQpath_new
        # new_data = \
        #     np.vstack((rpath_nn_init[:p.T2 + 1].reshape((1, p.T2 + 1)),
        #                wpath_nn_init[:p.T2 + 1].reshape((1, p.T2 + 1)),
        #                Ympath_init[:, :p.T2 + 1],
        #                KLratm_path[:, :p.T2 + 1], Km_path[:, :p.T2 + 1],
        #                Lm_path[:, :p.T2 + 1], pm_path_nn[:, :p.T2 + 1],
        #                p_path_nn[:p.T2 + 1].reshape((1, p.T2 + 1)),
        #                pm_path_norm[:, :p.T2 + 1],
        #                rpath_norm[:p.T2 + 1].reshape((1, p.T2 + 1)),
        #                wpath_norm[:p.T2 + 1].reshape((1, p.T2 + 1)),
        #                Cm_path, K_dem_path.reshape((1, p.T2 + 1)),
        #                K_sup_path.reshape((1, p.T2 + 1)),
        #                eps_r_path.reshape((1, p.T2 + 1)),
        #                L_dem_path.reshape((1, p.T2 + 1)),
        #                L_sup_path.reshape((1, p.T2 + 1)),
        #                eps_w_path.reshape((1, p.T2 + 1)),
        #                Ym_dem_path, Ym_sup_path, eps_Ym_path,
        #                dist * np.ones((1, p.T2 + 1)),
        #                rwYmpath_new[:, :p.T2 + 1]))
        # if iter_TPI == 1:
        #     variables, periods = new_data.shape
        #     data_by_iter = new_data.reshape((variables, periods, 1))
        # else:
        #     data_by_iter = \
        #         np.append(data_by_iter,
        #                   new_data.reshape((variables, periods, 1)),
        #                   axis=2)
        # Excel_filename = 'Excel_output' + str(iter_TPI) + '.csv'
        # Excel_path = os.path.join(output_dir, Excel_filename)
        # np.savetxt(Excel_path, new_data, delimiter=',',
        #            header=Excel_header_str)

    tpi_time = time.clock() - start_time
    # Print TPI computation time
    utils.print_time(tpi_time, 'TPI')

    if (iter_TPI == p.maxiter_TP) and (dist > p.TP_OutTol):
        print('TPI reached maxiter and did not converge.')
    elif (iter_TPI == p.maxiter_TP) and (dist <= p.TP_OutTol):
        print('TPI converged in the last iteration. ' +
              'Should probably increase maxiter_TP.')
    elif (iter_TPI < p.maxiter_TP) and (dist <= p.TP_OutTol):
        print('TPI SUCCESS: Converged on iteration', iter_TPI, '.')

    r_path = r_path_init
    BQ_path = BQ_path_init
    # Solve for equilibrium time paths of aggregate output, consumption,
    # investment, and net exports
    Y_path = firms.get_Yt(K_path, L_path, p)
    C_path = aggr.get_Ct(cs_path[:, :p.T2 + 1], p)
    I_path = aggr.get_It(K_path, p)
    NX_path = aggr.get_NXt(bs_path[1:, :p.T2 + 1], p)

    # Create TPI output dictionary
    tpi_output = {
        'cs_path': cs_path, 'ns_path': ns_path, 'bs_path': bs_path,
        'ns_err_path': ns_err_path, 'bs_err_path': bs_err_path,
        'r_path': r_path, 'w_path': w_path, 'BQ_path': BQ_path,
        'K_path': K_path, 'L_path': L_path, 'Y_path': Y_path,
        'C_path': C_path, 'I_path': I_path, 'NX_path': NX_path,
        'dist': dist, 'iter_TPI': iter_TPI, 'tpi_time': tpi_time}

    if graphs:
        create_graphs(tpi_output, p)

    return tpi_output


def create_graphs(tpi_output, p):
    '''
    --------------------------------------------------------------------
    Plot equilibrium time path results
    --------------------------------------------------------------------
    INPUTS:
    tpi_output = length 13 dict, equilibrium time paths and computation
                 time from TPI computation
    args       = length 2 tuple, (S, T2)

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:

    OBJECTS CREATED WITHIN FUNCTION:
    S           =
    T2          =
    Kpath       =
    Lpath       =
    rpath       =
    wpath       =
    Ypath       =
    Cpath       =
    cpath       =
    npath       =
    bpath       =
    cur_path    = string, path name of current directory
    output_fldr = string, folder in current path to save files
    output_dir  = string, total path of images folder
    output_path = string, path of file name of figure to be saved
    tvec        = (T2+S-1,) vector, time period vector
    tgridT      = (T2,) vector, time period vector from 1 to T2
    sgrid       = (S,) vector, all ages from 1 to S
    tmat        = (S, T2) matrix, time periods for decisions ages
                  (S) and time periods (T2)
    smat        = (S, T2) matrix, ages for all decisions ages (S)
                  and time periods (T2)
    cmap_c      =
    cmap_n      =
    bpath_full  =
    sgrid_b     =
    tmat_b      =
    smat_b      =
    cmap_b      =

    FILES CREATED BY THIS FUNCTION:
        Kpath.png
        Lpath.png
        rpath.png
        wpath.png
        Ypath.png
        C_aggr_path.png
        cpath.png
        npath.png
        bpath.png

    RETURNS: None
    --------------------------------------------------------------------
    '''
    cs_path = tpi_output['cs_path']
    ns_path = tpi_output['ns_path']
    bs_path = tpi_output['bs_path']
    r_path = tpi_output['r_path']
    w_path = tpi_output['w_path']
    BQ_path = tpi_output['BQ_path']
    K_path = tpi_output['K_path']
    L_path = tpi_output['L_path']
    Y_path = tpi_output['Y_path']
    C_path = tpi_output['C_path']
    I_path = tpi_output['I_path']
    NX_path = tpi_output['NX_path']

    # Create directory if images directory does not already exist
    cur_path = os.path.split(os.path.abspath(__file__))[0]
    image_fldr = 'OUTPUT/TP/images'
    image_dir = os.path.join(cur_path, image_fldr)
    if not os.access(image_dir, os.F_OK):
        os.makedirs(image_dir)

    # Plot time path of interest rate
    tvec = np.arange(0, p.T2 + 1)
    minorLocator = MultipleLocator(1)
    fig, ax = plt.subplots()
    plt.plot(tvec, r_path, label=r'$r_t$')
    # for the minor ticks, use no labels; default NullFormatter
    ax.xaxis.set_minor_locator(minorLocator)
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    plt.title(r'Time path of interest rate $r_t$')
    plt.xlabel(r'Period $t$')
    plt.ylabel(r'Interest rate $r_t$')
    output_path = os.path.join(image_dir, 'TP_r_path')
    plt.savefig(output_path)
    # plt.show()
    plt.close()

    # Plot time path of wage
    minorLocator = MultipleLocator(1)
    fig, ax = plt.subplots()
    plt.plot(tvec, w_path, label=r'$\hat{w}_t$')
    # for the minor ticks, use no labels; default NullFormatter
    ax.xaxis.set_minor_locator(minorLocator)
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    plt.title(r'Time path of wage $\hat{w}_t$')
    plt.xlabel(r'Period $t$')
    plt.ylabel(r'Wage $\hat{w}_t$')
    output_path = os.path.join(image_dir, 'TP_w_path')
    plt.savefig(output_path)
    # plt.show()
    plt.close()

    # Plot time path of total bequests
    minorLocator = MultipleLocator(1)
    fig, ax = plt.subplots()
    plt.plot(tvec, BQ_path, label=r'$\hat{BQ}_t$')
    # for the minor ticks, use no labels; default NullFormatter
    ax.xaxis.set_minor_locator(minorLocator)
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    plt.title(r'Time path of total bequests $\hat{BQ}_t$')
    plt.xlabel(r'Period $t$')
    plt.ylabel(r'Total bequests $\hat{BQ}_t$')
    output_path = os.path.join(image_dir, 'TP_BQ_path')
    plt.savefig(output_path)
    # plt.show()
    plt.close()

    # Plot time path of aggregate capital
    minorLocator = MultipleLocator(1)
    fig, ax = plt.subplots()
    plt.plot(tvec, K_path, label=r'$\hat{K}_t$')
    # for the minor ticks, use no labels; default NullFormatter
    ax.xaxis.set_minor_locator(minorLocator)
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    plt.title(r'Time path of aggregate capital $\hat{K}_t$')
    plt.xlabel(r'Period $t$')
    plt.ylabel(r'Aggregate capital $\hat{K}_t$')
    output_path = os.path.join(image_dir, 'TP_K_path')
    plt.savefig(output_path)
    # plt.show()
    plt.close()

    # Plot time path of aggregate labor
    minorLocator = MultipleLocator(1)
    fig, ax = plt.subplots()
    plt.plot(tvec, L_path, label=r'$\hat{L}_t$')
    # for the minor ticks, use no labels; default NullFormatter
    ax.xaxis.set_minor_locator(minorLocator)
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    plt.title(r'Time path of aggregate labor $\hat{L}_t$')
    plt.xlabel(r'Period $t$')
    plt.ylabel(r'Aggregate labor $\hat{L}_t$')
    output_path = os.path.join(image_dir, 'TP_L_path')
    plt.savefig(output_path)
    # plt.show()
    plt.close()

    # Plot time path of aggregate output
    minorLocator = MultipleLocator(1)
    fig, ax = plt.subplots()
    plt.plot(tvec, Y_path, label=r'$\hat{Y}_t$')
    # for the minor ticks, use no labels; default NullFormatter
    ax.xaxis.set_minor_locator(minorLocator)
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    plt.title(r'Time path of aggregate output $\hat{Y}_t$')
    plt.xlabel(r'Period $t$')
    plt.ylabel(r'Aggregate output $\hat{Y}_t$')
    output_path = os.path.join(image_dir, 'TP_Y_path')
    plt.savefig(output_path)
    # plt.show()
    plt.close()

    # Plot time path of aggregate consumption
    minorLocator = MultipleLocator(1)
    fig, ax = plt.subplots()
    plt.plot(tvec, C_path, label=r'$\hat{C}_t$')
    # for the minor ticks, use no labels; default NullFormatter
    ax.xaxis.set_minor_locator(minorLocator)
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    plt.title(r'Time path of aggregate consumption $\hat{C}_t$')
    plt.xlabel(r'Period $t$')
    plt.ylabel(r'Aggregate consumption $\hat{C}_t$')
    output_path = os.path.join(image_dir, 'TP_C_path')
    plt.savefig(output_path)
    # plt.show()
    plt.close()

    # Plot time path of aggregate investment
    minorLocator = MultipleLocator(1)
    fig, ax = plt.subplots()
    plt.plot(tvec, I_path, label=r'$\hat{I}_t$')
    # for the minor ticks, use no labels; default NullFormatter
    ax.xaxis.set_minor_locator(minorLocator)
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    plt.title(r'Time path of aggregate investment $\hat{I}_t$')
    plt.xlabel(r'Period $t$')
    plt.ylabel(r'Aggregate investment $\hat{I}_t$')
    output_path = os.path.join(image_dir, 'TP_I_path')
    plt.savefig(output_path)
    # plt.show()
    plt.close()

    # Plot time path of net exports
    minorLocator = MultipleLocator(1)
    fig, ax = plt.subplots()
    plt.plot(tvec, NX_path, label=r'$\hat{NX}_t$')
    # for the minor ticks, use no labels; default NullFormatter
    ax.xaxis.set_minor_locator(minorLocator)
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    plt.title(r'Time path of net exports $\hat{NX}_t$')
    plt.xlabel(r'Period $t$')
    plt.ylabel(r'Net exports $\hat{NX}_t$')
    output_path = os.path.join(image_dir, 'TP_NX_path')
    plt.savefig(output_path)
    # plt.show()
    plt.close()

    # Plot time path of individual consumption distribution
    tgridT = np.arange(0, p.T2 + 1)
    sgrid = np.arange(p.E + 1, p.E + p.S + 1)
    tmat, smat = np.meshgrid(tgridT, sgrid)
    cmap_c = cm.get_cmap('summer')
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel(r'period-$t$')
    ax.set_ylabel(r'age-$s$')
    ax.set_zlabel(r'individual consumption $\hat{c}_{s,t}$')
    strideval = max(int(1), int(round(p.S / 10)))
    ax.plot_surface(tmat, smat, cs_path[:, :p.T2 + 1],
                    rstride=strideval, cstride=strideval, cmap=cmap_c)
    output_path = os.path.join(image_dir, 'TP_cs_path')
    plt.savefig(output_path)
    # plt.show()
    plt.close()

    # Plot time path of individual labor supply distribution
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel(r'period-$t$')
    ax.set_ylabel(r'age-$s$')
    ax.set_zlabel(r'individual labor supply $n_{s,t}$')
    strideval = max(int(1), int(round(p.S / 10)))
    ax.plot_surface(tmat, smat, ns_path[:, :p.T2 + 1],
                    rstride=strideval, cstride=strideval, cmap=cmap_c)
    output_path = os.path.join(image_dir, 'TP_ns_path')
    plt.savefig(output_path)
    # plt.show()
    plt.close()

    # Plot time path of individual savings distribution
    sgrid2 = np.arange(p.E + 1, p.E + p.S + 2)
    tmat2, smat2 = np.meshgrid(tgridT, sgrid2)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel(r'period-$t$')
    ax.set_ylabel(r'age-$s$')
    ax.set_zlabel(r'individual labor supply $n_{s,t}$')
    strideval = max(int(1), int(round(p.S / 10)))
    ax.plot_surface(tmat2, smat2, bs_path[:, :p.T2 + 1],
                    rstride=strideval, cstride=strideval, cmap=cmap_c)
    output_path = os.path.join(image_dir, 'TP_bs_path')
    plt.savefig(output_path)
    # plt.show()
    plt.close()
