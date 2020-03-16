'''
------------------------------------------------------------------------
This module contains the functions used to solve the transition path
equilibrium using time path iteration (TPI) for the model with S-period
lived agents, endogenous labor supply, non-constant demographics,
bequests, and labor productivity growth.

This Python module imports the following module(s):
    aggregates.py
    firms.py
    households.py
    utilities.py

This Python module defines the following function(s):
    get_path()
    get_TP()
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
    xpath = (T,) vector, linear or parabolic xpath from x1 to xT

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
    ss_output = length 18 dictionary, steady-state output
    graphs    = boolean, =True if generate transition path equilibrium
                graphs

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        get_path()
        firms.get_wt()
        hh.get_cnb_paths()
        aggr.get_Lt()
        aggr.get_Kt()
        firms.get_rt()
        aggr.get_BQt()
        utils.print_time()
        firms.get_Yt()
        aggr.get_Ct()
        aggr.get_It()
        aggr.get_NXt()
        creat_graphs()

    OBJECTS CREATED WITHIN FUNCTION:
    start_time   = scalar, current processor time in seconds (float)
    r_ss         = scalar > -delta, steady-state interest rate
    BQ_ss        = scalar > 0, steady-state total bequests
    r_path_init  = (T2+S,) vector, initial guess of interest rate time
                   path
    BQ_path_init = (T2+S,) vector, initial guess of total bequests time
                   path
    r_0          = scalar > -delta, initial period interest rate guess
    BQ_0         = scalar > 0, initial period total bequests guess
    rBQpath_init = (2, T2+S) matrix, initial guess for time paths of
                   interest rate and total bequests
    iter_TPI     = integer >= 0, iteration number index for TPI
    dist         = scalar > 0, distance measure of
                   (rBQpath_new - rBQpath_init)
    cnb_args     = length 2 tuple, (ss_output, p) arguments passed to
                   hh.get_cnb_paths()
    w_path       = (T2+S,) vector, time path of wages
    cs_path      = (S, T2+S) matrix, time path of household consumption
                   by age
    ns_path      = (S, T2+S) matrix, time path of household labor supply
                   by age
    bs_path      = (S+1, T2+S+1) matrix, time path of household savings
                   and wealth by age
    ns_err_path  = (S, T2+S) matrix, time path of Euler errors by age
                   from household optimal labor supply decisions
    bs_err_path  = (S+1, T2+S+1) matrix, time path of Euler errors by
                   age from household optimal savings decisions
    L_path       = (T2+1,) vector, time path of aggregate labor
    K_path       = (T2+1,) vector, time path of aggregate capital stock
    r_path_new   = (T2+S,) vector, time path of interest rates implied
                   by household and firm optimization
    BQ_Path_new  = (T2+S,) vector, time path of total bequests implied
                   by household and firm optimization
    rBQpath_new  = (2, T2+S) matrix, time path of interest rates and
                   total bequests implied by household and firm
                   optimization
    tpi_time     = scalar, elapsed time for TPI computation
    r_path       = (T2+S,) vector, equilibrium interest rate time path
    BQ_path      = (T2+S,) vector, equilibrium total bequests time path
    Y_path       = (T2+1,) vector, equilibrium aggregate output time
                   path
    C_path       = (T2+1,) vector, equilibrium aggregate consumption
                   time path
    I_path       = (T2+1,) vector, equilibrium aggregate investment time
                   path
    NX_path      = (T2+1,) vector, equilibrium net exports time path
    tpi_output   = length 17 dictionary, tpi output objects {cs_path,
                   ns_path, bs_path, ns_err_path,  bs_err_path, r_path,
                   w_path,  BQ_path, K_path,  L_path,  Y_path, C_path,
                   I_path, NX_path, dist, iter_TPI, tpi_time}

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: tpi_output
    --------------------------------------------------------------------
    '''
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
            rBQpath_init = (p.xi_TP * rBQpath_new +
                            (1 - p.xi_TP) * rBQpath_init)

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
    tpi_output = length 17 dictionary, tpi output objects {cs_path,
                 ns_path, bs_path, ns_err_path,  bs_err_path, r_path,
                 w_path,  BQ_path, K_path,  L_path,  Y_path, C_path,
                 I_path, NX_path, dist, iter_TPI, tpi_time}
    p          = parameters class object

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    cur_path    = string, path name of current directory
    image_fldr  = string, folder in current path to save files
    image_dir   = string, total path of images folder
    output_path = string, path of file name of figure to be saved
    cs_path     = (S, T2+S) matrix, time path of household consumption
                  by age
    ns_path     = (S, T2+S) matrix, time path of household labor supply
                  by age
    bs_path     = (S+1, T2+S+1) matrix, time path of household savings
                  and wealth by age
    r_path      = (T2+S,) vector, equilibrium interest rate time path
    w_path      = (T2+S,) vector, equilibrium wage time path
    BQ_path     = (T2+S) vector, equilibrium total bequests time path
    K_path      = (T2+1,) vector, time path of aggregate capital stock
    L_path      = (T2+1,) vector, time path of aggregate labor
    Y_path      = (T2+1,) vector, time path of aggregate output
    C_path      = (T2+1,) vector, time path of aggregate consumption
    I_path      = (T2+1,) vector, time path of aggregate investment
    NX_path     = (T2+1,) vector, time path of net exports
    tvec        = (T2+1,) vector, vector of time periods to be plotted
    sgrid       = (S,) vector, all ages from 21 to 100
    tmat        = (S, T2+1) matrix, time periods for decisions ages
                  (S) and time periods (T2+1)
    smat        = (S, T2+1) matrix, ages for all decisions ages (S)
                  and time periods (T2+1)
    sgrid_b     = (S+1,) vector, all ages from 21 to 101
    tmat_b      = (S+1, T2+1) matrix, time periods for decisions ages
                  (S+1) and time periods (T2+1)
    smat_b      = (S+1, T2+1) matrix, ages for all decisions ages (S+1)
                  and time periods (T2+1)

    FILES CREATED BY THIS FUNCTION:
        TP_r_path.png
        TP_w_path.png
        TP_BQ_path.png
        TP_K_path.png
        TP_L_path.png
        TP_Y_path.png
        TP_C_path.png
        TP_I_path.png
        TP_NX_path.png
        TP_cs_path.png
        TP_ns_path.png
        TP_bs_path.png

    RETURNS: None
    --------------------------------------------------------------------
    '''
    # Create directory if images directory does not already exist
    cur_path = os.path.split(os.path.abspath(__file__))[0]
    image_fldr = 'OUTPUT/TP/images'
    image_dir = os.path.join(cur_path, image_fldr)
    if not os.access(image_dir, os.F_OK):
        os.makedirs(image_dir)

    # Unpack time path equilibrium objects to be plotted
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

    # Plot time path of interest rate
    tvec = np.arange(0, p.T2 + 1)
    minorLocator = MultipleLocator(1)
    fig, ax = plt.subplots()
    plt.plot(tvec, r_path[:p.T2 + 1], label=r'$r_t$')
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
    plt.plot(tvec, w_path[:p.T2 + 1], label=r'$\hat{w}_t$')
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
    plt.plot(tvec, BQ_path[:p.T2 + 1], label=r'$\hat{BQ}_t$')
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
    sgrid = np.arange(p.E + 1, p.E + p.S + 1)
    tmat, smat = np.meshgrid(tvec, sgrid)
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
    sgrid_b = np.arange(p.E + 1, p.E + p.S + 2)
    tmat_b, smat_b = np.meshgrid(tvec, sgrid_b)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel(r'period-$t$')
    ax.set_ylabel(r'age-$s$')
    ax.set_zlabel(r'individual savings $b_{s,t}$')
    strideval = max(int(1), int(round(p.S / 10)))
    ax.plot_surface(tmat_b, smat_b, bs_path[:, :p.T2 + 1],
                    rstride=strideval, cstride=strideval, cmap=cmap_c)
    output_path = os.path.join(image_dir, 'TP_bs_path')
    plt.savefig(output_path)
    # plt.show()
    plt.close()
