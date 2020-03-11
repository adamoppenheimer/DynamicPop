'''
------------------------------------------------------------------------
This program runs the steady state solver as well as the time path
iteration solution for the model with S-period lived agents, endogenous
labor supply and multiple static industries from from Chapter 17 of the
OG textbook.

This Python script imports the following module(s):
    MltIndStatLab_params.py
    SS.py
    TPI.py
    aggregates.py
    elliputil.py
    utilities.py

This Python script calls the following function(s):
    elp.fit_ellip_CFE()
    ss.get_SS()
    utils.compare_args()
    aggr.get_K()
    tpi.get_TPI()

Files created by this script:
    OUTPUT/SS/ss_vars.pkl
    OUTPUT/SS/ss_args.pkl
    OUTPUT/TPI/tpi_vars.pkl
    OUTPUT/TPI/tpi_args.pkl
------------------------------------------------------------------------
'''
# Import packages
import numpy as np
import pickle
import os
# import multiprocessing
# from dask import delayed
# from dask.distributed import Client
import parameters as params
import SS as ss
import TP as tp
# import aggregates as aggr
# import utilities as utils

'''
------------------------------------------------------------------------
Import parameters
------------------------------------------------------------------------
'''
p = params.parameters()

'''
------------------------------------------------------------------------
Set up parallel processing
------------------------------------------------------------------------
'''
# max_cores = multiprocessing.cpu_count()
# print('Cores available on this machine =', max_cores)
# num_workers = min(max_cores, p.S)
# print('Number of workers =', num_workers)
# client = Client(processes=False)

'''
------------------------------------------------------------------------
Solve for the steady-state solution
------------------------------------------------------------------------
cur_path       = string, current file path of this script
ss_output_fldr = string, cur_path extension of SS output folder path
ss_output_dir  = string, full path name of SS output folder
ss_outputfile  = string, path name of file for SS output objects
ss_paramsfile  = string, path name of file for SS parameter objects
ss_args        = length 15 tuple, arguments to pass in to ss.get_SS()
rss_init       = scalar > -delta, initial guess for r_ss
c1_init        = scalar > 0, initial guess for c1
init_vals      = length 2 tuple, initial guesses (r, c1) to be passed
                 in to ss.get_SS
ss_output      = length 14 dict, steady-state objects {n_ss, b_ss, c_ss,
                 b_Sp1_ss, w_ss, r_ss, K_ss, L_ss, Y_ss, C_ss, n_err_ss,
                 b_err_ss, RCerr_ss, ss_time}
ss_vars_exst   = boolean, =True if ss_vars.pkl exists
ss_args_exst   = boolean, =True if ss_args.pkl exists
err_msg        = string, error message
prev_ss_args   = length 12 tuple, previous arguments used to produce
                 saved steady-state output
args_same      = boolean, =True if ss_args == prev_ss_args
------------------------------------------------------------------------
'''
# Create OUTPUT/SS directory if does not already exist
cur_path = os.path.split(os.path.abspath(__file__))[0]
ss_output_fldr = 'OUTPUT/SS'
ss_output_dir = os.path.join(cur_path, ss_output_fldr)
if not os.access(ss_output_dir, os.F_OK):
    os.makedirs(ss_output_dir)
ss_outputfile = os.path.join(ss_output_dir, 'ss_vars.pkl')
ss_paramsfile = os.path.join(ss_output_dir, 'ss_args.pkl')

# Compute steady-state solution

if p.SS_solve:
    print('BEGIN EQUILIBRIUM STEADY-STATE COMPUTATION')
    rss_init = 0.13
    BQss_init = 0.03
    init_vals = np.array([rss_init, BQss_init])

    print('Solving SS outer loop using root finder method on r and BQ.')
    ss_output = ss.get_SS(init_vals, p, p.SS_graphs)

    # Save ss_output as pickle
    pickle.dump(ss_output, open(ss_outputfile, 'wb'))
    pickle.dump(p, open(ss_paramsfile, 'wb'))

# Don't compute steady-state, get it from pickle
else:
    # Make sure that the SS output files exist
    ss_vars_exst = os.path.exists(ss_outputfile)
    ss_args_exst = os.path.exists(ss_paramsfile)
    if (not ss_vars_exst) or (not ss_args_exst):
        # If the files don't exist, stop the program and run the steady-
        # state solution first
        err_msg = ('ERROR: The SS output files do not exist and ' +
                   'SS_solve=False. Must set SS_solve=True and ' +
                   'compute steady-state solution.')
        raise ValueError(err_msg)
    else:
        # If the files do exist, make sure that none of the parameters
        # changed from the parameters used in the solution for the saved
        # steady-state pickle
        prev_p = pickle.load(open(ss_paramsfile, 'rb'))
        prev_p_dict = prev_p.__dict__
        p_dict = p.__dict__
        keys_to_check = \
            ['S', 'yrs_in_per', 'T1', 'T2', 'beta_an', 'beta', 'sigma',
             'l_tilde', 'chi_n_vec', 'Frisch_elast', 'CFE_scale',
             'b_ellip', 'upsilon', 'A', 'alpha', 'delta_an',
             'delta', 'g_y', 'E', 'min_yr', 'max_yr', 'curr_year',
             'rho_ss', 'i_ss', 'omega_ss', 'omega_tp', 'rho_m1',
             'omega_m1', 'g_n_ss', 'g_n_tp', 'rho_st', 'i_st',
             'SS_OutTol', 'SS_EulTol', 'SS_EulDif']
        keys_nequal_list = []
        keys_equal = True
        for key in keys_to_check:
            if isinstance(p_dict[key], np.ndarray):
                if not np.array_equal(p_dict[key], prev_p_dict[key]):
                    keys_equal = False
                    keys_nequal_list.append({
                        'p_dict_' + key: p_dict[key],
                        'prev_p_dict_' + key: prev_p_dict[key]})
            else:
                if p_dict[key] != prev_p_dict[key]:
                    keys_equal = False
                    keys_nequal_list.append({
                        'p_dict_' + key: p_dict[key],
                        'prev_p_dict_' + key: prev_p_dict[key]})
        if keys_equal:
            # If none of the parameters changed, use saved pickle
            print('RETRIEVE STEADY-STATE SOLUTIONS FROM FILE')
            ss_output = pickle.load(open(ss_outputfile, 'rb'))
        else:
            # If any of the parameters changed, end the program and
            # compute the steady-state solution
            err_msg = ('ERROR: Current ss_args in class p are not ' +
                       'equal to the ss_args in class prev_p that ' +
                       'produced ss_output. Must solve for SS before ' +
                       'solving transition path. Set SS_solve=True.')
            print('Class objects that are not equal are the following:')
            print(keys_nequal_list)
            raise ValueError(err_msg)

'''
------------------------------------------------------------------------
Solve for the transition path equilibrium by time path iteration (TPI)
------------------------------------------------------------------------
tpi_output_fldr = string, cur_path extension of TPI output folder path
tpi_output_dir  = string, full path name of TPI output folder
tpi_outputfile  = string, path name of file for TPI output objects
tpi_paramsfile  = string, path name of file for TPI parameter objects
r_ss            = scalar > 0, steady-state aggregate interest rate
K_ss            = scalar > 0, steady-state aggregate capital stock
L_ss            = scalar > 0, steady-state aggregate labor
C_ss            = scalar > 0, steady-state aggregate consumption
b_ss            = (S,) vector, steady-state savings distribution
                  (b1, b2,... bS)
n_ss            = (S,) vector, steady-state labor supply distribution
                  (n1, n2,... nS)
init_wgts       = (S,) vector, weights representing the factor by which
                  the initial wealth distribution differs from the
                  steady-state wealth distribution
bvec1           = (S,) vector, initial period savings distribution
K1              = scalar, initial period aggregate capital stock
K1_cstr         = boolean, =True if K1 <= 0
tpi_params      = length 23 tuple, args to pass into c7tpf.get_TPI()
tpi_output      = length 14 dictionary, {cpath, npath, bpath, wpath,
                  rpath, Kpath, Lpath, Ypath, Cpath, bSp1_err_path,
                  b_err_path, n_err_path, RCerrPath, tpi_time}
tpi_args        = length 24 tuple, args that were passed in to get_TPI()
------------------------------------------------------------------------
'''
if p.TP_solve:
    print('BEGIN EQUILIBRIUM TRANSITION PATH COMPUTATION')

    # Create OUTPUT/TPI directory if does not already exist
    cur_path = os.path.split(os.path.abspath(__file__))[0]
    tp_output_fldr = 'OUTPUT/TP'
    tp_output_dir = os.path.join(cur_path, tp_output_fldr)
    if not os.access(tp_output_dir, os.F_OK):
        os.makedirs(tp_output_dir)
    tp_outputfile = os.path.join(tp_output_dir, 'tp_vars.pkl')
    tp_paramsfile = os.path.join(tp_output_dir, 'tp_args.pkl')

    b_ss = ss_output['b_ss']

    # Choose initial period distribution of wealth (b_s0) as function of
    # the steady-state distribution of wealth, which initial
    # distribution determines initial period total capital across
    # industries
    pct_s1 = 0.98
    pct_sS = 1.03
    init_wgts = (((pct_sS - pct_s1) / (p.S)) * np.arange(p.S + 1) +
                 pct_s1)
    p.b_s0_vec = init_wgts * b_ss

    # Re-save ss_args.pkl because p.b_s0_vec was changed
    pickle.dump(p, open(ss_paramsfile, 'wb'))

    tp_output = tp.get_TP(p, ss_output, p.TP_graphs)

    # Save tp_output as pickle
    pickle.dump(tp_output, open(tp_outputfile, 'wb'))
    pickle.dump(p, open(tp_paramsfile, 'wb'))
