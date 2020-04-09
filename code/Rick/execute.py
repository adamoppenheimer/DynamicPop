'''
------------------------------------------------------------------------
This program runs the steady state solver as well as the time path
iteration solution for the model with S-period lived agents, endogenous
labor supply, non-constant demographics, bequests, and productivity
growth.

This Python script imports the following module(s):
    parameters.py
    SS.py
    TP.py

This Python script calls the following function(s):
    params.parameters()
    ss.get_SS()
    tp.get_TP()

Files created by this script:
    OUTPUT/SS/ss_vars.pkl
    OUTPUT/SS/ss_args.pkl
    OUTPUT/TP/tp_vars.pkl
    OUTPUT/TP/tp_args.pkl
------------------------------------------------------------------------
'''
# Import packages
import numpy as np
import pickle
import os
import parameters as params
import SS as ss
import TP as tp

'''
------------------------------------------------------------------------
Import parameters
------------------------------------------------------------------------
'''
p = params.parameters()
for demog_type in ['dynamic_full_alternate']: # 'static', 'dynamic_partial', 'dynamic_full',
    p.set_demog(demog_type)

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
    BQss_init      = scalar > 0, initial guess for BQ_ss
    init_vals      = (2,) vector, initial guesses ([rss_init, BQss_init])
                    passed into ss.get_SS()
    ss_output      = length 18 dict, steady-state objects {c_ss, n_ss, b_ss,
                    n_err_ss, b_err_ss, r_ss, w_ss, BQ_ss, r_err_ss,
                    BQ_err_ss, L_ss, K_ss, Y_ss, C_ss, I_ss, NX_ss,
                    RCerr_ss, ss_time}
    ss_vars_exst   = boolean, =True if ss_vars.pkl exists
    ss_args_exst   = boolean, =True if ss_args.pkl exists
    err_msg        = string, error message
    prev_p         = parameters class object loaded from pickle
    prev_p_dict    = dictionary of objects in prev_p parameters class object
    p_dict         = dictionary of objects in p parameters class object
    keys_to_check  = length 27 list, string names of keys to compare between
                    prev_p_dict and p_dict
    keys_neql_list = list, container for keys that are not equal in p and
                    prev_p parameter class objects
    keys_equal     = boolean, =True if two keys are equal
    key            = index of elements in keys_to_check
    ------------------------------------------------------------------------
    '''
    # Create OUTPUT/SS directory if does not already exist
    cur_path = os.path.split(os.path.abspath(__file__))[0]
    ss_output_fldr = 'OUTPUT/SS/' + demog_type
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
                ['S', 'yrs_in_per', 'beta_an', 'beta', 'sigma', 'l_tilde',
                'chi_n_vec', 'Frisch_elast', 'CFE_scale', 'b_ellip',
                'upsilon', 'A', 'alpha', 'delta_an', 'delta', 'g_y', 'E',
                'min_yr', 'max_yr', 'curr_year', 'rho_ss', 'i_ss',
                'omega_ss', 'g_n_ss', 'SS_OutTol', 'SS_EulTol',
                'SS_EulDif']
            keys_neql_list = []
            keys_equal = True
            for key in keys_to_check:
                if isinstance(p_dict[key], np.ndarray):
                    if not np.array_equal(p_dict[key], prev_p_dict[key]):
                        keys_equal = False
                        keys_neql_list.append({
                            'p_dict_' + key: p_dict[key],
                            'prev_p_dict_' + key: prev_p_dict[key]})
                else:
                    if p_dict[key] != prev_p_dict[key]:
                        keys_equal = False
                        keys_neql_list.append({
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
                print(keys_neql_list)
                raise ValueError(err_msg)

    '''
    ------------------------------------------------------------------------
    Solve for the transition path equilibrium by time path iteration (TPI)
    ------------------------------------------------------------------------
    tp_output_fldr = string, cur_path extension of TPI output folder path
    tp_output_dir  = string, full path name of TPI output folder
    tp_outputfile  = string, path name of file for TPI output objects
    tp_paramsfile  = string, path name of file for TPI parameter objects
    b_ss           = (S+1,) vector, steady-state distribution of wealth or
                    savings [b_{E+1}, b_{E+2},...b_{E+S+1}]
    pct_s1         = scalar, percent difference of b_{E+1,0} from
                    b_ss_{s=E+1}
    pct_sSp1       = scalar, percent difference of b_{E+S+1,0} from
                    b_ss_{s=E+S+1}
    init_wgts      = (S+1,) vector, weights representing the factor by which
                    the initial wealth distribution differs from the
                    steady-state wealth distribution
    b_s0_vec       = (S+1,) vector, initial period savings distribution
    tp_output      = length 17 dictionary, tpi output objects {cs_path,
                    ns_path, bs_path, ns_err_path,  bs_err_path, r_path,
                    w_path,  BQ_path, K_path,  L_path,  Y_path, C_path,
                    I_path, NX_path, dist, iter_TPI, tpi_time}
    ------------------------------------------------------------------------
    '''
    if p.TP_solve:
        print('BEGIN EQUILIBRIUM TRANSITION PATH COMPUTATION')

        # Create OUTPUT/TPI directory if does not already exist
        cur_path = os.path.split(os.path.abspath(__file__))[0]
        tp_output_fldr = 'OUTPUT/TP/' + demog_type
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
        pct_sSp1 = 1.03
        init_wgts = (((pct_sSp1 - pct_s1) / (p.S)) * np.arange(p.S + 1) +
                    pct_s1)
        p.b_s0_vec = init_wgts * b_ss

        # Re-save ss_args.pkl because p.b_s0_vec was changed
        pickle.dump(p, open(ss_paramsfile, 'wb'))

        tp_output = tp.get_TP(p, ss_output, p.TP_graphs)

        # Save tp_output as pickle
        pickle.dump(tp_output, open(tp_outputfile, 'wb'))
        pickle.dump(p, open(tp_paramsfile, 'wb'))
