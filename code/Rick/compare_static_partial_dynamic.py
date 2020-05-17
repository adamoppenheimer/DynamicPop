'''
------------------------------------------------------------------------
This program generates SS and TPI figures

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
from matplotlib import pyplot as plt

'''
------------------------------------------------------------------------
Generate figures
------------------------------------------------------------------------
'''
cur_path = '/Users/adamalexanderoppenheimer/Desktop/DynamicPop/code/Rick' # os.path.split(os.path.abspath(__file__))[0]

p_list_ss = []
p_list_tp = []

for demog_type in ['static', 'dynamic_partial', 'dynamic_full', 'dynamic_full_alternate']:
    ss_output_fldr = 'OUTPUT/SS/' + demog_type
    ss_output_dir = os.path.join(cur_path, ss_output_fldr)
    ss_outputfile = os.path.join(ss_output_dir, 'ss_args.pkl')

    # Make sure that the SS output files exist
    ss_output_exst = os.path.exists(ss_outputfile)
    if not ss_output_exst:
        # If the files don't exist, stop the program
        err_msg = ('ERROR: The SS output files do not exist')
        raise ValueError(err_msg)
    else:
        ss_args = pickle.load(open(ss_outputfile, 'rb'))
    p_list_ss.append(ss_args)

    tp_output_fldr = 'OUTPUT/TP/' + demog_type
    tp_output_dir = os.path.join(cur_path, tp_output_fldr)
    tp_outputfile = os.path.join(tp_output_dir, 'tp_args.pkl')

    # Make sure that the TPI output files exist
    tp_output_exst = os.path.exists(tp_outputfile)
    if not tp_output_exst:
        # If the files don't exist, stop the program
        err_msg = ('ERROR: The TP output files do not exist')
        raise ValueError(err_msg)
    else:
        tp_args = pickle.load(open(tp_outputfile, 'rb'))
    p_list_tp.append(tp_args)

###########################
##### Start with SS p #####
###########################

static_p = p_list_ss[0]
partial_dynamic_p = p_list_ss[1]

##########################
##### Initial Values #####
##########################

# First, compare population
static_pop = static_p.omega_tp[:, 0]
partial_dynamic_pop = partial_dynamic_p.omega_tp[:, 0]
d_pop = static_pop - partial_dynamic_pop

plt.plot(static_pop, label='static')
plt.plot(partial_dynamic_pop, label='partial dynamic')
plt.legend()
plt.show()

# Next, compare immigration
static_imm = static_p.i_st[:, 0]
partial_dynamic_imm = partial_dynamic_p.i_st[:, 0]
d_imm = static_imm - partial_dynamic_imm

plt.plot(static_imm, label='static')
plt.plot(partial_dynamic_imm, label='partial dynamic')
plt.legend()
plt.show()

# Finally, compare mortality
static_mort = static_p.rho_st[:, 0]
partial_dynamic_mort = partial_dynamic_p.rho_st[:, 0]
d_mort = static_mort - partial_dynamic_mort

plt.plot(static_mort, label='static')
plt.plot(partial_dynamic_mort, label='partial dynamic')
plt.legend()
plt.show()

#####################
##### SS Values #####
#####################

# First, compare population
static_pop_ss = static_p.omega_ss
partial_dynamic_pop_ss = partial_dynamic_p.omega_ss

plt.plot(static_pop_ss, label='static')
plt.plot(partial_dynamic_pop_ss, label='partial dynamic')
plt.legend()
plt.show()

# Next, compare immigration
static_imm_ss = static_p.i_ss
partial_dynamic_imm_ss = partial_dynamic_p.i_ss

plt.plot(static_imm_ss, label='static')
plt.plot(partial_dynamic_imm_ss, label='partial dynamic')
plt.legend()
plt.show()

# Finally, compare mortality
static_mort_ss = static_p.rho_ss
partial_dynamic_mort_ss = partial_dynamic_p.rho_ss

plt.plot(static_mort_ss, label='static')
plt.plot(partial_dynamic_mort_ss, label='partial dynamic')
plt.legend()
plt.show()

##############################
##### Then Consider TP p #####
##############################

static_p = p_list_tp[0]
partial_dynamic_p = p_list_tp[1]

##########################
##### Initial Values #####
##########################

# First, compare population
static_pop = static_p.omega_tp[:, 0]
partial_dynamic_pop = partial_dynamic_p.omega_tp[:, 0]
d_pop = static_pop - partial_dynamic_pop

plt.plot(static_pop, label='static')
plt.plot(partial_dynamic_pop, label='partial dynamic')
plt.legend()
plt.show()

# Next, compare immigration
static_imm = static_p.i_st[:, 0]
partial_dynamic_imm = partial_dynamic_p.i_st[:, 0]
d_imm = static_imm - partial_dynamic_imm

plt.plot(static_imm, label='static')
plt.plot(partial_dynamic_imm, label='partial dynamic')
plt.legend()
plt.show()

# Finally, compare mortality
static_mort = static_p.rho_st[:, 0]
partial_dynamic_mort = partial_dynamic_p.rho_st[:, 0]
d_mort = static_mort - partial_dynamic_mort

plt.plot(static_mort, label='static')
plt.plot(partial_dynamic_mort, label='partial dynamic')
plt.legend()
plt.show()

#####################
##### SS Values #####
#####################

# First, compare population
static_pop_ss = static_p.omega_ss
partial_dynamic_pop_ss = partial_dynamic_p.omega_ss

plt.plot(static_pop_ss, label='static')
plt.plot(partial_dynamic_pop_ss, label='partial dynamic')
plt.legend()
plt.show()

# Next, compare immigration
static_imm_ss = static_p.i_ss
partial_dynamic_imm_ss = partial_dynamic_p.i_ss

plt.plot(static_imm_ss, label='static')
plt.plot(partial_dynamic_imm_ss, label='partial dynamic')
plt.legend()
plt.show()

# Finally, compare mortality
static_mort_ss = static_p.rho_ss
partial_dynamic_mort_ss = partial_dynamic_p.rho_ss

plt.plot(static_mort_ss, label='static')
plt.plot(partial_dynamic_mort_ss, label='partial dynamic')
plt.legend()
plt.show()
