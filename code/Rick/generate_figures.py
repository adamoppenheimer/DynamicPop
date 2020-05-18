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

'''
------------------------------------------------------------------------
Generate figures
------------------------------------------------------------------------
'''
cur_path = os.path.split(os.path.abspath(__file__))[0]
ss_results = []
tp_results = []

p = params.parameters('static')
figure_labels = ['Static', 'Partial-Dynamic', 'Full-Dynamic', 'Alternate Full-Dynamic']

for demog_type in ['static', 'dynamic_partial', 'dynamic_full', 'dynamic_full_alternate']:
    ss_output_fldr = 'OUTPUT/SS/' + demog_type
    ss_output_dir = os.path.join(cur_path, ss_output_fldr)
    ss_outputfile = os.path.join(ss_output_dir, 'ss_vars.pkl')

    # Make sure that the SS output files exist
    ss_output_exst = os.path.exists(ss_outputfile)
    if not ss_output_exst:
        # If the files don't exist, stop the program
        err_msg = ('ERROR: The SS output files do not exist')
        raise ValueError(err_msg)
    else:
        ss_output = pickle.load(open(ss_outputfile, 'rb'))
    ss_results.append(ss_output)

    tp_output_fldr = 'OUTPUT/TP/' + demog_type
    tp_output_dir = os.path.join(cur_path, tp_output_fldr)
    tp_outputfile = os.path.join(tp_output_dir, 'tp_vars.pkl')

    # Make sure that the TPI output files exist
    tp_output_exst = os.path.exists(tp_outputfile)
    if not tp_output_exst:
        # If the files don't exist, stop the program
        err_msg = ('ERROR: The TP output files do not exist')
        raise ValueError(err_msg)
    else:
        tp_output = pickle.load(open(tp_outputfile, 'rb'))
    tp_results.append(tp_output)

    graph = False
    if graph:
        p = params.parameters(demog_type)
        tp.create_graphs(tp_output, p)

tp.tp_pct_change_graphs(tp_results[0], tp_results[1:], p, figure_labels)

ss.ss_pct_change_graphs(ss_results[0], ss_results[1:], p, figure_labels)
