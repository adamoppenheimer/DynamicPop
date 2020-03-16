'''
------------------------------------------------------------------------
This module contains the functions that generate aggregate variables in
the steady-state or in the transition path of the overlapping
generations model with S-period lived agents, endogenous labor supply,
non-constant demographics, bequests, and labor productivity growth.

This Python module imports the following module(s): None

This Python module defines the following function(s):
    get_Lt()
    get_Kt()
    get_BQt()
    get_Ct()
    get_It()
    get_NXt()
------------------------------------------------------------------------
'''
# Import packages
import numpy as np

'''
------------------------------------------------------------------------
Functions
------------------------------------------------------------------------
'''


def get_Lt(n_st, p):
    '''
    --------------------------------------------------------------------
    Solve for aggregate labor
    --------------------------------------------------------------------
    INPUTS:
    n_st = (S,) vector or (S, T2+1) matrix, steady-state n_s or time
           path of household labor supply n_{s,t}
    p    = parameters object, exogenous parameters of the model

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    L_t = scalar or (T2+1,) vector, steady-state aggregate labor or time
          path of aggregate labor

    RETURNS: L_t
    --------------------------------------------------------------------
    '''
    if n_st.ndim == 1:  # steady-state case
        L_st = (p.omega_ss * n_st).sum()
    elif n_st.ndim == 2:  # transition path case
        L_st = (p.omega_tp * n_st).sum(axis=0)

    return L_st


def get_Kt(b_sp1, p):
    '''
    --------------------------------------------------------------------
    Solve for aggregate Capital
    --------------------------------------------------------------------
    INPUTS:
    b_sp1 = (S,) vector or (S, T2+1) matrix, steady-state savings
            b_sp1 or time path of household savings b_{s+1,t}
    p     = parameters object, exogenous parameters of the model

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    K_t          = scalar or (T2+1,) vector, steady-state aggregate
                   capital or time path of aggregate capital
    omega_sm1tm1 = (S, T2+1) matrix, omega_{s-1,t-1} time path of
                   population distribution for ages E+1 to E+S
    omega_stm1   = (S, T2+1) matrix, omega_{s,t-1} time path of
                   population distribution for ages E+2 to E+S+1
    i_st         = (S, T2+1) matrix, i_{s,t} time path of immigration
                   rates for ages E+2 to E+S+1

    RETURNS: K_t
    --------------------------------------------------------------------
    '''
    if b_sp1.ndim == 1:  # steady-state case
        K_st = ((1 / (1 + p.g_n_ss)) *
                (p.omega_ss * b_sp1 +
                 np.append(p.i_ss[1:], 0.0) *
                 np.append(p.omega_ss[1:], 0.0) * b_sp1).sum())
    elif b_sp1.ndim == 2:  # transition path case
        omega_sm1tm1 = np.append(p.omega_m1.reshape((p.S, 1)),
                                 p.omega_tp[:, :p.T2], axis=1)
        omega_stm1 = np.vstack((p.omega_tp[1:, :],
                                np.zeros((1, p.T2 + 1))))
        i_st = np.vstack((p.i_st[1:, :p.T2 + 1],
                          np.zeros((1, p.T2 + 1))))
        K_st = ((1 / (1 + p.g_n_tp[:p.T2 + 1])) *
                (omega_sm1tm1 * b_sp1 +
                 i_st * omega_stm1 * b_sp1)).sum(axis=0)

    return K_st


def get_BQt(b_sp1t, r_t, p):
    '''
    --------------------------------------------------------------------
    Solve for total bequests
    --------------------------------------------------------------------
    INPUTS:
    b_sp1t = (S,) vector or (S, T2+1) matrix, steady-state savings b_sp1
             or time path of household savings b_{s+1,t}
    r_t    = scalar > 0 or (T2+1,) vector, steady-state interest rate or
             time path of interest rates
    p      = parameters object, exogenous parameters of the model

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    BQ_t         = scalar or (T2+1,) vector, steady-state total bequests
                   or time path of total bequests
    rho_sm1tm1   = (S, T2+1) matrix, rho_{s,t} time path of mortality
                   rates for ages E+1 to E+S
    omega_sm1tm1 = (S, T2+1) matrix, omega_{s,t} time path of population
                   distribution for ages E+1 to E+S

    RETURNS: BQ_t
    --------------------------------------------------------------------
    '''
    if b_sp1t.ndim == 1:  # steady-state case
        BQ_t = (((1 + r_t) / (1 + p.g_n_ss)) *
                (p.rho_ss * p.omega_ss * b_sp1t).sum())
    elif b_sp1t.ndim == 2:  # transition path case
        rho_sm1tm1 = np.append(p.rho_m1.reshape((p.S, 1)),
                               p.rho_st[:, :p.T2], axis=1)
        omega_sm1tm1 = np.append(p.omega_m1[:].reshape((p.S, 1)),
                                 p.omega_tp[:, :p.T2], axis=1)
        BQ_t = (((1 + r_t) / (1 + p.g_n_tp[:p.T2 + 1])) *
                (rho_sm1tm1 * omega_sm1tm1 * b_sp1t).sum(axis=0))

    return BQ_t


def get_Ct(c_st, p):
    '''
    --------------------------------------------------------------------
    Solve for aggregate consumption C
    --------------------------------------------------------------------
    INPUTS:
    c_st = (S,) vector or (S, T2+1) matrix, steady-state consumption c_s
           or time path of household consumption c_{s,t}
    p    = parameters object, exogenous parameters of the model

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    C_t = scalar or (T2+1,) vector, steady-state aggregate consumption
          or time path of aggregate consumption

    RETURNS: C_t
    --------------------------------------------------------------------
    '''
    if c_st.ndim == 1:  # steady-state case
        C_t = (p.omega_ss * c_st).sum()
    elif c_st.ndim == 2:  # transition path case
        C_t = (p.omega_tp * c_st).sum(axis=0)

    return C_t


def get_It(K_t, p):
    '''
    --------------------------------------------------------------------
    Solve for aggregate investment I
    --------------------------------------------------------------------
    INPUTS:
    K_t = scalar > 0 or (T2+1,) vector, steady-state aggregate capital K
          or time path of aggregate capital K_t
    p   = parameters object, exogenous parameters of the model

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    K_tp1 = (T2+1,) vector, time series of aggregate capital stock
            iterated one period forward
    I_t   = scalar or (T2+1,) vector, steady-state aggregate investment
            I or time path of aggregate investment I_t

    RETURNS: I_t
    --------------------------------------------------------------------
    '''
    if np.isscalar(K_t):  # steady-state case
        I_t = (np.exp(p.g_y) * (1 + p.g_n_ss) - 1 + p.delta) * K_t
    else:  # transition path case
        K_tp1 = np.append(K_t[1:], K_t[-1])
        I_t = (np.exp(p.g_y) * (1 + p.g_n_tp[:p.T2 + 1]) * K_tp1 -
               (1 - p.delta) * K_t)

    return I_t


def get_NXt(b_sp1tp1, p):
    '''
    --------------------------------------------------------------------
    Solve for net exports NX
    --------------------------------------------------------------------
    INPUTS:
    b_stp1 = (S,) vector or (S, T2+1) matrix, steady-state savings
             or time path of next period savings
    p      = parameters object, exogenous parameters of the model

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    NX_t       = scalar or (T2+1,) vector, steady-state net exports NX
                 or time path of net exports NX_t
    omega_sp1t = (S, T2+1) matrix, omega_{s+1,t} time path of population
                 distribution for ages E+2 to E+S+1
    i_stp1     = (S, T2+2) matrix, i_{s,t} time path of immigration
                 rates for ages E+1 to E+S for one extra time period
    i_sp1tp1   = (S, T2+1) matrix, i_{s+1,t+1} time path of immigration
                 rates for ages E+2 to E+S+1 for periods t=1 to t=T2+1

    RETURNS: NX_t
    --------------------------------------------------------------------
    '''
    if b_sp1tp1.ndim == 1:  # steady-state case
        NX_t = (-np.exp(p.g_y) *
                (np.append(p.i_ss[1:], 0.0) *
                 np.append(p.omega_ss[1:], 0.0) * b_sp1tp1).sum())
    elif b_sp1tp1.ndim == 2:  # transition path case
        omega_sp1t = np.append(p.omega_tp[1:, :p.T2 + 1],
                               np.zeros((1, p.T2 + 1)), axis=0)
        i_stp1 = np.append(p.i_st, p.i_st[:, -1].reshape((p.S, 1)),
                           axis=1)
        i_sp1tp1 = np.append(i_stp1[1:, 1:p.T2 + 2],
                             np.zeros((1, p.T2 + 1)), axis=0)
        NX_t = -np.exp(p.g_y) * (i_sp1tp1 * omega_sp1t *
                                 b_sp1tp1).sum(axis=0)

    return NX_t
