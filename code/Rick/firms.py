'''
------------------------------------------------------------------------
This module contains the functions that generate the variables
associated with firms' optimization in the steady-state or in the
transition path of the overlapping generations model with S-period lived
agents, endogenous labor supply, non-constant demographic dynamics, and
productivity growth.

This Python module imports the following module(s): None

This Python module defines the following function(s):
    get_wt()
    get_rt()
    get_Yt()
------------------------------------------------------------------------
'''

'''
------------------------------------------------------------------------
    Functions
------------------------------------------------------------------------
'''


def get_wt(r_t, p):
    '''
    --------------------------------------------------------------------
    Solve for wage w_t given assumed interest rate r_t
    --------------------------------------------------------------------
    INPUTS:
    r_t = scalar > 0 or (T2+S,) vector, steady-state interest rate or
          time path of interest rates
    p   = parameters class object

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    w_t = scalar or (T2+S,) vector, steady-state wage w or time path of
          wages w_t

    RETURNS: w_t
    --------------------------------------------------------------------
    '''
    w_t = (1 - p.alpha) * p.A * (((p.alpha * p.A) / (r_t + p.delta)) **
                                 (p.alpha / (1 - p.alpha)))

    return w_t


def get_rt(K_t, L_t, p):
    '''
    --------------------------------------------------------------------
    Solve for interest rate r_t given aggregate capital stock K_t and
    aggregate labor L_t
    --------------------------------------------------------------------
    INPUTS:
    K_t = scalar > 0 or (T2+1,) vector, steady-state aggregate capital
          stock K or time path of aggregate capital K_t
    L_t = scalar > 0 or (T2+1,) vector, steady-state aggregate labor L
          or time path of aggregate labor L_t
    p   = parameters class object

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    r_t = scalar or (T2+1,) vector, steady-state interest rate r or time
          path of interest rate r_t

    RETURNS: r_t
    --------------------------------------------------------------------
    '''
    r_t = p.alpha * p.A * ((L_t / K_t) ** (1 - p.alpha)) - p.delta

    return r_t


def get_Yt(K_t, L_t, p):
    '''
    --------------------------------------------------------------------
    Solve for aggregate output Y_t given aggregate capital stock K_t and
    aggregate labor L_t
    --------------------------------------------------------------------
    INPUTS:
    K_t = scalar > 0 or (T2+1,) vector, steady-state aggregate capital
          stock K or time path of aggregate capital K_t
    L_t = scalar > 0 or (T2+1,) vector, steady-state aggregate labor L
          or time path of aggregate labor L_t
    p   = parameters class object

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    Y_t = scalar or (T2+1,) vector, steady-state aggregate output Y or
          time path of aggregate output Y_t

    RETURNS: Y_t
    --------------------------------------------------------------------
    '''
    Y_t = p.A * (K_t ** p.alpha) * (L_t ** (1 - p.alpha))

    return Y_t
