'''
Create a parameters class that instantiates initial values of all
exogenous parameters
'''

# Import packages
import numpy as np
import elliputil as elp
import demographics as demog


class parameters:
    '''
    Parameters class for exogenous objects
    '''

    def __init__(self):
        '''
        Instantiate the parameters class with the input name
        paramclass_name
        '''
        '''
        Period parameters
        ----------------------------------------------------------------
        E          = integer >= 1, number of periods an individual is
                     economically inactive (youth)
        S          = integer >= 3, number of periods an individual lives
        yrs_in_per = scalar > 0, number of years in model period
        T1         = integer > S, number of periods until demographics
                     hit steady state
        T2         = integer > T1, number of periods until economy is in
                     steady state
        ----------------------------------------------------------------
        '''
        self.E = int(20)
        self.S = int(80)
        self.yrs_in_per = 80 / self.S
        self.T1 = int(round(3.0 * self.S))
        self.T2 = int(round(4.0 * self.S))

        '''
        Household parameters
        ----------------------------------------------------------------
        beta_an      = scalar in (0,1), discount factor for one year
        beta         = scalar in (0,1), discount factor for each model
                       period
        sigma        = scalar > 0, coefficient of relative risk aversion
        l_tilde      = scalar > 0, per-period time endowment for every
                       agent
        chi_n_vec    = (S,) vector, values for chi_{n,s}
        chi_b        = scalar >= 0, scalar parameter on utility of dying
                       with positive savings (warm glow bequest motive)
        b_ellip_init = scalar > 0, initial guess for b
        upsilon_init = scalar > 1, initial guess for upsilon
        ellip_init   = (2,) vector, initial guesses for b and upsilon
        Frisch_elast = scalar > 0, Frisch elasticity of labor supply for
                       CFE disutility of labor
        CFE_scale    = scalar > 0, scale parameter for CFE disutility of
                       labor
        cfe_params   = (2,) vector, values for (Frisch, CFE_scale)
        b_ellip      = scalar > 0, fitted value of b for elliptical
                       disutility of labor
        upsilon      = scalar > 1, fitted value of upsilon for
                       elliptical disutility of labor
        b_s0_vec     = (S,) vector, initial wealth for each aged
                       individual in the first model period
        ----------------------------------------------------------------
        '''
        self.beta_an = 0.96
        self.beta = self.beta_an ** self.yrs_in_per
        self.sigma = 2.2
        self.l_tilde = 1.0
        self.chi_n_vec = 1.0 * np.ones(self.S)
        self.chi_b = 1.0
        b_ellip_init = 1.0
        upsilon_init = 2.0
        ellip_init = np.array([b_ellip_init, upsilon_init])
        self.Frisch_elast = 0.9
        self.CFE_scale = 1.0
        cfe_params = np.array([self.Frisch_elast, self.CFE_scale])
        b_ellip, upsilon = elp.fit_ellip_CFE(ellip_init, cfe_params,
                                             self.l_tilde)
        self.b_ellip = b_ellip
        self.upsilon = upsilon
        self.b_s0_vec = 0.1 * np.ones(self.S)

        '''
        Demographic parameters
        ----------------------------------------------------------------
        min_yr        = integer > 0,
        max_yr        = integer > min_yr,
        curr_year     = integer >= 2020
        omega_tp      = (S, T2+1) matrix, transition path of the
                        stationarized population distribution by age for
                        economically active ages
        omega_m1      = (S,) vector, stationarized economically active
                        population distribution in period t=-1 right
                        before current period
        g_n_ss        = (T2+1,) vector,
        omega_ss      = (S,) vector, steady-state stationarized
                        population distribution by age for economically
                        active ages
        surv_rates    =
        mort_rates    =
        g_n_path      =
        imm_rates_mat =
        rho_ss        = (S,) vector, steady-state mortality rates by age
                        for economically active ages
        rho_m1        =
        rho_st        =
        i_ss          = (S,) vector, steady-state immigration rates by
                        age for economically active ages
        i_st          =
        ----------------------------------------------------------------
        '''
        self.min_yr = 1
        self.max_yr = self.E + self.S
        self.curr_year = 2020
        (omega_tp, g_n_ss, omega_ss, surv_rates, mort_rates, g_n_path,
            imm_rates_mat, omega_m1) = \
            demog.get_pop_objs(self.E, self.S, self.T1, self.min_yr,
                               self.max_yr, self.curr_year,
                               GraphDiag=False)
        self.rho_ss = mort_rates
        self.i_ss = imm_rates_mat.T[:, self.T1]
        self.omega_ss = omega_ss
        self.omega_tp = \
            np.append(omega_tp.T, np.tile(omega_ss.reshape((self.S, 1)),
                                          (1, self.T2 - self.T1 + 1)),
                      axis=1)
        self.rho_m1 = mort_rates
        self.omega_m1 = omega_m1
        self.g_n_ss = g_n_ss
        self.g_n_tp = \
            np.append(g_n_path.reshape((1, self.T1 + self.S)),
                      g_n_ss * np.ones((1, self.T2 - self.T1 +
                                        1)), axis=1).flatten()
        self.rho_st = np.tile(self.rho_ss.reshape((self.S, 1)),
                              (1, self.T2 + 1))
        self.i_st = imm_rates_mat.T[:, :self.T2 + 1]

        '''
        Industry parameters
        ----------------------------------------------------------------
        A        = scalar > 0, total factor productivity
        alpha    = scalar in (0,1), capital share of income
        delta_an = scalar in [0,1], annual capital depreciation rate
        delta    = scalar in [0,1], model period capital depreciation
                   rate
        g_y_an   = scalar, annual net growth rate of labor productivity
        g_y      = scalar, net growth rate of labor productivity in each
                   model period
        ----------------------------------------------------------------
        '''
        self.A = 1.0
        self.alpha = 0.35
        self.delta_an = 0.05
        self.delta = 1 - ((1 - self.delta_an) ** self.yrs_in_per)
        g_y_an = 0.03
        self.g_y = ((1 + g_y_an) ** self.yrs_in_per) - 1

        '''
        Set steady-state solution method parameters
        ----------------------------------------------------------------
        SS_solve   = boolean, =True if want to solve for steady-state
                     solution, otherwise retrieve solutions from pickle
        SS_OutTol  = scalar > 0, tolerance for outer loop iterative
                     solution convergence criterion
        SS_EulTol  = scalar > 0, tolerance level for steady-state
                     inner-loop Euler equation root finder
        SS_graphs  = boolean, =True if want graphs of steady-state
                     objects
        SS_EulDif  = boolean, =True if use simple differences in Euler
                     errors. Otherwise, use percent deviation form.
        ----------------------------------------------------------------
        '''
        self.SS_solve = False
        self.SS_OutTol = 1e-13
        self.SS_EulTol = 1e-13
        self.SS_graphs = True
        self.SS_EulDif = True

        '''
        Set transition path solution method parameters
        ----------------------------------------------------------------
        TP_solve   = boolean, =True if want to solve TP equilibrium
        TP_OutTol  = scalar > 0, tolerance level for outer-loop
                     bisection method in TPI
        TP_EulTol  = scalar > 0, tolerance level for inner-loop root
                     finder in each iteration of TPI
        TP_graphs  = boolean, =True if want graphs of TPI objects
        TP_EulDif  = boolean, =True if want difference version of Euler
                     errors beta*(1+r)*u'(c2) - u'(c1), =False if want
                     ratio version [beta*(1+r)*u'(c2)]/[u'(c1)] - 1
        xi_TP      = scalar in (0,1], TPI path updating parameter
        maxiter_TP = integer >= 1, Maximum number of iterations for TPI
        ----------------------------------------------------------------
        '''
        self.TP_solve = True
        self.TP_OutTol = 1e-7
        self.TP_EulTol = 1e-10
        self.TP_graphs = True
        self.TP_EulDif = True
        self.xi_TP = 0.2
        self.maxiter_TP = 1000
