# DynamicPop
This repository holds the code, documentation, and package files for population dynamics package. We propose a general set of methods and options for modeling population dynamics that adjust over the near- and medium-run but settle down in the long-run.

1. The percent of the population in each age group `omega_{s+1,t+1}` at a particular period `t+1` is a function of fertility rates `f_{s,t}`, mortality rates `rho_{s,t}`, immigration rates `i_{s,t}`, and population distribution `omega_{s,t}` in the previous period.
2. Theory and intuition require a limit or long-run steady-state in the determinants of the population `f_bar_s`, `rho_bar_s`, and `i_bar_s` by some period in the future `T_omega`.
3. Choose 3-parameter generalized gamma distribution plus a scale parameter (4 parameters total) to approximate population distributions.
4. Fit a 3-parameter parametric function (exponential or arctangent) to the time series of the parameters in the scaled generalized gamma distribution.
5. Interpolate parameters between most recent period and period `T_omega` to interpolate the population distribution into the future.
