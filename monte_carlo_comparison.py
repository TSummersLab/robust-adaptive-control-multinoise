"""
Robust adaptive control via multiplicative noise from bootstrapped uncertainty estimates.
"""
# Authors: Ben Gravell and Tyler Summers

import numpy as np
import numpy.linalg as la
import numpy.random as npr

from problem_data_gen import gen_system_omni, save_system
from utility.matrixmath import mdot, specrad, minsv, lstsqb, dlyap, dare, dare_gain
from utility.ltimult import dare_mult
from utility.lti import ctrb, dctg

from utility.pickle_io import pickle_export
from utility.path_utility import create_directory
from utility.user_input import yes_or_no
from utility.printing import printcolors, create_tag
from time import time
import os

import multiprocessing as mp


def groupdot(A, x):
    """
    Perform dot product over groups of matrices,
    suitable for performing many LTI state transitions in a vectorized fashion
    """
    return np.einsum('...ik,...k', A, x)


def generate_offline_data(n, m, A, B, W, Ns, T, u_explore_var, x0, seed=None):
    """
    Generate state and input data
    :param n: Number of states
    :param m: Number of inputs
    :param A: State matrix
    :param B: Input matrix
    :param W: Noise covariance matrix
    :param Ns: Number of Monte Carlo samples
    :param T: Number of time steps to simulate
    :param u_explore_var: Control input exploration noise variance
    :param x0: Initial state
    :param seed: Seed for NumPy random number generator
    :returns:
    x_hist: State history
    u_hist: Control input history
    w_hist: Additive noise history
    """
    print("Generating offline sample trajectory data...")
    npr.seed(seed)

    # Generate random input
    u_hist = np.sqrt(u_explore_var)*npr.randn(Ns, T, m)
    # Generate noise
    w_hist = npr.multivariate_normal(np.zeros(n), W, size=(Ns, T))
    # Preallocate state history
    x_hist = np.zeros([Ns, T+1, n])
    # Initial state
    x_hist[:, 0] = x0
    # Loop over time
    for t in range(T):
        # Update state
        x_hist[:, t+1] = groupdot(A, x_hist[:, t]) + groupdot(B, u_hist[:, t]) + w_hist[:, t]
    return x_hist, u_hist, w_hist


def least_squares(x_train_hist, u_train_hist, t=None, n=None, m=None, A=None, B=None, idx=None):
    """
    Compute estimated system matrices from training data using ordinary least squares.
    :param t: Time up to which to use the available data.
    :param idx: List of indices of data to use e.g. for bootstrap sampling.
    """
    # Get sizes
    if n is None:
        n = x_train_hist.shape[1]
    if m is None:
        m = u_train_hist.shape[1]

    # Get time horizon
    if t is None:
        t = u_train_hist.shape[0]

    if A is None:
        A_known = False
    else:
        A_known = True
    if B is None:
        B_known = False
    else:
        B_known = True

    # Choose the indexing of the data e.g. for bootstrap samples
    if idx is None:
        idx_before = slice(t)
        idx_after = slice(1, t+1)
    else:
        idx_before = idx
        idx_after = idx+1

    if not A_known and not B_known:
        X = x_train_hist[idx_after]
        Z = np.hstack([x_train_hist[idx_before], u_train_hist[idx_before]])
        Thetahat = lstsqb(mdot(X.T, Z), mdot(Z.T, Z))
        Ahat = Thetahat[:, 0:n]
        Bhat = Thetahat[:, n:n+m]
    elif not A_known and B_known:
        X = x_train_hist[idx_after] - groupdot(B, u_train_hist[idx_before])
        Z = x_train_hist[idx_before]
        Ahat = lstsqb(mdot(X.T, Z), mdot(Z.T, Z))
        Bhat = np.copy(B)
    elif A_known and not B_known:
        X = x_train_hist[idx_after] - groupdot(A, x_train_hist[idx_before])
        Z = u_train_hist[idx_before]
        Ahat = np.copy(A)
        Bhat = lstsqb(mdot(X.T, Z), mdot(Z.T, Z))

    return Ahat, Bhat


def sample_transition_bootstrap(x_train_hist, u_train_hist, t=None, n=None, m=None, Nb=None, log_diagnostics=True):
    """
    Compute estimate of model uncertainty (covariance) via sample-transition bootstrap
    :param t: Time up to which to use the available data.
    :param Nb: Number of bootstrap samples
    """
    # Get sizes
    if n is None:
        n = x_train_hist.shape[1]
    if m is None:
        m = u_train_hist.shape[1]

    # Get time horizon
    if t is None:
        t = u_train_hist.shape[0]

    # Preallocate bootstrap sample arrays
    Ahat_boot = np.zeros([Nb, n, n])
    Bhat_boot = np.zeros([Nb, n, m])

    # Diagnostic logging
    tag_str_list = []

    # Form bootstrap estimates
    for i in range(Nb):
        # Sample transitions iid with replacement
        # Intuitively is OK due to the Markov property of the dynamic system
        idx = npr.randint(t, size=t)
        # Reject degenerate samples
        while np.unique(idx).size < n+m:
            if log_diagnostics:
                tag_str_list.append(create_tag('Bootstrap drew a noninvertible sample, resampling.'))
            idx = npr.randint(t, size=t)
        Ahat_boot[i], Bhat_boot[i] = least_squares(x_train_hist, u_train_hist, t, idx=idx)

    # Sample variance of bootstrap estimates
    Ahat_boot_reshaped = np.reshape(Ahat_boot, [Nb, n*n], order='F')
    Bhat_boot_reshaped = np.reshape(Bhat_boot, [Nb, n*m], order='F')
    Ahat_boot_mean_reshaped = np.mean(Ahat_boot_reshaped, axis=0)
    Bhat_boot_mean_reshaped = np.mean(Bhat_boot_reshaped, axis=0)
    Abar = Ahat_boot_reshaped - Ahat_boot_mean_reshaped
    Bbar = Bhat_boot_reshaped - Bhat_boot_mean_reshaped
    SigmaA = mdot(Abar.T, Abar)/(Nb-1)
    SigmaB = mdot(Bbar.T, Bbar)/(Nb-1)

    SigmaAeigvals, SigmaAeigvecs = la.eigh(SigmaA)
    SigmaBeigvals, SigmaBeigvecs = la.eig(SigmaB)
    alpha = SigmaAeigvals
    beta = SigmaBeigvals
    Aa = np.reshape(SigmaAeigvecs, [n*n, n, n], order='F') # These uncertainty directions have unit Frobenius norm
    Bb = np.reshape(SigmaBeigvecs, [n*m, n, m], order='F') # These uncertainty directions have unit Frobenius norm
    return alpha, Aa, beta, Bb, tag_str_list


def semiparametric_bootstrap(Ahat, Bhat, x_train_hist, u_train_hist, t=None, n=None, m=None, Nb=None,
                             log_diagnostics=True):
    """
    Compute estimate of model uncertainty (covariance) via semiparametric bootstrap
    :param Ahat: Nominal A matrix
    :param Bhat: Nominal B matrix
    :param t: Time up to which to use the available data.
    :param Nb: Number of bootstrap samples
    """
    # Get sizes
    if n is None:
        n = x_train_hist.shape[1]
    if m is None:
        m = u_train_hist.shape[1]

    # Get time horizon
    if t is None:
        t = u_train_hist.shape[0]

    # Preallocate bootstrap sample arrays
    x_boot_hist = np.zeros([Nb, t+1, n])
    u_boot_hist = np.zeros([Nb, t, m])
    w_boot_hist = np.zeros([Nb, t, n])
    Ahat_boot = np.zeros([Nb, n, n])
    Bhat_boot = np.zeros([Nb, n, m])

    # Diagnostic logging
    tag_str_list = []

    # Compute residuals under nominal model Ahat, Bhat
    w_hist = np.zeros([t, n])
    for i in range(t):
        x = x_train_hist[i]
        u = u_train_hist[i]
        xp_true = x_train_hist[i+1]
        xp_pred = np.dot(Ahat, x) + np.dot(Bhat, u)
        w_hist[i] = xp_true - xp_pred

    # Initialize bootstrap training data
    for i in range(Nb):
        # Initialize state
        x_boot_hist[i, 0] = x_train_hist[0]

        # Copy input sequence
        u_boot_hist[i] = u_train_hist[0:t]

        # Sample residuals iid with replacement
        idx = npr.randint(t, size=t)
        w_boot_hist[i] = w_hist[idx]

    # Form bootstrap training data
    for t_samp in range(t):
        # Update state
        x_boot_hist[:, t_samp+1] = (groupdot(Ahat, x_boot_hist[:, t_samp])
                                    + groupdot(Bhat, u_boot_hist[:, t_samp])
                                    + w_boot_hist[:, t_samp])

    # Form bootstrap estimates
    for i in range(Nb):
        Ahat_boot[i], Bhat_boot[i] = least_squares(x_boot_hist[i], u_boot_hist[i])

    # Sample variance of bootstrap estimates
    Ahat_boot_reshaped = np.reshape(Ahat_boot, [Nb, n*n], order='F')
    Bhat_boot_reshaped = np.reshape(Bhat_boot, [Nb, n*m], order='F')
    Ahat_boot_mean_reshaped = np.mean(Ahat_boot_reshaped, axis=0)
    Bhat_boot_mean_reshaped = np.mean(Bhat_boot_reshaped, axis=0)
    Abar = Ahat_boot_reshaped - Ahat_boot_mean_reshaped
    Bbar = Bhat_boot_reshaped - Bhat_boot_mean_reshaped
    SigmaA = mdot(Abar.T, Abar)/(Nb-1)
    SigmaB = mdot(Bbar.T, Bbar)/(Nb-1)
    SigmaAeigvals, SigmaAeigvecs = la.eigh(SigmaA)
    SigmaBeigvals, SigmaBeigvecs = la.eigh(SigmaB)
    alpha = SigmaAeigvals
    beta = SigmaBeigvals
    Aa = np.reshape(SigmaAeigvecs, [n*n, n, n], order='F')  # These uncertainty directions have unit Frobenius norm
    Bb = np.reshape(SigmaBeigvecs, [n*m, n, m], order='F')  # These uncertainty directions have unit Frobenius norm
    return alpha, Aa, beta, Bb, tag_str_list


def compute_robust_gain(Ahat, Bhat, Q, R, a, Aa, b, Bb, noise_pre_scale, noise_post_scale,
                        bisection_epsilon, log_diagnostics):
    """
    Compute gain using multiplicative noise to augment robustness.
    """

    noise_limit_scale = 1.0
    tag_str_list = []

    # Compute gains from generalized Riccati equation
    P, K = dare_mult(Ahat, Bhat, noise_pre_scale*a, Aa, noise_pre_scale*b, Bb, Q, R)
    if K is None:
        # If assumed multiplicative noise variance is too high to admit solution, decrease noise variance
        # Bisection on the noise variance scaling to find the control
        # when the noise just touches the stability boundary
        c_upr = 1.0
        c_lwr = 0.0
        while c_upr-c_lwr > bisection_epsilon:
            cmid = (c_upr + c_lwr)/2
            P, K = dare_mult(Ahat, Bhat, cmid*noise_pre_scale*a, Aa, cmid*noise_pre_scale*b, Bb, Q, R)
            if K is None:
                c_upr = cmid
            else:
                c_lwr = cmid
        noise_limit_scale = c_lwr

        if log_diagnostics:
            tag_str_list.append(create_tag('Scaled noise variance by %.3f' % noise_limit_scale))
        if c_lwr > 0:
            P, K = dare_mult(Ahat, Bhat, c_lwr*noise_pre_scale*noise_post_scale*a, Aa,
                             c_lwr*noise_pre_scale*noise_post_scale*b, Bb, Q, R)
            if K is None:
                tag_str_list.append(create_tag('GAIN NOT FOUND BY DARE_MULT, INCREASE SOLVER PRECISION',
                                               message_type='fail'))
                tag_str_list.append(create_tag('Falling back on cert-equiv gain', message_type='fail'))
                P, K = dare_gain(Ahat, Bhat, Q, R)
        else:
            P, K = dare_gain(Ahat, Bhat, Q, R)
            if log_diagnostics:
                tag_str_list.append(create_tag('Bisection collapsed to cert-equiv'))
    else:
        if not noise_post_scale == 1:
            P, K = dare_mult(Ahat, Bhat, noise_pre_scale*noise_post_scale*a, Aa,
                             noise_pre_scale*noise_post_scale*b, Bb, Q, R)

    return P, K, noise_limit_scale, tag_str_list


def monte_carlo_sample(training_type, testing_type, control_scheme, uncertainty_estimator, required_args,
                       x_train_hist, u_train_hist, w_hist,
                       monte_carlo_idx, print_diagnostics=False, log_diagnostics=False):
    log_str = ''
    if log_diagnostics:
        code_start_time = time()
        log_str += 'Monte Carlo sample %d \n' % (monte_carlo_idx+1)

    # Unpack arguments from dictionary
    n = required_args['n']
    m = required_args['m']
    A = required_args['A']
    B  = required_args['B']
    Q = required_args['Q']
    R = required_args['R']
    Ns = required_args['Ns']
    Nb = required_args['Nb']
    T = required_args['T']
    x0 = required_args['x0']
    bisection_epsilon = required_args['bisection_epsilon']
    t_start_estimate = required_args['t_start_estimate']
    t_explore = required_args['t_explore']
    t_cost_fh = required_args['t_cost_fh']
    cost_horizon = required_args['cost_horizon']
    Kare_true = required_args['Kare_true']
    u_explore_var = required_args['u_explore_var']
    u_exploit_var = required_args['u_exploit_var']
    noise_pre_scale = required_args['noise_pre_scale']
    noise_post_scale = required_args['noise_post_scale']

    if control_scheme == 'certainty_equivalent':
        noise_post_scale = 0

    # Preallocate history arrays
    # State and input
    if testing_type == 'online':
        x_test_hist = np.zeros([T+1, n])
        u_test_hist = np.zeros([T, m])
        x_opt_test_hist = np.zeros([T+1, n])
        u_opt_test_hist = np.zeros([T, m])
    else:
        x_test_hist = None
        u_test_hist = None
        x_opt_test_hist = None
        u_opt_test_hist = None
    # Gain
    K_hist = np.zeros([T, m, n])
    # Nominal model
    Ahat_hist = np.full([T, n, n], np.inf)
    Bhat_hist = np.full([T, n, m], np.inf)
    # Model uncertainty
    alpha_hist = np.full([T, n*n], np.inf)
    beta_hist = np.full([T, n*m], np.inf)
    Aahist = np.full([T, n*n, n, n], np.inf)
    Bbhist = np.full([T, n*m, n, m], np.inf)
    gamma_reduction_hist = np.ones(T)
    # Spectral radius
    specrad_hist = np.full(T, np.inf)
    # Cost
    cost_future_hist = np.full(T, np.inf)
    cost_adaptive_hist = np.full(T, np.inf)
    cost_optimal_hist = np.full(T, np.inf)
    # Model error
    Aerr_hist = np.full(T, np.inf)
    Berr_hist = np.full(T, np.inf)

    # Reset state to x0
    if training_type == 'online':
        x_train = np.copy(x0)
        x_train_hist[0] = x_train
    if testing_type == 'online':
        x_test = np.copy(x0)
        x_opt_test = np.copy(x0)
        x_test_hist[0] = x_test
        x_opt_test_hist[0] = x_opt_test

    # Loop over time
    for t in range(T):
        if log_diagnostics:
            stable_str = printcolors.Green+'Stabilized'+printcolors.Default
            tag_str_list = []
        if t < t_explore:
            # Explore only until time t_explore
            if training_type == 'online' or testing_type == 'online':
                u = np.sqrt(u_explore_var)*npr.randn(m)
                u_opt = mdot(Kare_true, x_opt_test)
            if log_diagnostics:
                u_str = "Explore"
                stable_str = 'Exploring'
            # Assign inf to future cost since we have no optimal controller yet
            cost_future_hist[t] = np.inf
        else:
            # Exploit estimated model
            if log_diagnostics:
                u_str = 'Exploit'
            a = alpha_hist[t-1]
            b = beta_hist[t-1]
            Aa = Aahist[t-1]
            Bb = Bbhist[t-1]

            # Check if estimated system is controllable within a tolerance
            # If not, perturb the estimated B matrix until it is
            ctrb_tol = 1e-4
            while minsv(ctrb(Ahat, Bhat)) < ctrb_tol:
                if log_diagnostics:
                    tag_str_list.append(create_tag('Estimated system uncontrollable, adjusted Bhat'))
                Bhat += 0.00001*npr.randn(n, m)

            # Compute gains
            if noise_post_scale > 0:
                # Robust
                P, K, gamma_reduction, tag_str_list_cg = compute_robust_gain(Ahat, Bhat, Q, R, a, Aa, b, Bb,
                                                                             noise_pre_scale, noise_post_scale,
                                                                             bisection_epsilon, log_diagnostics)
                gamma_reduction_hist[t] = gamma_reduction
                if log_diagnostics:
                    tag_str_list += tag_str_list_cg
            else:
                # Certainty equivalent
                P, K = dare_gain(Ahat, Bhat, Q, R)

            # Compute exploration control component
            if training_type == 'online':
                if control_scheme == 'robust':
                    u_explore_scale = np.sqrt(np.max(a)) + np.sqrt(np.max(b))
                    u_explore = u_explore_scale*np.sqrt(u_exploit_var)*npr.randn(m)
                else:
                    u_explore = np.sqrt(u_exploit_var)*npr.randn(m)
            else:
                u_explore = np.zeros(m)

            # Compute control using optimal control using estimate of system
            u_optimal_estimated = mdot(K, x_test)

            # Apply the sum of optimal and exploration controls
            u = u_optimal_estimated + u_explore

            # Compute control using optimal control given knowledge of true system
            u_opt = mdot(Kare_true, x_opt_test)

            # Evaluate spectral radius of true closed-loop system with current control
            AK = A + mdot(B, K)
            specrad_hist[t] = specrad(AK)
            if log_diagnostics:
                if specrad_hist[t] > 1:
                    stable_str = printcolors.Red+'Unstable'+printcolors.Default

            # Record gains
            K_hist[t] = K

            # Compute cost of current control on actual system
            if cost_horizon == 'infinite':
                # Infinite-horizon cost
                if specrad_hist[t] > 1:
                    cost_future_hist[t] = np.inf
                else:
                    P = dlyap(AK.T, Q + mdot(K.T, R, K))
                    cost_future_hist[t] = np.trace(P)
            elif cost_horizon == 'finite':
                # Finite-horizon cost
                P = dctg(AK, Q + mdot(K.T, R, K), t_cost_fh)
                cost_future_hist[t] = np.trace(P)

            if log_diagnostics:
                if la.norm(x_test) > 1e4:
                    tag_str_list.append(create_tag('x_test = %e > 1e3' % (la.norm(x_test)), message_type='fail'))

        # Accumulate cost
        if testing_type == 'online':
            cost_adaptive_hist[t] = mdot(x_test.T, Q, x_test) + mdot(u.T, R, u)
            cost_optimal_hist[t] = mdot(x_opt_test.T, Q, x_opt_test) + mdot(u_opt.T, R, u_opt)

        # Look up noise
        w = w_hist[t]

        # Update training state and input
        if training_type == 'online':
            x_train = np.dot(A, x_train) + np.dot(B, u) + w
            # Record train-time state and control history
            x_train_hist[t+1] = x_train
            u_train_hist[t] = u

        # Update test-time state
        if testing_type == 'online':
            if training_type == 'online':
                x_test = np.copy(x_train)
            else:
                # Update state under adaptive control
                x_test = np.dot(A, x_test) + np.dot(B, u) + w

            # Update the test-time state under optimal control
            x_opt_test = np.dot(A, x_opt_test) + np.dot(B, u_opt) + w

            # Record test-time state and control history
            x_test_hist[t+1] = x_test
            x_opt_test_hist[t+1] = x_opt_test
            u_test_hist[t] = u
            u_opt_test_hist[t] = u_opt

        # Start generating model and uncertainty estimates once there is enough data to get non-degenerate estimates
        if t >= t_start_estimate:
            # Compute nominal model estimate via least squares
            Ahat, Bhat = least_squares(x_train_hist, u_train_hist, t)
            Adiff = Ahat-A
            Bdiff = Bhat-B
            Aerr = la.norm(Adiff, ord='fro')
            Berr = la.norm(Bdiff, ord='fro')

            # Record model estimates and errors
            Ahat_hist[t] = Ahat
            Bhat_hist[t] = Bhat
            Aerr_hist[t] = Aerr
            Berr_hist[t] = Berr

            # Compute model uncertainty estimate
            if control_scheme == 'robust':
                if uncertainty_estimator == 'sample_transition_bootstrap':
                    # Estimate model uncertainty via non-parametric bootstrap
                    alpha, Aa, beta, Bb, tag_str_list_bp = sample_transition_bootstrap(x_train_hist, u_train_hist,
                                                                                       t, Nb=Nb,
                                                                                       log_diagnostics=log_diagnostics)
                elif uncertainty_estimator == 'semiparametric_bootstrap':
                    alpha, Aa, beta, Bb, tag_str_list_bp = semiparametric_bootstrap(Ahat, Bhat,
                                                                                    x_train_hist, u_train_hist,
                                                                                    t, Nb=Nb,
                                                                                    log_diagnostics=log_diagnostics)
                elif uncertainty_estimator == 'exact':
                    # "Cheat" by using the true error as the multiplicative noise
                    Aa = np.zeros([n*n, n, n])
                    Bb = np.zeros([n*m, n, m])
                    Aa[0] = Adiff/Aerr
                    Bb[0] = Bdiff/Berr
                    alpha = np.zeros(n*n)
                    beta = np.zeros(n*m)
                    alpha[0] = Aerr**2
                    beta[0] = Berr**2
                    tag_str_list_bp = []

                if log_diagnostics:
                    tag_str_list += tag_str_list_bp

                # Record multiplicative noise history
                alpha_hist[t] = alpha
                beta_hist[t] = beta
                Aahist[t] = Aa
                Bbhist[t] = Bb

        # Print and log diagnostic messages
        if log_diagnostics:
            time_whole_str = ''
            time_header_str = "Time = %4d  %s. %s." % (t, u_str, stable_str)
            time_whole_str += time_header_str + '\n'
            for tag_str in tag_str_list:
                time_whole_str += tag_str + '\n'
            log_str += time_whole_str
        if print_diagnostics:
            print(time_whole_str)
    if log_diagnostics:
        code_end_time = time()
        code_elapsed_time = code_end_time - code_start_time
        time_elapsed_str = '%12.6f' % code_elapsed_time
        log_str += "Completed Monte Carlo sample %6d / %6d in %s seconds\n" % (monte_carlo_idx+1, Ns, time_elapsed_str)
    else:
        time_elapsed_str = '?'

    return {'cost_adaptive_hist': cost_adaptive_hist,
            'cost_optimal_hist': cost_optimal_hist,
            'cost_future_hist': cost_future_hist,
            'specrad_hist': specrad_hist,
            'Ahat_hist': Ahat_hist,
            'Bhat_hist': Bhat_hist,
            'Aerr_hist': Aerr_hist,
            'Berr_hist': Berr_hist,
            'alpha_hist': alpha_hist,
            'beta_hist': beta_hist,
            'Aahist': Aahist,
            'Bbhist': Bbhist,
            'gamma_reduction_hist': gamma_reduction_hist,
            'x_train_hist': x_train_hist,
            'u_train_hist': u_train_hist,
            'x_test_hist': x_test_hist,
            'u_test_hist': u_test_hist,
            'x_opt_test_hist': x_opt_test_hist,
            'u_opt_test_hist': u_opt_test_hist,
            'K_hist': K_hist,
            'monte_carlo_idx': np.array(monte_carlo_idx),
            'log_str': log_str,
            'time_elapsed_str': time_elapsed_str}


def monte_carlo_group(training_type, testing_type, control_scheme, uncertainty_estimator, required_args,
                      conditional_args, w_hist, parallel_option='serial'):
    print("Evaluating control scheme: "+control_scheme)

    # Unpack arguments from dictionaries
    n = required_args['n']
    m = required_args['m']
    Ns = required_args['Ns']
    T = required_args['T']

    # Preallocate history arrays
    if training_type == 'online':
        x_train_hist = np.zeros([Ns, T+1, n])
        u_train_hist = np.zeros([Ns, T, m])
    elif training_type == 'offline':
        x_train_hist = conditional_args['x_train_hist']
        u_train_hist = conditional_args['u_train_hist']

    # Simulate each Monte Carlo trial
    shape_dict = {'x_train_hist': [T+1, n],
                  'u_train_hist': [T, m],
                  'x_test_hist': [T+1, n],
                  'x_opt_test_hist': [T+1, n],
                  'u_test_hist': [T, m],
                  'u_opt_test_hist': [T, m],
                  'K_hist': [T, m, n],
                  'Ahat_hist': [T, n, n],
                  'Bhat_hist': [T, n, m],
                  'alpha_hist': [T, n*n],
                  'beta_hist': [T, n*m],
                  'Aahist': [T, n*n, n, n],
                  'Bbhist': [T, n*m, n, m],
                  'gamma_reduction_hist': [T],
                  'specrad_hist': [T],
                  'cost_future_hist': [T],
                  'cost_adaptive_hist': [T],
                  'cost_optimal_hist': [T],
                  'Aerr_hist': [T],
                  'Berr_hist': [T],
                  'monte_carlo_idx': [1],
                  'log_str': None,
                  'time_elapsed_str': None}
    fields = shape_dict.keys()
    output_dict = {}
    for field in fields:
        if field == 'log_str' or field == 'time_elapsed_str':
            output_dict[field] = [None]*Ns
        else:
            output_field_shape = [Ns] + shape_dict[field]
            output_dict[field] = np.zeros(output_field_shape)

    def collect_result(sample_dict):
        k = sample_dict['monte_carlo_idx']
        time_elapsed_str = sample_dict['time_elapsed_str']
        for field in fields:
            output_dict[field][k] = sample_dict[field]
        print("Completed Monte Carlo sample %6d / %6d in %s seconds" % (k+1, Ns, time_elapsed_str))

    # Serial single-threaded processing
    if parallel_option == 'serial':
        for k in range(Ns):
            sample_dict = monte_carlo_sample(training_type, testing_type, control_scheme, uncertainty_estimator,
                                              required_args, x_train_hist[k], u_train_hist[k], w_hist[k], k)
            collect_result(sample_dict)
    # Asynchronous parallel CPU processing
    elif parallel_option == 'parallel':
        num_cpus_to_use = mp.cpu_count() - 1
        pool = mp.Pool(num_cpus_to_use)
        for k in range(Ns):
            pool.apply_async(monte_carlo_sample,
                             args=(training_type, testing_type, control_scheme, uncertainty_estimator,
                                   required_args, x_train_hist[k], u_train_hist[k], w_hist[k], k),
                             callback=collect_result)
        pool.close()
        pool.join()
    print('')
    return output_dict


def compute_derived_data(output_dict, testing_type, receding_horizon=5):
    """
    Compute derived cost data quantities from the results.
    These derived data can be computed and stored for faster loading/plotting,
    or can be calculated after loading to reduce data storage requirements.

    output_dict is modified/mutated
    """
    if not testing_type == 'online':
        return
    for control_scheme in output_dict.keys():
        cost_adaptive_hist = output_dict[control_scheme]['cost_adaptive_hist']
        cost_optimal_hist = output_dict[control_scheme]['cost_optimal_hist']

        # Compute receding horizon data
        Ns, T = cost_adaptive_hist.shape
        cost_adaptive_hist_receding = np.full([Ns, T], np.inf)
        cost_optimal_hist_receding = np.full([Ns, T], np.inf)
        for k in range(Ns):
            for t in range(T):
                if t > receding_horizon:
                    cost_adaptive_hist_receding[k, t] = np.mean(cost_adaptive_hist[k, t-receding_horizon:t])
                    cost_optimal_hist_receding[k, t] = np.mean(cost_optimal_hist[k, t-receding_horizon:t])

        # Compute accumulated cost
        cost_adaptive_hist_accum = np.full([Ns, T], np.inf)
        cost_optimal_hist_accum = np.full([Ns, T], np.inf)
        for k in range(Ns):
            for t in range(T):
                cost_adaptive_hist_accum[k, t] = np.sum(cost_adaptive_hist[k, 0:t])
                cost_optimal_hist_accum[k, t] = np.sum(cost_optimal_hist[k, 0:t])

        # Compute regret and regret_ratio
        regret_hist = cost_adaptive_hist - np.mean(cost_optimal_hist, axis=0)
        regret_hist_receding = cost_adaptive_hist_receding - np.mean(cost_optimal_hist_receding, axis=0)
        regret_hist_accum = cost_adaptive_hist_accum - np.mean(cost_optimal_hist_accum, axis=0)

        regret_ratio_hist = cost_adaptive_hist / np.mean(cost_optimal_hist, axis=0)
        regret_ratio_hist_receding = cost_adaptive_hist_receding / np.mean(cost_optimal_hist_receding, axis=0)
        regret_ratio_hist_accum = cost_adaptive_hist_accum / np.mean(cost_optimal_hist_accum, axis=0)

        output_dict[control_scheme]['cost_adaptive_hist_receding'] = cost_adaptive_hist_receding
        output_dict[control_scheme]['cost_optimal_hist_receding'] = cost_optimal_hist_receding
        output_dict[control_scheme]['cost_adaptive_hist_accum'] = cost_adaptive_hist_accum
        output_dict[control_scheme]['cost_optimal_hist_accum'] = cost_optimal_hist_accum
        output_dict[control_scheme]['regret_hist'] = regret_hist
        output_dict[control_scheme]['regret_hist_receding'] = regret_hist_receding
        output_dict[control_scheme]['regret_hist_accum'] = regret_hist_accum
        output_dict[control_scheme]['regret_ratio_hist'] = regret_ratio_hist
        output_dict[control_scheme]['regret_ratio_hist_receding'] = regret_ratio_hist_receding
        output_dict[control_scheme]['regret_ratio_hist_accum'] = regret_ratio_hist_accum


def mainfun(training_type, testing_type, uncertainty_estimator, Ns, Nb, T, noise_pre_scale, noise_post_scale,
            cost_horizon, horizon_method, t_cost_fh, system_idx, system_kwargs, seed, parallel_option):
    # Set up output directory
    timestr = str(time()).replace('.','p')
    dirname_out = timestr+'_Ns_'+str(Ns)+'_T_'+str(T)+'_system_'+str(system_idx)+'_seed_'+str(seed)
    dirname_out = os.path.join('experiments', dirname_out)
    create_directory(dirname_out)

    # Seed the random number generator
    npr.seed(seed)

    # Problem data
    n, m, A, B, Q, R, W = gen_system_omni(system_idx, **system_kwargs)
    filename_out = 'system_dict.pickle'
    save_system(n, m, A, B, Q, R, W, dirname_out, filename_out)

    # Catch numerical error-prone case when system is open-loop unstable and not using control during training
    if specrad(A) > 1 and training_type == 'offline':
        response = yes_or_no("System is open-loop unstable, offline trajectories may cause numerical issues. Continue?")
        if not response:
            return

    # Initial state
    x0 = np.zeros(n)

    # Calculate the true optimal control given perfect information of the system
    Pare_true, Kare_true = dare_gain(A, B, Q, R)
    AKare_true = A + mdot(B, Kare_true)
    QKare_true = Q + mdot(Kare_true.T, R, Kare_true)

    # Compute cost horizon and optimal cost
    if cost_horizon == 'infinite':
        cost_are_true = np.trace(Pare_true)
    elif cost_horizon == 'finite':
        if horizon_method == 'fixed':
            pass # t_cost_fh = t_cost_fh
        elif horizon_method == 'from_opt':
            cost_gap = 0.01
            t_cost_fh = 0
            Pare_true_fh = np.copy(QKare_true)
            Ptare_true_fh = np.copy(QKare_true)
            while np.trace(Pare_true) / np.trace(Pare_true_fh) - 1 > cost_gap:
                Ptare_true_fh = mdot(AKare_true.T, Ptare_true_fh, AKare_true)
                Pare_true_fh += Ptare_true_fh
                t_cost_fh += 1
        Pare_true_fh = dctg(AKare_true, QKare_true, t_cost_fh)
        cost_are_true = np.trace(Pare_true_fh)

    # Time history
    t_hist = np.arange(T)

    # Time to begin forming model estimates
    t_start_estimate = 2*(n+m)
    if t_start_estimate < n+m:
        response = yes_or_no("t_start_estimate chosen < n+m. Continue?")
        if not response:
            return

    # Time to switch from exploration to exploitation
    t_explore = t_start_estimate+1
    if t_explore < n+m+1:
        response = yes_or_no("t_explore chosen < n+m+1. Continue?")
        if not response:
            return

    # Input exploration noise during explore and exploit phases
    u_explore_var = np.max(la.eig(W)[0])
    u_exploit_var = np.max(la.eig(W)[0])


    # Bisection tolerance
    bisection_epsilon = 0.01

    # Export the simulation options for later reference
    sim_options = {'training_type': training_type,
                   'testing_type': testing_type,
                   'uncertainty_estimator': uncertainty_estimator,
                   'Ns': Ns,
                   'Nb': Nb,
                   'T': T,
                   'system_idx': system_idx,
                   'seed': seed,
                   'bisection_epsilon': bisection_epsilon,
                   't_start_estimate': t_start_estimate,
                   't_explore': t_explore,
                   'u_explore_var': u_explore_var,
                   'u_exploit_var': u_exploit_var}
    filename_out = 'sim_options.pickle'
    pickle_export(dirname_out, filename_out, sim_options)

    control_schemes = ['certainty_equivalent', 'robust']
    output_dict = {}

    if training_type == 'offline':
        # Generate sample trajectory data (pure exploration)
        x_train_hist, u_train_hist, w_hist = generate_offline_data(n, m, A, B, W, Ns, T, u_explore_var, x0, seed)
    elif training_type == 'online':
        # Generate the additive noise sequence to apply on all algorithms
        w_hist = npr.multivariate_normal(mean=np.zeros(n), cov=W, size=[Ns, T])

    # Evaluate control schemes
    required_args = {'n': n,
                     'm': m,
                     'A': A,
                     'B': B,
                     'Q': Q,
                     'R': R,
                     'W': W,
                     'Ns': Ns,
                     'Nb': Nb,
                     'T': T,
                     'x0': x0,
                     'bisection_epsilon': bisection_epsilon,
                     't_start_estimate': t_start_estimate,
                     't_explore': t_explore,
                     't_cost_fh': t_cost_fh,
                     'cost_horizon': cost_horizon,
                     'Kare_true': Kare_true,
                     'u_explore_var': u_explore_var,
                     'u_exploit_var': u_exploit_var,
                     'noise_pre_scale': noise_pre_scale,
                     'noise_post_scale': noise_post_scale}

    if training_type == 'online':
        conditional_args = {}
    elif training_type == 'offline':
        conditional_args = {'x_train_hist': x_train_hist,
                            'u_train_hist': u_train_hist}
    for control_scheme in control_schemes:
        output_dict[control_scheme] = monte_carlo_group(training_type=training_type,
                                                        testing_type=testing_type,
                                                        control_scheme=control_scheme,
                                                        uncertainty_estimator=uncertainty_estimator,
                                                        required_args=required_args,
                                                        conditional_args=conditional_args,
                                                        w_hist=w_hist,
                                                        parallel_option=parallel_option)
    compute_derived_data(output_dict, testing_type)

    # Export relevant data
    filename_out = training_type+'_training_'+testing_type+'_testing_'+'comparison_results'+'.pickle'
    data_out = [output_dict, cost_are_true, t_hist, t_explore]
    pickle_export(dirname_out, filename_out, data_out)

    # Plotting
    show_plots = True
    if show_plots:
        from plotting import multi_plot
        multi_plot(output_dict, cost_are_true, t_hist, t_start_estimate)

    return output_dict


if __name__ == "__main__":
    # Choose between offline and online training data
    training_type = 'offline'

    # Choose between offline and online testing
    testing_type = 'online'

    # Choose the uncertainty estimation scheme
    # uncertainty_estimator = 'exact'
    # uncertainty_estimator = 'sample_transition_bootstrap'
    uncertainty_estimator = 'semiparametric_bootstrap'

    # Number of Monte Carlo samples
    Ns = 100000
    # Ns = 100

    # Number of bootstrap samples
    Nb = 100

    # Simulation time
    T = 200

    # Choose noise_pre_scale (AKA gamma), the pre-limit multiplicative noise scaling parameter, should be >= 1
    # "How much mult noise do you want?"
    noise_pre_scale = 1

    # Choose the post-limit multiplicative noise scaling parameter, must be between 0 and 1
    # "How much of the max possible mult noise do you want?"
    noise_post_scale = 1
    # noise_post_scale = 1 / noise_pre_scale

    # Choose cost horizon
    # cost_horizon = 'infinite'
    # horizon_method = None
    # t_cost_fh = None

    # cost_horizon = 'finite'
    # horizon_method = 'from_opt'
    # t_cost_fh = None

    cost_horizon = 'finite'
    horizon_method = 'fixed'
    t_cost_fh = 10

    # Random number generator seed
    seed = 1
    # seed = npr.randint(1000)

    # System to choose
    system_idx = 'scalar'

    if system_idx == 'scalar':
        system_kwargs = {'A': 1,
                         'B': 1,
                         'Q': 1,
                         'R': 0,
                         'W': 1}
    elif system_idx == 'rand':
        system_kwargs = {'n': 2,
                         'm': 1,
                         'spectral_radius': 1.1,
                         'noise_scale': 1,
                         'seed': seed}
    else:
        system_kwargs = {}

    # Parallel computation option
    # parallel_option = 'serial'
    parallel_option = 'parallel'

    # Run main
    output_dict = mainfun(training_type, testing_type, uncertainty_estimator, Ns, Nb, T, noise_pre_scale, noise_post_scale,
                          cost_horizon, horizon_method, t_cost_fh, system_idx, system_kwargs, seed, parallel_option)
