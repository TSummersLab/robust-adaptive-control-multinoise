"""
Problem data generation.
Generic outputs are:
:param n: Number of states, integer
:param m: Number of inputs, integer
:param A: System state matrix, n x n matrix
:param B: System control input matrix, n x m matrix
:param Q: State-dependent quadratic cost, n x n matrix
:param R: Control-dependent quadratic cost, m x m matrix
:param W: Additive noise covariance, n x n matrix
"""

import numpy as np
import numpy.random as npr
import numpy.linalg as la
import scipy.linalg as sla

from utility.matrixmath import specrad, mdot
from utility.pickle_io import pickle_import, pickle_export


def gen_rand_system(n=3, m=2, spectral_radius=0.9, noise_scale=1, noise_shape='eye', seed=1):
    """
    Generate a random system
    :param n: Number of states, integer
    :param m: Number of inputs, integer
    :param spectral_radius: Open-loop spectral radius of A, float
    :param noise_scale: Scaling of noise covariance, float
    :param noise_shape: Shape of noise covariance, string, valid options are 'full', 'diag', 'eye'
    :param seed: Seed for random number generator, positive integer
    :returns: Number of states, number of inputs, state matrix, input matrix, state cost matrix, input cost matrix,
              additive noise covariance matrix
    """

    npr.seed(seed)
    A = npr.randn(n, n)
    A *= spectral_radius/specrad(A)
    B = npr.randn(n, m)
    U = npr.randn(n, n)
    V = npr.randn(m, m)
    Y = npr.randn(n, n)
    Q = np.dot(U, U.T)
    R = np.dot(V, V.T)
    if noise_shape == 'full':
        W = noise_scale*np.dot(Y, Y.T)
    elif noise_shape == 'diag':
        W = noise_scale*np.diag(np.diag(np.dot(Y, Y.T)))
    elif noise_shape == 'eye':
        W = noise_scale*np.eye(n)
    return n, m, A, B, Q, R, W


def gen_scalar_system(A=1, B=1, Q=1, R=1, W=1):
    # Scalar system
    n = 1
    m = 1
    # Ensure arrays are two-dimensional in case A, B, Q, R, W are scalars
    A, B, Q, R, W = [var*np.eye(1) for var in [A, B, Q, R, W]]
    return n, m, A, B, Q, R, W


def gen_pendulum_system(inverted, mass=10, damp=2, dt=0.1, Q=None, R=None, W=None):
    # Pendulum with forward Euler discretization
    n = 2
    m = 1
    if inverted:
        sign = 1
    else:
        sign = -1
    A = np.array([[1, dt], [sign*mass*dt, 1-damp*dt]])
    B = np.array([[0], [dt]])
    if Q is None:
        Q = np.eye(n)
    if R is None:
        R = np.eye(m)
    if W is None:
        W = 0.001*np.diag([0.01, 1])

    return n, m, A, B, Q, R, W


def gen_example_system(idx):
    """
    Example systems
    :param idx: Selection integer to pick the example system.
    """

    n, m, A, B, Q, R, W = None, None, None, None, None, None, None

    if idx == 1:
        # Stable
        n = 2
        m = 1
        A = np.array([[ 0.9, 0.2],
                      [-0.4, 0.9]])
        B = np.array([[-0.5],
                       [0.8]])
        Q = np.eye(n)
        R = np.eye(m)
        # W = 50*np.eye(n)
        # W = 10*np.eye(n)
        W = 1*np.eye(n)


    elif idx == 2:
        n = 2
        m = 1
        A = np.array([[ 1.4, 0.5],
                      [-0.8, 0.1]])
        B = np.array([[-0.3],
                       [0.6]])
        Q = np.eye(n)
        R = np.eye(m)
        W = np.eye(n)

    # another possibly interesting example?
    # A = [1.43 0.46
    #     -0.83 0.12]
    # B = [-0.34 0.62]

    elif idx == 3:
        # Unstable, marginally controllable
        # Kare is high
        n = 2
        m = 1
        A = np.array([[0.24, 2.60],
                      [2.84, 2.00]])
        B = np.array([[1.6],
                      [2.3]])
        Q = np.eye(n)
        R = np.eye(m)
        W = np.eye(n)

    elif idx == 4:
        # Unstable, marginally controllable, more controllable than above
        # Kare is high
        n = 2
        m = 1
        A = np.array([[0.2, 2.6],
                      [2.8, 2.0]])
        B = np.array([[1.6],
                      [2.3]])
        Q = np.eye(n)
        R = np.eye(m)
        W = 0.1*np.eye(n)

    elif idx == 5:
        # Marginally stable, marginally controllable
        # Kare is medium
        n = 2
        m = 1
        A = np.array([[0.05, 0.65],
                      [0.70, 0.50]])
        B = np.array([[1.6],
                      [2.3]])
        Q = np.eye(n)
        R = np.eye(m)
        W = 1*np.eye(n)

    elif idx == 100:
        # Non-inverted pendulum with forward Euler discretization
        # Kare is medium
        n = 2
        m = 1
        dt = 0.1
        mass = 10
        damp = 2
        A = np.array([[1, dt], [-mass*dt, 1-damp*dt]])
        B = np.array([[0], [dt]])
        Q = np.eye(n)
        R = np.eye(m)
        W = 0.01*np.diag([0.01, 1])

    elif idx == 101:
        # Inverted pendulum with forward Euler discretization
        # Kare is medium
        n = 2
        m = 1
        dt = 0.1
        mass = 10
        A = np.array([[1, dt], [mass*dt, 1]])
        B = np.array([[0], [dt]])
        Q = np.eye(n)
        R = np.eye(m)
        W = 0.01*np.diag([0.01, 1])

    elif idx == 102:
        # Unstable Laplacian from Dean et al.
        n = 3
        m = 3
        A = np.array([[1.01, 0.01, 0.00],
                      [0.01, 1.01, 0.01],
                      [0.00, 0.01, 1.01]])
        B = np.eye(m)
        Q = 10*np.eye(n)
        R = np.eye(m)
        W = np.eye(n)

    elif idx == 103:
        # Example from "On the Robustness of LQ Regulators" by Soroka and Shaked
        # https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=1103612&tag=1

        n = 2
        m = 1
        Ac = np.array([[-1, 0],
                       [0, -2]])
        Bc = np.array([[1],
                       [1]])
        dt = 0.01
        A = sla.expm(Ac*dt)
        B = mdot(A-np.eye(n), la.solve(Ac, Bc))
        C = np.ones(n)
        Qc = np.outer(C, C)
        Rc = 0.001*np.eye(m)
        Q = Qc*dt
        R = Rc*dt
        A = np.array([[0.99, 0.00],
                      [0.00, 0.98]])
        B = np.array([[0.01],
                      [0.01]])
        Q = np.eye(n)
        R = 0.001*np.eye(m)
        W = 0.01*np.eye(n)

    elif idx == 201:
        # Scalar system, stable
        n = 1
        m = 1
        A = 0.9*np.eye(n)
        B = np.eye(m)
        Q = np.eye(n)
        R = np.eye(m)
        W = 10*np.eye(n)

    else:
        raise Exception('Invalid system index chosen, please choose a different one')

    return n, m, A, B, Q, R, W


def gen_system_omni(system_idx, **kwargs):
    """
    Wrapper for system generation functions.
    """
    if system_idx == 'inverted_pendulum':
        return gen_pendulum_system(inverted=True, **kwargs)
    elif system_idx == 'noninverted_pendulum':
        return gen_pendulum_system(inverted=False, **kwargs)
    elif system_idx == 'scalar':
        return gen_scalar_system(**kwargs)
    elif system_idx == 'rand':
        return gen_rand_system(**kwargs)
    else:
        return gen_example_system(idx=system_idx)


def save_system(n, m, A, B, Q, R, W, dirname_out, filename_out):
    variables = [n, m, A, B, Q, R, W]
    variable_names = ['n', 'm', 'A', 'B', 'Q', 'R', 'W']
    system_data = dict(((variable_name, variable) for variable_name, variable in zip(variable_names, variables)))
    pickle_export(dirname_out, filename_out, system_data)


def load_system(filename_in):
    system_data = pickle_import(filename_in)
    variable_names = ['n', 'm', 'A', 'B', 'Q', 'R', 'W']
    return [system_data[variable] for variable in variable_names]