"""Extra math functions"""
# Author: Ben Gravell

import numpy as np


def quadratic_formula(a, b, c):
    """
    Solve the quadratic equation 0 = a*x**2 + b*x + c using the quadratic formula.
    """
    if a == 0:
        return [-c/b, np.nan]
    disc = b**2 - 4*a*c
    disc_sqrt = disc**0.5
    den = 2*a
    roots = [(-b+disc_sqrt)/den, (-b-disc_sqrt)/den]
    return roots


def symlog(X,scale=1):
    """Symmetric log transform"""
    return np.multiply(np.sign(X),np.log(1+np.abs(X)/(10**scale)))