# -*- coding: utf-8 -*-


import numpy as np
import numpy.linalg as la
import typing

__author__ = "aeiwz"


def nipals(x: np.ndarray, y: np.ndarray,
           tol: float = 1e-10,
           max_iter: int = 1000,
           dot=np.dot) -> typing.Tuple:
    """
    Non-linear Iterative Partial Least Squares

    Parameters
    ----------
    x: np.ndarray
        Variable matrix with size n by p, where n number of samples,
        p number of variables.
    y: np.ndarray
        Dependent variable with size n by 1.
    tol: float
        Tolerance for the convergence.
    max_iter: int
        Maximal number of iterations.

    Returns
    -------
    w: np.ndarray
        Weights with size p by 1.
    u: np.ndarray
        Y-scores with size n by 1.
    c: float
        Y-weight
    t: np.ndarray
        Scores with size n by 1

    """
    if x.shape[0] != y.shape[0]:
        raise ValueError("Input arrays x and y must have the same number of samples.")

    u = y
    i = 0
    d = tol * 10
    while d > tol and i <= max_iter:
        w = dot(u, x) / dot(u, u)
        w /= la.norm(w)
        t = dot(x, w)
        c = dot(t, y) / dot(t, t)
        u_new = y * c / (c * c)
        d = la.norm(u_new - u) / la.norm(u_new)
        u = u_new
        i += 1

    return w, u, c, t
