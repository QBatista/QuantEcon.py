"""
Implements a linear programming solver using the "Simplex" method.

"""

import numpy as np
from numba import jit
from .util import gauss_jordan_pivoting, min_ratio_test


@jit(nopython=True, cache=True)
def linprog_simplex(c, A, b, maxiter=20000):
    """
    Solve the following linear programming problem using the "Simplex" method:

    Minimize:    c.T @ x
    Subject to:  A @ x ≤ b
                 x ≥ 0

    Jit-compiled in `nopython` mode.

    Parameters
    ----------
    c : ndarray(float, ndim=(n, 1))
        Coefficients of the linear objective function to be maximized.
    A : ndarray(float)
        2-D array which when multiplied by `x` gives the upper bound of the
        inequality constraints.
    b : ndarray(float)
        1-D array of values representing the upper-bound of each inequality
        constraint.
    maxiter : scalar(int)
        Maximum number of iterations.

    Return
    ----------
    x : ndarray(float)
        A basic solution to the linear programming problem.

    References
    ----------

    http://mat.gsia.cmu.edu/classes/QUANT/NOTES/chap7.pdf

    """
    tableau = _init_tableau(c, A, b)

    nrows, ncols = tableau.shape

    evar_idx = _search_evar(tableau)

    _iter = 0

    # If all coefficients are positive
    if evar_idx == -1:
        for evar_idx in range(1, c.shape[0]+1):
            not_found = True
            row_idx = 1
            while not_found:
                if tableau[row_idx, evar_idx] != 0:
                    not_found = False
                row_idx += 1
            gauss_jordan_pivoting(tableau, evar_idx, row_idx)

    while evar_idx != -1 and _iter < maxiter:
        argmins = np.arange(1, nrows)
        min_ratio_test(tableau, evar_idx, -1, argmins, nrows-1)
        gauss_jordan_pivoting(tableau, evar_idx, argmins[0])
        evar_idx = _search_evar(tableau)
        _iter += 1

    x = _find_solution(tableau)

    return x[:c.shape[0]]


@jit(nopython=True, cache=True)
def _init_tableau(c, A, b):
    """
    Create a tableau for a linear programming problem given an objective and
    constaints. Jit-compiled in `nopython` mode. See `linprog_simplex` for
    further details.

    Parameters
    ----------
    c : ndarray(float, ndim=(n, 1))
        Coefficients of the linear objective function to be maximized.
    A : ndarray(float)
        2-D array which when multiplied by `x` gives the upper bound of the
        inequality constraints.
    b : ndarray(float)
        1-D array of values representing the upper-bound of each inequality
        constraint.

    Return
    ----------
    tableau : ndarray(float)
        Matrix containing the standardized linear programming problem
        coefficients.

    """

    nrows, ncols = A.shape

    tableau = np.empty((nrows+1, ncols+nrows+2))

    tableau[0, 0] = 1.
    tableau[1:, 0] = 0.

    tableau[0, 1:(ncols+1)] = c.ravel()
    tableau[1:, 1:(ncols+1)] = A

    _range = np.arange(nrows+1)

    for i in range(nrows):
        tableau[i+1, ncols+1+i] = 1.
        tableau[_range != (i+1), ncols+1+i] = 0.

    tableau[0, -1] = 0.
    tableau[1:, -1] = b.ravel()

    return tableau


@jit(nopython=True, cache=True)
def _search_evar(tableau):
    """
    Search for a variable with a negative coefficient in `tableau`.
    Jit-compiled in `nopython` mode.

    Parameters
    ----------
    tableau : ndarray(float)
        Matrix containing the standardized linear programming problem
        coefficients.

    Return
    ----------
    idx : scalar(int)
        The index of a variable with a negative coefficient. If all variables
        have positive coefficients, return -1.

    """
    nrows, ncols = tableau.shape

    for i in range(ncols-2):
        idx = 1 + i
        if tableau[0, idx] < 0.:
            return idx
    idx = -1
    return idx


@jit(nopython=True, cache=True)
def _find_solution(tableau):
    """
    Return a basic solution to the linear programming problem. See
    `linprog_simplex` for further details.

    Parameters
    ----------
    tableau : ndarray(float)
        Matrix containing the standardized linear programming problem
        coefficients.

    Return
    ----------
    x : ndarray(float)
        A basic solution to the linear programming problem.

    """
    x = tableau[0, 1:-1]
    for i, x_i in enumerate(x):
        if x_i == 0.:
            not_found = True
            row = 1
            while not_found:
                if tableau[row, (i+1)] == 1.:
                    x[i] = tableau[row, -1]
                    not_found = False
                row += 1
        else:
            x[i] = 0.

    return x
