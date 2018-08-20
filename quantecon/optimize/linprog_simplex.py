"""
Implements a linear programming solver using the "Simplex" method.

"""

import numpy as np
from numba import jit
from collections import namedtuple
from ..util import pivot_operation, min_ratio_test, lex_min_ratio_test


results = namedtuple('results', 'fun sol dual_vars status')

@jit(nopython=True, cache=True)
def linprog_simplex(tableau, N, M_ub=0, M_eq=0, tie_breaking_rule=0,
                    maxiter=10000, tolpiv=1e-9, featol=1e-6,
                    tol_ratio_diff=1e-15, verbose=False):
    """
    Solve the following LP problem using the Simplex method:

    Maximize:    c.T @ x
    Subject to:  A_ub @ x ≤ b_ub
                 A_eq @ x = b_eq
                 x ≥ 0

    Jit-compiled in `nopython` mode.

    Parameters
    ----------
    tableau : ndarray(float, ndim=2)
        2-D array containing the standardized LP problem in detached
        coefficients form augmented with the artificial variables, the
        infeasibility form and with nonnegative constant terms. The
        infeasibility form is assumed to be placed in the last row, and the
        objective function in the second last row.

    N : scalar(int)
        Number of control variables.

    M_ub : scalar(int)
        Number of inequality constraints.

    M_eq : scalar(int)
        Number of equality constraints.

    tie_breaking_rule : scalar(int), optional(default=0)
        An integer representing the rule used to break ties:
        0 : Bland's rule
        1 : Lexicographic rule

    maxiter : scalar(int), optional(default=10000)
        Maximum number of pivot operation for each phase.

    tolpiv : scalar(float), optional(default=1e-9)
        Tolerance for treating an element as nonpositive during the
        minimum-ratio test.

    featol : scalar(float), optional(default=1e-6)
        Tolerance for treating an element as nonpositive when choosing the
        pivot column.

    tol_ratio_diff : scalar(float), optional(default=1e-15)
        Tolerance for comparing candidate minimum ratios.

    Return
    ----------
    results : namedtuple
        fun - The value of the maximized objective.
        sol - If `status` is 0, a basic solution to the linear programming
               problem.
        dual_vars - If `status` is 0, a basic solution to the dual linear
                    programming problem.
        status - An integer representing the exit status of the optimization:
                 0 : Optimization terminated successfully
                 1 : Iteration limit reached

    Examples
    --------
    >>> M_ub = 3
    >>> N = 2
    >>> tableau = np.array([[2., 1., 1., 0., 0., 1., 0., 0., 0., 10.],
                            [1., 1., 0., 1., 0., 0., 1., 0., 0., 8.],
                            [1., 0., 0., 0., 1., 0., 0., 1., 0., 4.],
                            [3., 2., 0., 0., 0., 0., 0., 0., 1., 0.],
                            [-4., -2., -1., -1., -1., 0., 0., 0., 0., -22.]])

    >>> linprog_simplex(tableau, N, M_ub)
    results(fun=-0.0, sol=array([2., 6.]), dual_vars=array([-0., -0., -0.]),
            status=0)

    References
    ----------

    [1] Dantzig, George B., "Linear programming and extensions". Rand
        Corporation Research Study Princeton Univ. Press, Princeton, NJ, 1963

    [2] Bland, Robert G. (May 1977). "New finite pivoting tie_breaking_rules
        for the simplex method". Mathematics of Operations Research.
        2 (2): 103–107.

    [3] Pan, Ping-Qi., "Linear Programming Computation". Springer,
        Berlin (2014).

    [4] https://www.whitman.edu/Documents/Academics/Mathematics/lewis.pdf

    [5] http://mat.gsia.cmu.edu/classes/QUANT/NOTES/chap7.pdf

    """
    M = M_ub + M_eq

    tableau[M, 0:N] = -tableau[M, 0:N] # Maximize

    basis = np.empty(M, dtype=np.int64)
    for i in range(M):  # Warning: the artificial variables are used as a basis
        basis[i] = N + M_ub + i

    # Phase I
    status = simplex_algorithm(tableau, basis, M, tie_breaking_rule,
                               maxiter, tolpiv, featol, tol_ratio_diff)

    if status == 1:
        if verbose:
            print("WARNING: The maximum number of iterations has been" +
                  "reached in Phase I")

        return results(0., np.empty(1), np.empty(1), status)

    if abs(tableau[-1, -1]) > featol:
        raise ValueError("The problem appears to be infeasible")

    # Update `tableau`
    updated_tableau = tableau[:-1, tableau[-1, :] <= featol]

    # Phase II
    status = simplex_algorithm(updated_tableau, basis, M, tie_breaking_rule,
                               maxiter, tolpiv, featol, tol_ratio_diff)

    tableau[:-1, tableau[-1, :] <= featol] = updated_tableau[:, :]

    # Find solution
    x, lambd = np.zeros(N, dtype=np.float64), np.zeros(M, dtype=np.float64)

    fun = get_solution(tableau, basis, x, lambd)

    if status == 1 and verbose:
        print("WARNING: The maximum number of iterations has been" +
              "reached in Phase II")

    return results(fun, x, lambd, status)


@jit(nopython=True, cache=True)
def simplex_algorithm(tableau, basis, M, tie_breaking_rule, maxiter,
                      tolpiv, featol, tol_ratio_diff):
    """
    Execute the simplex algorithm on `tableau` using `tie_breaking_rule`.
    Jit-compiled in `nopython` mode.

    Parameters
    ----------
    tableau : ndarray(float, ndim=2)
        2-D array to be modified inplace which contains the LP problem in
        detached coefficients form. The objective is assumed to be placed in
        the last row.

    basis : ndarray(int, ndim=1)
        1-D array containing the indices of the basic variables in `tableau`.

    M : scalar(int)
        Total number of constraints in the LP problem.

    tie_breaking_rule : scalar(int)
        Rule used to break ties when choosing pivot elements.
        0 : Bland's rule
        1 : Lexicographic rule

    maxiter : scalar(int), optional(default=10000)
        Maximum number of pivot operation for each phase.

    tolpiv : scalar(float)
        Tolerance for treating an element as nonpositive during the
        minimum-ratio test.

    featol : scalar(float)
        Tolerance for treating an element as nonpositive when choosing the
        pivot column.

    tol_ratio_diff : scalar(float)
        Tolerance for comparing candidate minimum ratios.

    Return
    ----------
    status : scalar(int)
        An integer representing the exit status of the optimization:
        0 : Optimization terminated successfully
        1 : Iteration limit reached

    """
    pivot_col = _choose_pivot_col(tableau, basis, tie_breaking_rule, featol)

    argmins = np.arange(0, M)

    for num_iter in range(maxiter):
        if pivot_col == -1:
            return 0

        pivot_row, num_argmins = _choose_pivot_row(tableau, pivot_col,
                                                   argmins, M,
                                                   tie_breaking_rule, tolpiv,
                                                   tol_ratio_diff)

        # Check if there is no lower bound
        if num_argmins == 0:
            raise ValueError("The problem appears to be unbounded.")

        pivot_operation(tableau, (pivot_row, pivot_col))

        # Update `basis`
        basis[pivot_row] = pivot_col

        # Update `pivot_col`
        pivot_col = _choose_pivot_col(tableau, basis, tie_breaking_rule,
                                      featol)

    return 1


@jit(nopython=True, cache=True)
def _choose_pivot_col(tableau, basis, tie_breaking_rule, featol):
    """
    Choose the column index of the pivot element in `tableau` using
    `tie_breaking_rule`. Jit-compiled in `nopython` mode.

    Parameters
    ----------
    tableau : ndarray(float, ndim=2)
        2-D array which contains the LP problem in detached coefficients form.

    basis : ndarray(int, ndim=1)
        1-D array containing the indices of the basic variables in `tableau`.

    tie_breaking_rule : scalar(int)
        An integer representing the rule used to break ties:
        0 : Bland's rule
        1 : Lexicographic rule

    featol : scalar(float)
        Tolerance for treating an element as nonpositive.

    Return
    ----------
    idx : scalar(int)
        The index of the variable with a negative coefficient and the lowest
        column index. If all variables have nonnegative coefficients, return
        -1.

    """
    if tie_breaking_rule == 0:  # Bland's tie_breaking_rule
        for idx in range(tableau.shape[1]-1):
            if tableau[-1, idx] < -featol and (idx != basis).all():
                return idx

    if tie_breaking_rule == 1:  # Lexicographic rule
        idx = tableau[-1, :-1].argmin()
        if tableau[-1, idx] < -featol:
            return idx

    return -1


@jit(nopython=True, cache=True)
def _choose_pivot_row(tableau, pivot_col, argmins, M, tie_breaking_rule,
                      tolpiv, tol_ratio_diff):
    """
    Choose the row index of the pivot element in `tableau` using
    `tie_breaking_rule`. Jit-compiled in `nopython` mode.

    Parameters
    ----------
    tableau : ndarray(float, ndim=2)
        2-D array which contains the LP problem in detached coefficients form.

    pivot_col : scalar(int)
        The index of the chosen pivot column.

    argmins : ndarray(int, ndim=1)
        Array containing the indices of the candidate rows. Modified in
        place to store the indices of minimizing rows.

    M : scalar(int)
        Total number of constraints in the LP problem.

    tie_breaking_rule : scalar(int)
        An integer representing the rule used to break ties:
        0 : Bland's rule
        1 : Lexicographic rule

    tolpiv : scalar(float)
        Tolerance for treating an element as nonpositive during the
        minimum-ratio test.

    tol_ratio_diff : scalar(float)
        Tolerance for comparing candidate minimum ratios.

    Return
    ----------
    pivot_row : scalar(int)
        The index of the chosen pivot row.

    num_argmins : scalar(int)
        Number of minimizing rows.

    """
    if tie_breaking_rule == 0:  # Bland's tie_breaking_rule
        num_argmins = min_ratio_test(tableau, pivot_col, -1, argmins, M,
                                     tolpiv, tol_ratio_diff)
        pivot_row = argmins[:num_argmins].min()

        # Restore `argmins`
        for i in range(M):
            argmins[i] = i

        return pivot_row, num_argmins

    if tie_breaking_rule == 1:  # Lexicographic rule
        return lex_min_ratio_test(tableau, pivot_col, 0, tableau.shape[1],
                                  argmins, M, tolpiv, tol_ratio_diff)

    else:
        return -1, -1


@jit(nopython=True, cache=True)
def get_solution(tableau, basis, x, lambd):
    """
    Find a basic solution to the LP problem in `tableau`. Jit-compiled in
    `nopython` mode.

    Parameters
    ----------
    tableau : ndarray(float, ndim=2)
        2-D array which contains the LP problem in detached coefficients form.

    basis : ndarray(int, ndim=1)
        1-D array containing the indices of the basic variables in `tableau`.

    x : ndarray(float, ndim=1)
        1-D array to be filled with the solution values.

    lambd : ndarray(float, ndim=1)
        1-D array to be filled with the dual solution values.

    Return
    ----------
    fun : scalar(float)
        The value of the maximized objective function

    """
    N, M = x.size, lambd.size

    for i in range(M):
        if basis[i] < N:
            x[basis[i]] = tableau[i, -1]

    for j in range(M):
        lambd[j] = tableau[-1, N+j] * (-1)

    fun = tableau[-1, -1] * (-1)

    return fun
