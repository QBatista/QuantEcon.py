"""
Tools for repeated games.

"""

import numpy as np
from scipy.spatial import HalfspaceIntersection
from quantecon.optimize import linprog_simplex
from quantecon.util import make_tableau
from .pure_nash import pure_nash_brute_gen
from ..ce_util import gridmake


class RepeatedGame:
    """
    Class representing an N-player repeated game.

    Parameters
    ----------
    stage_game : NormalFormGame
                 The stage game used to create the repeated game.

    delta : scalar(float)
            The common discount rate at which all players discount the future.

    """
    def __init__(self, stage_game, delta):
        self.sg = stage_game
        self.delta = delta
        self.N = stage_game.N
        self.nums_actions = stage_game.nums_actions

    def outerapproximation(self, nH=32, tol=1e-8, maxiter=1000,
                           check_pure_nash=True, verbose=False, nskipprint=50):
        """
        Approximates the set of equilibrium value set for a repeated game with
        the outer hyperplane approximation described by Judd, Yeltekin,
        Conklin 2002.

        Parameters
        ----------
        rpd : RepeatedGame
            Two player repeated game.

        nH : scalar(int), optional(default=32)
            Number of subgradients used for the approximation.

        tol : scalar(float), optional(default=1e-8)
            Tolerance in differences of set.

        maxiter : scalar(int), optional(default=500)
            Maximum number of iterations.

        check_pure_nash : bool, optional(default=True)
            Whether to perform a check about whether a pure Nash equilibrium
            exists.

        verbose : bool, optional(default=False)
            Whether to display updates about iterations and distance.

        nskipprint : scalar(int), optional(default=50)
            Number of iterations between printing information (assuming
            verbose=true).

        Returns
        -------
        vertices : ndarray(float, ndim=2)
                   Vertices of the outer approximation of the value set.

        """
        sg, delta = self.sg, self.delta
        p0, p1 = sg.players
        flat_payoff_array0 = p0.payoff_array.flatten()
        flat_payoff_array1 = p1.payoff_array.flatten()

        try:
            next(pure_nash_brute_gen(sg))
        except StopIteration:
            raise ValueError("No pure action Nash equilibrium exists in" +
                             " stage game")

        # Get number of actions for each player and create action space
        nA0, na0 = p0.num_actions, p1.num_actions
        nAS = nA0 * na0
        AS = gridmake(np.array(range(nA0)), np.array(range(na0)))

        # Create the unit circle, points, and hyperplane levels
        C, H, Z = initialize_sg_hpl(flat_payoff_array0, flat_payoff_array1, nH)
        Cnew = C.copy()

        # Create matrices for linear programming
        c, A, b = initialize_LP_matrices(delta, H)

        # bounds on w are [-Inf, Inf] while bounds on slack are [0, Inf]
        lb = -np.inf
        ub = np.inf

        # Allocate space to store all solutions
        Cia = np.zeros(nAS)
        Wia = np.zeros([2, nAS])

        # Set iterative parameters and iterate until converged
        itr, dist = 0, 10.0

        while (itr < maxiter) and (dist > tol):
            # Compute the current worst values for each agent
            _w0 = worst_value_i(self, H, C, 0)
            _w1 = worst_value_i(self, H, C, 1)

            # Iterate over all subgradients
            for ih in range(nH):
                #
                # Subgradient specific instructions
                #
                h0, h1 = H[ih, :]

                # Update all set constraints - Copies elements 1:nH of C into b
                b[:nH] = C

                # Put the right objective into c
                c[0] = h0
                c[1] = h1

                for ia in range(nAS):
                    # Action specific instruction
                    a0, a1 = AS[ia, :]

                    # Update incentive constraints
                    b[nH] = (1 - delta) * flow_u_0(self, a0, a1) - \
                            (1 - delta) * best_day_payoff_0(self, a1) - \
                        delta * _w0
                    b[nH + 1] = (1 - delta) * flow_u_1(self, a0, a1) - \
                                (1 - delta) * best_dev_payoff_1(self, a0) - \
                        delta * _w1

                    # Pull out optimal value and compute
                    M_ub, N = A.shape

                    tableau = make_tableau(c, A, b)
                    w_sol = linprog_simplex(tableau, N, M_ub)[1]

                    value = (1-delta)*flow_u(self, a0, a1) + delta * w_sol

                    # Save hyperplane level and continuation promises
                    Cia[ia] = h0 * value[0] + h1 * value[1]
                    Wia[:, ia] = value

                # Action which pushes furthest in direction h_i
                astar = np.argmax(Cia)
                a0star, a1star = AS[astar, :]

                # Get hyperplane level and continuation value
                Cstar = Cia[astar]
                Wstar = Wia[:, astar]
                if Cstar > -1e10:
                    Cnew[ih] = Cstar
                else:
                    raise Error("Failed to find feasible action/continuation" +
                                " pair")

                # Update the points
                Z[:, ih] = \
                    (1 - delta) * flow_u(self, a0star, a1star) + \
                    delta * np.array([Wstar[0], Wstar[1]])

            # Update distance and iteration counter
            dist = np.max(abs(C - Cnew))
            itr += 1

            if verbose and (itr % nskipprint == 0):
                print("\riter {} -- dist {}.".format(itr, dist), end="")

            if itr >= maxiter:
                raise ValueError("Maximum Iteration Reached")

            # Update hyperplane levels
            C[:] = Cnew

        # Given the H-representation `(H, C)` of the computed polytope of
        # equilibrium payoff profiles, we obtain its V-representation
        # `vertices` using Scipy
        p = HalfspaceIntersection(np.column_stack((H, -C)), np.mean(Z, axis=1))
        vertices = p.intersections

        # Reduce the number of vertices by rounding points to the tolerance
        tol_int = int(round(abs(np.log10(tol))) - 1)

        # Find vertices that are unique within tolerance level
        vertices = np.vstack({tuple(row) for row in
                              np.round(vertices, tol_int)})

        return vertices


def unitcircle(npts):
    """
    Places `npts` equally spaced points along the 2 dimensional circle and
    returns the points with x coordinates in first column and y coordinates
    in second column.

    Parameters
    ----------
    npts : scalar(float)
        Number of points.

    Returns
    -------
    pts : ndarray(float, dim=2)
        The coordinates of equally spaced points.

    """
    degrees = np.linspace(0, 2 * np.pi, npts + 1)

    pts = np.zeros((npts, 2))
    for i in range(npts):
        x = degrees[i]
        pts[i, 0] = np.cos(x)
        pts[i, 1] = np.sin(x)
    return pts


def initialize_hpl(nH, o, r):
    """
    Initializes subgradients, extreme points and hyperplane levels for the
    approximation of the convex value set of a 2 player repeated game.

    Parameters
    ----------
    nH : scalar(int)
        Number of subgradients used for the approximation.

    o : ndarray(float, ndim=2)
        Origin for the approximation.

    r: scalar(float)
       Radius for the approximation.

    Returns
    -------
    C : ndarray(float, ndim=1)
        The array containing the hyperplane levels.

    H : ndarray(float, ndim=2)
        The array containing the subgradients.

    Z : ndarray(float, ndim=2)
        The array containing the extreme points of the value set.

    """
    # Create unit circle
    H = unitcircle(nH)
    HT = H.T

    # Choose origin and radius for big approximation
    Z = np.zeros((2, nH))
    C = np.zeros(nH)

    for i in range(nH):
        temp_val0 = o[0] + r * HT[0, i]
        temp_val1 = o[1] + r * HT[1, i]
        Z[0, i] = temp_val0
        Z[1, i] = temp_val1
        C[i] = temp_val0 * HT[0, i] + temp_val1 * HT[1, i]

    return C, H, Z


def initialize_sg_hpl(flat_payoff_array0, flat_payoff_array1, nH):
    """
    Initializes subgradients, extreme points and hyperplane levels for the
    approximation of the convex value set of a 2 player repeated game by
    choosing an appropriate origin and radius.

    Parameters
    ----------
    flat_payoff_array0 : ndarray(float, ndim=2)
        Flattened payoff array for player 0.

    flat_payoff_array1 : ndarray(float, ndim=2)
        Flattened payoff array for player 1.

    nH : scalar(int)
        Number of subgradients used for the approximation.

    Returns
    -------
    C : ndarray(float, ndim=1)
        The array containing the hyperplane levels.

    H : ndarray(float, ndim=2)
        The array containing the subgradients.

    Z : ndarray(float, ndim=2)
        The array containing the extreme points of the value set.

    """
    # Choose the origin to be mean of max and min payoffs
    p0_min, p0_max = min(flat_payoff_array0), max(flat_payoff_array0)
    p1_min, p1_max = min(flat_payoff_array1), max(flat_payoff_array1)

    o = np.array([(p0_min + p0_max) / 2.0, (p1_min + p1_max) / 2.0])
    r0 = np.max(((p0_max - o[0]) ** 2, (o[0] - p0_min) ** 2))
    r1 = np.max(((p1_max - o[1]) ** 2, (o[1] - p1_min) ** 2))
    r = np.sqrt(r0 + r1)

    return initialize_hpl(nH, o, r)


def initialize_LP_matrices(delta, H):
    """
    Initializes matrices for linear programming problems.

    Parameters
    ----------
    delta : scalar(float)
        The common discount rate at which all players discount the future.

    H : ndarray(float, ndim=2)
        Subgradients used to approximate value set.

    Returns
    -------
    c : ndarray(float, ndim=1)
        Vector used to determine which subgradient is being used.

    A : ndarray(float, ndim=2)
        Matrix with nH set constraints and to be filled with 2 additional
        incentive compatibility constraints.

    b : ndarray(float, ndim=1)
        Vector to be filled with the values for the constraints.

    """
    # Total number of subgradients
    nH = H.shape[0]

    # Create the c vector (objective)
    c = np.zeros(2)

    # Create the A matrix (constraints)
    A = np.zeros((nH + 2, 2))
    A[0:nH, :] = H
    A[nH, :] = -delta, 0.
    A[nH+1, :] = 0., -delta

    # Create the b vector (constraints)
    b = np.zeros(nH + 2)

    return c, A, b


# Flow utility in terms of the players actions
def flow_u_0(rpd, a0, a1): return rpd.sg.players[0].payoff_array[a0, a1]


def flow_u_1(rpd, a0, a1): return rpd.sg.players[1].payoff_array[a1, a0]


def flow_u(rpd, a0, a1):
    return np.array([flow_u_0(rpd, a0, a1), flow_u_1(rpd, a0, a1)])


# Computes the payoff of the best deviation
def best_day_payoff_0(rpd, a1):
    return max(rpd.sg.players[0].payoff_array[:, a1])


def best_dev_payoff_1(rpd, a0):
    return max(rpd.sg.players[1].payoff_array[:, a0])


def worst_value_i(rpd, H, C, i):
    """
    Returns the worst possible payoff for player i.

    Parameters
    ----------
    rpd : RepeatedGame
        Two player repeated game instance.
    H : ndarray(float, ndim=2)
        Subgradients used to approximate value set.
    C : ndarray(float, ndim=1)
        Hyperplane levels used to approximate the value set.
    i : scalar(int)
        The player of interest.

    Returns
    -------
    out : scalar(float)
          Worst possible payoff of player i.

    """
    # Objective depends on which player we are minimizing
    c = np.zeros(2)
    c[i] = -1.0

    M_ub, N = H.shape
    tableau = make_tableau(c, A_ub=H, b_ub=C)
    out = linprog_simplex(tableau, N, M_ub)

    return out[1][i]
