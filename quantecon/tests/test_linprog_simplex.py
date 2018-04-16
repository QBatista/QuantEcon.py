"""
Tests for simplex_solver.py

"""

import numpy as np
from numpy.testing import assert_allclose
from quantecon import linprog_simplex


class TestSimplexSolver:
    def test_linear_upper_bound(self):
        # Maximize a linear function subject to only linear upper bound
        # constraints.
        # http://www.dam.brown.edu/people/huiwang/classes/am121/Archive/simplex_121_c.pdf
        c = np.array([3, 2])
        A_ub =np.array([[2, 1],
                        [1, 1],
                        [1, 0]])
        b_ub = np.array([10, 8, 4])

        x = linprog_simplex(-c, A_ub, b_ub) # Maximize

        solution = np.array([2., 6.])

        assert_allclose(x, solution)

    def test_mixed_constraints(self):
        # Minimize linear function subject to non-negative variables.
        # http://www.statslab.cam.ac.uk/~ff271/teaching/opt/notes/notes8.pdf
        c = np.array([6, 3])
        A_ub = np.array([[0, 3],
                         [-1, -1],
                         [-2, 1]])
        b_ub = np.array([2, -1, -1])

        x = linprog_simplex(c, A_ub, b_ub)

        solution = np.array([2/3, 1/3])

        assert_allclose(x, solution)

    def test_degeneracy(self):
        # http://mat.gsia.cmu.edu/classes/QUANT/NOTES/chap7.pdf
        c = np.array([2, 1])
        A_ub =np.array([[3, 1],
                        [1, -1],
                        [0, 1]])
        b_ub = np.array([6, 2, 3])

        x = linprog_simplex(-c, A_ub, b_ub) # Maximize

        solution = np.array([1., 3.])

        assert_allclose(x, solution)

    def test_cyclic_recovery(self):
        # http://www.math.ubc.ca/~israel/m340/kleemin3.pdf

        c = np.array([100, 10, 1]) * -1  # Maximize
        A_ub = np.array([[1, 0, 0],
                         [20, 1, 0],
                         [200, 20, 1]])
        b_ub = np.array([1, 100, 10000])

        x = linprog_simplex(c, A_ub, b_ub)

        solution = np.array([0., 0., 10000.])

        assert_allclose(x, solution)



if __name__ == '__main__':
    import sys
    import nose

    argv = sys.argv[:]
    argv.append('--verbose')
    argv.append('--nocapture')
    nose.main(argv=argv, defaultTest=__file__)
