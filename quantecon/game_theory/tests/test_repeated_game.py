"""
Tests for repeated_game.py

"""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal, assert_equal
from quantecon.game_theory import NormalFormGame, Player
from nose.tools import raises
from ..repeated_game import RepeatedGame, unitcircle


class TestRepeatedGame:
    def setUp(self):
        pd_payoff = np.array([[9.0, 1.0], [10.0, 3.0]])

        self.A = Player(pd_payoff)
        self.B = Player(pd_payoff)
        nfg = NormalFormGame((self.A, self.B))

        # Tests construction of repeated game
        delta = 0.75

        self.rpd = RepeatedGame(nfg, delta)

    def test_unitcircle(self):
        incr = 2 * np.pi / 5
        pts = np.array([[np.cos(0. * incr), np.sin(0. * incr)],
                        [np.cos(1. * incr), np.sin(1. * incr)],
                        [np.cos(2. * incr), np.sin(2. * incr)],
                        [np.cos(3. * incr), np.sin(3. * incr)],
                        [np.cos(4. * incr), np.sin(4. * incr)]])

        npts = 5
        circle = unitcircle(npts)

        assert_array_equal(circle, pts)

    def test_outerapproximation(self):
        vertices = self.rpd.outerapproximation()
        expected_output = np.array([[10.0, 3.97266],
                                    [9.0, 9.0],
                                    [3.97266, 10.0],
                                    [3.0, 3.0],
                                    [10.0, 3.0]])

        assert_allclose(vertices, expected_output)

    @raises(ValueError)
    def test_no_pure_nash(self):
        matching_pennies_bimatrix = [[(1, -1), (-1, 1)],
                                     [(-1, 1), (1, -1)]]

        g_MP = NormalFormGame(matching_pennies_bimatrix)
        rpd = RepeatedGame(g_MP, np.random.rand())

        rpd.outer_approximation()


if __name__ == '__main__':
    import sys
    import nose

    argv = sys.argv[:]
    argv.append('--verbose')
    argv.append('--nocapture')
    nose.main(argv=argv, defaultTest=__file__)
