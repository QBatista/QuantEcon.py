"""
Author: Quentin Batista

Tests for repeated_game.py

"""

import numpy as np
from nose.tools import ok_
from quantecon.game_theory import NormalFormGame, Player, flow_u_1, flow_u_2, \
                                  flow_u, best_dev_i, worst_value_i, \
                                  initialize_hpl, outerapproximation, \
                                  RepeatedGame, RGUtil


class TestRepeatedGame:
    def setUp(self):
        pd_payoff = np.array([[9.0, 1.0], [10.0, 3.0]])

        self.A = Player(pd_payoff)
        self.B = Player(pd_payoff)
        nfg = NormalFormGame((self.A, self.B))

        # Tests construction of repeated game
        self.rpd = RepeatedGame(nfg, 0.75)
        self.C, self.H, self.Z = initialize_hpl(4, [0.0, 0.0], 1.0)

    def test_flow_utilities(self):
        ok_(abs(flow_u_1(self.rpd, 0, 0) - 9.0) < 1e-14)
        ok_(abs(flow_u_2(self.rpd, 1, 1) - 3.0) < 1e-14)
        ok_(max(abs(flow_u(self.rpd, 1, 1) - [3.0, 3.0])) < 1e-14)

    def test_best_deviation(self):
        ok_(best_dev_i(self.rpd, 0, 0) == 1)

    def test_unit_circle(self):
        H = RGUtil.unitcircle(4)
        points = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
        ok_(max(abs(H - points).flatten()) < 1e-12)

    def test_subgradient_hyperplane_init(self):
        ok_(max(abs(self.C - np.ones(4))) < 1e-12)
        ok_(max(abs(self.H - self.Z.T).flatten()) < 1e-12)

    def test_worst_value(self):
        ok_(abs(worst_value_i(self.rpd, self.H, self.C, 0) + 1.0) < 1e-12)

    def test_outer_approximation(self):
        vertices = outerapproximation(self.rpd)
        expected_output = [[10.0, 3.97266],
                           [9.0, 9.0],
                           [3.97266, 10.0],
                           [3.0, 3.0],
                           [10.0, 3.0]]

        for value in expected_output:
            ok_(value in vertices)


if __name__ == '__main__':
    import sys
    import nose

    argv = sys.argv[:]
    argv.append('--verbose')
    argv.append('--nocapture')
    nose.main(argv=argv, defaultTest=__file__)