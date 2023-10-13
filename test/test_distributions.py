import unittest

import numpy as np

from fglib2.distributions import Multinomial
from fglib2.variables import Symbolic
from fglib import nodes, rv


class MultinomialTestCase(unittest.TestCase):

    x: Symbolic
    y: Symbolic
    z: Symbolic

    @classmethod
    def setUpClass(cls):
        np.random.seed(69)
        cls.x = Symbolic("X", range(2))
        cls.y = Symbolic("Y", range(3))
        cls.z = Symbolic("Z", range(5))

    def test_creation(self):
        distribution = Multinomial([self.x, self.y, self.z], np.random.rand(len(self.x.domain),
                                                                            len(self.y.domain),
                                                                            len(self.z.domain)))
        self.assertTrue(distribution)

    def test_marginal(self):
        distribution = Multinomial([self.x, self.y, self.z], np.random.rand(len(self.x.domain),
                                                                            len(self.y.domain),
                                                                            len(self.z.domain)))

        marginal = distribution.marginal([self.x, self.y], normalize=True)
        self.assertAlmostEqual(marginal.probabilities.sum(), 1)

    def test_expand(self):
        distribution_1 = Multinomial([self.x], np.random.rand(len(self.x.domain)))
        distribution_2 = Multinomial([self.y], np.random.rand(len(self.y.domain)))

        x1 = nodes.VNode("x1", rv.Discrete)
        x2 = nodes.VNode("x2", rv.Discrete)
        f_x1 = rv.Discrete(distribution_1.probabilities, x1)
        f_x2 = rv.Discrete(distribution_2.probabilities, x2)

        self.assertTrue(np.all(f_x1.pmf == distribution_1.probabilities))
        self.assertTrue(np.all(f_x2.pmf == distribution_2.probabilities))

        print(f_x1.pmf, distribution_1.probabilities)

        f_x1._expand(f_x2.dim, f_x2.pmf.shape)
        expanded = distribution_1.expand(distribution_2.variables)
        print(f_x1)
        print(expanded.probabilities)


if __name__ == '__main__':
    unittest.main()
