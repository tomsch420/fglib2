import unittest

import numpy as np

from fglib2.distributions import Multinomial
from fglib2.variables import Symbolic


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
        distribution = Multinomial([self.x, self.y, self.z], np.random.rand(len(self.x.domain),
                                                                            len(self.y.domain),
                                                                            len(self.z.domain)))


if __name__ == '__main__':
    unittest.main()
