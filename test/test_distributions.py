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

    def test_to_tabulate(self):
        distribution = Multinomial([self.x, self.y, self.z], np.random.rand(len(self.x.domain),
                                                                            len(self.y.domain),
                                                                            len(self.z.domain)))
        table = distribution.to_tabulate()
        self.assertTrue(table)

    def test_mode(self):
        distribution = Multinomial([self.x, self.y], np.random.rand(len(self.x.domain),
                                                                    len(self.y.domain)))
        mode, probability = distribution.mode()
        self.assertEqual(probability, distribution.likelihood([0, 1]))
        self.assertEqual(mode, [0, 1])

    def test_multiple_modes(self):
        distribution = Multinomial([self.x, self.y], np.array([[0.1, 0.7, 0.3], [0.7, 0.4, 0.1]]),
                                   normalize=False)
        mode, likelihood = distribution.mode()
        self.assertEqual(likelihood, 0.7)
        self.assertEqual(mode, [[0, 1], [1, 0]])

    def test_max_message(self):
        distribution = Multinomial([self.x, self.y], np.array([[0.1, 0.2, 0.3], [0.7, 0.4, 0.1]]),
                                   normalize=False)
        max_message = distribution.max_message(self.x)
        self.assertTrue(np.allclose(max_message.probabilities, np.array([0.3, 0.7])))




if __name__ == '__main__':
    unittest.main()
