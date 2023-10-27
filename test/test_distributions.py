import unittest

import numpy as np
from random_events.events import Event
from random_events.variables import Symbolic

from fglib2.distributions import Multinomial


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

    def test__mode(self):
        distribution = Multinomial([self.x, self.y], np.array([[0.1, 0.2, 0.3], [0.7, 0.4, 0.1]]),
                                   normalize=False)
        mode, probability = distribution._mode()
        self.assertEqual(probability, 0.7)
        self.assertEqual(mode[0]["X"], (1,))
        self.assertEqual(mode[0]["Y"], (0,))

    def test_mode(self):
        distribution = Multinomial([self.x, self.y], np.array([[0.05298733, 0.18055962, 0.04123557],
                                                               [0.30759066, 0.31958457, 0.09804226]]))

        mode, probability = distribution.mode()
        mode = mode[0]
        self.assertEqual(probability, distribution.probabilities.max())
        self.assertEqual(mode["X"], (1,))
        self.assertEqual(mode["Y"], (1,))

    def test_multiple_modes(self):
        distribution = Multinomial([self.x, self.y], np.array([[0.1, 0.7, 0.3], [0.7, 0.4, 0.1]]),
                                   normalize=False)
        mode, likelihood = distribution.mode()
        self.assertEqual(likelihood, 0.7)
        self.assertEqual(len(mode), 2)
        self.assertEqual(mode[0]["X"], (0,))
        self.assertEqual(mode[0]["Y"], (1,))
        self.assertEqual(mode[1]["X"], (1,))
        self.assertEqual(mode[1]["Y"], (0,))

    def test_max_message(self):
        distribution = Multinomial([self.x, self.y], np.array([[0.1, 0.2, 0.3], [0.7, 0.4, 0.1]]),
                                   normalize=False)
        max_message = distribution.max_message(self.x)
        self.assertTrue(np.allclose(max_message.probabilities, np.array([0.3, 0.7])))

    def test_probability(self):
        distribution = Multinomial([self.x, self.y], np.array([[0.1, 0.2, 0.3], [0.7, 0.4, 0.1]]),
                                   normalize=True)
        event = Event()
        self.assertAlmostEqual(distribution.probability(event), 1)

        event[self.x] = 0
        self.assertAlmostEqual(distribution.probability(event), 1 / 3)

        event[self.y] = (0, 1)
        self.assertAlmostEqual(distribution.probability(event), 0.3 / 1.8)

    def test_conditional(self):
        distribution = Multinomial([self.x, self.y], np.array([[0.1, 0.2, 0.3], [0.7, 0.4, 0.1]]),
                                   normalize=True)
        event = Event({self.y: (0, 1)})
        conditional = distribution.conditional(event)
        self.assertEqual(conditional.probability(event), 1)
        self.assertEqual(conditional.probability(Event()), 1.)
        self.assertEqual(conditional.probability(Event({self.y: 2})), 0.)


if __name__ == '__main__':
    unittest.main()
