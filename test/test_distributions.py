import itertools
import unittest

import numpy as np
from random_events.events import Event
from random_events.variables import Symbolic

from fglib2.distributions import Multinomial


class MultinomialConstructionTestCase(unittest.TestCase):
    x: Symbolic
    y: Symbolic
    z: Symbolic

    @classmethod
    def setUpClass(cls):
        np.random.seed(69)
        cls.x = Symbolic("X", range(2))
        cls.y = Symbolic("Y", range(3))
        cls.z = Symbolic("Z", range(5))

    def test_creation_with_probabilities(self):
        distribution = Multinomial([self.x, self.y, self.z], np.random.rand(len(self.x.domain),
                                                                            len(self.y.domain),
                                                                            len(self.z.domain)))
        self.assertTrue(distribution)

    def test_creation_without_probabilities(self):
        distribution = Multinomial([self.x])
        self.assertTrue(np.allclose(1., distribution.probabilities))

    def test_creation_with_invalid_probabilities_shape(self):
        probabilities = np.array([[0.1, 0.1], [0.2, 0.2]])
        with self.assertRaises(ValueError):
            distribution = Multinomial([self.x, self.y], probabilities)

    def test_copy(self):
        distribution_1 = Multinomial([self.x, self.y], np.array([[0.1, 0.2, 0.3], [0.7, 0.4, 0.1]]))
        distribution_2 = distribution_1.__copy__()
        self.assertEqual(distribution_1, distribution_2)
        distribution_2.probabilities = np.zeros_like(distribution_2.probabilities)
        self.assertNotEqual(distribution_2, distribution_1)

    def test_to_tabulate(self):
        distribution = Multinomial([self.x, self.y, self.z], np.random.rand(len(self.x.domain),
                                                                            len(self.y.domain),
                                                                            len(self.z.domain)))
        table = distribution.to_tabulate()
        self.assertTrue(table)

    def test_to_str(self):
        distribution = Multinomial([self.x, self.y, self.z])
        self.assertTrue(str(distribution))


class MultinomialEncodingTestCase(unittest.TestCase):

    animal: Symbolic
    color: Symbolic
    distribution: Multinomial

    @classmethod
    def setUpClass(cls):
        cls.animal = Symbolic("animal", {"cat", "dog", "mouse"})
        cls.color = Symbolic("color", {"grey", "brown", "black"})
        cls.distribution = Multinomial((cls.animal, cls.color))

    def test_encode(self):
        event = ["cat", "grey"]
        self.assertEqual(self.distribution.encode(event), [0, 2])

    def test_decode(self):
        event = [1, 0]
        self.assertEqual(self.distribution.decode(event), ["dog", "black"])

    def test_encode_raises(self):
        event = ["bob", "linda"]
        with self.assertRaises(ValueError):
            self.distribution.encode(event)

    def test_decode_raises(self):
        event = [3, 0]
        with self.assertRaises(IndexError):
            self.distribution.decode(event)

    def test_encode_many(self):
        event = [["cat", "grey"], ["dog", "brown"]]
        self.assertEqual(self.distribution.encode_many(event), [[0, 2], [1, 1]])

    def test_decode_many(self):
        event = [[0, 2], [1, 1]]
        self.assertEqual(self.distribution.decode_many(event), [["cat", "grey"], ["dog", "brown"]])


class MultinomialInferenceTestCase(unittest.TestCase):
    x: Symbolic
    y: Symbolic
    z: Symbolic
    random_distribution: Multinomial
    random_distribution_mass: float
    crafted_distribution: Multinomial
    crafted_distribution_mass: float

    @classmethod
    def setUpClass(cls):
        np.random.seed(69)
        cls.x = Symbolic("X", range(2))
        cls.y = Symbolic("Y", range(3))
        cls.z = Symbolic("Z", range(5))
        cls.random_distribution = Multinomial([cls.x, cls.y, cls.z], np.random.rand(len(cls.x.domain),
                                                                                    len(cls.y.domain),
                                                                                    len(cls.z.domain)))
        cls.random_distribution_mass = cls.random_distribution.probabilities.sum()

        cls.crafted_distribution = Multinomial([cls.x, cls.y], np.array([[0.1, 0.2, 0.3], [0.7, 0.4, 0.1]]))
        cls.crafted_distribution_mass = cls.crafted_distribution.probabilities.sum()

    def test_normalize_random(self):
        distribution = self.random_distribution.normalize()
        self.assertNotAlmostEqual(self.random_distribution.probabilities.sum(),1.)
        self.assertAlmostEqual(distribution.probabilities.sum(), 1.)

    def test_normalize_crafted(self):
        distribution = self.random_distribution.normalize()
        self.assertNotAlmostEqual(self.random_distribution.probabilities.sum(), self.crafted_distribution_mass)
        self.assertAlmostEqual(distribution.probabilities.sum(), 1.)

    def test_random_marginal_with_normalize(self):
        marginal = self.random_distribution.marginal([self.x, self.y]).normalize()
        self.assertAlmostEqual(marginal.probabilities.sum(), 1)

    def test_crafted_marginal_with_normalize(self):
        marginal = self.crafted_distribution.marginal([self.x]).normalize()
        self.assertAlmostEqual(marginal.probabilities.sum(), 1)
        self.assertAlmostEqual(marginal.probabilities[0], 0.6 / self.crafted_distribution_mass)
        self.assertAlmostEqual(marginal.probabilities[1], 1.2 / self.crafted_distribution_mass)

    def test_random_mode(self):
        mode, probability = self.random_distribution.mode()
        mode = mode[0]
        self.assertEqual(probability, self.random_distribution.probabilities.max())
        self.assertEqual(mode["X"], (0,))
        self.assertEqual(mode["Y"], (0,))

    def test_crafted_mode(self):
        mode, probability = self.crafted_distribution.mode()
        mode = mode[0]
        self.assertEqual(probability, self.crafted_distribution.probabilities.max())
        self.assertEqual(mode["X"], (1,))
        self.assertEqual(mode["Y"], (0,))

    def test_multiple_modes(self):
        distribution = Multinomial([self.x, self.y], np.array([[0.1, 0.7, 0.3], [0.7, 0.4, 0.1]]),)
        mode, likelihood = distribution.mode()
        self.assertEqual(likelihood, 0.7)
        self.assertEqual(len(mode), 2)
        self.assertEqual(mode[0]["X"], (0,))
        self.assertEqual(mode[0]["Y"], (1,))
        self.assertEqual(mode[1]["X"], (1,))
        self.assertEqual(mode[1]["Y"], (0,))

    def test_random_max_message(self):
        max_message = self.random_distribution.max_message(self.z)
        self.assertEqual(max_message.probabilities.shape, (5, ))

    def test_crafted_max_message(self):
        max_message = self.crafted_distribution.max_message(self.x)
        self.assertTrue(np.allclose(max_message.probabilities, np.array([0.3, 0.7])))

    def test_max_message_wrong_variable(self):
        with self.assertRaises(ValueError):
            self.crafted_distribution.max_message(self.z)

    def test_crafted_probability(self):
        distribution = self.crafted_distribution.normalize()
        event = Event()
        self.assertAlmostEqual(distribution.probability(event), 1)

        event[self.x] = 0
        self.assertAlmostEqual(distribution.probability(event), 1 / 3)

        event[self.y] = (0, 1)
        self.assertAlmostEqual(distribution.probability(event), 0.3 / self.crafted_distribution_mass)

    def test_random_probability(self):
        distribution = self.random_distribution.normalize()
        event = Event()
        self.assertAlmostEqual(distribution.probability(event), 1)

        event[self.x] = 0
        self.assertLessEqual(distribution.probability(event), 1.)

        event[self.y] = (0, 1)
        self.assertLessEqual(distribution.probability(event), 1.)

    def test_crafted_conditional(self):
        event = Event({self.y: (0, 1)})
        conditional = self.crafted_distribution.conditional(event).normalize()
        self.assertEqual(conditional.probability(event), 1)
        self.assertEqual(conditional.probability(Event()), 1.)
        self.assertEqual(conditional.probability(Event({self.y: 2})), 0.)

    def test_random_conditional(self):
        event = Event({self.y: (0, 1)})
        conditional = self.random_distribution.conditional(event).normalize()
        self.assertAlmostEqual(conditional.probability(event), 1)
        self.assertAlmostEqual(conditional.probability(Event()), 1.)
        self.assertEqual(conditional.probability(Event({self.y: 2})), 0.)


class MultinomialMultiplicationTestCase(unittest.TestCase):
    x: Symbolic
    y: Symbolic
    z: Symbolic

    distribution_x: Multinomial
    distribution_y: Multinomial
    distribution_z: Multinomial
    distribution_xy: Multinomial
    distribution_xz: Multinomial
    distribution_yz: Multinomial

    @classmethod
    def setUpClass(cls):
        np.random.seed(69)
        cls.x = Symbolic("X", range(2))
        cls.y = Symbolic("Y", range(3))
        cls.z = Symbolic("Z", range(5))

        # create univariate distributions
        cls.distribution_x = Multinomial([cls.x], np.random.rand(len(cls.x.domain)))
        cls.distribution_y = Multinomial([cls.y], np.random.rand(len(cls.y.domain)))
        cls.distribution_z = Multinomial([cls.z], np.random.rand(len(cls.z.domain)))

        # create bi-variate distributions
        cls.distribution_xy = Multinomial([cls.x, cls.y], np.random.rand(len(cls.x.domain), len(cls.y.domain)))
        cls.distribution_xz = Multinomial([cls.x, cls.y], np.random.rand(len(cls.x.domain), len(cls.y.domain)))
        cls.distribution_yz = Multinomial([cls.x, cls.y], np.random.rand(len(cls.x.domain), len(cls.y.domain)))

    def test_identical_variables(self):
        result = self.distribution_x * self.distribution_x
        self.assertEqual(result.variables, self.distribution_x.variables)
        self.assertTrue(np.allclose(result.probabilities, self.distribution_x.probabilities ** 2))

    def test_left_subset_variables(self):
        result = self.distribution_x * self.distribution_xy
        self.assertEqual(result.variables, self.distribution_xy.variables)

        # adjust x distribution by marginal from joint over xy
        adjusted_distribution_x = self.distribution_x * self.distribution_xy.marginal([self.x])

        # check that all probabilities are the same
        for x_value, y_value in itertools.product(self.x.domain, self.y.domain):

            # create condition and event
            condition = Event({self.x: x_value})
            event = condition.intersection(Event({self.y: y_value}))

            # manual result as P(Y,X) = P(X) * P(Y|X)
            manual_result = (adjusted_distribution_x.normalize().probability(condition) *
                             result.conditional(condition).normalize().probability(event))
            product_result = result.normalize().probability(event)
            self.assertAlmostEqual(manual_result, product_result)

    def test_right_subset_variables(self):
        self.assertEqual(self.distribution_xy * self.distribution_x, self.distribution_x * self.distribution_xy)

    def test_disjoint_variables(self):
        with self.assertRaises(AssertionError):
            self.distribution_x * self.distribution_y

if __name__ == '__main__':
    unittest.main()
