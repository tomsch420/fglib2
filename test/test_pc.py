import unittest

import networkx as nx
import numpy as np
from random_events.variables import Continuous
from random_events.events import Event
from probabilistic_model.probabilistic_circuit.units import DeterministicSumUnit
from probabilistic_model.probabilistic_circuit.distributions import UniformDistribution
from fglib2.distributions import Multinomial
from fglib2.probabilistic_circuits import SumUnitFactor
import portion
from fglib2.graphs import FactorGraph, FactorNode
import matplotlib.pyplot as plt


class SumUnitTestCase(unittest.TestCase):

    variables = [
        Continuous("x1"),
        Continuous("x2"),
        # Continuous("x3"),
        # Continuous("x4"),
        # Continuous("x5")
        ]

    model: FactorGraph

    def setUp(self):
        np.random.seed(69)
        interval_1 = portion.closed(-1.25, -0.75)
        interval_2 = portion.closed(0.75, 1.25)

        model = FactorGraph()

        for variable in self.variables:
            distribution = (UniformDistribution(variable, interval_1) +
                            UniformDistribution(variable, interval_2))
            factor = SumUnitFactor(distribution)
            model *= factor

        for f1, f2 in zip(model.factor_nodes[:-1], model.factor_nodes[1:]):
            model *= FactorNode(Multinomial([f1.latent_variable, f2.latent_variable],
                                            np.array([[0, 0.5], [0.5, 0]])))

        self.model = model

    def test_creation(self):
        nx.draw(self.model, with_labels=True)
        plt.show()

    def test_latent_distribution(self):
        distribution = self.model.factor_nodes[0].latent_distribution()
        self.assertTrue(np.all(distribution.probabilities == np.array([0.5, 0.5])))
        self.assertEqual(distribution.variables[0], self.model.factor_nodes[0].latent_variable)

    def test_marginal(self):
        self.model.sum_product()
        latent_variables = self.model.variables
        for variable in latent_variables:
            belief = self.model.belief(variable).normalize()
            self.assertTrue(np.all(belief.probabilities == np.array([0.5, 0.5])))

    def test_marginal_with_evidence(self):
        factor_node = self.model.factor_nodes[0]
        conditional, probability = factor_node.distribution.conditional(Event({self.variables[0]:
                                                                                    portion.closed(-1.25, -0.75)}))
        self.assertEqual(probability, 0.5)
        factor_node.distribution = conditional
        self.model.factor_nodes[0] = factor_node

        self.model.sum_product()
        latent_variables = self.model.variables
        latent_variable_0 = latent_variables[0]
        latent_variable_0_distribution = self.model.belief(latent_variable_0).normalize()
        self.assertTrue(np.all(latent_variable_0_distribution.probabilities == np.array([0., 1.])))

        latent_variable_1 = latent_variables[1]
        latent_variable_1_distribution = self.model.belief(latent_variable_1).normalize()
        self.assertTrue(np.all(latent_variable_1_distribution.probabilities == np.array([1., 0.])))

    def test_mode(self):
        mode_by_hand = Event({self.model.factor_nodes[0].latent_variable: (0, 1),
                              self.model.factor_nodes[1].latent_variable: (0, 1)})
        modes = self.model.max_product()
        self.assertEqual(modes, mode_by_hand)

    def test_conditional_mode(self):
        factor_node = self.model.factor_nodes[0]
        conditional, probability = factor_node.distribution.conditional(Event({self.variables[0]:
                                                                                    portion.closed(-1.25, -0.75)}))
        self.assertEqual(probability, 0.5)
        factor_node.distribution = conditional
        self.model.factor_nodes[0] = factor_node

        mode_by_hand = Event({self.model.factor_nodes[0].latent_variable: 0,
                              self.model.factor_nodes[1].latent_variable: 1})
        modes = self.model.max_product()
        self.assertEqual(modes, mode_by_hand)



if __name__ == '__main__':
    unittest.main()
