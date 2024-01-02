import unittest

import networkx as nx
import numpy as np
from random_events.variables import Continuous
from probabilistic_model.probabilistic_circuit.units import DeterministicSumUnit
from probabilistic_model.probabilistic_circuit.distributions import UniformDistribution
from fglib2.distributions import Multinomial
from fglib2.probabilistic_circuits import SumUnitFactor
import portion
from fglib2.graphs import FactorGraph, FactorNode
import matplotlib.pyplot as plt


class JPTTestCase(unittest.TestCase):

    variables = [
        Continuous("x1"),
        Continuous("x2"),
        # Continuous("x3"),
        # Continuous("x4"),
        # Continuous("x5")
        ]

    model: FactorGraph

    def setUp(self):
        interval_1 = portion.closed(-1.25, -0.75)
        interval_2 = portion.closed(0.75, 1.25)

        model = FactorGraph()

        for variable in self.variables:
            distribution = SumUnitFactor(UniformDistribution(variable, interval_1) +
                                         UniformDistribution(variable, interval_2))
            factor = FactorNode(distribution)
            model *= factor

        for f1, f2 in zip(model.factor_nodes[:-1], model.factor_nodes[1:]):
            model *= FactorNode(Multinomial([f1.variables[0], f2.variables[0]], np.array([[0, 0.5], [0.5, 0]])))

        self.model = model

    def test_creation(self):
        nx.draw(self.model, with_labels=True)
        plt.show()

    def test_marginal(self):
        self.model.sum_product()
        latent_variables = self.model.variables
        print(latent_variables)
        for variable in latent_variables:
            print(self.model.belief(variable))
        # self.model.marginal(self.variables[:1])


    def test_sum_product(self):
        self.model.sum_product()


if __name__ == '__main__':
    unittest.main()
