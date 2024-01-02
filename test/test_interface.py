import random
import unittest
from typing import List

from random_events.variables import Symbolic
from random_events.events import Event
from fglib2.graphs import FactorGraph, FactorNode
from fglib2.distributions import Multinomial
import networkx as nx
import numpy as np


def generate_random_tree_over_variables(variables: List[Symbolic]) -> FactorGraph:
    """
    Create a random factor tree over the variables with random parameters.
    :param variables: The variables to create the factor tree over
    :return: The model
    """
    graph = nx.random_labeled_tree(len(variables))
    model = None
    for u, v in graph.edges():
        v1 = variables[u]
        v2 = variables[v]
        probabilities = np.random.uniform(low=0, high=1, size=(len(v1.domain), len(v2.domain)))
        distribution = Multinomial([v1, v2], probabilities=probabilities).normalize()
        factor = FactorNode(distribution)
        if model is None:
            model = factor
        else:
            model = model * factor

    return model


def generate_random_event_over_variables(variables: List[Symbolic]) -> Event:
    """
    Generate a random event that can be used for inference.
    :param variables: The variables to create the random event over.
    :return: The random event
    """
    event = Event()
    for variable in variables:
        event[variable] = random.sample(variable.domain, random.randint(1, len(variable.domain)))
    return event


def generate_random_likelihood_event_over_variables(variables: List[Symbolic]) -> List:
    """
    Generate a random full evidence event that can be used for likelihood inference.
    :param variables: The variables to create the random event over
    :return: The random event
    """
    return [random.choice(variable.domain) for variable in variables]

@unittest.skip("Not implemented")
class InterfaceTestCase(unittest.TestCase):

    variables = [Symbolic(f"x_{i}", range(random.randrange(2, 4))) for i in range(9)]

    model: FactorGraph
    event: Event
    joint_distribution: Multinomial

    @classmethod
    def setUpClass(cls):
        cls.model = generate_random_tree_over_variables(cls.variables)
        cls.event = generate_random_event_over_variables(cls.variables)
        cls.joint_distribution = cls.model.brute_force_joint_distribution()

    def test_probability(self):
        p_event_joint_distribution = self.joint_distribution.probability(self.event)
        p_event_factor_graph = self.model.probability(self.event)
        self.assertAlmostEqual(p_event_joint_distribution, p_event_factor_graph)

    def test_mode(self):
        mode_joint_distribution, max_joint_distribution = self.joint_distribution.mode()
        mode_factor_graph, max_factor_graph = self.model.mode()
        self.assertEqual(max_factor_graph, max_joint_distribution)
        self.assertEqual(mode_factor_graph, mode_joint_distribution)

    def test_conditional(self):
        conditional_joint_distribution, p_joint_distribution = self.joint_distribution.conditional(self.event)
        conditional_factor_graph, p_factor_graph = self.model.conditional(self.event)
        self.assertAlmostEqual(p_joint_distribution, p_factor_graph)
        self.assertEqual(conditional_joint_distribution, conditional_factor_graph.brute_force_joint_distribution())

    def test_marginal(self):
        marginal_variables = self.variables[:random.randint(1, len(self.variables) - 1)]
        marginal_joint_distribution = self.joint_distribution.marginal(marginal_variables)
        marginal_factor_graph = self.model.marginal(marginal_variables)
        self.assertEqual(marginal_joint_distribution, marginal_factor_graph.brute_force_joint_distribution())

    def test_likelihood(self):
        event = generate_random_likelihood_event_over_variables(self.variables)
        likelihood_joint_distribution = self.joint_distribution.likelihood(event)
        likelihood_graph = self.model.likelihood(event)
        self.assertAlmostEqual(likelihood_graph, likelihood_joint_distribution)


if __name__ == '__main__':
    unittest.main()
