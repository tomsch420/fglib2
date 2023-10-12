import unittest

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from fglib2.variables import Symbolic
from fglib2.nodes import VariableNode, FactorNode, Edge
from fglib2.graphs import FactorGraph
from fglib2.distributions import Multinomial


class FactorGraphTestCase(unittest.TestCase):

    x: Symbolic
    y: Symbolic
    z: Symbolic

    @classmethod
    def setUpClass(cls):
        cls.x = Symbolic("X", range(2))
        cls.y = Symbolic("Y", range(3))
        cls.z = Symbolic("Z", range(5))

    def test_creation(self):
        fg = FactorGraph()
        x = VariableNode(self.x)
        y = VariableNode(self.y)
        z = VariableNode(self.z)
        fg.add_node(x)
        fg.add_node(y)
        fg.add_node(z)
        fg.add_edge(x, y)
        fg.add_edge(y, z)
        fg.add_edge(z, x)
        self.assertEqual(len(fg.nodes), 3)
        self.assertEqual(len(fg.edges), 3)
        self.assertEqual(fg.variable_nodes, [x, y, z])


class InferenceTestCase(unittest.TestCase):

    x: Symbolic
    y: Symbolic
    z: Symbolic
    factor_graph: FactorGraph

    @classmethod
    def setUpClass(cls):
        np.random.seed(69)
        cls.x = Symbolic("X", range(2))
        cls.y = Symbolic("Y", range(3))
        cls.z = Symbolic("Z", range(5))
        cls.factor_graph = FactorGraph()
        x = VariableNode(cls.x)
        y = VariableNode(cls.y)
        z = VariableNode(cls.z)
        cls.factor_graph.add_nodes_from([x, y, z])

        f_x = FactorNode([cls.x], Multinomial([cls.x], np.random.rand(2)))

        f_xy = FactorNode([cls.x, cls.y],
                          Multinomial([cls.x, cls.y], np.random.rand(2, 3)))
        f_yz = FactorNode([cls.y, cls.z],
                          Multinomial([cls.y, cls.z], np.random.rand(3, 5)))

        cls.factor_graph.add_nodes_from([f_x, f_xy, f_yz])
        cls.factor_graph.add_edges_from([(x, f_x), (x, f_xy), (y, f_xy),
                                         (y, f_yz), (z, f_yz)])

    def test_sum_product(self):
        # plot graph
        nx.draw(self.factor_graph, with_labels=True)
        # plt.show()

        self.factor_graph.sum_product()
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
