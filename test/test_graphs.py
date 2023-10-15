import unittest

import networkx as nx
from networkx.algorithms import bipartite
import numpy as np
import matplotlib.pyplot as plt
import numpy.testing as npt

from fglib2.variables import Symbolic
from fglib2.nodes import VariableNode, FactorNode, Edge
from fglib2.graphs import FactorGraph
from fglib2.distributions import Multinomial

import fglib.nodes
import fglib.graphs
import fglib.inference
import fglib.rv


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


class FglibCompareTestCase(unittest.TestCase):

    def setUp(self):
        # Create factor graph
        self.fg = fglib.graphs.FactorGraph()

        # Create variable nodes
        self.x1 = fglib.nodes.VNode("x1", fglib.rv.Discrete)
        self.x2 = fglib.nodes.VNode("x2", fglib.rv.Discrete)
        self.x3 = fglib.nodes.VNode("x3", fglib.rv.Discrete)
        self.x4 = fglib.nodes.VNode("x4", fglib.rv.Discrete)

        # Create factor nodes (with joint distributions)
        dist_fa = [[0.3, 0.4],
                   [0.3, 0.0]]
        self.fa = fglib.nodes.FNode("fa",
                                    fglib.rv.Discrete(dist_fa, self.x1, self.x2))

        dist_fb = [[0.3, 0.4],
                   [0.3, 0.0]]
        self.fb = fglib.nodes.FNode("fb", fglib.rv.Discrete(dist_fb, self.x2, self.x3))

        dist_fc = [[0.3, 0.4],
                   [0.3, 0.0]]
        self.fc = fglib.nodes.FNode("fc", fglib.rv.Discrete(dist_fc, self.x2, self.x4))

        # Add nodes to factor graph
        self.fg.set_nodes([self.x1, self.x2, self.x3, self.x4])
        self.fg.set_nodes([self.fa, self.fb, self.fc])

        # Add edges to factor graph
        self.fg.set_edge(self.x1, self.fa)
        self.fg.set_edge(self.fa, self.x2)
        self.fg.set_edge(self.x2, self.fb)
        self.fg.set_edge(self.fb, self.x3)
        self.fg.set_edge(self.x2, self.fc)
        self.fg.set_edge(self.fc, self.x4)

    def test_spa(self):
        fglib.inference.sum_product(self.fg, query_node=self.x1)

        # Test belief of variable node x1
        belief = self.x1.belief(normalize=False)
        res = np.array([0.183, 0.147])
        npt.assert_almost_equal(belief.pmf, res)
        self.assertEqual(belief.dim, (self.x1,))

        belief = self.x1.belief()
        res /= np.sum(res)
        npt.assert_almost_equal(belief.pmf, res)
        self.assertEqual(belief.dim, (self.x1,))

        # Test belief of variable node x2
        belief = self.x2.belief(normalize=False)
        res = np.array([0.294, 0.036])
        npt.assert_almost_equal(belief.pmf, res)
        self.assertEqual(belief.dim, (self.x2,))

        belief = self.x2.belief()
        res /= np.sum(res)
        npt.assert_almost_equal(belief.pmf, res)
        self.assertEqual(belief.dim, (self.x2,))

        # Test belief of variable node x3
        belief = self.x3.belief(normalize=False)
        res = np.array([0.162, 0.168])
        npt.assert_almost_equal(belief.pmf, res)
        self.assertEqual(belief.dim, (self.x3,))

        belief = self.x3.belief()
        res /= np.sum(res)
        npt.assert_almost_equal(belief.pmf, res)
        self.assertEqual(belief.dim, (self.x3,))

        # Test belief of variable node x4
        belief = self.x4.belief(normalize=False)
        res = np.array([0.162, 0.168])
        npt.assert_almost_equal(belief.pmf, res)
        self.assertEqual(belief.dim, (self.x4,))

        belief = self.x4.belief()
        res /= np.sum(res)
        npt.assert_almost_equal(belief.pmf, res)
        self.assertEqual(belief.dim, (self.x4,))

    def test_mpa(self):
        fglib.inference.max_product(self.fg, query_node=self.x1)

        # Test maximum of variable node x1
        maximum = self.x1.maximum(normalize=False)
        res = 0.048
        npt.assert_almost_equal(maximum, res)

        maximum = self.x1.maximum()
        res /= np.sum([0.048, 0.048])
        npt.assert_almost_equal(maximum, res)

        # Test maximum of variable node x2
        maximum = self.x2.maximum(normalize=False)
        res = 0.048
        npt.assert_almost_equal(maximum, res)

        maximum = self.x2.maximum()
        res /= np.sum([0.036, 0.048])
        npt.assert_almost_equal(maximum, res)

        # Test maximum of variable node x3
        maximum = self.x3.maximum(normalize=False)
        res = 0.048
        npt.assert_almost_equal(maximum, res)

        maximum = self.x3.maximum()
        res /= np.sum([0.048, 0.036])
        npt.assert_almost_equal(maximum, res)

        # Test maximum of variable node x4
        maximum = self.x4.maximum(normalize=False)
        res = 0.048
        npt.assert_almost_equal(maximum, res)

        maximum = self.x4.maximum()
        res /= np.sum([0.036, 0.048])
        npt.assert_almost_equal(maximum, res)

    def test_msa(self):
        fglib.inference.max_sum(self.fg, query_node=self.x1)

        # Test maximum of variable node x1
        maximum = self.x1.maximum(normalize=False)
        res = -3.036
        npt.assert_almost_equal(maximum, res, decimal=3)

        maximum = self.x1.maximum()
        res /= np.abs(np.sum([-3.036, -3.036]))
        npt.assert_almost_equal(maximum, res, decimal=3)

        # Test maximum of variable node x2
        maximum = self.x2.maximum(normalize=False)
        res = -3.036
        npt.assert_almost_equal(maximum, res, decimal=3)

        maximum = self.x2.maximum()
        res /= np.abs(np.sum([-3.036, -3.324]))
        npt.assert_almost_equal(maximum, res, decimal=3)

        # Test maximum of variable node x3
        maximum = self.x3.maximum(normalize=False)
        res = -3.036
        npt.assert_almost_equal(maximum, res, decimal=3)

        maximum = self.x3.maximum()
        res /= np.abs(np.sum([-3.324, -3.036]))
        npt.assert_almost_equal(maximum, res, decimal=3)

        # Test maximum of variable node x4
        maximum = self.x4.maximum(normalize=False)
        res = -3.036
        npt.assert_almost_equal(maximum, res, decimal=3)

        maximum = self.x4.maximum()
        res /= np.abs(np.sum([-3.324, -3.036]))
        npt.assert_almost_equal(maximum, res, decimal=3)


class TestExample(unittest.TestCase):

    def test_readme(self):
        # Create factor graph
        fg = fglib.graphs.FactorGraph()

        # Create variable nodes
        x1 = fglib.nodes.VNode("x1", fglib.rv.Discrete)  # with 2 states (Bernoulli)
        x2 = fglib.nodes.VNode("x2", fglib.rv.Discrete)  # with 3 states
        x3 = fglib.nodes.VNode("x3", fglib.rv.Discrete)
        x4 = fglib.nodes.VNode("x4", fglib.rv.Discrete)

        # Create factor nodes (with joint distributions)
        dist_fa = [[0.3, 0.2, 0.1],
                   [0.3, 0.0, 0.1]]
        fa = fglib.nodes.FNode("fa", fglib.rv.Discrete(dist_fa, x1, x2))

        dist_fb = [[0.3, 0.2],
                   [0.3, 0.0],
                   [0.1, 0.1]]
        fb = fglib.nodes.FNode("fb", fglib.rv.Discrete(dist_fb, x2, x3))

        dist_fc = [[0.3, 0.2],
                   [0.3, 0.0],
                   [0.1, 0.1]]
        fc = fglib.nodes.FNode("fc", fglib.rv.Discrete(dist_fc, x2, x4))

        # Add nodes to factor graph
        fg.set_nodes([x1, x2, x3, x4])
        fg.set_nodes([fa, fb, fc])

        # Add edges to factor graph
        fg.set_edge(x1, fa)
        fg.set_edge(fa, x2)
        fg.set_edge(x2, fb)
        fg.set_edge(fb, x3)
        fg.set_edge(x2, fc)
        fg.set_edge(fc, x4)

        # Perform sum-product algorithm on factor graph
        # and request belief of variable node x4
        belief = fglib.inference.sum_product(fg, x4)

        # Print belief of variables
        # print("Belief of variable node x4:")
        # print(belief)

        npt.assert_almost_equal(belief.pmf, np.array([0.63, 0.36]), decimal=2)


if __name__ == "__main__":
    unittest.main()
