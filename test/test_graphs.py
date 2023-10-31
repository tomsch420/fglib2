import unittest

import fglib.graphs
import fglib.inference
import fglib.nodes
import fglib.rv
import networkx as nx
import numpy as np

from fglib2.distributions import Multinomial
from fglib2.graphs import FactorGraph, VariableNode, FactorNode
from random_events.variables import Symbolic


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
        fxy = FactorNode(Multinomial([self.x, self.y]))
        fyz = FactorNode(Multinomial([self.y, self.z]))

        fg = fxy * fyz

        self.assertEqual(len(fg.nodes), 5)
        self.assertEqual(len(fg.edges), 4)
        self.assertEqual(fg.variables, [self.x, self.y, self.z])


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

        f_x = FactorNode(Multinomial([cls.x], np.random.rand(2)))

        f_xy = FactorNode(Multinomial([cls.x, cls.y], np.random.rand(2, 3)))
        f_yz = FactorNode(Multinomial([cls.y, cls.z], np.random.rand(3, 5)))

        cls.factor_graph.add_nodes_from([f_x, f_xy, f_yz])
        cls.factor_graph.add_edges_from([(x, f_x), (x, f_xy), (y, f_xy), (y, f_yz), (z, f_yz)])

    def test_sum_product(self):
        # plot graph
        nx.draw(self.factor_graph, with_labels=True)
        # plt.show()

        self.factor_graph.sum_product()
        self.assertTrue(True)


class FglibCompareTestCase(unittest.TestCase):

    def setUp(self):
        # Create factor graph
        self.fglib_graph = fglib.graphs.FactorGraph()

        # Create variable nodes
        self.fglib_x1 = fglib.nodes.VNode("x1", fglib.rv.Discrete)
        self.x1 = Symbolic("x1", range(2))

        self.fglib_x2 = fglib.nodes.VNode("x2", fglib.rv.Discrete)
        self.x2 = Symbolic("x2", range(2))

        self.fglib_x3 = fglib.nodes.VNode("x3", fglib.rv.Discrete)
        self.x3 = Symbolic("x3", range(2))

        self.fglib_x4 = fglib.nodes.VNode("x4", fglib.rv.Discrete)
        self.x4 = Symbolic("x4", range(2))

        # Create factor nodes (with joint distributions)
        dist_fa = [[0.3, 0.4], [0.3, 0.0]]
        fa = fglib.nodes.FNode("fa", fglib.rv.Discrete(dist_fa, self.fglib_x1, self.fglib_x2))

        fa_own = FactorNode(Multinomial([self.x1, self.x2], np.array(dist_fa)))

        dist_fb = [[0.3, 0.4], [0.3, 0.0]]
        fb = fglib.nodes.FNode("fb", fglib.rv.Discrete(dist_fb, self.fglib_x2, self.fglib_x3))

        fb_own = FactorNode(Multinomial([self.x2, self.x3], np.array(dist_fb)))

        dist_fc = [[0.3, 0.4], [0.3, 0.0]]
        fc = fglib.nodes.FNode("fc", fglib.rv.Discrete(dist_fc, self.fglib_x2, self.fglib_x4))

        fc_own = FactorNode(Multinomial([self.x2, self.x4], np.array(dist_fc)))

        self.graph = FactorGraph() * fa_own * fb_own * fc_own

        # Add nodes to factor graph
        self.fglib_graph.set_nodes([self.fglib_x1, self.fglib_x2, self.fglib_x3, self.fglib_x4])
        self.fglib_graph.set_nodes([fa, fb, fc])

        # Add edges to factor graph
        self.fglib_graph.set_edge(self.fglib_x1, fa)
        self.fglib_graph.set_edge(fa, self.fglib_x2)
        self.fglib_graph.set_edge(self.fglib_x2, fb)
        self.fglib_graph.set_edge(fb, self.fglib_x3)
        self.fglib_graph.set_edge(self.fglib_x2, fc)
        self.fglib_graph.set_edge(fc, self.fglib_x4)

    def test_graph(self):
        self.assertEqual(len(self.fglib_graph.nodes), len(self.graph.nodes))
        self.assertEqual(len(self.fglib_graph.edges), len(self.graph.edges))

    def test_brute_force(self):
        distribution = self.graph.brute_force_joint_distribution()
        print(distribution.to_tabulate())

    def test_calculation_by_hand(self):
        x1_to_fa = self.graph.node_of(self.x1).unity()
        fa = self.graph.factor_of([self.x1, self.x2])
        fa_to_x2 = (fa.distribution * x1_to_fa).marginal([self.x2])
        x2_to_fb = fa_to_x2

        fb = self.graph.factor_of([self.x2, self.x3])
        fb_to_x3 = (x2_to_fb * fb.distribution).marginal([self.x3])

    def test_spa_x1(self):
        nx.draw(self.graph, with_labels=True)
        # plt.show()
        fglib.inference.sum_product(self.fglib_graph, query_node=self.fglib_x1)
        # Test belief of variable node x1
        fglib_belief = self.fglib_x1.belief(normalize=True)
        self.graph.sum_product()
        belief = self.graph.belief(self.x1)
        self.assertTrue(np.allclose(belief.probabilities, fglib_belief.pmf))

    def test_spa_x2(self):
        fglib.inference.sum_product(self.fglib_graph, query_node=self.fglib_x2)
        # Test belief of variable node x2
        fglib_belief = self.fglib_x2.belief(normalize=True)
        self.graph.sum_product()
        belief = self.graph.belief(self.x2)
        self.assertTrue(np.allclose(belief.probabilities, fglib_belief.pmf))

    def test_spa_x3(self):
        fglib.inference.sum_product(self.fglib_graph, query_node=self.fglib_x3)
        # Test belief of variable node x3
        fglib_belief = self.fglib_x3.belief(normalize=True)
        self.graph.sum_product()
        belief = self.graph.belief(self.x3)
        self.assertTrue(np.allclose(belief.probabilities, fglib_belief.pmf))

    def test_spa_x4(self):
        fglib.inference.sum_product(self.fglib_graph, query_node=self.fglib_x4)
        # Test belief of variable node x4
        fglib_belief = self.fglib_x4.belief(normalize=True)
        self.graph.sum_product()
        belief = self.graph.belief(self.x4)
        self.assertTrue(np.allclose(belief.probabilities, fglib_belief.pmf))

    def test_latex_equation(self):
        print(self.graph.to_latex_equation())

    def test_max_product(self):
        mode = self.graph.max_product()
        self.assertEqual(mode[self.x1], (0, 1))
        self.assertEqual(mode[self.x2], (0, ))
        self.assertEqual(mode[self.x3], (1, ))
        self.assertEqual(mode[self.x4], (1, ))


if __name__ == "__main__":
    unittest.main()
