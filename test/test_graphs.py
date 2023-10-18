import itertools
import unittest

import networkx as nx
from networkx.algorithms import bipartite
import numpy as np
import matplotlib.pyplot as plt
import numpy.testing as npt

from fglib2.variables import Symbolic
from fglib2.nodes import VariableNode, FactorNode, Edge
from fglib2.graphs import FactorGraph, ForneyFactorGraph
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
        dist_fa = [[0.3, 0.4],
                   [0.3, 0.0]]
        fa = fglib.nodes.FNode("fa",
                                    fglib.rv.Discrete(dist_fa, self.fglib_x1, self.fglib_x2))

        fa_own = FactorNode([self.x1, self.x2], Multinomial([self.x1, self.x2],
                                                            np.array(dist_fa)))

        dist_fb = [[0.3, 0.4],
                   [0.3, 0.0]]
        fb = fglib.nodes.FNode("fb", fglib.rv.Discrete(dist_fb, self.fglib_x2, self.fglib_x3))

        fb_own = FactorNode([self.x2, self.x3], Multinomial([self.x2, self.x3], np.array(dist_fb)))

        dist_fc = [[0.3, 0.4],
                   [0.3, 0.0]]
        fc = fglib.nodes.FNode("fc", fglib.rv.Discrete(dist_fc, self.fglib_x2, self.fglib_x4))

        fc_own = FactorNode([self.x2, self.x4], Multinomial([self.x2, self.x4], np.array(dist_fc)))

        self.graph = ForneyFactorGraph() * fa_own * fb_own # * fc_own

        # Add nodes to factor graph
        self.fglib_graph.set_nodes([self.fglib_x1, self.fglib_x2, self.fglib_x3, self.fglib_x4])
        # self.fglib_graph.set_nodes([fa, fb, fc])
        self.fglib_graph.set_nodes([self.fglib_x1, self.fglib_x2, self.fglib_x3])
        self.fglib_graph.set_nodes([fa, fb])

        # Add edges to factor graph
        self.fglib_graph.set_edge(self.fglib_x1, fa)
        self.fglib_graph.set_edge(fa, self.fglib_x2)
        self.fglib_graph.set_edge(self.fglib_x2, fb)
        self.fglib_graph.set_edge(fb, self.fglib_x3)
        # self.fglib_graph.set_edge(self.fglib_x2, fc)
        # self.fglib_graph.set_edge(fc, self.fglib_x4)

    def test_likelihood(self):
        world = [0, 0, 0, 0]
        self.graph.likelihood(world)

    def test_graph(self):
        self.assertEqual(len(self.fglib_graph.nodes), len(self.graph.nodes))
        self.assertEqual(len(self.fglib_graph.edges), len(self.graph.edges))

    def test_brute_force(self):
        worlds = list(itertools.product(*[variable.domain for variable in self.graph.variables]))
        worlds = np.array(worlds)
        print(worlds.shape)
        potentials = np.ones(len(worlds))

        for idx, world in enumerate(worlds):

            for factor in self.graph.factor_nodes:
                indices = [self.graph.variables.index(variable) for variable in factor.variables]
                potentials[idx] *= factor.distribution.likelihood(world[indices])

        for index, variable in enumerate(self.graph.variables):
            for value in variable.domain:
                indices = np.where(worlds[:, index] == value)[0]
                print("P({} = {}) = {}".format(variable.name, value, np.sum(potentials[indices])/np.sum(potentials)))

    def test_retardation(self):
        x1_to_fa = self.graph.node_of(self.x1).unity()
        print(x1_to_fa)
        fa = self.graph.factor_of([self.x1, self.x2])
        fa_to_x2 = (fa.distribution * x1_to_fa).marginal([self.x2])
        x2_to_fb = fa_to_x2

        fb = self.graph.factor_of([self.x2, self.x3])
        fb_to_x3 = (x2_to_fb* fb.distribution).marginal([self.x3])
        print(fb_to_x3)

    def test_spa_x1(self):
        nx.draw(self.graph, with_labels=True)
        # plt.show()
        fglib.inference.sum_product(self.fglib_graph, query_node=self.fglib_x1)

        # Test belief of variable node x1
        fglib_belief = self.fglib_x1.belief(normalize=True)

        self.graph.sum_product()

        print([self.graph.edges[e[0], e[1]]["edge"] for e in self.graph.edges])
        belief = self.graph.belief(self.x1)
        #import fglib.rv
        def getval(message):
            a = []
            for i,row in enumerate(message):
                if row[0]:
                    print(f"R{i} C0 dim ", *row[0].dim)
                if row[1]:
                    print(f"R{i} C1 dim ", *row[1].dim)
                a.append(str(row[0]) + ", " + str(row[1]))
            return "[" + "\n".join(a) + "]"

        fglib_edges = [getval(self.fglib_graph.edges[e[0], e[1]]["object"].message) for e in self.fglib_graph.edges]


        own_edges = [self.graph.edges[e[0], e[1]]["edge"] for e in self.graph.edges]
        print("-" * 80)
        print(*fglib_edges, sep="\n============================\n")
        print("-" * 80)
        print(own_edges)
        print("-" * 80)
        print(belief.probabilities, fglib_belief.pmf)
        self.assertTrue(np.allclose(belief.probabilities, fglib_belief.pmf))

    def test_spa_x2(self):
        fglib.inference.sum_product(self.fglib_graph, query_node=self.fglib_x2)
        # Test belief of variable node x2
        fglib_belief = self.fglib_x2.belief(normalize=True)
        self.graph.sum_product()
        belief = self.graph.belief(self.x2)
        print(belief.probabilities, fglib_belief.pmf)
        self.assertTrue(np.allclose(belief.probabilities, fglib_belief.pmf))

    def test_spa_x3(self):
        fglib.inference.sum_product(self.fglib_graph, query_node=self.fglib_x3)
        # Test belief of variable node x3
        fglib_belief = self.fglib_x3.belief(normalize=True)
        self.graph.sum_product()
        belief = self.graph.belief(self.x3)

        print(belief.probabilities, fglib_belief.pmf)

    def test_spa_x4(self):
        fglib.inference.sum_product(self.fglib_graph, query_node=self.fglib_x4)
        # Test belief of variable node x4
        fglib_belief = self.fglib_x4.belief(normalize=True)
        self.graph.sum_product()
        belief = self.graph.belief(self.x4)

        print(belief.probabilities, fglib_belief.pmf)

    def test_spa(self):
        fglib.inference.sum_product(self.fglib_graph, query_node=self.fglib_x1)

        # Test belief of variable node x1
        belief = self.fglib_x1.belief(normalize=False)
        res = np.array([0.183, 0.147])
        npt.assert_almost_equal(belief.pmf, res)
        self.assertEqual(belief.dim, (self.fglib_x1,))

        belief = self.fglib_x1.belief()
        res /= np.sum(res)
        npt.assert_almost_equal(belief.pmf, res)
        self.assertEqual(belief.dim, (self.fglib_x1,))

        # Test belief of variable node x2
        belief = self.fglib_x2.belief(normalize=False)
        res = np.array([0.294, 0.036])
        npt.assert_almost_equal(belief.pmf, res)
        self.assertEqual(belief.dim, (self.fglib_x2,))

        belief = self.fglib_x2.belief()
        res /= np.sum(res)
        npt.assert_almost_equal(belief.pmf, res)
        self.assertEqual(belief.dim, (self.fglib_x2,))

        # Test belief of variable node x3
        belief = self.fglib_x3.belief(normalize=False)
        res = np.array([0.162, 0.168])
        npt.assert_almost_equal(belief.pmf, res)
        self.assertEqual(belief.dim, (self.fglib_x3,))

        belief = self.fglib_x3.belief()
        res /= np.sum(res)
        npt.assert_almost_equal(belief.pmf, res)
        self.assertEqual(belief.dim, (self.fglib_x3,))

        # Test belief of variable node x4
        belief = self.fglib_x4.belief(normalize=False)
        res = np.array([0.162, 0.168])
        npt.assert_almost_equal(belief.pmf, res)
        self.assertEqual(belief.dim, (self.fglib_x4,))

        belief = self.fglib_x4.belief()
        res /= np.sum(res)
        npt.assert_almost_equal(belief.pmf, res)
        self.assertEqual(belief.dim, (self.fglib_x4,))

    def test_mpa(self):
        fglib.inference.max_product(self.fglib_graph, query_node=self.fglib_x1)

        # Test maximum of variable node x1
        maximum = self.fglib_x1.maximum(normalize=False)
        res = 0.048
        npt.assert_almost_equal(maximum, res)

        maximum = self.fglib_x1.maximum()
        res /= np.sum([0.048, 0.048])
        npt.assert_almost_equal(maximum, res)

        # Test maximum of variable node x2
        maximum = self.fglib_x2.maximum(normalize=False)
        res = 0.048
        npt.assert_almost_equal(maximum, res)

        maximum = self.fglib_x2.maximum()
        res /= np.sum([0.036, 0.048])
        npt.assert_almost_equal(maximum, res)

        # Test maximum of variable node x3
        maximum = self.fglib_x3.maximum(normalize=False)
        res = 0.048
        npt.assert_almost_equal(maximum, res)

        maximum = self.fglib_x3.maximum()
        res /= np.sum([0.048, 0.036])
        npt.assert_almost_equal(maximum, res)

        # Test maximum of variable node x4
        maximum = self.fglib_x4.maximum(normalize=False)
        res = 0.048
        npt.assert_almost_equal(maximum, res)

        maximum = self.fglib_x4.maximum()
        res /= np.sum([0.036, 0.048])
        npt.assert_almost_equal(maximum, res)

    def test_msa(self):
        fglib.inference.max_sum(self.fglib_graph, query_node=self.fglib_x1)

        # Test maximum of variable node x1
        maximum = self.fglib_x1.maximum(normalize=False)
        res = -3.036
        npt.assert_almost_equal(maximum, res, decimal=3)

        maximum = self.fglib_x1.maximum()
        res /= np.abs(np.sum([-3.036, -3.036]))
        npt.assert_almost_equal(maximum, res, decimal=3)

        # Test maximum of variable node x2
        maximum = self.fglib_x2.maximum(normalize=False)
        res = -3.036
        npt.assert_almost_equal(maximum, res, decimal=3)

        maximum = self.fglib_x2.maximum()
        res /= np.abs(np.sum([-3.036, -3.324]))
        npt.assert_almost_equal(maximum, res, decimal=3)

        # Test maximum of variable node x3
        maximum = self.fglib_x3.maximum(normalize=False)
        res = -3.036
        npt.assert_almost_equal(maximum, res, decimal=3)

        maximum = self.fglib_x3.maximum()
        res /= np.abs(np.sum([-3.324, -3.036]))
        npt.assert_almost_equal(maximum, res, decimal=3)

        # Test maximum of variable node x4
        maximum = self.fglib_x4.maximum(normalize=False)
        res = -3.036
        npt.assert_almost_equal(maximum, res, decimal=3)

        maximum = self.fglib_x4.maximum()
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
