from random import random
from typing import List

import networkx as nx
from .nodes import VariableNode, FactorNode


class FactorGraph(nx.Graph):

    def __init__(self):
        super().__init__(self)

    @property
    def variable_nodes(self) -> List[VariableNode]:
        """
        Return a list of all variable nodes in the factor graph.
        """
        return [node for node in self.nodes if isinstance(node, VariableNode)]

    @property
    def factor_nodes(self) -> List[FactorNode]:
        """
        Return a list of all factor nodes in the factor graph.
        """
        return [node for node in self.nodes if isinstance(node, FactorNode)]

    def sum_product(self):
        """
        Apply the sum product algorithm to the factor graph.
        The variables and factors are modified in place.
        """
        root = self.variable_nodes[0]

        # Depth First Search to determine edges
        dfs = nx.dfs_edges(self, root)

        # Convert tuple to a reversed list
        backward_path = list(dfs)
        forward_path = reversed(backward_path)

        # Messages in forward phase
        for (v, u) in forward_path:  # Edge direction: v -> u
            print(v, u)
            print(type(v), type(u))
            msg = u.spa(v)
            self[u][v]['object'].set_message(u, v, msg)

        # Messages in backward phase
        for (u, v) in backward_path:  # Edge direction: u -> v
            msg = u.spa(v)
            self[u][v]['object'].set_message(u, v, msg)

    def max_product(self):
        """
        Apply the max product algorithm to the factor graph.
        The variables and factors are modified in place.
        """

