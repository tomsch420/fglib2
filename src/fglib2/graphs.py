from random import random
from typing import List, Tuple, Iterable

import networkx as nx
from .nodes import VariableNode, FactorNode, Edge, Node


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

    def add_edge(self, u_of_edge: Node, v_of_edge: Node, **attr):
        """
        Add an edge between u and v.
        :param u_of_edge: Source node.
        :param v_of_edge: target node.
        :param attr: edge attributes.
        """
        edge = Edge(u_of_edge, v_of_edge)
        super().add_edge(u_of_edge, v_of_edge, edge=edge)

    def add_edges_from(self, edges: Iterable[Tuple[Node, Node]], **attr):
        """
        Add all the edges in edges.
        :param edges: List of edges.
        :param attr: Edge attributes.
        """
        for (u, v) in edges:
            self.add_edge(u, v, **attr)

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
        forward_path = list(reversed(backward_path))
        print(*forward_path)
        import fglib.edges
        # Messages in forward phase
        for (v, u) in forward_path:  # Edge direction: v -> u
            print(v, u)
            print(*self.neighbors(v))
            incoming_messages = [self[w][v]['edge'].message_to_source for w in nx.all_neighbors(self, v) if w != v]
            msg = u.sum_product(incoming_messages)
            self[u][v]['edge'].message_to_target = msg

        # Messages in backward phase
        for (u, v) in backward_path:  # Edge direction: u -> v
            incoming_messages = [self[w][v]['edge'].message_to_source for w in self.neighbors(v) if w != u]
            msg = u.sum_product(incoming_messages)
            self[u][v]['edge'].message_to_source(u, v, msg)

    def max_product(self):
        """
        Apply the max product algorithm to the factor graph.
        The variables and factors are modified in place.
        """

