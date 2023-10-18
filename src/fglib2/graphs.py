from typing import List, Tuple, Iterable, Optional

import networkx as nx

from .distributions import Multinomial
from .nodes import VariableNode, FactorNode, Edge, Node
from .variables import Variable, Symbolic


class FactorGraph(nx.Graph):

    def __init__(self):
        super().__init__(self)

    @property
    def variables(self) -> List[Symbolic]:
        """
        Return a list of all variables in the factor graph.
        """
        return list(sorted(set([node.variable for node in self.variable_nodes])))

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

        # Messages in forward phase
        for (target, source) in forward_path:  # Edge direction: u -> v
            print(source, target)
            if isinstance(source, VariableNode):
                incoming_messages = [self.edges[neighbour, source]['edge'].factor_to_variable for neighbour in
                                     self.neighbors(source) if neighbour != target]
                msg = source.sum_product(incoming_messages)

                self.edges[source, target]['edge'].variable_to_factor = msg
                print(self.edges[source, target]['edge'])

            elif isinstance(source, FactorNode):
                incoming_messages = [self.edges[neighbour, source]['edge'].variable_to_factor for neighbour in
                                     self.neighbors(source) if neighbour != target]
                msg = source.sum_product(incoming_messages)
                self.edges[source, target]['edge'].factor_to_variable = msg.marginal([target.variable], normalize=False)

        # Messages in backward phase
        for (source, target) in backward_path:  # Edge direction: u -> v

            if isinstance(source, VariableNode):
                incoming_messages = [self.edges[neighbour, source]['edge'].factor_to_variable for neighbour in
                                     self.neighbors(source) if neighbour != target]
                msg = source.sum_product(incoming_messages)
                self.edges[source, target]['edge'].variable_to_factor = msg

            elif isinstance(source, FactorNode):
                incoming_messages = [self.edges[neighbour, source]['edge'].variable_to_factor for neighbour in
                                     self.neighbors(source) if neighbour != target]
                msg = source.sum_product(incoming_messages)
                self.edges[source, target]['edge'].factor_to_variable = msg.marginal([target.variable], normalize=False)

    def node_of(self, variable: Variable) -> VariableNode:
        """
        Return the variable node of a variable.
        :param variable: The variable.
        :return: The variable node.
        """
        return [node for node in self.variable_nodes if node.variable == variable][0]

    def factor_of(self, variables: Iterable[Symbolic]) -> FactorNode:
        return [node for node in self.factor_nodes if set(node.variables) == set(variables)][0]

    def max_product(self):
        """
        Apply the max product algorithm to the factor graph.
        The variables and factors are modified in place.
        """

    def belief(self, variable: Symbolic) -> Multinomial:
        """
        Compute the belief of a variable.
        :param variable: The variable
        :return: The distribution over the variable.
        """

        variable_node = self.node_of(variable)
        neighbors = self.neighbors(variable_node)

        belief: Optional[Multinomial] = None

        for neighbor in neighbors:
            if belief is None:
                belief = self.edges[neighbor, variable_node]['edge'].factor_to_variable
            else:
                belief *= self.edges[neighbor, variable_node]['edge'].factor_to_variable

        return belief.marginal([variable])

    def likelihood(self, event: List[int]) -> float:
        """
        Calculate the likelihood of an event. The event is a list of values for the variables in the same order
        as the sorted variables used in this graph.
        :param event:
        :return: The likelihood of such event.
        """
        result = 1.

        assert len(self.variables) == len(event)

        event = dict(zip(self.variables, event))

        root = self.variable_nodes[0]

        # Depth First Search to determine edges
        dfs = nx.dfs_edges(self, root)

        # Convert tuple to a reversed list
        path = list(dfs)

        for source, target in path:
            source: VariableNode
            target: FactorNode
            p_source = target.distribution.marginal([source.variable])
            result = p_source.likelihood([event[source.variable]])
            print(result)
            exit()


class ForneyFactorGraph(FactorGraph):

    def __init__(self):
        super().__init__()

    def __mul__(self, other: FactorNode):
        """
        Add a factor to the graph.
        :param other: Factor node.
        """

        self.add_node(other)

        for variable in other.variables:
            if variable not in self.variables:
                self.add_node(VariableNode(variable))

            v_node = [node for node in self.variable_nodes if node.variable == variable][0]
            self.add_edge(other, v_node)

        return self
