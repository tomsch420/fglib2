import itertools
from abc import ABC
from typing import List, Optional
from typing import Tuple, Iterable

import networkx as nx
import numpy as np

from .distributions import Multinomial
from random_events.variables import Discrete
from random_events.events import Event


class Node(ABC):
    """Abstract Class for a nodes in a factor graph."""

    def sum_product(self, messages: List[Multinomial]) -> Multinomial:
        """
        Calculate the sum product algorithms step at this node.

        :param messages: The input messages
        :return: The output message
        """
        raise NotImplementedError()


class VariableNode(Node):
    """Variable node in a factor graph."""

    variable: Discrete
    """
    The variable asserted with this node.
    """

    def __init__(self, variable: Discrete):
        """
        Create a variable node.
        :param variable: The variable asserted with this node.
        """
        self.variable = variable

    def unity(self) -> Multinomial:
        """
        Create a uniform distribution over the variable of this node.
        :return:
        """
        return Multinomial([self.variable])

    def __repr__(self):
        return self.variable.name

    def sum_product(self, messages: List[Multinomial]) -> Multinomial:
        """
        Apply the sum product algorithm of a variable node.

        All the input messages are multiplied together.

        :param messages: The incoming messages.
        :return: The product of all input messages.
        """
        message = self.unity()
        for msg in messages:
            if msg is not None:
                message *= msg

        return message


class FactorNode(Node):
    """Factor node in a factor graph."""

    def __init__(self, distribution: Multinomial):
        """
        Create a factor node.
        :param distribution: The distribution asserted with this factor.
        """
        self.distribution = distribution

    @property
    def variables(self) -> List[Discrete]:
        return self.distribution.variables

    def __repr__(self):
        return "f({})".format(", ".join([v.name for v in self.variables]))

    def sum_product(self, messages: List[Multinomial]) -> Multinomial:
        """
        Apply the sum product algorithm of a factor node.

        The product of all incoming messages is calculated and multiplied with the distribution of this factor node.

        :param messages: The incoming messages.
        :return: The resulting, multivariate distribution.
        """

        message = self.distribution

        # Product over incoming messages
        for msg in messages:
            if msg is not None:
                message *= msg

        return message

    def __mul__(self, other: 'FactorNode') -> 'FactorGraph':
        """
        Multiply factor nodes to get a factor graph.

        :param other: The other factor node.
        :return: The resulting factor graph.
        """
        return FactorGraph() * self * other

    def max_message(self, variable) -> 'Multinomial':
        """
        Construct a message that contains the maximum likelihood for each value of the variable.

        .. Note::
            The message is not normalized. The reason is the purpose of a max message. In every entry of the
            `probabilities` array is the maximum possible likelihood for the corresponding event. Therefore,
            this message should not be normalized.

        :param variable: The variable to construct it over.
        :return: A not normalized distribution over the variable with the maximum likelihood for each value.
        """
        if variable not in self.variables:
            raise ValueError("The variable {} is not in the distribution."
                             "The distributions variables are {}".format(variable, self.variables))
        axis = tuple(index for index, var in enumerate(self.variables) if var != variable)
        probabilities = np.max(self.distribution.probabilities, axis=axis)
        return Multinomial([variable], probabilities)


class Edge:
    """
    Edge in a factor graph.

    Edges always have to be directed from a variable node to a factor node or vice versa.
    """

    def __init__(self, source: Node, target: Node):
        """
        Create an edge.

        :param source: The source node.
        :param target: The target node.
        """
        self.source = source
        self.target = target

        if ((isinstance(source, VariableNode) and isinstance(target, VariableNode)) or (
                isinstance(source, FactorNode) and isinstance(target, FactorNode))):
            raise ValueError("Edges can only be created between variable and factor nodes. Tried to create an edge"
                             "from {} to {}.".format(source, target))

        self._variable_to_factor = None
        self._factor_to_variable = None

    @property
    def variable_node(self):
        """
        Get the variable node of this edge.
        """
        if isinstance(self.source, VariableNode):
            return self.source
        elif isinstance(self.target, VariableNode):
            return self.target
        else:
            raise ValueError("Edge does not contain a variable node.")

    @property
    def factor_node(self):
        """
        Get the factor node of this edge.
        """
        if isinstance(self.source, FactorNode):
            return self.source
        elif isinstance(self.target, FactorNode):
            return self.target
        else:
            raise ValueError("Edge does not contain a factor node.")

    @property
    def variable_to_factor(self):
        """
        Get the message from variable to factor.
        """
        return self._variable_to_factor

    @variable_to_factor.setter
    def variable_to_factor(self, message: Multinomial):
        """
        Set the message from variable to factor.
        """
        self._variable_to_factor = message

    @property
    def factor_to_variable(self):
        """
        Get the message from factor to variable.
        """
        return self._factor_to_variable

    @factor_to_variable.setter
    def factor_to_variable(self, message: Multinomial):
        """
        Set the message from factor to variable.
        """
        # assert message.variables == [self.variable_node.variable]
        self._factor_to_variable = message

    def __str__(self):
        return "{} -> {}: {} \n".format(self.variable_node, self.factor_node,
                                        self.variable_to_factor) + "{} -> {}: {} ".format(self.factor_node,
                                                                                          self.variable_node,
                                                                                          self.factor_to_variable)

    def __repr__(self):
        return str(self)


class FactorGraph(nx.Graph):
    """
    A factor graph.

    A factor graph is a bipartite graph representing the factorization of a function. In probability theory and its
    applications, factor graphs are used to represent factorization of a probability distribution function, enabling
    efficient computations, such as the computation of marginal distributions through the sum–product algorithm.

    .. Note::
        Only factor trees are efficient to compute with the sum/max product algorithm. Other graphs rely on approximate
        inference.
    """

    @property
    def variables(self) -> List[Discrete]:
        """
        Return a list of all variables in the factor graph.
        """
        return list(sorted([node.variable for node in self.variable_nodes]))

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

        if self.has_edge(u_of_edge, v_of_edge):
            raise ValueError("Edge from {} to {} already exists.".format(u_of_edge, v_of_edge))

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
        The messages of the edges are set in place.
        """
        root = self.variable_nodes[0]

        # Depth First Search to determine edges
        dfs = nx.dfs_edges(self, root)

        # Convert tuple to a reversed list
        backward_path = list(dfs)
        forward_path = list(reversed(backward_path))

        # Messages in forward phase
        for i, path in enumerate([forward_path, backward_path]):  # Edge direction: u -> v
            for (target, source) in path:
                if i == 1:
                    # Reverse the direction of the edges in the backward path
                    target, source = source, target
                if isinstance(source, VariableNode):
                    incoming_messages = [self.edges[neighbour, source]['edge'].factor_to_variable for neighbour in
                                         self.neighbors(source) if neighbour != target]
                    msg = source.sum_product(incoming_messages)
                    self.edges[source, target]['edge'].variable_to_factor = msg

                elif isinstance(source, FactorNode):
                    incoming_messages = [self.edges[neighbour, source]['edge'].variable_to_factor for neighbour in
                                         self.neighbors(source) if neighbour != target]
                    msg = source.sum_product(incoming_messages)
                    self.edges[source, target]['edge'].factor_to_variable = msg.marginal([target.variable])

    def max_product(self) -> Event:
        """
        Apply the max product algorithm to the factor graph.
        The messages of the edges are set in place.

        :return: The mode of the joint distribution as an Event.
        """

        root = self.variable_nodes[0]

        # Depth First Search to determine edges
        dfs = nx.dfs_edges(self, root)

        # Convert tuple to a reversed list
        backtracking_path = list(dfs)
        forward_path = list(reversed(backtracking_path))

        for target, source in forward_path:
            if isinstance(source, VariableNode):
                incoming_messages = [self.edges[neighbour, source]['edge'].factor_to_variable for neighbour in
                                     self.neighbors(source) if neighbour != target]
                msg = source.sum_product(incoming_messages)
                self.edges[source, target]['edge'].variable_to_factor = msg

            elif isinstance(source, FactorNode):
                incoming_messages = [self.edges[neighbour, source]['edge'].variable_to_factor for neighbour in
                                     self.neighbors(source) if neighbour != target]
                msg = source.sum_product(incoming_messages)

                msg = FactorNode(msg).max_message(target.variable)
                self.edges[source, target]['edge'].factor_to_variable = msg

        for source, target in backtracking_path:
            if isinstance(source, VariableNode):
                incoming_messages = [self.edges[neighbour, source]['edge'].factor_to_variable for neighbour in
                                     self.neighbors(source)]
                # get the posterior distribution
                msg = source.sum_product(incoming_messages)

                # get the mode and likelihood
                mode, likelihood = msg.mode()

                # construct dirac message for the mode
                probabilities = np.zeros(len(source.variable.domain))
                for mode_ in mode:
                    probabilities[tuple(*mode_.values())] = 1.
                msg = Multinomial([source.variable], probabilities)

                # set message
                self.edges[source, target]['edge'].variable_to_factor = msg

            elif isinstance(source, FactorNode):
                incoming_messages = [self.edges[neighbour, source]['edge'].variable_to_factor for neighbour in
                                     self.neighbors(source)]
                msg = source.sum_product(incoming_messages)

                msg = FactorNode(msg).max_message(target.variable)
                self.edges[source, target]['edge'].factor_to_variable = msg

        result = Event()
        for variable in self.variables:
            mode, likelihood = self.belief(variable).mode()

            mode_ = mode[0]
            for mode__ in mode[1:]:
                mode_ |= mode_ | mode__
            result = mode_ & result
        return result

    def node_of(self, variable: Discrete) -> VariableNode:
        """
        Return the variable node of a variable.
        :param variable: The variable.
        :return: The variable node.
        """
        return [node for node in self.variable_nodes if node.variable == variable][0]

    def factor_of(self, variables: Iterable[Discrete]) -> FactorNode:
        """
        Return the factor node of a set of factors.
        :param variables: The variables of the factor.
        :return: The factor node.
        """
        return [node for node in self.factor_nodes if set(node.variables) == set(variables)][0]

    def belief(self, variable: Discrete) -> Multinomial:
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

    def __mul__(self, other: FactorNode) -> 'FactorGraph':
        """
        Add a factor to the graph.

        The variables that are not yet in the graph are added and the required edges are created.
        :param other: The factor to add.

        :return: The factor graph with the added factor.
        """

        self.add_node(other)

        for variable in other.variables:
            if variable not in self.variables:
                self.add_node(VariableNode(variable))

            v_node = [node for node in self.variable_nodes if node.variable == variable][0]
            self.add_edge(other, v_node)

        return self

    def to_latex_equation(self) -> str:
        """
        :return: a latex representation of the equation represented by this factor graph.
        """
        return r"P({}) = {}".format(", ".join(tuple(variable.name for variable in self.variables)),
                                    r" \cdot ".join([str(factor) for factor in self.factor_nodes]))

    def brute_force_joint_distribution(self) -> Multinomial:
        """
        Compute the joint distribution of the factor graphs variables by brute force.

        .. Warning::
            This method is only feasible for a small number of variables as it has exponential runtime.

        :return: A Multinomial distribution over all the variables.
        """
        worlds = list(itertools.product(*[variable.domain for variable in self.variables]))
        worlds = np.array(worlds)
        potentials = np.zeros(tuple(len(variable.domain) for variable in self.variables))

        for idx, world in enumerate(worlds):
            potential = 1.
            for factor in self.factor_nodes:
                indices = [self.variables.index(variable) for variable in factor.variables]
                potential *= factor.distribution.likelihood(world[indices])
            potentials[tuple(world)] = potential

        return Multinomial(self.variables, potentials)

    def reset(self):
        """
        Clear all messages in the graph in place.
        """
        for edge in self.edges:
            self.edges[edge]['edge'].variable_to_factor = None
            self.edges[edge]['edge'].factor_to_variable = None
