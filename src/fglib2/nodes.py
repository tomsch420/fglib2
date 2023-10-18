from abc import ABC
from typing import List, Optional

from .distributions import Multinomial
from .variables import Symbolic
import numpy as np


class Node(ABC):

    def sum_product(self, messages: List[Multinomial]) -> Multinomial:
        """
        Calculate the sum product algorithms step at this node.

        :param messages: The input messages
        :return: The output message
        """


class VariableNode(Node):
    """Variable node in a factor graph."""

    variable: Symbolic

    def __init__(self, variable: Symbolic):
        self.variable = variable

    def unity(self) -> Multinomial:
        return Multinomial([self.variable], normalize=False)

    def __repr__(self):
        return self.variable.name

    def sum_product(self, messages: List[Multinomial]) -> Multinomial:
        print(*messages)
        message = self.unity()
        print(message)
        for msg in messages:
            if msg is not None:
                message *= msg

        return message

    def create_dirac_message(self, value) -> Multinomial:
        """
        Create a dirac message for the given value.
        :param value: The value to create the dirac message for.
        :return: The dirac message as distribution
        """
        return self._create_dirac_message(self.variable.encode(value))

    def _create_dirac_message(self, value) -> Multinomial:
        """
        Create a dirac message for the given value.
        :param value: The value that is already encoded
        :return: The dirac message as distribution
        """
        distribution = np.zeros(len(self.variable.domain))
        distribution[value] = 1.
        return Multinomial([self.variable], distribution)


class FactorNode(Node):
    """Factor node in a factor graph."""

    def __init__(self, variables: List[Symbolic], distribution: Multinomial):
        self.variables = variables
        self.distribution = distribution

    def __repr__(self):
        return "f({})".format(", ".join([v.name for v in self.variables]))

    def sum_product(self, messages: List[Multinomial]) -> Multinomial:

        message = self.distribution

        # Product over incoming messages
        for msg in messages:
            if msg is not None:
                message *= msg

        return message


class Edge:

    """Edge.

    Base class for all edges.
    Each edge class contains a message attribute,
    which stores the corresponding message in forward and backward direction.

    """

    def __init__(self, source, target):
        """Create an edge."""
        self.source = source
        self.target = target

        self._variable_to_factor = None
        self._factor_to_variable = None

    @property
    def variable_node(self):
        if isinstance(self.source, VariableNode):
            return self.source
        elif isinstance(self.target, VariableNode):
            return self.target
        else:
            raise ValueError("Edge does not contain a variable node.")

    @property
    def factor_node(self):
        if isinstance(self.source, FactorNode):
            return self.source
        elif isinstance(self.target, FactorNode):
            return self.target
        else:
            raise ValueError("Edge does not contain a factor node.")

    @property
    def variable_to_factor(self):
        return self._variable_to_factor

    @variable_to_factor.setter
    def variable_to_factor(self, message: Multinomial):
        # assert message.variables == [self.variable_node.variable]
        self._variable_to_factor = message

    @property
    def factor_to_variable(self):
        return self._factor_to_variable

    @factor_to_variable.setter
    def factor_to_variable(self, message: Multinomial):
        print(message)
        # assert message.variables == [self.variable_node.variable]
        self._factor_to_variable = message

    def __str__(self):
        return "{} -> {}: {} \n".format(self.variable_node, self.factor_node, self.variable_to_factor) + \
                "{} -> {}: {} ".format(self.factor_node, self.variable_node, self.factor_to_variable)

    def __repr__(self):
        return str(self)
