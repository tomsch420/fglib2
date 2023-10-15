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
        return Multinomial([self.variable])

    def __repr__(self):
        return self.variable.name

    def sum_product(self, messages: List[Multinomial]) -> Multinomial:
        message = self.unity()

        for msg in messages:
            if msg is not None:
                message *= msg

        return message


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

        # Integration/Summation over incoming variables
        for msg in messages:
            if msg is not None:
                message = message.marginal(msg.variables, normalize=False)

        return message


class Edge:

    """Edge.

    Base class for all edges.
    Each edge class contains a message attribute,
    which stores the corresponding message in forward and backward direction.

    """

    def __init__(self, source, target, message_to_target: Optional[Multinomial] = None,
                 message_to_source: Optional[Multinomial] = None):
        """Create an edge."""
        self.source = source
        self.target = target

        self.message_to_target = message_to_target
        self.message_to_source = message_to_source
