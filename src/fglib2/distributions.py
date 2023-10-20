import itertools
from typing import Iterable, List, Optional, Tuple

import numpy as np

from .variables import Symbolic

import tabulate


class Multinomial:
    """
    A multinomial distribution over symbolic random variables.
    """

    variables = List[Symbolic]
    """
    The variables in the distribution.
    """

    probabilities: np.ndarray
    """
    The probability mass function. The dimensions correspond to the variables in the same order.
    The first dimension indexes over the first variable and so on.
    """

    def __init__(self, variables: Iterable[Symbolic], probabilities: Optional[np.ndarray] = None,
                 normalize: bool = True):
        self.variables = list(sorted(variables))

        shape = tuple(len(variable.domain) for variable in self.variables)

        if probabilities is None:
            probabilities = np.ones(shape)

        if shape != probabilities.shape:
            raise ValueError("The number of variables must match the number of dimensions in the probability array."
                             "Variables: {}".format(self.variables), "Dimensions: {}".format(probabilities.shape))

        self.probabilities = probabilities / np.sum(probabilities) if normalize else probabilities

    def marginal(self, variables: Iterable[Symbolic], normalize: bool = True) -> 'Multinomial':
        """
        Compute the marginal distribution over the given variables.

        :param variables: The variables to keep over.
        :param normalize: Rather to return a normalized distribution or not.

        :return: The marginal distribution over variables.
        """

        # calculate which variables to marginalize over
        axis = tuple(self.variables.index(variable) for variable in self.variables if variable not in variables)

        probabilities = np.sum(self.probabilities, axis=axis)

        return Multinomial(variables, probabilities, normalize=normalize)

    def _mode(self) -> Tuple[List, float]:
        """
        Calculate the most likely event.
        :return: The mode of the distribution as index-list and its likelihood.
        """

        likelihood = np.max(self.probabilities)
        mode = [event.tolist() for event in np.where(self.probabilities == likelihood)]
        return mode, likelihood

    def mode(self) -> Tuple[List, float]:
        """
        Calculate the most likely event.
        :return: The mode of the distribution and its likelihood.
        """
        event, probability = self._mode()
        return self.decode(event), probability

    def decode(self, event: List[int]) -> List:
        """
        Decode an event from a list of indices to a list of values.
        :param event: The event to decode as a list of indices
        :return: The decoded event
        """
        return [variable.decode(value) for variable, value in zip(self.variables, event)]

    def __copy__(self) -> 'Multinomial':
        """Return a shallow copy of the distribution."""
        return Multinomial(self.variables, self.probabilities)

    def __mul__(self, other: 'Multinomial') -> 'Multinomial':
        """Multiply two Multinomial distributions and return the result.

        :param other: The other distribution to multiply.

        :return: The sum of the two distributions.

        """

        # if the distributions are over the same variables, multiply the probability element-wise
        if set(other.variables) == set(self.variables):
            return Multinomial(self.variables, self.probabilities * other.probabilities)

        # if the other distribution is over more variables than this one, flip order
        if len(self.variables) < len(other.variables):
            return other * self

        assert set(other.variables).issubset(set(self.variables))

        assert len(other.variables) == 1

        # Multiply the probabilities along the dimension of the other variable
        dimension = self.variables.index(other.variables[0])
        shape = [1, ] * len(self.variables)
        shape[dimension] = -1
        probabilities = self.probabilities * other.probabilities.reshape(shape)

        return Multinomial(self.variables, probabilities, normalize=False)

    def __eq__(self, other: 'Multinomial') -> bool:
        """Compare self with other and return the boolean result.

        Two discrete random variables are equal only if the probability mass
        functions are equal and the order of dimensions are equal.

        """
        return (self.variables == other.variables and
                self.probabilities.shape == other.probabilities.shape and
                np.all(self.probabilities == other.probabilities))

    def __str__(self):
        return "P({}): \n".format(", ".join(var.name for var in self.variables)) + str(self.probabilities)

    def to_tabulate(self) -> str:
        """
        :return: a pretty table of the distribution.
        """
        columns = [[var.name for var in self.variables] + ["P"]]
        events: List[List] = list(list(event) for event in itertools.product(*[var.domain for var in self.variables]))

        for idx, event in enumerate(events):
            events[idx].append(self.likelihood(event))
        table = columns + events

        return tabulate.tabulate(table, headers="firstrow", tablefmt="fancy_grid")

    def encode(self, event: List) -> List[int]:
        """
        Encode an event into a list of indices within the respective domains.
        :param event: The event to encode as a list of elements of the respective variables domains
        :return: The encoded event
        """
        return [variable.encode(value) for variable, value in zip(self.variables, event)]

    def _likelihood(self, event: List[int]) -> float:
        """
        Calculate the likelihood of an event.
        The event is a list of indices for the variable values in the same order
        :param event:
        :return: P(event)
        """
        return float(self.probabilities[tuple(event)])

    def likelihood(self, event: List) -> float:
        """
        Calculate the likelihood of an event.
        The event is a list of values for the variables in the same order
        :param event:
        :return: P(event)
        """
        return self._likelihood(self.encode(event))

    def max_message(self, variable) -> 'Multinomial':
        """
        Construct a message that contains the maximum likelihood for each value of the variable.
        :param variable: The variable to construct it over.
        :return: A not normalized distribution over the variable with the maximum likelihood for each value.
        """
        if variable not in self.variables:
            raise ValueError("The variable {} is not in the distribution."
                             "The distributions variables are {}".format(variable, self.variables))
        axis = tuple(index for index, var in enumerate(self.variables) if var != variable)
        probabilities = np.max(self.probabilities, axis=axis)
        return Multinomial([variable], probabilities, normalize=False)
