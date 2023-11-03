import itertools
from typing import Iterable, List, Optional, Tuple

import numpy as np

from random_events.variables import Discrete
from random_events.events import Event, EncodedEvent

import tabulate


class Multinomial:
    """
    A multinomial distribution over discrete random variables.
    """

    variables = Tuple[Discrete]
    """
    The variables in the distribution.
    """

    probabilities: np.ndarray
    """
    The probability mass function. The dimensions correspond to the variables in the same order.
    The first dimension indexes over the first variable and so on. If no probabilities are provided in the constructor,
    the probabilities are initialized with ones.
    """

    def __init__(self, variables: Iterable[Discrete], probabilities: Optional[np.ndarray] = None):
        self.variables = tuple(sorted(variables))

        shape = tuple(len(variable.domain) for variable in self.variables)

        if probabilities is None:
            probabilities = np.ones(shape)

        if shape != probabilities.shape:
            raise ValueError("The number of variables must match the number of dimensions in the probability array."
                             "Variables: {}".format(self.variables), "Dimensions: {}".format(probabilities.shape))

        self.probabilities = probabilities

    def marginal(self, variables: Iterable[Discrete]) -> 'Multinomial':
        """
        Compute the marginal distribution over the given variables.

        :param variables: The variables to keep over.

        :return: The marginal distribution over variables.
        """

        # calculate which variables to marginalize over as the difference between variables and self.variables
        axis = tuple(self.variables.index(variable) for variable in self.variables if variable not in variables)

        # marginalize the probabilities over the axis
        probabilities = np.sum(self.probabilities, axis=axis)

        return Multinomial(variables, probabilities)

    def _mode(self) -> Tuple[List[EncodedEvent], float]:
        """
        Calculate the most likely event.
        :return: The mode of the distribution as EncodedEvent and its likelihood.
        """
        likelihood = np.max(self.probabilities)
        events = np.transpose(np.asarray(self.probabilities == likelihood).nonzero())
        mode = [EncodedEvent(zip(self.variables, event)) for event in events.tolist()]
        return mode, likelihood

    def mode(self) -> Tuple[List, float]:
        """
        Calculate the most likely event.
        :return: The mode of the distribution as Event and its likelihood.
        """
        mode, likelihood = self._mode()
        return [mode_.decode() for mode_ in mode], likelihood

    def __copy__(self) -> 'Multinomial':
        """
        :return: a shallow copy of the distribution.
        """
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

        return Multinomial(self.variables, probabilities)

    def __eq__(self, other: 'Multinomial') -> bool:
        """Compare self with other and return the boolean result.

        Two discrete random variables are equal only if the probability mass
        functions are equal and the order of dimensions are equal.

        """
        return (self.variables == other.variables and
                self.probabilities.shape == other.probabilities.shape and
                np.allclose(self.probabilities, other.probabilities))

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

    def encode(self, event: Iterable) -> List[int]:
        """
        Encode an event into a list of indices within the respective domains.
        :param event: The event to encode as a list of elements of the respective variables domains
        :return: The encoded event
        """
        return [variable.encode(value) for variable, value in zip(self.variables, event)]

    def encode_many(self, events: Iterable[Iterable]) -> List[List[int]]:
        """
        Encode multiple events into a list of indices within the respective domains.
        :param events: The events to encode as a list of elements of the respective variables domains
        :return: The encoded events
        """
        return [self.encode(event) for event in events]

    def decode(self, event: Iterable[int]) -> List:
        """
        Decode an event from a list of indices to a list of values.
        :param event: The event to decode as a list of indices
        :return: The decoded event
        """
        return [variable.decode(value) for variable, value in zip(self.variables, event)]

    def decode_many(self, events: Iterable[Iterable[int]]) -> List[List]:
        """
        Decode multiple events from a list of indices to a list of values.
        :param events: The events to decode as a list of indices
        :return: The decoded events
        """
        return [self.decode(event) for event in events]

    def _probability(self, event: EncodedEvent) -> float:
        """
        Calculate the probability of an event encoded.
        The encoded event has to contain information about all variables in the distribution.
        :param event: The event to calculate the probability of.
        :return: P(event)
        """
        indices = tuple(event[variable] for variable in self.variables)
        return self.probabilities[np.ix_(*indices)].sum()

    def probability(self, event: Event) -> float:
        """
        Calculate the probability of an event.
        :param event: The event to calculate the probability of.
        :return: P(event)
        """
        event = Event({variable: variable.domain for variable in self.variables}) & event
        return self._probability(event.encode())

    def _likelihood(self, event: List[int]) -> float:
        """
        Calculate the likelihood of a full evidence query.
        The event is a list of indices for the variable values in the same order
        :param event:
        :return: P(event)
        """
        return float(self.probabilities[tuple(event)])

    def likelihood(self, event: List) -> float:
        """
        Calculate the likelihood of a full evidence query.
        The event is a list of values for the variables in the same order
        :param event:
        :return: P(event)
        """
        return self._likelihood(self.encode(event))

    def _conditional(self, event: EncodedEvent) -> 'Multinomial':
        """
        Calculate the conditional distribution given an event encoded.
        The encoded event has to contain information about all variables in the distribution.
        :param event: The event to condition on.
        :return: The conditional distribution
        """
        indices = tuple(event[variable] for variable in self.variables)
        indices = np.ix_(*indices)
        probabilities = np.zeros_like(self.probabilities)
        probabilities[indices] = self.probabilities[indices]
        return Multinomial(self.variables, probabilities)

    def conditional(self, event: Event) -> 'Multinomial':
        """
        Calculate the conditional distribution given an event.
        :param event: The event to condition on
        :return: The conditional distribution
        """
        event = Event({variable: variable.domain for variable in self.variables}) & event
        return self._conditional(event.encode())

    def normalize(self) -> 'Multinomial':
        """
        Normalize the distribution.
        :return: The normalized distribution
        """
        normalized_probabilities = self.probabilities / np.sum(self.probabilities)
        return Multinomial(self.variables, normalized_probabilities)
