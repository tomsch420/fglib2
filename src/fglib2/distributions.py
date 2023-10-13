from typing import Iterable, List, Optional

import numpy as np

from .variables import Symbolic

import copy


class Multinomial:

    variables = List[Symbolic]

    def __init__(self, variables: Iterable[Symbolic], probabilities: Optional[np.ndarray] = None,
                 normalize: bool = True):
        self.variables = list(variables)

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
        axis = tuple(self.variables.index(variable) for variable in set(self.variables) - set(variables))

        probabilities = np.sum(self.probabilities, axis=axis)

        return Multinomial(variables, probabilities, normalize=normalize)

    def expand(self, variables: Iterable[Symbolic]) -> 'Multinomial':
        """
        Expand the distribution to include the given variables.
        :param variables: The variables to include
        :return: The expanded multinomial distribution.
        """

        # Extract missing dimensions and their cardinality
        diff = [var for var in variables if var not in self.variables]
        reps = [1, ] * (len(self.variables) + len(diff))

        # copy probabilities
        probabilities = self.probabilities

        # Expand missing dimensions
        for index, variable in enumerate(diff):
            probabilities = np.expand_dims(probabilities, axis=-1)
            reps[len(self.variables) + index] = len(variable.domain)

        # Repeat missing dimensions
        probabilities = np.tile(probabilities, reps)

        # return final distribution
        return Multinomial(self.variables + diff, probabilities, normalize=False)

    def __copy__(self):
        return Multinomial(self.variables, self.probabilities)

    def merge(self, other: 'Multinomial') -> 'Multinomial':
        """
        Merge the dimensions of two multinomial distributions.
        :param other: The other distribution.
        :return: A new multinomial distribution that contains variables from both distributions.
        """
        # Verify the dimensions of summand and summand.
        if len(self.probabilities.shape) < len(other.probabilities.shape):
            return self.expand(other.variables)
        elif len(self.probabilities.shape) > len(other.probabilities.shape):
            return other.expand(self.variables)
        else:
            return copy.copy(self)

    def __add__(self, other: 'Multinomial') -> 'Multinomial':
        """Add two Multinomial distributions and return the result.

        :param other: The other distribution to add.

        :return: The sum of the two distributions.
        """
        result = self.merge(other)
        result.probabilities = self.probabilities + other.probabilities
        return result

    def __mul__(self, other):
        """Multiply two Multinomial distributions and return the result.

        :param other: The other distribution to multiply.

        :return: The sum of the two distributions.

        """
        result = self.merge(other)
        result.probabilities = self.probabilities * other.probabilities
        return result

    def __eq__(self, other: 'Multinomial') -> bool:
        """Compare self with other and return the boolean result.

        Two discrete random variables are equal only if the probability mass
        functions are equal and the order of dimensions are equal.

        """
        return (self.variables == other.variables and
                self.probabilities.shape == other.probabilities.shape and
                np.all(self.probabilities == other.probabilities))
