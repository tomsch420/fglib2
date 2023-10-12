from typing import Iterable, List, Optional

import numpy as np

from .variables import Symbolic


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

    def expand(self, variables: Iterable[Symbolic], states) -> 'Multinomial':
        reps = [1, ] * len(list(variables))

        # Extract missing dimensions
        diff = [i for i, d in enumerate(variables) if d not in self.variables]

        probabilities = self.probabilities

        # Expand missing dimensions
        for d in diff:
            probabilities = np.expand_dims(probabilities, axis=d)
            reps[d] = states[d]

        # Repeat missing dimensions
        probabilities = np.tile(probabilities, reps)
        return Multinomial(variables, probabilities)

    def __mul__(self, other: 'Multinomial') -> 'Multinomial':
        ...

    def __add__(self, other: 'Multinomial') -> 'Multinomial':
        ...
