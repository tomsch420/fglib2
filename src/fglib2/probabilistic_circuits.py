from typing import Iterable, Optional, Union, List, Tuple

import numpy as np
from random_events.events import Event, EncodedEvent
from typing_extensions import Self

from probabilistic_model.probabilistic_model import ProbabilisticModel
from probabilistic_model.probabilistic_circuit import SmoothSumUnit, DeterministicSumUnit, Unit
from random_events.variables import Variable, Symbolic, Discrete
from .distributions import Multinomial
from .graphs import FactorNode


class SumUnitWrapper(DeterministicSumUnit):

    @staticmethod
    def from_sum_unit(unit: SmoothSumUnit) -> 'DeterministicSumUnit':
        result = SumUnitWrapper(unit.variables, unit.weights)
        result.children = unit.children
        return result

    def _conditional(self, event: EncodedEvent) -> Tuple[Optional[Self], float]:
        # conditional weights of new sum unit
        conditional_weights = []

        # conditional children of new sum unit
        conditional_children = []

        # initialize probability
        probability = 0.

        for weight, child in zip(self.weights, self.children):
            conditional_child, conditional_probability = child._conditional(event)

            if conditional_child is None:
                conditional_child = child

            conditional_probability = conditional_probability * weight
            probability += conditional_probability

            conditional_weights.append(conditional_probability)
            conditional_children.append(conditional_child)

        result = self._parameter_copy()
        result.weights = conditional_weights
        result.children = conditional_children
        return result.normalize(), probability


class SumUnitFactor(FactorNode):
    """
    A sum unite (mixture model) that can be used as factor for variables in a factor graph.

    Example use-case:
        Imagine you have a set of variables that expand over some template, e.g. time.
        You learn a mixture for each time step and then use the latent variable interpretation of the
        mixture model to create a factor graph. The factors for the transition model are multinomial distributions
        over the latent variables. The factors for the emission model are the joint probability trees.
    """

    latent_variable: Symbolic

    def __init__(self, distribution: SmoothSumUnit):
        super().__init__(SumUnitWrapper.from_sum_unit(distribution))
        self.latent_variable = Symbolic(f"latent_{str(id(self.distribution))}",
                                        range(len(self.distribution.weights)))

    def marginal(self, variables: List[Variable]) -> Union[Multinomial, Self]:
        if variables[0] == self.latent_variable:
            return self.latent_distribution()

    @property
    def variables(self) -> List[Discrete]:
        return [self.latent_variable]

    def latent_distribution(self):
        return Multinomial([self.latent_variable], np.array(self.distribution.weights))

    def sum_product(self, messages: List[Multinomial]) -> Multinomial:

        message = self.latent_distribution()

        # Product over incoming messages
        for msg in messages:
            if msg is not None:
                message *= msg

        return message