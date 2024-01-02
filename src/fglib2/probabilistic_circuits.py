from typing import Iterable, Optional, Union, List

import numpy as np
from typing_extensions import Self

from probabilistic_model.probabilistic_model import ProbabilisticModel
from probabilistic_model.probabilistic_circuit import SmoothSumUnit
from random_events.variables import Variable, Symbolic
from .distributions import Multinomial


class SumUnitFactor(ProbabilisticModel):
    """
    A sum unite (mixture model) that can be used as factor for variables in a factor graph.

    Example use-case:
        Imagine you have a set of variables that expand over some template, e.g. time.
        You learn a mixture for each time step and then use the latent variable interpretation of the
        mixture model to create a factor graph. The factors for the transition model are multinomial distributions
        over the latent variables. The factors for the emission model are the joint probability trees.
    """

    def __init__(self, model: SmoothSumUnit):
        self.model = model
        latent_variable = Symbolic(f"latent_{str(id(model))}", range(len(self.model.weights)))
        super().__init__([latent_variable])

    @property
    def latent_variable(self) -> Symbolic:
        return self.variables[0]

    def marginal(self, variables: List[Variable]) -> Union[Multinomial, Self]:
        if variables[0] == self.latent_variable:
            return Multinomial([self.latent_variable], np.array(self.model.weights))