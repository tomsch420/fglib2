import dataclasses
from typing import Tuple, Iterable, Union, List, Any


@dataclasses.dataclass(unsafe_hash=True, eq=True)
class Variable:
    """
    An abstract random variable.
    """

    name: str
    """The name of this random variable."""

    def __lt__(self, other: 'Variable'):
        return self.name < other.name

    def __gt__(self, other: 'Variable'):
        return self.name > other.name

    def __le__(self, other: 'Variable'):
        return self.name <= other.name

    def __ge__(self, other: 'Variable'):
        return self.name >= other.name


@dataclasses.dataclass(unsafe_hash=True, eq=True)
class Symbolic(Variable):
    """
    Symbolic (unordered) random variable.
    """

    domain: Tuple = dataclasses.field(repr=False)
    """The domain of this variable. Every object in the domain must implement the equal operator."""

    def __init__(self, name: str, domain: Iterable):
        super().__init__(name)
        self.domain = tuple(domain)

    def encode(self, values: Union[Any, Iterable]) -> Union[List[int], int]:
        """
        Convert values from the domain to their respective indices in the domain, such that they can be used for
            indexing.

        :param values: The values to convert.

        :return: List of indices or just the index if a single element was given.
        """
        if values in self.domain:
            return self.encode([values])[0]
        else:
            return [self.domain.index(value) for value in values]
