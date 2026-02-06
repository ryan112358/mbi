"""Defines the CliqueVector class for managing collections of factors over cliques.

This module introduces the `CliqueVector`, a data structure designed to hold
and manipulate sets of `Factor` objects, each associated with a specific clique
(a subset of attributes) within a domain. It facilitates operations common in
graphical models, such as projecting onto sub-cliques, expanding to larger cliques,
and performing arithmetic operations on these collections.
"""

from __future__ import annotations

import functools
import operator

import attr
import chex
import jax
import jax.numpy as jnp

from .clique_utils import Clique, reverse_clique_mapping
from .domain import Domain
from .factor import Factor, Projectable


@functools.partial(
    jax.tree_util.register_dataclass,
    meta_fields=["domain", "cliques"],
    data_fields=["arrays"],
)
@attr.dataclass
class CliqueVector:
    """Manages a collection of factors, each associated with a clique.

    This class provides a structure for holding and operating on multiple
    `Factor` objects, where each factor corresponds to a clique (a subset of
    attributes) within a larger domain. It's particularly useful in the context
    of graphical models for representing sets of potentials or marginals.

    Attributes:
        domain (Domain): The overall domain that encompasses all cliques.
        cliques (list[Clique]): A list of cliques (tuples of attribute names)
            for which factors are stored.
        arrays (dict[Clique, Factor]): A dictionary mapping each clique in
            `cliques` to its corresponding `Factor` object.
    """

    domain: Domain
    cliques: list[Clique]
    arrays: dict[Clique, Factor]

    def __attrs_post_init__(self):
        if set(self.cliques) != set(self.arrays):
            raise ValueError("Cliques must be equal to keys of array.")
        if len(self.cliques) != len(set(self.cliques)):
            raise ValueError("Cliques must be unique.")

    @classmethod
    def zeros(cls, domain: Domain, cliques: list[Clique]) -> CliqueVector:
        """Creates a CliqueVector initialized with zero factors for each clique."""
        arrays = {cl: Factor.zeros(domain.project(cl)) for cl in cliques}
        return cls(domain, cliques, arrays)

    @classmethod
    def ones(cls, domain: Domain, cliques: list[Clique]) -> CliqueVector:
        """Creates a CliqueVector initialized with one factors for each clique."""
        arrays = {cl: Factor.ones(domain.project(cl)) for cl in cliques}
        return cls(domain, cliques, arrays)

    @classmethod
    def random(cls, domain: Domain, cliques: list[Clique]) -> CliqueVector:
        """Creates a CliqueVector initialized with random factors for each clique."""
        arrays = {cl: Factor.random(domain.project(cl)) for cl in cliques}
        return cls(domain, cliques, arrays)

    @classmethod
    def abstract(cls, domain: Domain, cliques: list[Clique]) -> CliqueVector:
        arrays = {cl: Factor.abstract(domain.project(cl)) for cl in cliques}
        return cls(domain, cliques, arrays)

    @classmethod
    def from_projectable(cls, data: Projectable, cliques: list[Clique]):
        """Creates a CliqueVector by projecting a data source onto the specified cliques."""
        arrays = {cl: data.project(cl) for cl in cliques}
        return cls(data.domain, cliques, arrays)

    @functools.cached_property
    def active_domain(self):
        """Returns the merged domain encompassing all attributes across all cliques."""
        domains = [self.domain.project(cl) for cl in self.cliques]
        return functools.reduce(lambda a, b: a.merge(b), domains, Domain([], []))

    # @functools.lru_cache(maxsize=None)
    def parent(self, clique: Clique) -> Clique | None:
        """Finds a clique in this vector that is a superset of the given clique."""
        for result in self.cliques:
            if set(clique) <= set(result):
                return result

    def supports(self, clique: Clique) -> bool:
        """Checks if the given clique is supported (is a subset of any clique in the vector)."""
        return self.parent(clique) is not None

    def project(self, clique: Clique, log: bool = False) -> Factor:
        clique = tuple(clique)
        if clique in self.arrays:
            return self[clique]
        if self.supports(clique):
            return self[self.parent(clique)].project(clique, log=log)
        raise ValueError(f"Cannot project onto unsupported clique {clique}.")

    def expand(self, cliques: list[Clique]) -> CliqueVector:
        """Re-expresses this CliqueVector over an expanded set of cliques.

        If the original CliqueVector represents the potentials of a Graphical Model,
        the given cliques support the cliques in the original CliqueVector, then
        the distribution represented by the new CliqueVector will be identical.

        Args:
            cliques: The new cliques the clique vector will be defined over.

        Returns:
            An expanded CliqueVector defined over the given set of cliques.
        """
        mapping = reverse_clique_mapping(cliques, self.cliques, domain=self.domain)
        arrays = {}
        for cl in cliques:
            dom = self.domain.project(cl)
            if len(mapping[cl]) == 0:
                arrays[cl] = Factor.zeros(dom)
            else:
                arrays[cl] = sum(self[cl2] for cl2 in mapping[cl]).expand(dom)
        return CliqueVector(self.domain, cliques, arrays)

    def contract(self, cliques: list[Clique], log: bool = False) -> CliqueVector:
        """Computes a new CliqueVector by projecting this one onto a smaller set of cliques."""
        arrays = {cl: self.project(cl, log=log) for cl in cliques}
        return CliqueVector(self.domain, cliques, arrays)

    def normalize(self, total: float = 1, log: bool = True):
        """Normalizes each factor within the CliqueVector."""
        return jax.tree.map(
            lambda f: f.normalize(total, log), self, is_leaf=Factor.__instancecheck__
        )

    def __mul__(self, const: chex.Numeric) -> CliqueVector:
        """Multiplies each factor in the vector by a constant."""
        return jax.tree.map(lambda f: f * const, self)

    def __rmul__(self, const: chex.Numeric) -> CliqueVector:
        """Right-multiplies each factor in the vector by a constant."""
        return self.__mul__(const)

    def __truediv__(self, const: chex.Numeric) -> CliqueVector:
        """Divides each factor in the vector by a constant."""
        return self.__mul__(1 / const)

    def __add__(self, other: chex.Numeric | CliqueVector) -> CliqueVector:
        """Adds another CliqueVector or a constant to this vector elementwise."""
        if isinstance(other, CliqueVector):
            return jax.tree.map(jnp.add, self, other)
        return jax.tree.map(lambda f: f + other, self)

    def __sub__(self, other: chex.Numeric | CliqueVector) -> CliqueVector:
        """Subtracts another CliqueVector or a constant from this vector elementwise."""
        return self + -1 * other

    def exp(self) -> CliqueVector:
        """Applies elementwise exponentiation (jnp.exp) to each factor."""
        return jax.tree.map(jnp.exp, self)

    def log(self) -> CliqueVector:
        """Applies elementwise logarithm (jnp.log) to each factor."""
        return jax.tree.map(jnp.log, self)

    def dot(self, other: CliqueVector) -> chex.Numeric:
        """Computes the dot product between this CliqueVector and another."""
        dots = jax.tree.map(Factor.dot, self, other, is_leaf=Factor.__instancecheck__)
        return jax.tree.reduce(operator.add, dots, 0)

    def size(self):
        """Calculates the total number of parameters across all factors in the vector."""
        return sum(self.domain.size(cl) for cl in self.cliques)

    def __getitem__(self, clique: Clique) -> Factor:
        """Retrieves the factor associated with the given clique."""
        return self.arrays[clique]

    def __setitem__(self, clique: Clique, value: Factor):
        """Sets the factor for a given clique, replacing the existing one if present."""
        if clique in self.cliques:
            self.arrays[clique] = value
        else:
            raise ValueError(f"Clique {clique} not in CliqueVector.")

    def apply_sharding(self, mesh: jax.sharding.Mesh | None) -> CliqueVector:
        """Apply sharding constraint to each factor in the CliqueVector.

        The sharding strategy is automatically determined based on the provided
        mesh and the factor domains.

        Args:
            mesh: The mesh over which the factors should be sharded.

        Returns:
            A new CliqueVector identical to self with sharding constraints applied to
            the underlying factors.
        """
        arrays = {cl: self.arrays[cl].apply_sharding(mesh) for cl in self.cliques}
        return CliqueVector(self.domain, self.cliques, arrays)
