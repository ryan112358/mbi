from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from typing import Literal, Protocol

import attr
import chex
import jax
import jax.numpy as jnp
import numpy as np

from .domain import Domain

jax.config.update("jax_enable_x64", True)


@functools.partial(
    jax.tree_util.register_dataclass, meta_fields=["domain"], data_fields=["values"]
)
@attr.dataclass(frozen=True)
class Factor:
    """Represents a factor defined over a discrete domain.

    A factor can be thought of as a potential function or an unnormalized
    probability distribution over a set of discrete variables defined by a
    `Domain` object. It maps each configuration of the domain to a value.

    Attributes:
        domain (Domain): The discrete domain over which the factor is defined.
        values (jax.Array): A JAX array containing the factor's values. The shape
            of this array matches the shape specified by the `domain`.

    Supported Operations:
        - Creation: `zeros`, `ones`, `random` for creating factors.
        - Reshaping: `transpose`, `expand` for modifying the domain/shape.
        - Aggregation: `sum`, `logsumexp`, `project` for marginalizing attributes.
        - Element-wise: `exp`, `log`, `normalize` for value transformations.
        - Binary Ops: `+`, `-`, `*`, `/`, `dot` for combining factors.

    Example Usage:
        >>> from mbi import Domain  # Needed for doctest context
        >>> domain = Domain.fromdict({'X': 2, 'Y': 3})
        >>> factor = Factor.ones(domain)
        >>> print(factor.domain)
        Domain(X: 2, Y: 3)
    """

    domain: Domain
    values: jax.Array

    def __post_init__(self):
        if self.values.shape != self.domain.shape:
            raise ValueError("values must be same shape as domain.")

    # Constructors
    @classmethod
    def zeros(cls, domain: Domain) -> Factor:
        """Creates a Factor object with all values initialized to zero."""
        return cls(domain, jnp.zeros(domain.shape))

    @classmethod
    def ones(cls, domain: Domain) -> Factor:
        """Creates a Factor object with all values initialized to one."""
        return cls(domain, jnp.ones(domain.shape))

    @classmethod
    def random(cls, domain: Domain) -> Factor:
        """Creates a Factor object with random values (uniform 0-1)."""
        return cls(domain, jnp.asarray(np.random.rand(*domain.shape)))

    @classmethod
    def abstract(cls, domain: Domain) -> Factor:
        return cls(domain, jax.ShapeDtypeStruct(domain.shape, jnp.float64))

    # Reshaping operations
    def transpose(self, attrs: Sequence[str]) -> Factor:
        """Rearranges the factor's axes according to the new attribute order."""
        if set(attrs) != set(self.domain.attrs):
            raise ValueError("attrs must be same as domain attributes")
        newdom = self.domain.project(attrs)
        ax = newdom.axes(self.domain.attrs)
        values = jnp.moveaxis(self.values, range(len(ax)), ax)
        return Factor(newdom, values)

    def expand(self, domain):
        """Expands the factor's domain to include new attributes."""
        if not domain.contains(self.domain):
            raise ValueError("Expanded domain must contain domain.")
        dims = len(domain) - len(self.domain)
        values = self.values.reshape(self.domain.shape + tuple([1] * dims))
        ax = domain.axes(self.domain.attrs)
        values = jnp.moveaxis(values, range(len(ax)), ax)
        values = jnp.broadcast_to(values, domain.shape)
        return Factor(domain, values)

    # Functions that aggregate along some subset of axes
    def _aggregate(self, fn: Callable, attrs: Sequence[str] | None = None) -> Factor:
        """Helper for aggregating values along specified attribute axes."""
        attrs = self.domain.attrs if attrs is None else attrs
        axes = self.domain.axes(attrs)
        values = fn(self.values, axis=axes)
        newdom = self.domain.marginalize(attrs)
        return Factor(newdom, values)

    def max(self, attrs: Sequence[str] | None = None) -> Factor:
        """Computes the maximum value along specified attribute axes."""
        return self._aggregate(jnp.max, attrs)

    def sum(self, attrs: Sequence[str] | None = None) -> Factor:
        """Computes the sum along specified attribute axes."""
        return self._aggregate(jnp.sum, attrs)

    def logsumexp(self, attrs: Sequence[str] | None = None) -> Factor:
        """Computes the log-sum-exp along specified attribute axes."""
        return self._aggregate(jax.scipy.special.logsumexp, attrs)

    def project(self, attrs: str | Sequence[str], log: bool = False) -> "Factor":
        """Computes the marginal distribution by summing/logsumexp'ing out other attributes."""
        if isinstance(attrs, str):
            attrs = (attrs,)
        marginalized = self.domain.marginalize(attrs).attrs
        result = self.logsumexp(marginalized) if log else self.sum(marginalized)
        return result.transpose(attrs)

    def slice(self, evidence: dict[str, int | np.ndarray | jax.Array]) -> "Factor":
        """Slices the factor by fixing specific attribute values.

        If at least one attribute has numpy-valued evidence, the returned factor will
        have a new leading dimension called '_mbi_evidence' corresponding to the
        number of evidence points.

        Args:
            evidence: A dictionary mapping attribute names to the values they should be fixed to.

        Returns:
            A new Factor with the specified attributes fixed and removed from the domain.
        """
        slices = [slice(None)] * len(self.domain)
        relevant = [e for e in evidence if e in self.domain.attrs]

        adv_indices = []
        has_vector = False
        ev_size = None

        for attr in relevant:
            axis = self.domain.axes((attr,))[0]
            val = evidence[attr]
            slices[axis] = val
            adv_indices.append(axis)

            is_arr = hasattr(val, "ndim") and val.ndim > 0
            if is_arr:
                has_vector = True
                if ev_size is None:
                    ev_size = val.shape[0]
                elif ev_size != val.shape[0]:
                    raise ValueError("All vector evidence must have same size.")

        values = self.values[tuple(slices)]
        domain = self.domain.marginalize(relevant)

        if has_vector:
            adv_indices.sort()
            # If advanced indices are contiguous, numpy puts the new dimension at the start of the block
            is_contiguous = (adv_indices[-1] - adv_indices[0] + 1) == len(adv_indices)
            target_axis = adv_indices[0] if is_contiguous else 0

            # We want the evidence dimension to be at axis 0
            if target_axis != 0:
                values = jnp.moveaxis(values, target_axis, 0)

            new_labels = None
            if self.domain.labels is not None:
                new_labels = (tuple(range(ev_size)),)

            new = Domain(["_mbi_evidence"], [ev_size], labels=new_labels)
            domain = new.merge(domain)

        return Factor(domain, values)

    def supports(self, attrs: str | Sequence[str]) -> bool:
        return self.domain.supports(attrs)

    # Functions that operate element-wise
    def exp(self, out=None) -> Factor:
        """Applies element-wise exponentiation (jnp.exp) to the factor's values."""
        return Factor(self.domain, jnp.exp(self.values))

    def log(self, out=None) -> Factor:
        """Applies element-wise logarithm (jnp.log) to the factor's values."""
        return Factor(self.domain, jnp.log(self.values))

    def normalize(self, total: float = 1.0, log: bool = False) -> Factor:
        """Normalizes the factor so its values sum to `total` (or log-normalize)."""
        if log:
            return self + jnp.log(total) - self.logsumexp()
        return self * total / self.sum()

    def copy(self) -> Factor:
        """Returns a copy of the factor (potentially shallow due to JAX)."""
        return self

    def __float__(self):
        if len(self.domain) > 0:
            raise ValueError("Domain must be empty to convert to float.")
        return float(self.values)

    # Binary operations between two factors
    def _binaryop(self, fn: Callable, other: Factor | chex.Numeric) -> Factor:
        """Helper for applying binary operations between this factor and another factor or scalar."""
        if not isinstance(other, Factor) and jnp.ndim(other) == 0:
            other = Factor(Domain([], []), jnp.asarray(other))
        newdom = self.domain.merge(other.domain)
        factor1 = self.expand(newdom)
        factor2 = other.expand(newdom)
        return Factor(newdom, fn(factor1.values, factor2.values))

    def __sub__(self, other: Factor | chex.Numeric) -> Factor:
        return self._binaryop(jnp.subtract, other)

    def __truediv__(self, other: Factor | chex.Numeric) -> Factor:
        return self._binaryop(jnp.divide, other)

    def __mul__(self, other: Factor | chex.Numeric) -> Factor:
        """Multiply two factors together.

        Example Usage:
            >>> f1 = Factor.ones(Domain(['a','b'], [2,3]))
            >>> f2 = Factor.ones(Domain(['b','c'], [3,4]))
            >>> f3 = f1 * f2
            >>> print(f3.domain)
            Domain(a: 2, b: 3, c: 4)

        Args:
          other: the other factor to multiply

        Returns:
          the product of the two factors
        """
        return self._binaryop(jnp.multiply, other)

    def __add__(self, other: Factor | chex.Numeric) -> Factor:
        return self._binaryop(jnp.add, other)

    def __radd__(self, other: chex.Numeric) -> Factor:
        return self + other

    def __rsub__(self, other: chex.Numeric) -> Factor:
        return self + (-1 * other)

    def __rmul__(self, other: chex.Numeric) -> Factor:
        return self * other

    def dot(self, other: Factor) -> Factor:
        if self.domain != other.domain:
            raise ValueError(f"Domains do not match {self.domain} != {other.domain}")
        return jnp.sum(self.values * other.values)

    def datavector(self, flatten: bool = True) -> jax.Array:
        """Returns the factor's values as a flattened vector or original array."""
        return self.values.flatten() if flatten else self.values

    def pad(
        self, mesh: jax.sharding.Mesh | None, pad_value: Literal[0, "-inf"]
    ) -> Factor:
        if mesh is None:
            return self
        pad_amounts = [0] * len(self.domain)
        for i, ax in enumerate(self.domain):
            if ax in mesh.axis_names:
                size = self.domain[ax]
                num_shards = mesh.axis_sizes[mesh.axis_names.index(ax)]
                pad_amounts[i] = -size % num_shards

        values = jnp.pad(
            self.values,
            pad_width=tuple((0, w) for w in pad_amounts),
            constant_values=0.0 if pad_value == 0 else -jnp.inf,
        )
        # We keep the domain as-is here, even though values is now larger.
        # We have a couple of options
        #   1. Explicitly unpad when we are done.
        #   2. Never unpad, just expand the domain, and make sure new elements are impossible.
        #   3. Allow values to be an array where each dim is >= the domain implies, and truncate
        #       when necessary.
        return Factor(self.domain, values)

    def apply_sharding(self, mesh: jax.sharding.Mesh | None) -> Factor:
        """Apply sharding constraint to the factor values.

        The sharding strategy is automatically determined based on the provided
        mesh, and the factor domain.

        Args:
            mesh: The mesh over which the factor should be sharded.

        Returns:
            A new factor identical to self with sharding constraints applied to the values.
        """
        if mesh is None:
            return self
        pspec = [None] * len(self.domain)
        for i, ax in enumerate(self.domain):
            if ax in mesh.axis_names:
                pspec[i] = ax
        sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(*pspec))

        return Factor(
            domain=self.domain,
            values=jax.lax.with_sharding_constraint(self.values, sharding),
        )


class Projectable(Protocol):
    """A projectable is an object that can be projected onto a subset of attributes to compute a marginal.

    Example projectables:
        * Dataset
        * Factor
        * CliqueVector
        * MarkovRandomField
    """

    @property
    def domain(self) -> Domain:
        """Returns the domain over which this projectable is defined."""

    def project(self, attrs: str | Sequence[str]) -> Factor:
        """Projection onto a subset of attributes."""

    def supports(self, attrs: str | Sequence[str]) -> bool:
        """Returns true if the given attributes can be projected onto."""
