"""Structural constraints on allowed value combinations.

Provides the :class:`Constraint` dataclass for declaring which combinations of
attribute values are valid (or invalid) in a dataset domain.  Constraints can
be folded into graphical model potentials via :meth:`Constraint.as_potential`.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from .domain import Attribute
from .domain import Domain
from .factor import Factor


def _validate_combos(arr, n_attrs, name):
  shape = np.shape(arr)
  if len(shape) != 2 or shape[1] != n_attrs:
    raise ValueError(f"{name} must have shape (*, {n_attrs}), got {shape}.")


def _validate_mapping(mapping, domain):
  if len(domain.attributes) != 2:
    raise ValueError("mapping requires exactly 2 attributes (fine, coarse).")
  shape = np.shape(mapping)
  if len(shape) != 1 or shape[0] != domain.shape[0]:
    raise ValueError(
        f"mapping must have shape ({domain.shape[0]},), got {shape}."
    )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class Constraint:
  """A structural constraint on allowed value combinations.

  Exactly one of ``valid``, ``invalid``, or ``mapping`` must be specified.

  Attributes:
      domain: The sub-domain this constraint covers.
      valid: Array of shape ``(n, len(domain))`` listing allowed combos.
      invalid: Array of shape ``(n, len(domain))`` listing forbidden combos.
      mapping: Array of shape ``(domain.shape[0],)`` defining a functional
          dependency from the first attribute (fine) to the second (coarse).
          Requires exactly two attributes in the domain.
  """

  domain: Domain = jax.tree.static()
  valid: np.ndarray | None = None
  invalid: np.ndarray | None = None
  mapping: np.ndarray | None = None

  def __post_init__(self):
    # During JAX pytree reconstruction, fields are tracers.
    for f in ("valid", "invalid", "mapping"):
      if isinstance(getattr(self, f), jax.Array):
        return

    n_set = sum(x is not None for x in (self.valid, self.invalid, self.mapping))
    if n_set != 1:
      raise ValueError(
          "Specify exactly one of 'valid', 'invalid', or 'mapping'."
      )

    n = len(self.domain.attributes)
    if self.valid is not None:
      _validate_combos(self.valid, n, "valid")
    elif self.invalid is not None:
      _validate_combos(self.invalid, n, "invalid")
    else:
      _validate_mapping(self.mapping, self.domain)

  @property
  def clique(self) -> tuple[Attribute, ...]:
    """Sorted attribute names, for junction tree construction."""
    return tuple(sorted(self.domain.attributes))

  @property
  def is_deterministic(self) -> bool:
    """Whether this constraint is a functional dependency."""
    return self.mapping is not None

  @property
  def potential(self) -> Factor:
    """Return a log-space potential: 0 for allowed, -inf for forbidden."""
    if self.mapping is not None:
      values = jnp.full(self.domain.shape, -jnp.inf)
      fine_idx = jnp.arange(self.domain.shape[0])
      values = values.at[fine_idx, self.mapping].set(0.0)
    elif self.valid is not None:
      valid = jnp.asarray(self.valid)
      idx = tuple(valid[:, i] for i in range(valid.shape[1]))
      values = jnp.full(self.domain.shape, -jnp.inf).at[idx].set(0.0)
    else:
      invalid = jnp.asarray(self.invalid)
      idx = tuple(invalid[:, i] for i in range(invalid.shape[1]))
      values = jnp.zeros(self.domain.shape).at[idx].set(-jnp.inf)
    return Factor(self.domain, values)
