"""Defines loss functions based on linear measurements of marginals.

This module provides structures and functions for defining and calculating loss
based on potentially noisy linear measurements of marginal distributions. Key
components include the ``LinearMeasurement`` class to represent individual
measurements and the ``MarginalLossFn`` class to define loss functions over
``CliqueVector`` objects, enabling the evaluation of model fit against observed
or noisy data. Utilities for clique manipulation and feasibility checks are also
included.
"""

import dataclasses
from collections.abc import Callable, Sequence
from typing import Any, SupportsFloat, cast

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
import numpy as np
import optax

from .clique_utils import Clique, clique_mapping, maximal_subset
from .clique_vector import CliqueVector
from .domain import Domain
from .factor import Factor


def _weighted_datavector(weights: np.ndarray, x: Factor) -> jax.Array:
  """Datavector scaled by pre-baked weights (pickle-safe).

  .. deprecated::
      Use :class:`WeightedQuery` instead for serialization support.
  """
  return x.datavector() * weights


# ---------------------------------------------------------------------------
# Serializable query types
# ---------------------------------------------------------------------------
# These callable dataclasses can be used as the ``query`` field of
# :class:`LinearMeasurement`.  Because they are structured data (not opaque
# closures), ``save_pytree`` can round-trip them via pickle.


@dataclasses.dataclass(frozen=True)
class DatavectorQuery:
  """Identity query: ``f.datavector()``.

  Attributes:
      use_for_total_estimation: If ``False``, exclude measurements using this
          query from ``minimum_variance_unbiased_total``. Defaults to ``True``.
  """

  use_for_total_estimation: bool = True

  def __call__(self, f: Factor) -> jax.Array:
    return f.datavector()

  def op_norm_sq(self) -> float:
    return 1.0


@dataclasses.dataclass(frozen=True, eq=False)
class WeightedQuery:
  """Datavector scaled by fixed weights: ``f.datavector() * weights``."""

  weights: np.ndarray

  def __call__(self, f: Factor) -> jax.Array:
    return f.datavector() * self.weights

  def op_norm_sq(self) -> float:
    return float(np.max(np.asarray(self.weights) ** 2))

  # Identity-based hash/eq so JAX static tracing works (ndarray isn't hashable).
  def __hash__(self) -> int:
    return id(self)

  def __eq__(self, other: object) -> bool:
    return self is other


@dataclasses.dataclass(frozen=True)
class SlicedQuery:
  """Datavector with leading elements removed: ``f.datavector()[start:]``."""

  start: int = 1

  def __call__(self, f: Factor) -> jax.Array:
    return f.datavector()[self.start :]

  def op_norm_sq(self) -> float:
    return 1.0


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class LinearMeasurement:
  """A class for representing a private linear measurement of a marginal.

  Attributes:
      noisy_measurement: The noisy measurement of the marginal.
      clique: The clique (a tuple of attribute names) defining the marginal.
      stddev: The standard deviation of the noise added to the measurement.
      query: A linear function that, when applied to a Factor, extracts a
      a vector with the same shape and interpretation as `noisy_measurement`.
  """

  noisy_measurement: ArrayLike
  clique: Clique = jax.tree.static()
  # Dynamic (traced) leaf, not static aux-data: a static value gets baked into
  # the treedef, changing the jit cache key and defeating precompile() reuse.
  stddev: float = 1.0
  query: Callable[[Factor], jax.Array] = jax.tree.static(
      default=DatavectorQuery()
  )

  def __post_init__(self):
    object.__setattr__(self, "clique", tuple(self.clique))

  def compress(
      self, mapping: dict[str, np.ndarray], domain: Domain
  ) -> "LinearMeasurement":
    """Compress this measurement by merging domain values.

    Args:
        mapping: Maps attribute names to 1D integer arrays.
            ``mapping[attr][i]`` gives the compressed value for
            original value ``i``.
        domain: The full dataset domain (needed to reshape the
            measurement vector into the clique's shape).

    Returns:
        A new ``LinearMeasurement`` over the compressed domain.
    """
    if not any(a in mapping for a in self.clique):
      return self

    y = np.asarray(self.noisy_measurement).flatten()

    # Build per-attribute index arrays (mapping or identity).
    indices, compressed_sizes = [], []
    for a in self.clique:
      if a in mapping:
        m = np.asarray(mapping[a])
        indices.append(m)
        compressed_sizes.append(int(m.max()) + 1)
      else:
        indices.append(np.arange(domain[a]))
        compressed_sizes.append(domain[a])

    # Flat compressed index for every element of y.
    grids = np.meshgrid(*indices, indexing="ij")
    flat_idx = np.ravel_multi_index(
        [g.ravel() for g in grids], compressed_sizes
    )

    total = int(np.prod(compressed_sizes))
    y_compressed = np.bincount(flat_idx, weights=y, minlength=total)
    inv_coefs = 1.0 / np.sqrt(
        np.bincount(flat_idx, minlength=total).astype(float)
    )

    return LinearMeasurement(
        y_compressed * inv_coefs,
        self.clique,
        self.stddev,
        query=WeightedQuery(inv_coefs),
    )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class MarginalLossFn:
  """A loss function over the concatenated vector of marginals.

  Separates the computation (``loss_fn``, static) from the captured data
  (``data``, dynamic JAX pytree leaves).  This allows ``MarginalLossFn``
  to be passed through ``jax.jit`` as a regular traced argument rather
  than a static one, so that different measurement values with the same
  structure reuse the same compiled program.

  Attributes:
      cliques: Cliques defining the scope of the marginals.
      loss_fn: A pure function ``(marginals, data) -> loss``.  This is
          treated as static metadata for JIT compilation.
      data: Arbitrary pytree of arrays captured by ``loss_fn``.  This is
          traced through JIT and can change without recompilation.
      lipschitz: Lipschitz constant of the gradient. Defaults to ``1.0``.
  """

  cliques: Sequence[Clique] = jax.tree.static()
  loss_fn: Callable[[CliqueVector, Any], ArrayLike] = jax.tree.static()
  data: Any = ()
  # Dynamic (traced) leaf, always a concrete float (never None): a static value
  # gets baked into the treedef, changing the jit cache key and defeating reuse.
  lipschitz: float = 1.0

  def __post_init__(self):
    object.__setattr__(self, "cliques", tuple(self.cliques))

  def __call__(self, marginals: CliqueVector) -> ArrayLike:
    return self.loss_fn(marginals, self.data)


def _l2_loss(
    marginals: CliqueVector,
    measurements: tuple[LinearMeasurement, ...],
) -> ArrayLike:
  """Weighted L2 loss over linear measurements."""
  loss = 0.0
  for M in measurements:
    mu = marginals.project(M.clique)
    stddev = jnp.maximum(M.stddev, 1e-12)
    diff = (M.query(mu) - jnp.asarray(M.noisy_measurement)) / stddev
    loss += 0.5 * jnp.vdot(diff, diff)
  return loss


def _l1_loss(
    marginals: CliqueVector,
    measurements: tuple[LinearMeasurement, ...],
) -> ArrayLike:
  """Weighted L1 loss over linear measurements."""
  loss = 0.0
  for M in measurements:
    mu = marginals.project(M.clique)
    stddev = jnp.maximum(M.stddev, 1e-12)
    diff = (M.query(mu) - jnp.asarray(M.noisy_measurement)) / stddev
    loss += jnp.sum(jnp.abs(diff))
  return loss


def _normalized_l2_loss(
    marginals: CliqueVector,
    measurements: tuple[LinearMeasurement, ...],
) -> ArrayLike:
  """Normalized L2 loss over linear measurements."""
  loss = _l2_loss(marginals, measurements)
  total = marginals.project(()).datavector(flatten=False)
  return jnp.sqrt(loss / len(measurements) / total)


def _normalized_l1_loss(
    marginals: CliqueVector,
    measurements: tuple[LinearMeasurement, ...],
) -> ArrayLike:
  """Normalized L1 loss over linear measurements."""
  loss = _l1_loss(marginals, measurements)
  total = marginals.project(()).datavector(flatten=False)
  return loss / len(measurements) / total


def calculate_l2_lipschitz(
    domain: Domain,
    cliques: list[Clique],
    loss_fn: Callable[[CliqueVector], ArrayLike],
) -> float:
  """Estimate the Lipschitz constant of L(x) = || f(x) - y ||_2^2 where f is a linear function.

  The Lipschitz constant can usually be obtained via the largest eigenvalue of the Hessian, which
  for linear functions represented in matrix form is A^T A.  This function computes the same
  value without materializing this n x n matrix by using power iteration and leveraging jax.jvp.

  Args:
      domain: The domain over which the loss_fn is defined.
      loss_fn: The loss function, assumed to be of the form || f(x) - y ||_2^2 where f is linear.

  Returns:
      An estimate of the Lipschitz constant of the grad(L).
  """
  x0 = CliqueVector.zeros(domain, cliques)

  @jax.jit
  def compute_Hv(v: CliqueVector) -> CliqueVector:
    return jax.jvp(jax.grad(loss_fn), (x0,), (v,))[1]

  v = CliqueVector.ones(domain, cliques)
  v = v / optax.tree.norm(v)
  for _ in range(50):
    Hv = compute_Hv(v)
    estimate = optax.tree.norm(Hv)
    v = Hv / (estimate + 1e-12)
  return float(estimate)


def _query_op_norm_sq(query: Callable[[Factor], jax.Array]) -> float | None:
  """Squared operator norm ||Q||_2^2 of a linear query, or None if unknown."""
  fn = getattr(query, "op_norm_sq", None)
  return float(cast(SupportsFloat, fn())) if callable(fn) else None


def calculate_l2_lipschitz_from_metadata(
    domain: Domain,
    measurements: Sequence[LinearMeasurement],
) -> float | None:
  """Lipschitz constant of the (unnormalized) l2 loss gradient from metadata.

  The loss is quadratic, so its gradient's Lipschitz constant is lambda_max of a
  Hessian that is block-diagonal by maximal clique::

      H = sum_M (1 / sigma_M^2) (Q_M P_{c_M})^T (Q_M P_{c_M})

  where P_{c_M} marginalizes a maximal-clique table down to clique c_M, with
  ||P_c||^2 = |C| / |c| (the product of the summed-out domain sizes). Bounding
  each block by the sum of its terms' norms gives::

      L <= max_C sum_{M->C} ||Q_M||^2 / sigma_M^2 * |C| / |c_M|.

  The uniform vector is the top eigenvector of every marginalization operator,
  so this is *exact* when all queries are identity marginals (the common case)
  and a safe overestimate otherwise (a larger L only shrinks the step size).

  Returns None if any query's operator norm is unknown (e.g. a nonlinear or
  user-supplied callable query), signalling the caller to fall back to the
  power-iteration estimate.
  """
  cliques = [m.clique for m in measurements]
  maximal = maximal_subset(cliques)
  routing = clique_mapping(maximal, cliques, domain)

  blocks: dict[Clique, float] = {}
  for m in measurements:
    q_norm_sq = _query_op_norm_sq(m.query)
    if q_norm_sq is None:
      return None
    containing = routing[m.clique]
    proj_norm_sq = domain.size(containing) / domain.size(m.clique)
    stddev = max(float(m.stddev), 1e-12)
    contribution = q_norm_sq / stddev**2 * proj_norm_sq
    blocks[containing] = blocks.get(containing, 0.0) + contribution
  return max(blocks.values()) if blocks else 1.0


def from_linear_measurements(
    measurements: list[LinearMeasurement],
    domain: Domain,
    norm: str = "l2",
    normalize: bool = False,
) -> MarginalLossFn:
  """Construct a MarginalLossFn from a list of LinearMeasurements.

  Args:
      measurements: A list of LinearMeasurements.
      norm: Either "l1" or "l2".
      normalize: Flag determining if the loss function should be normalized
          by the length of linear measurements and estimated total.
      domain: The domain over which the measurements were made, necessary for calcualting the Lipschitz parameter.

  Returns:
      The MarginalLossFn L(mu) = sum_{c} || Q_c mu_c - y_c || (possibly squared or normalized).
  """
  if norm not in ["l1", "l2"]:
    raise ValueError(f"Unknown norm {norm}.")
  cliques = [m.clique for m in measurements]
  maximal_cliques = maximal_subset(cliques)
  data = tuple(measurements)

  if normalize:
    loss_fn = _normalized_l2_loss if norm == "l2" else _normalized_l1_loss
  else:
    loss_fn = _l2_loss if norm == "l2" else _l1_loss

  loss = MarginalLossFn(maximal_cliques, loss_fn, data)

  # Lipschitz requires concrete measurement values.
  has_abstract = any(
      isinstance(m.noisy_measurement, jax.ShapeDtypeStruct)
      for m in measurements
  )
  if norm == "l2" and not normalize and not has_abstract:
    lipschitz = calculate_l2_lipschitz_from_metadata(domain, measurements)
    if lipschitz is None:
      lipschitz = calculate_l2_lipschitz(domain, maximal_cliques, loss)
    return MarginalLossFn(maximal_cliques, loss_fn, data, lipschitz)

  return loss


def primal_feasibility(mu: CliqueVector) -> ArrayLike:
  """Calculates the average L1 distance between overlapping marginals in `mu` (consistency)."""
  ans = 0
  count = 0
  for r in mu.cliques:
    for s in mu.cliques:
      if r == s:
        break
      d = tuple(set(r) & set(s))
      if len(d) > 0:
        x = mu[r].project(d).datavector()
        y = mu[s].project(d).datavector()
        denom = 0.5 * x.sum() + 0.5 * y.sum()
        err = jnp.linalg.norm(x - y, 1) / denom
        ans += err
        count += 1
  try:
    return ans / count
  except Exception:  # pylint: disable=broad-exception-caught
    return 0


def mle_loss_fn(marginals: CliqueVector) -> "MarginalLossFn":
  """MLE loss: ``-marginals.dot(mu.log())``."""
  return MarginalLossFn(
      cliques=marginals.cliques,
      loss_fn=lambda mu, target: -target.dot(mu.log()),
      data=marginals,
  )
