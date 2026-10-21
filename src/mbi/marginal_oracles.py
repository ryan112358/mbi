"""Marginal oracles for computing marginals from graphical model potentials.

This module provides marginal oracles that compute marginals from the
log-space potentials of a discrete graphical model.  Three junction tree
schedules are available:

- ``message_passing_implicit``: Implicit-factor message passing using a
  configurable contraction function (default: ``einsum_fused``).
- ``message_passing_hugin``: Classic HUGIN algorithm with belief subtraction.
- ``message_passing_shafer_shenoy``: Shafer-Shenoy algorithm with neighbor collection.

Use ``default_oracle()`` to automatically select the best oracle for your
hardware and clique structure::

    oracle = marginal_oracles.default_oracle(cliques=cliques, domain=domain)
    marginals = oracle(potentials)
"""

import collections
from collections.abc import Callable, Sequence
import concurrent.futures
import functools
import itertools
import math
import string
from typing import Protocol
import warnings

import jax
import jax.numpy as jnp
import networkx as nx
import numpy as np

from . import junction_tree
from ._future import RobustFuture
from .clique_utils import Clique, clique_mapping
from .clique_vector import CliqueVector
from .constraint import Constraint
from .domain import Attribute, Domain
from .einsum import custom_einsum
from .factor import Factor

_EINSUM_LETTERS = list(string.ascii_lowercase) + list(string.ascii_uppercase)

_COMPILE_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=2)


class MarginalOracle(Protocol):
  """Callable signature for stateless marginal oracle functions.

  A marginal oracle consumes log-space potentials of a graphical model
  and returns its marginals over the same set of cliques.  Different
  oracles may have different time/space complexities and numerical
  stabilities.

  Available oracles:

  - ``message_passing_implicit``: Implicit-factor message passing (configurable).
  - ``message_passing_hugin``: Classic HUGIN algorithm with belief subtraction.
  - ``message_passing_shafer_shenoy``: Shafer-Shenoy algorithm with neighbor collection.
  - ``brute_force_marginals``: Materializes the full joint distribution.
  - ``einsum_marginals``: Direct einsum (not recommended for large models).
  """

  def __call__(
      self,
      potentials: CliqueVector,
      total: jax.Array | float | None = 1.0,
      *,
      constraints: Sequence[Constraint] = (),
      reduction_fn: Callable = jax.scipy.special.logsumexp,
  ) -> CliqueVector:
    """Compute marginals from potentials.

    Args:
        potentials: Potentials of a graphical model.
        total: Normalization constant, or ``None`` to return unnormalized
            log-space beliefs. Defaults to 1.0.
        constraints: Structural constraints. Oracles that support
            constraints fold them into the potentials or handle
            them natively; unsupported oracles raise ValueError.
        reduction_fn: Reduction callable, either ``jax.scipy.special.logsumexp``
            (default) for standard marginal inference or ``jnp.max`` for
            max-marginals in log space (scaled by ``total``).

    Returns:
        Marginals as a CliqueVector.
    """


def sum_product(
    factors: list[Factor], dom: Domain, einsum_fn: Callable = jnp.einsum
) -> Factor:
  """Compute the sum-of-products of a list of Factors using einsum.

  Args:
      factors: A list of Factors.
      dom: The target domain of the output factor.

  Returns:
      sum_{S - D} prod_i F_i,
      where
          * F_i = factors[i]
          * D = dom
          * S = union of domains of F_i
  """

  attrs = sorted(set.union(*[set(f.domain) for f in factors]).union(set(dom)))
  mapping = dict(zip(attrs, _EINSUM_LETTERS))

  def convert(d):
    return "".join(mapping[a] for a in d.attributes)

  formula = ",".join(convert(f.domain) for f in factors) + "->" + convert(dom)
  values = einsum_fn(
      formula,
      *[f.values for f in factors],
      optimize="auto",
      precision=jax.lax.Precision.HIGHEST,
  )
  return Factor(dom, values)


def einsum_semistable(
    log_factors: list[Factor],
    dom: Domain,
    einsum_fn: Callable = jnp.einsum,
) -> Factor:
  """Compute the log-space sum-product via the exp-normalize trick.

  Subtracts per-factor maxima for numerical stability, exponentiates,
  runs a standard multiply-sum einsum, then maps back to log space.
  Fast and memory-efficient (no super-factor materialization), but
  can produce NaN when factor magnitudes span a very wide range.

  Args:
      log_factors: Factors in log space.
      dom: Target domain for the output.
      einsum_fn: Einsum implementation to use (default: ``jnp.einsum``).

  Returns:
      A Factor over *dom* containing the result in log space.
  """
  maxes = [
      f.max(f.domain.marginalize(dom.attributes).attributes)
      for f in log_factors
  ]
  stable_factors = [(f - m).exp() for f, m in zip(log_factors, maxes)]
  return sum_product(stable_factors, dom, einsum_fn).log() + sum(maxes)


def einsum_materialized(
    log_factors: list[Factor],
    dom: Domain,
    reduction_fn: Callable = jax.scipy.special.logsumexp,
) -> Factor:
  """Compute the log-space sum-product by materializing the combined factor.

  Combines all log-factors into a single super-factor (via addition in
  log-space), then reduces over the non-target dimensions with ``reduction_fn``.
  Numerically stable because no exp/log round-trip is needed, but may
  require O(product of all domain sizes) memory for the intermediate
  super-factor.

  Args:
      log_factors: Factors in log space.
      dom: Target domain for the output.
      reduction_fn: Reduction callable (default: ``jax.scipy.special.logsumexp``).

  Returns:
      A Factor over *dom* in log space.
  """
  combined = functools.reduce(lambda a, b: a + b, log_factors)
  return combined.project(dom.attributes, reduction_fn=reduction_fn)


def einsum_fused(
    log_factors: list[Factor],
    dom: Domain,
    reduction_fn: Callable = jax.scipy.special.logsumexp,
) -> Factor:
  """Compute the log-space sum-product using ``custom_dot_general``.

  Delegates to ``custom_einsum`` with add/logsumexp operations,
  allowing the XLA compiler to fuse the combine and reduce steps.
  This avoids both the exp/log round-trip of ``einsum_semistable``
  and the super-factor materialization of ``einsum_materialized``.

  On GPU, XLA often fuses the operations resulting in both good
  stability and good performance.

  Args:
      log_factors: Factors in log space.
      dom: Target domain for the output.
      reduction_fn: Reduction callable (default: ``jax.scipy.special.logsumexp``).

  Returns:
      A Factor over *dom* in log space.
  """

  def _custom_einsum(formula, *arrays, **kwargs):
    kwargs.pop("optimize", None)
    kwargs.pop("precision", None)
    return custom_einsum(
        formula,
        *arrays,
        combine_fn=jnp.add,
        reduce_fn=reduction_fn,
    )

  return sum_product(log_factors, dom, einsum_fn=_custom_einsum)


def _shard_mesh(merged: Domain, dom: Domain) -> jax.sharding.Mesh:
  """Construct a device mesh that shards attributes of ``merged`` across JAX devices.

  Greedily assigns prime factors of ``jax.device_count()`` to attributes in
  ``merged``, preferring eliminated attributes (those not in ``dom``) so
  cross-device reductions use in-place ``psum``/``pmax`` collectives over
  ``all_gather``, then minimizing per-device padded chunk growth
  (``ceil(size / shards)``), and breaking ties by larger current chunk size.

  In the 4-device example below, ``'c'`` is in ``dom``, so eliminated
  attributes ``'a'`` and ``'b'`` are preferred. The first factor of 2 shards
  ``'a'`` (chunk ``6 -> 3``), and the second shards ``'b'`` (chunk ``4 -> 2``
  with no padding) rather than padding ``'a'`` from 6 to 8::

      merged = Domain(['a', 'b', 'c'], [6, 4, 10])
      dom = Domain(['c'], [10])
      mesh = _shard_mesh(merged, dom)  # dict(mesh.shape) == {'a': 2, 'b': 2}

  Args:
      merged: Joint domain across all input factors and the output domain.
      dom: Target output domain of the contraction.

  Returns:
      A ``jax.sharding.Mesh`` whose axis names are a subset of ``merged``'s
      attributes and whose axis sizes multiply to ``jax.device_count()``.
  """
  axes = {a: 1 for a in merged.attributes}
  n, d = jax.device_count(), 2
  chunk = lambda a, k: math.ceil(merged[a] / k)
  while n > 1:
    while n % d == 0:
      ratio = lambda a: chunk(a, axes[a] * d) / chunk(a, axes[a])
      key = lambda a: (ratio(a) == 1, a in dom, ratio(a), -chunk(a, axes[a]))
      axes[min(merged.attributes, key=key)] *= d
      n //= d
    d += 1
  axes = {a: k for a, k in axes.items() if k > 1}
  devices = np.array(jax.devices()).reshape(tuple(axes.values()))
  return jax.sharding.Mesh(devices, tuple(axes))


def einsum_sharded(
    log_factors: list[Factor],
    dom: Domain,
    reduction_fn: Callable = jax.scipy.special.logsumexp,
    *,
    contraction: Callable = einsum_materialized,
) -> Factor:
  """Compute the log-space sum-product sharded across all JAX devices.

  Splits attributes across devices via ``Factor.pad`` and ``jax.shard_map``,
  runs ``contraction`` on each device's local slice, and combines the partial
  results with a collective reduction or ``all_gather``.

  Args:
      log_factors: Factors in log space.
      dom: Target domain for the output.
      reduction_fn: Reduction callable (default:
        ``jax.scipy.special.logsumexp``).
      contraction: Per-device contraction function (default:
        ``einsum_materialized``).

  Returns:
      A Factor over *dom* in log space.
  """
  merged = functools.reduce(lambda a, b: a.merge(b.domain), log_factors, dom)
  if jax.device_count() <= 1 or len(merged) == 0:
    return contraction(log_factors, dom, reduction_fn=reduction_fn)

  mesh = _shard_mesh(merged, dom)
  chunks = {a: math.ceil(merged[a] / d) for a, d in mesh.shape.items()}
  local_dom = Domain.fromdict({**merged.config, **chunks})
  elim = tuple(a for a in mesh.axis_names if a not in dom)
  kept = [a for a in mesh.axis_names if a in dom]

  def _slice(f: Factor) -> Factor:
    if not any(a in mesh.shape for a in f.domain.attributes):
      return f
    val = f.pad(mesh, "-inf").values
    for i, a in enumerate(f.domain.attributes):
      if a in mesh.shape:
        val = jax.lax.dynamic_slice_in_dim(
            val, jax.lax.axis_index(a) * chunks[a], chunks[a], axis=i
        )
    return Factor(local_dom.project(f.domain.attributes), val)

  @functools.partial(
      jax.shard_map,
      mesh=mesh,
      in_specs=jax.sharding.PartitionSpec(),
      out_specs=jax.sharding.PartitionSpec(),
      check_vma=False,
  )
  def _kernel(factors: list[Factor]) -> Factor:
    local = [_slice(f) for f in factors]
    out_dom = local_dom.project(dom.attributes)
    val = contraction(local, out_dom, reduction_fn=reduction_fn).values
    if elim:
      if reduction_fn is jnp.max:
        val = jax.lax.pmax(val, elim)
      else:
        m = jax.lax.pmax(jax.lax.stop_gradient(val), elim)
        shift = jnp.where(jnp.isneginf(m), 0.0, m)
        val = m + jnp.log(jax.lax.psum(jnp.exp(val - shift), elim))
    for a in kept:
      axis = dom.attributes.index(a)
      val = jax.lax.all_gather(val, a, axis=axis, tiled=True)
      val = jax.lax.slice_in_dim(val, 0, dom[a], axis=axis)
    return Factor(dom, val)

  return _kernel(log_factors)


def _einsum_sharded_multi(
    log_factors: list[Factor],
    doms: Sequence[Domain],
    reduction_fn: Callable = jax.scipy.special.logsumexp,
    *,
    contraction: Callable = einsum_materialized,
) -> list[Factor]:
  """Project combined ``log_factors`` onto multiple target domains in one pass."""
  if not doms:
    return []
  if len(doms) == 1 or contraction is not einsum_materialized:
    return [
        einsum_sharded(
            log_factors, d, reduction_fn=reduction_fn, contraction=contraction
        )
        for d in doms
    ]
  merged = functools.reduce(
      lambda a, b: a.merge(b.domain), log_factors, Domain([], [])
  )
  for d in doms:
    merged = merged.merge(d)
  if jax.device_count() <= 1 or len(merged) == 0:
    combined = functools.reduce(lambda a, b: a + b, log_factors)
    return [
        combined.project(d.attributes, reduction_fn=reduction_fn) for d in doms
    ]

  mesh = _shard_mesh(merged, Domain([], []))
  chunks = {a: math.ceil(merged[a] / d) for a, d in mesh.shape.items()}
  local_dom = Domain.fromdict({**merged.config, **chunks})

  def _slice(f: Factor) -> Factor:
    if not any(a in mesh.shape for a in f.domain.attributes):
      return f
    val = f.pad(mesh, "-inf").values
    for i, a in enumerate(f.domain.attributes):
      if a in mesh.shape:
        val = jax.lax.dynamic_slice_in_dim(
            val, jax.lax.axis_index(a) * chunks[a], chunks[a], axis=i
        )
    return Factor(local_dom.project(f.domain.attributes), val)

  @functools.partial(
      jax.shard_map,
      mesh=mesh,
      in_specs=jax.sharding.PartitionSpec(),
      out_specs=jax.sharding.PartitionSpec(),
      check_vma=False,
  )
  def _kernel(factors: list[Factor]) -> list[Factor]:
    local = [_slice(f) for f in factors]
    combined = functools.reduce(lambda a, b: a + b, local)
    results = []
    for dom in doms:
      out_dom = local_dom.project(dom.attributes)
      val = combined.project(
          out_dom.attributes, reduction_fn=reduction_fn
      ).values
      elim = tuple(a for a in mesh.axis_names if a not in dom)
      kept = [a for a in mesh.axis_names if a in dom]
      if elim:
        if reduction_fn is jnp.max:
          val = jax.lax.pmax(val, elim)
        else:
          m = jax.lax.pmax(jax.lax.stop_gradient(val), elim)
          shift = jnp.where(jnp.isneginf(m), 0.0, m)
          val = m + jnp.log(jax.lax.psum(jnp.exp(val - shift), elim))
      for a in kept:
        axis = dom.attributes.index(a)
        val = jax.lax.all_gather(val, a, axis=axis, tiled=True)
        val = jax.lax.slice_in_dim(val, 0, dom[a], axis=axis)
      results.append(Factor(dom, val))
    return results

  return _kernel(log_factors)


def _fold_constraints(
    potentials: CliqueVector,
    constraints: Sequence[Constraint],
) -> tuple[CliqueVector, Sequence[Clique]]:
  """Fold constraints into potentials as -inf/0 log-space factors.

  Returns:
      A tuple of (expanded_potentials, input_cliques) where input_cliques
      are the original cliques before constraint factors were added.
  """
  input_cliques = potentials.cliques
  if not constraints:
    return potentials, input_cliques
  domain = potentials.domain
  cliques = list(input_cliques)
  arrays = {cl: potentials[cl] for cl in cliques}
  for c in constraints:
    cl = domain.canonical(c.clique)
    factor = c.potential
    if cl in arrays:
      arrays[cl] = arrays[cl] + factor
    else:
      cliques.append(cl)
      arrays[cl] = factor
  return CliqueVector(domain, cliques, arrays), input_cliques


def _finalize_beliefs(
    beliefs: CliqueVector | Factor,
    cliques: Sequence[Clique],
    total: jax.Array | float | None,
    fn: Callable = jax.scipy.special.logsumexp,
) -> CliqueVector:
  # Project maximal-clique beliefs onto the target cliques and convert from
  # log space to the requested marginal or max-marginal scale.
  tables = {cl: beliefs.project(cl, reduction_fn=fn) for cl in cliques}
  beliefs = CliqueVector(beliefs.domain, cliques, tables)
  if total is None:
    return beliefs
  if fn is jnp.max:
    return beliefs * total
  return beliefs.normalize(total, log=True).exp()


@jax.jit(static_argnames=["jtree", "return_messages", "reduction_fn"])
def message_passing_hugin(
    potentials: CliqueVector,
    total: jax.Array | float | None = 1.0,
    jtree: nx.Graph | None = None,
    *,
    constraints: Sequence[Constraint] = (),
    return_messages: bool = False,
    reduction_fn: Callable = jax.scipy.special.logsumexp,
) -> CliqueVector | tuple[CliqueVector, dict]:
  """HUGIN message passing with belief subtraction.

  Expands potentials to super-cliques and uses belief subtraction for
  message computation.  Unstable when potentials contain ``-inf``.

  Args:
      potentials: Potentials of a graphical model.
      total: Normalization constant.
      jtree: Pre-computed junction tree (optional).
      constraints: Structural constraints folded into potentials as
          ``-inf``/``0`` factors before inference.
      return_messages: If True, return ``(marginals, messages)`` where
          *messages* is a dict mapping ``(clique_i, clique_j)`` to the
          log-space message Factor sent from *i* to *j*.
      reduction_fn: Either ``jax.scipy.special.logsumexp`` (default) or ``jnp.max``.
  """
  if constraints:
    warnings.warn(
        "HUGIN uses belief subtraction which is unstable with -inf"
        " potentials introduced by constraints. Consider using"
        " message_passing_shafer_shenoy instead.",
        stacklevel=2,
    )
  potentials, input_cliques = _fold_constraints(potentials, constraints)
  if len(potentials.cliques) == 0:
    result = CliqueVector(potentials.domain, [], {})
    return (result, {}) if return_messages else result

  domain, cliques = potentials.domain, potentials.cliques

  if jtree is None:
    jtree = junction_tree.make_junction_tree(domain, cliques, connected=True)[0]
  message_order = junction_tree.message_passing_order(jtree)
  maximal_cliques = junction_tree.maximal_cliques(jtree)

  clique_mapping(maximal_cliques, cliques, domain=domain)
  beliefs = potentials.expand(maximal_cliques)

  messages = {}
  for i, j in message_order:
    sep = beliefs[i].domain.invert(tuple(set(i) & set(j)))
    if (j, i) in messages:
      tau = beliefs[i] - messages[(j, i)]
    else:
      tau = beliefs[i]
    messages[(i, j)] = tau.aggregate(reduction_fn, sep)
    beliefs[j] = beliefs[j] + messages[(i, j)]

  marginals = _finalize_beliefs(beliefs, input_cliques, total, reduction_fn)
  return (marginals, messages) if return_messages else marginals


@jax.jit(static_argnames=["jtree", "return_messages", "reduction_fn"])
def message_passing_shafer_shenoy(
    potentials: CliqueVector,
    total: jax.Array | float | None = 1.0,
    jtree: nx.Graph | None = None,
    *,
    constraints: Sequence[Constraint] = (),
    return_messages: bool = False,
    reduction_fn: Callable = jax.scipy.special.logsumexp,
) -> CliqueVector | tuple[CliqueVector, dict]:
  """Shafer-Shenoy message passing with neighbor collection.

  Expands potentials to super-cliques and collects messages from all
  neighbors except the target.  More stable than HUGIN for ``-inf``
  potentials.

  Args:
      potentials: Potentials of a graphical model.
      total: Normalization constant.
      jtree: Pre-computed junction tree (optional).
      constraints: Structural constraints folded into potentials as
          ``-inf``/``0`` factors before inference.
      return_messages: If True, return ``(marginals, messages)`` where
          *messages* is a dict mapping ``(clique_i, clique_j)`` to the
          log-space message Factor sent from *i* to *j*.
      reduction_fn: Either ``jax.scipy.special.logsumexp`` (default) or ``jnp.max``.
  """
  potentials, input_cliques = _fold_constraints(potentials, constraints)
  if len(potentials.cliques) == 0:
    result = CliqueVector(potentials.domain, [], {})
    return (result, {}) if return_messages else result

  domain, cliques = potentials.domain, potentials.cliques

  if jtree is None:
    jtree = junction_tree.make_junction_tree(domain, cliques, connected=True)[0]
  message_order = junction_tree.message_passing_order(jtree)
  maximal_cliques = junction_tree.maximal_cliques(jtree)

  initial_beliefs = potentials.expand(maximal_cliques)

  messages = {}
  neighbors = {cl: list(jtree.neighbors(cl)) for cl in maximal_cliques}

  for i, j in message_order:
    tau = initial_beliefs[i]
    for k in neighbors[i]:
      if k == j:
        continue
      tau = tau + messages[(k, i)]

    sep = tau.domain.invert(tuple(set(i) & set(j)))
    messages[(i, j)] = tau.aggregate(reduction_fn, sep)

  beliefs = {}
  for cl in maximal_cliques:
    b = initial_beliefs[cl]
    for k in neighbors[cl]:
      b = b + messages[(k, cl)]
    beliefs[cl] = b

  beliefs = CliqueVector(potentials.domain, maximal_cliques, beliefs)
  marginals = _finalize_beliefs(beliefs, input_cliques, total, reduction_fn)
  return (marginals, messages) if return_messages else marginals


@jax.jit(
    static_argnames=["jtree", "contraction", "return_messages", "reduction_fn"]
)
def message_passing_implicit(
    potentials: CliqueVector,
    total: jax.Array | float | None = 1.0,
    jtree: nx.Graph | None = None,
    *,
    constraints: Sequence[Constraint] = (),
    contraction: Callable = einsum_sharded,
    return_messages: bool = False,
    reduction_fn: Callable = jax.scipy.special.logsumexp,
) -> CliqueVector | tuple[CliqueVector, dict]:
  """Implicit-factor message passing using a contraction function.

  Keeps potentials in their original factored form and computes messages
  via the given contraction function.  Most memory-efficient — never
  materializes super-clique tables.

  The default contraction (``einsum_sharded``) shards large contractions
  across available devices and delegates to ``einsum_materialized`` on a
  single device.  For better ``-inf`` tolerance, use ``einsum_fused``.

  Args:
      potentials: Potentials of a graphical model.
      total: Normalization constant.
      jtree: Pre-computed junction tree (optional).
      constraints: Structural constraints folded into potentials as
          ``-inf``/``0`` factors before inference.
      contraction: Contraction function for log-space sum-product.
      return_messages: If True, return ``(marginals, messages)`` where
          *messages* is a dict mapping ``(clique_i, clique_j)`` to the
          log-space message Factor sent from *i* to *j*.
      reduction_fn: Either ``jax.scipy.special.logsumexp`` (default) or ``jnp.max``.
  """
  if constraints and contraction is einsum_semistable:
    raise ValueError(
        "einsum_semistable is not compatible with constraints"
        " (-inf potentials). Use einsum_fused or the default"
        " einsum_sharded instead."
    )
  potentials, input_cliques = _fold_constraints(potentials, constraints)
  if len(potentials.cliques) == 0:
    result = CliqueVector(potentials.domain, [], {})
    return (result, {}) if return_messages else result

  domain, cliques = potentials.active_domain, potentials.cliques

  if jtree is None:
    jtree = junction_tree.make_junction_tree(domain, cliques, connected=True)[0]
  message_order = junction_tree.message_passing_order(jtree)
  maximal_cliques = junction_tree.maximal_cliques(jtree)

  mapping = clique_mapping(maximal_cliques, cliques, domain=domain)
  inverse_mapping = collections.defaultdict(list)
  incoming_messages = collections.defaultdict(list)
  potential_mapping = collections.defaultdict(list)

  for cl in cliques:
    potential_mapping[mapping[cl]].append(potentials[cl])
    inverse_mapping[mapping[cl]].append(cl)

  for i, msg in enumerate(message_order):
    for j in range(i):
      msg2 = message_order[j]
      if msg[0] == msg2[1] and msg[1] != msg2[0]:
        incoming_messages[msg].append(msg2)

  is_default_sharded = contraction is einsum_sharded
  if reduction_fn is not jax.scipy.special.logsumexp:
    contraction = functools.partial(contraction, reduction_fn=reduction_fn)

  def _barrier(x: jax.Array) -> jax.Array:
    v = x.ravel()[0]
    return jax.lax.stop_gradient(jnp.where(jnp.isfinite(v), v * 0.0, 0.0))

  messages = {}
  barrier = jnp.asarray(0.0)
  for i, j in message_order:
    shared = domain.project([a for a in i if a in j])
    input_potentials = potential_mapping[i]
    input_messages = [messages[key] for key in incoming_messages[(i, j)]]
    inputs = input_potentials + input_messages

    for attr in shared.attributes:
      if not any(attr in input.domain.attributes for input in inputs):
        inputs.append(Factor.zeros(domain.project([attr])))

    is_large = domain.size(i) >= 100_000
    if inputs and is_large:
      inputs = [f + barrier for f in inputs]
    msg = contraction(inputs, shared)
    if is_large:
      barrier = _barrier(msg.values)
    messages[(i, j)] = msg

  if return_messages:
    return CliqueVector(potentials.domain, [], {}), messages

  beliefs = {}
  for cl in maximal_cliques:
    input_potentials = potential_mapping[cl]
    input_messages = [val for key, val in messages.items() if key[1] == cl]
    inputs = input_potentials + input_messages
    target_cliques = [
        cl2 for cl2 in inverse_mapping[cl] if cl2 in input_cliques
    ]
    if not target_cliques:
      continue
    is_large = domain.size(cl) >= 100_000
    if inputs and is_large:
      inputs = [f + barrier for f in inputs]
    if is_default_sharded:
      target_doms = [domain.project(cl2) for cl2 in target_cliques]
      outs = _einsum_sharded_multi(
          inputs, target_doms, reduction_fn=reduction_fn
      )
      for cl2, out in zip(target_cliques, outs):
        beliefs[cl2] = out
      if is_large:
        barrier = _barrier(outs[-1].values)
    elif contraction is einsum_materialized:
      combined = functools.reduce(lambda a, b: a + b, inputs)
      for cl2 in target_cliques:
        beliefs[cl2] = combined.project(cl2, reduction_fn=reduction_fn)
      if is_large:
        barrier = _barrier(beliefs[target_cliques[-1]].values)
    else:
      for cl2 in target_cliques:
        out = contraction(inputs, domain.project(cl2))
        beliefs[cl2] = out
        if is_large:
          barrier = _barrier(out.values)

  beliefs = CliqueVector(potentials.domain, input_cliques, beliefs)
  if total is None and reduction_fn is jax.scipy.special.logsumexp:
    unused = potentials.domain.invert(domain.attributes)
    beliefs = beliefs + sum(math.log(potentials.domain[a]) for a in unused)
  marginals = _finalize_beliefs(beliefs, input_cliques, total, reduction_fn)
  return (marginals, messages) if return_messages else marginals


# Backward-compatible aliases.
message_passing_fast = functools.partial(
    message_passing_implicit, contraction=einsum_semistable
)
message_passing_stable = message_passing_hugin


def default_oracle(
    cliques: Sequence[Clique] | None = None,
    domain: Domain | None = None,
    backend: str | None = None,
    has_constraints: bool = True,
) -> MarginalOracle:
  """Select the best oracle for the given setting.

  Chooses based on hardware backend, clique structure, and whether the
  potentials may contain ``-inf`` entries:

  - **CPU**: ``message_passing_shafer_shenoy`` (has_constraints=True) or
    ``message_passing_hugin`` (has_constraints=False). Implicit is not used on
    CPU because the XLA compiler is less effective there.
  - **GPU/TPU, memory-limited**: ``message_passing_implicit`` when the total
    beliefs memory would exceed ~40% of estimated device HBM.  Implicit
    keeps factors in their original low-dimensional form and only
    materializes one super-clique at a time, using orders of magnitude
    less memory than Hugin/SS which materialize all super-clique arrays
    simultaneously.
  - **GPU/TPU, huge but few super-cliques**: ``message_passing_implicit``
    when individual super-cliques exceed 50M entries but there are fewer
    than 30 nodes.  With few messages, implicit moves less data per
    message by contracting small input factors rather than operating on
    huge belief arrays.
  - **GPU/TPU, has_constraints=True** (default):
    ``message_passing_shafer_shenoy`` (numerically robust).
  - **GPU/TPU, has_constraints=False**: ``message_passing_hugin`` (faster when
    ``-inf`` is absent).

  The heuristic is based on systematic A100 benchmarks across varied
  graph structures (cardinality 4-256, bandwidth 1-3, 32-256 attributes).
  When super-cliques are large but the junction tree has many nodes, Hugin
  and Shafer-Shenoy outperform implicit by 30-40% because their uniform
  add/logsumexp kernel pattern fuses better in XLA over many sequential
  messages.  Implicit only wins when individual super-cliques are very
  large (>50M entries) and the tree is small (<30 nodes), or when
  Hugin/SS would exceed device memory.

  Args:
      cliques: Cliques of the graphical model. Used to estimate max clique
          size when ``domain`` is also provided.
      domain: Domain of the graphical model. Used with ``cliques`` to
          estimate max clique size.
      backend: JAX backend string ('cpu', 'gpu', 'tpu'). If None, uses
          ``jax.default_backend()``.
      has_constraints: Whether potentials may contain ``-inf`` entries (e.g. from
          deterministic constraints or structural zeros). When True
          (default), Shafer-Shenoy is preferred for numerical robustness.
          When False, Hugin may be used for better performance.

  Returns:
      A callable satisfying the ``MarginalOracle`` protocol.

  Example::

      oracle = default_oracle(cliques=model.cliques, domain=model.domain)
      marginals = oracle(potentials)
  """
  if backend is None:
    backend = jax.default_backend()

  # Consider memory pressure and per-message data movement across all backends.
  if cliques is not None and domain is not None:
    jtree = junction_tree.make_junction_tree(domain, cliques)[0]
    max_cliques = junction_tree.maximal_cliques(jtree)
    sc_sizes = [domain.project(cl).size() for cl in max_cliques]
    max_size = max(sc_sizes, default=0)
    num_nodes = len(max_cliques)

    # Estimate total beliefs memory for Hugin/SS (4 bytes per float32).
    # For large junction trees (>= 100M cells / 400 MB), materializing all
    # super-clique tables simultaneously in Hugin/SS causes excessive memory
    # overhead and cannot shard super-clique contractions across devices.
    total_beliefs_bytes = sum(sc_sizes) * 4
    if total_beliefs_bytes >= 400e6:
      return message_passing_implicit

    # Few large super-cliques: implicit wins on data movement.
    # With <30 SC nodes there are few messages, so kernel launch overhead
    # is negligible, and implicit contracts small input factors instead
    # of operating on huge belief arrays.
    if max_size >= 50_000_000 and num_nodes < 30:
      return message_passing_implicit

  # Moderate or unknown cliques: SS for constraints, Hugin otherwise.
  if has_constraints:
    return message_passing_shafer_shenoy
  return message_passing_hugin


def brute_force_marginals(
    potentials: CliqueVector,
    total: jax.Array | float | None = 1,
    *,
    constraints: Sequence[Constraint] = (),
    reduction_fn: Callable = jax.scipy.special.logsumexp,
) -> CliqueVector:
  """Compute marginals from (log-space) potentials by materializing the full joint distribution.

  Args:
      potentials: Potentials of a graphical model.
      total: Normalization constant.
      constraints: Structural constraints folded into potentials as
          ``-inf``/``0`` factors before inference.
      reduction_fn: Either ``jax.scipy.special.logsumexp`` (default) or ``jnp.max``.
  """
  potentials, input_cliques = _fold_constraints(potentials, constraints)
  if len(potentials.cliques) == 0:
    return CliqueVector(potentials.domain, [], {})

  joint = sum(potentials.tables.values(), Factor.zeros(potentials.domain))
  return _finalize_beliefs(joint, input_cliques, total, reduction_fn)


def einsum_marginals(
    potentials: CliqueVector,
    total: jax.Array | float = 1,
    einsum_fn: Callable = jnp.einsum,
    *,
    constraints: Sequence[Constraint] = (),
) -> CliqueVector:
  """Compute marginals from (log-space) potentials by using einsum.

  This is a "brute-force" approach and is not recommended in practice.

  Args:
      potentials: Potentials of a graphical model.
      total: Normalization constant.
      einsum_fn: Einsum function to use.
      constraints: Structural constraints. Raises ValueError if non-empty.
  """
  if constraints:
    raise ValueError(
        "einsum_marginals does not support constraints. Use"
        " message_passing_shafer_shenoy or"
        " extensions.message_passing.shafer_shenoy."
    )
  # not strictly necessary, but consistent
  if len(potentials.cliques) == 0:
    return CliqueVector(potentials.domain, [], {})

  inputs = list(potentials.tables.values())
  return CliqueVector(
      potentials.domain,
      potentials.cliques,
      {
          cl: (
              einsum_semistable(
                  inputs, potentials[cl].domain, einsum_fn=einsum_fn
              )
              .normalize(total, log=True)
              .exp()
          )
          for cl in potentials.cliques
      },
  )


@jax.jit(static_argnames=("clique",))
def variable_elimination(
    potentials: CliqueVector,
    clique: tuple[str, ...],
    total: float = 1.0,
    evidence: dict[Attribute, int | jax.Array] | None = None,
    *,
    constraints: Sequence[Constraint] = (),
) -> Factor:
  """Compute an out-of-model/unsupported marginal from the potentials.

  Args:
      potentials: The (log-space) potentials of a Graphical Model.
      clique: The subset of attributes whose marginal you want.
      total: The normalization factor.
      evidence: Mapping from attribute names to observed scalar int values.
      constraints: Structural constraints folded into potentials as
          ``-inf``/``0`` factors before inference.

  Returns:
      The marginal defined over the domain of the input clique, where
      each entry is non-negative and sums to the input total.
  """
  potentials, _ = _fold_constraints(potentials, constraints)
  clique = tuple(clique)
  evidence_dict = evidence if evidence is not None else {}
  if set(clique) & set(evidence_dict.keys()):
    raise ValueError("Evidence attributes cannot be in the query clique.")

  k = len(potentials.cliques)
  psi = dict(zip(range(k), potentials.tables.values()))

  if evidence_dict:
    for i in list(psi.keys()):
      psi[i] = psi[i].slice(evidence_dict)

  domain = potentials.active_domain.marginalize(list(evidence_dict.keys()))
  cliques = [psi[i].domain.attributes for i in psi] + [clique]
  elim = domain.invert(clique)
  elim_order, _ = junction_tree.greedy_order(domain, cliques, elim=elim)

  for z in elim_order:
    psi2 = [psi.pop(i) for i in list(psi.keys()) if z in psi[i].domain]
    psi[k] = sum(psi2, Factor.zeros(Domain([], []))).logsumexp([z])
    k += 1

  newdom = potentials.domain.project(clique)
  zero = Factor(Domain([], []), jnp.asarray(0.0))
  unnormalized = sum(psi.values(), start=zero).expand(newdom)
  return unnormalized.normalize(total, log=True).exp().project(clique)


def precompile_bulk_variable_elimination(
    domain: Domain,
    potential_cliques: Sequence[Clique],
    marginal_queries: Sequence[Clique],
    *,
    constraints: Sequence[Constraint] = (),
) -> concurrent.futures.Future:
  """Warm the JIT cache for ``bulk_variable_elimination`` asynchronously.

  Fire and forget. Only requires domain and clique structure.

  Args:
    domain: The Domain of the graphical model.
    potential_cliques: The cliques that potentials are defined over.
    marginal_queries: A list of cliques to obtain marginals for.
    constraints: Structural constraints.
  """
  abstract_potentials = CliqueVector.abstract(domain, potential_cliques)
  abstract_constraints = jax.eval_shape(lambda x: x, tuple(constraints))

  jitted = jax.jit(variable_elimination, static_argnames=("clique",))

  def _compile_all():
    for query in marginal_queries:
      jitted.lower(
          abstract_potentials,
          query,
          total=1.0,
          constraints=abstract_constraints,
      ).compile()

  return RobustFuture(_COMPILE_POOL.submit(_compile_all))


def bulk_variable_elimination(
    potentials: CliqueVector,
    marginal_queries: Sequence[Clique],
    total: float = 1.0,
    *,
    constraints: Sequence[Constraint] = (),
) -> CliqueVector:
  """Compute the marginals of the graphical model with the given potentials.

  Unlike other marginal oracles, which only compute marginals for cliques
  in the potentials vector, this function can compute arbitrary marginals
  from an arbitrary model. Both runtime and compilation time can be expensive
  when there are a large number of marginal queries. This function compiles
  and runs variable_elimination for one query at a time, using parallelism
  and asyncronous computation do do the compilation in the background, while
  running variable_eliminatoin sequentially one query at a time.

  Args:
    potentials: The (log-space) potentials of a Graphical Model.
    marginal_queries: A list of cliques to obtain marginals for.
    total: The normalization factor.
    constraints: Structural constraints folded into potentials as
        ``-inf``/``0`` factors before inference.

  Returns:
    A CliqueVector with the marginals computed over the specified cliques.
  """
  constraints = tuple(constraints)
  jitted = jax.jit(variable_elimination, static_argnames=("clique",))

  def _evaluate(query):
    return query, jitted(
        potentials, query, total=total, constraints=constraints
    )

  futures = [_COMPILE_POOL.submit(_evaluate, cl) for cl in marginal_queries]
  results: dict[tuple[int | str, ...], Factor] = {}
  for future in concurrent.futures.as_completed(futures):
    query, result = future.result()
    results[query] = result

  out_domain = _fold_constraints(potentials, constraints)[0].domain
  return CliqueVector(out_domain, marginal_queries, results)


def calculate_many_marginals(
    potentials: CliqueVector,
    marginal_queries: Sequence[Clique],
    total: float = 1.0,
    belief_propagation_oracle: MarginalOracle = message_passing_stable,
    *,
    constraints: Sequence[Constraint] = (),
) -> CliqueVector:
  """Calculate marginals for all projections using belief propagation.

  Implements Algorithm from section 10.3 in Koller and Friedman.
  This method may be faster than calling variable_elimination many times.
  Note: this implementation is experimental, and further work may be needed
  to optimize it. Contributions are welcome.

  Args:
      potentials: Potentials of a graphical model.
      marginal_queries: a list of cliques whose marginals are desired.
      constraints: Structural constraints folded into potentials as
          ``-inf``/``0`` factors before inference.

  Returns:
      A CliqueVector, where each defined over the list of input marginal_queries.
  """
  potentials, _ = _fold_constraints(potentials, constraints)

  domain = potentials.domain
  jtree = junction_tree.make_junction_tree(
      potentials.domain, potentials.cliques
  )[0]
  max_cliques = junction_tree.maximal_cliques(jtree)
  neighbors = {i: tuple(jtree.neighbors(i)) for i in max_cliques}

  # TODO: let's see if we can get rid of this similar to message_passing_fast
  potentials = potentials.expand(max_cliques)

  # TODO: allow these to take in an optional junction tree
  marginals = belief_propagation_oracle(potentials, total)

  # first calculate P(Cj | Ci) for all neighbors Ci, Cj
  conditional = {}
  for Ci in max_cliques:
    for Cj in neighbors[Ci]:
      Cj: Clique  # networkx lacks a precise type annotation.
      Sij = tuple(set(Cj) & set(Ci))
      Z = marginals.project(Cj)
      Z_sep = Z.project(Sij)
      denom = Z_sep.expand(Z.domain)
      safe_div = jnp.where(denom.values != 0, Z.values / denom.values, 0.0)
      conditional[(Cj, Ci)] = Factor(Z.domain, safe_div)

  # now iterate through pairs of cliques in order of distance
  # not sure why this API changed and why we need to do this hack.
  nx.set_edge_attributes(jtree, values=1.0, name="weight")  # type: ignore
  pred, dist = nx.floyd_warshall_predecessor_and_distance(
      jtree
  )  # , weight=None)

  def order_fn(x):
    return dist[x[0]][x[1]]

  results = {}
  for Ci, Cj in sorted(itertools.combinations(max_cliques, 2), key=order_fn):
    if dist[Ci][Cj] == math.inf:
      continue
    Cl = pred[Ci][Cj]
    Y = conditional[(Cj, Cl)]
    if Cl == Ci:
      X = marginals[Ci]
      results[(Ci, Cj)] = results[(Cj, Ci)] = X * Y
    else:
      X = results[(Ci, Cl)]
      S = set(Cl) - set(Ci) - set(Cj)
      results[(Ci, Cj)] = results[(Cj, Ci)] = (X * Y).sum(tuple(S))

  results = {
      domain.canonical(key[0] + key[1]): val for key, val in results.items()
  }

  answers = {}
  for cl in marginal_queries:
    for attr in results:
      if set(cl) <= set(attr):
        answers[cl] = results[attr].project(cl)
        break
    if cl not in answers:
      # just use variable elimination
      answers[cl] = variable_elimination(potentials, cl, total)

  return CliqueVector(domain, marginal_queries, answers)


def kron_query(
    potentials: CliqueVector,
    query_factors: dict[str, jax.Array],
    total: float = 1,
    suffix: str = "_answer",
    *,
    constraints: Sequence[Constraint] = (),
) -> Factor:
  """Compute a Kronecker-product query.

  Args:
      potentials: Potentials of a graphical model.
      query_factors: Mapping from attribute names to query matrices.
      total: Normalization constant.
      suffix: Suffix for the answer attributes.
      constraints: Structural constraints. Raises ValueError if non-empty.
  """
  if constraints:
    raise ValueError(
        "kron_query does not support constraints. Fold them into"
        " potentials before calling kron_query."
    )
  new_factors: dict[Clique, Factor] = {}
  extra_domain: dict[Attribute, int] = {}
  extra_cliques: list[Clique] = []
  target_clique: list[Attribute] = []

  for key in query_factors:
    key2 = key + suffix
    values = query_factors[key]
    dom = Domain([key2, key], values.shape)
    new_factors[(key2, key)] = Factor(dom, values).log()
    extra_domain[key2] = values.shape[0]
    extra_cliques.append((key2, key))
    target_clique.append(key2)

  domain = potentials.domain.merge(Domain.fromdict(extra_domain))
  cliques = tuple(potentials.cliques) + tuple(extra_cliques)
  inputs = CliqueVector(domain, cliques, new_factors)
  return variable_elimination(inputs, tuple(target_clique), total)
