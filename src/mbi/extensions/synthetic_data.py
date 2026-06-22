"""Fused JAX synthetic data generation with async precompilation.

This module provides ``SyntheticDataGenerator``, a class that generates
synthetic data from a ``MarkovRandomField`` using JAX-accelerated column
generation.  All per-column operations (marginal computation, Gumbel
rounding / categorical sampling) stay on the GPU, eliminating the
GPU↔CPU synchronization barriers of the baseline NumPy path.

The class mirrors the ``Estimator.precompile()`` pattern: callers can
fire off ``precompile()`` as soon as the clique structure is known
(before estimation finishes) and overlap JIT compilation with mirror
descent iterations.

Typical usage::

    generator = SyntheticDataGenerator(method='round')

    # precompile can start before estimation — only needs domain + cliques.
    compile_future = generator.precompile(domain, cliques, rows=10_000_000)

    # ... run mirror descent ...
    model = estimator.estimate(domain, measurements)

    compile_future.result()               # block until compilation finishes
    data = generator.generate(model, rows=10_000_000)
"""

from __future__ import annotations

import collections
import concurrent.futures
import dataclasses
import functools
import logging
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .. import junction_tree, marginal_oracles
from ..clique_utils import Clique, clique_mapping
from ..clique_vector import CliqueVector
from ..dataset import Dataset
from ..domain import Domain
from ..factor import Factor
from ..markov_random_field import MarkovRandomField

# Shared thread pool for background JIT compilation.
_COMPILE_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=4)


# ---------------------------------------------------------------------------
# Generation plan: a static description of *how* to generate each column.
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _ColumnPlan:
  """Describes how to generate a single column."""

  col: str
  has_parents: bool
  query: tuple[str, ...]
  parents: tuple[str, ...]
  maximal_clique: tuple[str, ...]
  domain_size: int
  parent_sizes: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class _GenerationPlan:
  """Full generation plan for all columns."""

  order: tuple[str, ...]
  columns: dict[str, _ColumnPlan]
  jtree: Any  # nx.Graph — kept for message passing
  maximal_cliques: list[tuple[str, ...]]
  elimination_order: tuple[str, ...]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class SyntheticDataGenerator:
  """Generates synthetic data from a MarkovRandomField using JAX.

  The generator separates the concerns of *planning* (junction tree
  analysis, JIT compilation) from *execution* (message passing,
  per-column generation).  The planning phase only depends on the
  ``Domain`` and clique structure, both of which are known before
  estimation finishes.  This allows JIT compilation to run in the
  background while mirror descent is still iterating.

  Args:
    method: ``'round'`` for randomized rounding (default) or ``'sample'``
      for i.i.d. categorical sampling.
  """

  def __init__(self, method: str = 'round'):
    assert method in ('round', 'sample'), (
        f"method must be 'round' or 'sample', got {method!r}"
    )
    self.method = method
    self._plan: _GenerationPlan | None = None
    self._msg_fn = None

  # ------------------------------------------------------------------
  # Precompilation
  # ------------------------------------------------------------------

  def precompile(
      self,
      domain: Domain,
      cliques: list[Clique],
      rows: int,
  ) -> concurrent.futures.Future:
    """Warm up the JIT cache for ``generate`` asynchronously.

    Only requires the domain and clique structure — both are available
    before estimation finishes.  Returns a ``Future`` that resolves
    when all per-column JIT compilations have completed.  Callers may
    ignore the return value (fire-and-forget) or call
    ``future.result()`` to block.

    Args:
      domain: The Domain over which the model is defined.
      cliques: The cliques of the model (known after workload selection).
      rows: Number of records that will be generated.

    Returns:
      A ``Future`` that resolves to ``None`` when compilation is done.
    """
    rows = max(1, int(rows))
    plan = _build_plan(domain, cliques)
    self._plan = plan
    self._msg_fn = _make_message_passing_fn(plan.jtree)

    futures: list[concurrent.futures.Future] = []

    # 1. Precompile message passing (1 XLA program).
    futures.append(_COMPILE_POOL.submit(
        _precompile_message_passing, self._msg_fn, domain, plan,
    ))

    # 2. Precompile per-column generation functions (N XLA programs).
    for cp in plan.columns.values():
      futures.append(_COMPILE_POOL.submit(
          _precompile_column, cp, rows, self.method,
      ))

    def _await_all() -> None:
      for f in futures:
        f.result()

    return _COMPILE_POOL.submit(_await_all)

  # ------------------------------------------------------------------
  # Generation
  # ------------------------------------------------------------------

  def generate(
      self,
      model: MarkovRandomField,
      rows: int,
      *,
      seed: int = 0,
  ) -> Dataset:
    """Generate synthetic data from a fitted model.

    If ``precompile`` was called earlier with the same domain, cliques,
    and ``rows``, the JIT cache will be warm and this call is fast.

    Args:
      model: A fitted ``MarkovRandomField``.
      rows: Number of records to generate.
      seed: Random seed for the JAX PRNG.

    Returns:
      A ``Dataset`` whose marginals closely match the model.
    """
    rows = max(1, int(rows))
    domain = model.domain
    cliques = [set(cl) for cl in model.cliques]

    # Use cached plan or build a new one.
    if self._plan is not None:
      plan = self._plan
    else:
      plan = _build_plan(domain, cliques)
      self._plan = plan

    # Phase 1: Compute junction tree messages (single JIT'd program).
    if self._msg_fn is None:
      self._msg_fn = _make_message_passing_fn(plan.jtree)
    expanded_potentials = model.potentials.expand(list(plan.jtree.nodes))
    _, messages = self._msg_fn(expanded_potentials, 1.0)

    # Build per-maximal-clique potential lookup.
    mapping = clique_mapping(
        plan.maximal_cliques, model.cliques, domain=domain,
    )
    potential_mapping: dict[tuple, list[Factor]] = (
        collections.defaultdict(list)
    )
    for cl in model.cliques:
      potential_mapping[mapping[cl]].append(model.potentials[cl])

    # Build per-maximal-clique message lookup.
    message_lookup: dict[tuple, list[Factor]] = (
        collections.defaultdict(list)
    )
    for (i, j), msg in messages.items():
      message_lookup[j].append(msg)

    # Phase 2: Generate columns (all on GPU).
    rng = jax.random.PRNGKey(seed)
    data_jax: dict[str, jax.Array] = {}

    for step, col in enumerate(plan.order):
      rng, col_rng = jax.random.split(rng)
      cp = plan.columns[col]

      # Compute marginal on GPU.
      marg_jax = _compute_marginal_jax(
          cp.maximal_clique, cp.query, domain,
          potential_mapping, message_lookup, rows,
      )

      if not cp.has_parents:
        data_jax[col] = _dispatch_no_parents(
            col_rng, marg_jax, rows, self.method,
        )
      else:
        parent_data = tuple(data_jax[p] for p in cp.parents)
        data_jax[col] = _dispatch_with_parents(
            col_rng, marg_jax, parent_data, cp.parent_sizes,
            rows, self.method,
        )

      if (step + 1) % 10 == 0 or step + 1 == len(plan.order):
        logging.info(
            'Col %d/%d: %s done', step + 1, len(plan.order), col,
        )

    # Bulk transfer to CPU.
    data = {col: np.asarray(arr) for col, arr in data_jax.items()}
    return Dataset(data, domain)


# ---------------------------------------------------------------------------
# Plan construction (pure graph analysis, no JAX)
# ---------------------------------------------------------------------------


def _build_plan(
    domain: Domain,
    cliques: list[Clique],
) -> _GenerationPlan:
  """Analyse the junction tree and build the per-column generation plan."""
  clique_sets = [set(cl) for cl in cliques]
  jtree, elimination_order = junction_tree.make_junction_tree(
      domain, clique_sets,
  )
  maximal_cliques = junction_tree.maximal_cliques(jtree)
  order = tuple(reversed(elimination_order))

  columns: dict[str, _ColumnPlan] = {}
  used: set[str] = set()
  for col in order:
    relevant = [cl for cl in clique_sets if col in cl]
    parents = tuple(sorted(used.intersection(set().union(*relevant))))
    used.add(col)

    query = parents + (col,) if parents else (col,)
    query_set = set(query)
    mc = _find_maxclique(query_set, maximal_cliques)

    columns[col] = _ColumnPlan(
        col=col,
        has_parents=bool(parents),
        query=query,
        parents=parents,
        maximal_clique=mc,
        domain_size=domain[col],
        parent_sizes=tuple(domain[p] for p in parents),
    )

  return _GenerationPlan(
      order=order,
      columns=columns,
      jtree=jtree,
      maximal_cliques=maximal_cliques,
      elimination_order=tuple(elimination_order),
  )


# ---------------------------------------------------------------------------
# Precompilation helpers
# ---------------------------------------------------------------------------


def _make_message_passing_fn(jtree):
  """Create a JIT'd message passing function for a specific junction tree.

  The junction tree is captured in the closure, so all graph-structural
  operations (message ordering, clique mapping, neighbor lookups) are
  evaluated at trace time and compiled into a single XLA program.
  """

  @jax.jit
  def fn(potentials, total):
    return marginal_oracles.message_passing_implicit(
        potentials, total, jtree=jtree, return_messages=True,
    )

  return fn


def _precompile_message_passing(msg_fn, domain, plan):
  """Warm the JIT cache for the message passing program."""
  dummy_arrays = {}
  for cl in plan.jtree.nodes:
    shape = tuple(domain[attr] for attr in cl)
    dummy_arrays[cl] = jnp.zeros(shape, dtype=jnp.float32)
  dummy_potentials = CliqueVector(
      domain, list(plan.jtree.nodes), dummy_arrays,
  )
  msg_fn(dummy_potentials, 1.0)

def _precompile_column(
    cp: _ColumnPlan,
    rows: int,
    method: str,
) -> None:
  """Trace and compile the JIT'd generation function for one column."""
  dummy_rng = jax.random.PRNGKey(0)
  if not cp.has_parents:
    dummy_marg = jnp.ones(cp.domain_size, dtype=jnp.float32)
    _dispatch_no_parents(dummy_rng, dummy_marg, rows, method)
  else:
    shape = cp.parent_sizes + (cp.domain_size,)
    dummy_marg = jnp.ones(shape, dtype=jnp.float32)
    dummy_parents = tuple(
        jnp.zeros(rows, dtype=jnp.int32) for _ in cp.parent_sizes
    )
    _dispatch_with_parents(
        dummy_rng, dummy_marg, dummy_parents, cp.parent_sizes,
        rows, method,
    )


# ---------------------------------------------------------------------------
# JIT-compiled generation functions
# ---------------------------------------------------------------------------


def _dispatch_no_parents(rng_key, marg_counts, total, method):
  """Route to the appropriate JIT'd no-parents function."""
  if method == 'sample':
    return _sample_no_parents(rng_key, marg_counts, total)
  return _round_no_parents(rng_key, marg_counts, total)


def _dispatch_with_parents(
    rng_key, marg_counts, parent_data, parent_sizes, total, method,
):
  """Route to the appropriate JIT'd with-parents function."""
  if method == 'sample':
    return _sample_with_parents(rng_key, marg_counts, parent_data, parent_sizes)
  return _round_with_parents(
      rng_key, marg_counts, parent_data, parent_sizes, total,
  )


@functools.partial(jax.jit, static_argnums=(2,))
def _sample_no_parents(rng_key, marg_counts, total):
  """Sample method for no-parents case."""
  probs = marg_counts / marg_counts.sum()
  return jax.random.choice(
      rng_key, marg_counts.shape[0], shape=(total,), p=probs,
  ).astype(jnp.int32)


@functools.partial(jax.jit, static_argnums=(2,))
def _round_no_parents(rng_key, marg_counts, total):
  """Gumbel rounding for no-parents case."""
  domain_size = marg_counts.shape[0]
  counts = marg_counts * (total / marg_counts.sum())
  integ = jnp.floor(counts).astype(jnp.int32)
  frac = counts - jnp.floor(counts)
  extra = total - integ.sum()

  rng1, rng2 = jax.random.split(rng_key)
  u = jax.random.uniform(rng1, (domain_size,))
  scores = jnp.log(frac + 1e-30) - jnp.log(-jnp.log(u + 1e-30))
  scores = jnp.where(frac == 0, -jnp.inf, scores)

  ranked = jnp.argsort(-scores)
  roundup = jnp.where(jnp.arange(domain_size) < extra, 1, 0)
  integ = integ.at[ranked].add(roundup)

  vals = jnp.repeat(
      jnp.arange(domain_size, dtype=jnp.int32), integ,
      total_repeat_length=total,
  )
  perm = jax.random.permutation(rng2, total)
  return vals[perm]


@jax.jit
def _sample_with_parents(rng_key, marg_counts, parent_data, parent_sizes):
  """Per-record categorical sampling conditioned on parents."""
  marg_parents = marg_counts.sum(axis=-1, keepdims=True)
  cond_probs = jnp.where(marg_parents != 0, marg_counts / marg_parents, 0.0)
  domain_size = marg_counts.shape[-1]
  parent_product = cond_probs.size // domain_size
  cond_probs_2d = cond_probs.reshape(parent_product, domain_size)

  flat_parent_idx = _ravel_multi_index_jax(parent_data, parent_sizes)
  per_record_probs = cond_probs_2d[flat_parent_idx]
  return jax.random.categorical(
      rng_key, jnp.log(per_record_probs + 1e-30),
  ).astype(jnp.int32)


@functools.partial(jax.jit, static_argnums=(4,))
def _round_with_parents(
    rng_key, marg_counts, parent_data, parent_sizes, total,
):
  """Gumbel rounding conditioned on parents, vectorized on GPU."""
  rng1, rng2 = jax.random.split(rng_key)

  domain_size = marg_counts.shape[-1]
  parent_product = marg_counts.size // domain_size

  marg_parents = marg_counts.sum(axis=-1, keepdims=True)
  cond_probs = jnp.where(marg_parents != 0, marg_counts / marg_parents, 0.0)
  cond_probs_2d = cond_probs.reshape(parent_product, domain_size)

  flat_parent_idx = _ravel_multi_index_jax(parent_data, parent_sizes)
  counts_per_parent = jnp.bincount(flat_parent_idx, length=parent_product)

  expected = counts_per_parent[:, None] * cond_probs_2d
  integ = jnp.floor(expected).astype(jnp.int32)
  frac = expected - jnp.floor(expected)
  extra = counts_per_parent - integ.sum(axis=1)

  u = jax.random.uniform(rng1, (parent_product, domain_size))
  scores = jnp.log(frac + 1e-30) - jnp.log(-jnp.log(u + 1e-30))
  scores = jnp.where(frac == 0, -jnp.inf, scores)

  ranked = jnp.argsort(-scores, axis=1)
  positions = jnp.broadcast_to(
      jnp.arange(domain_size)[None, :], (parent_product, domain_size),
  )
  roundup_mask = (positions < extra[:, None]).astype(jnp.int32)
  row_indices = jnp.broadcast_to(
      jnp.arange(parent_product)[:, None], ranked.shape,
  )
  roundup = jnp.zeros((parent_product, domain_size), dtype=jnp.int32)
  roundup = roundup.at[row_indices, ranked].set(roundup_mask)
  integ = jnp.maximum(integ + roundup, 0)

  value_options = jnp.arange(domain_size, dtype=jnp.int32)
  value_options_tiled = jnp.tile(value_options, parent_product)
  all_values = jnp.repeat(
      value_options_tiled, integ.ravel(), total_repeat_length=total,
  )

  group_ids = jnp.repeat(
      jnp.arange(parent_product, dtype=jnp.float32),
      counts_per_parent,
      total_repeat_length=total,
  )
  shuffle_keys = jax.random.uniform(rng2, (total,))
  composite = group_ids + shuffle_keys
  shuffle_perm = jnp.argsort(composite)
  all_values = all_values[shuffle_perm]

  sort_order = jnp.argsort(flat_parent_idx)
  result = jnp.empty(total, dtype=jnp.int32)
  result = result.at[sort_order].set(all_values)
  return result


# ---------------------------------------------------------------------------
# Marginal computation and utility functions
# ---------------------------------------------------------------------------


def _compute_marginal_jax(
    maximal_clique, query, domain, potential_mapping, message_lookup, total,
):
  """Compute marginal as a JAX array, staying on device."""
  inputs = list(potential_mapping[maximal_clique]) + list(
      message_lookup[maximal_clique]
  )
  query_domain = domain.project(query)

  for attr in query_domain.attributes:
    if not any(attr in inp.domain.attributes for inp in inputs):
      inputs.append(Factor.zeros(domain.project([attr])))

  result = marginal_oracles.einsum_semistable(inputs, query_domain)
  result = result.normalize(total, log=True).exp()
  return result.datavector(flatten=False)


def _ravel_multi_index_jax(arrays, sizes):
  """JAX version of np.ravel_multi_index."""
  result = arrays[0].astype(jnp.int32)
  for arr, size in zip(arrays[1:], sizes[1:]):
    result = result * size + arr.astype(jnp.int32)
  return result


def _find_maxclique(query_vars, maximal_cliques):
  """Find a maximal clique containing all query variables."""
  for mc in maximal_cliques:
    if query_vars.issubset(set(mc)):
      return mc
  assert False, f'No maximal clique contains {query_vars}'
