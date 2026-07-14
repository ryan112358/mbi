"""JAX-accelerated synthetic data generation with async precompilation."""

from __future__ import annotations

import collections
import concurrent.futures
import dataclasses
import functools
import logging
import time
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

_COMPILE_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=1)


# precompile() and synthetic_data() are coupled: both must build the same
# _GenerationPlan and call _generate_column with matching static args for
# the JIT cache to hit.  Keep their signatures and internal logic in sync.


def precompile(
    domain: Domain,
    cliques: list[Clique],
    rows: int,
) -> concurrent.futures.Future:
  """Warm the JIT cache for ``synthetic_data`` asynchronously.

  Only requires domain and clique structure — both available after
  measurement selection.  Fire and forget; ``synthetic_data`` benefits
  from whatever has compiled so far.

  No concrete arrays are allocated (except a PRNG key); all inputs use
  abstract ``ShapeDtypeStruct`` values via ``Factor.abstract`` so the
  compiler sees shapes and dtypes without materializing memory.

  Args:
    domain: The Domain over which the model is defined.
    cliques: The cliques of the model (known after measurement selection).
    rows: Number of records that will be generated.
  """
  rows = max(1, int(rows))
  plan = _build_plan(domain, cliques)

  def _compile_all():
    # --- Compile message_passing_implicit (abstract, no execution) ---
    abstract_potentials = CliqueVector.abstract(domain, cliques)
    marginal_oracles.message_passing_implicit.lower(
        abstract_potentials,
        1.0,
        jtree=plan.jtree,
        return_messages=True,
    ).compile()

    # --- Build abstract inputs for each column from structure alone ---
    # Potentials grouped by maximal clique.
    mapping = clique_mapping(
        plan.maximal_cliques,
        cliques,
        domain=domain,
    )
    pot_map: dict[tuple, list[Factor]] = collections.defaultdict(list)
    for cl in cliques:
      pot_map[mapping[cl]].append(Factor.abstract(domain.project(cl)))

    # Messages grouped by destination clique (shapes from separators).
    message_order = junction_tree.message_passing_order(plan.jtree)
    msg_map: dict[tuple, list[Factor]] = collections.defaultdict(list)
    for i, j in message_order:
      sep = domain.project(tuple(sorted(set(i) & set(j))))
      msg_map[j].append(Factor.abstract(sep))

    # --- Compile each column's _generate_column ---
    for cp in plan.columns.values():
      # Same logic as _gather_inputs but using abstract Factors.
      inputs = list(pot_map[cp.maximal_clique]) + list(
          msg_map[cp.maximal_clique]
      )
      query_domain = domain.project(cp.query)
      for attr in query_domain.attributes:
        if not any(attr in inp.domain.attributes for inp in inputs):
          inputs.append(Factor.abstract(domain.project([attr])))

      # int32 matches _generate_column's ravel dtype; see the parent_arrays
      # comment in synthetic_data() for why the boundary is int32 rather than
      # the compact storage dtype.
      abstract_parents = tuple(
          jax.ShapeDtypeStruct((rows,), jnp.int32) for _ in cp.parents
      )
      _generate_column.lower(
          jax.random.PRNGKey(0),
          inputs,
          abstract_parents,
          query=cp.query,
          parent_sizes=cp.parent_sizes,
          total=rows,
      ).compile()

  return _COMPILE_POOL.submit(_compile_all)


def synthetic_data(
    model: MarkovRandomField,
    rows: int,
    seed: int = 0,
) -> Dataset:
  """Generate synthetic data from a fitted model using Gumbel rounding.

  Output marginals match the model within ±2 counts per cell.  If
  ``precompile`` was called earlier with matching domain, cliques, and
  ``rows``, the JIT cache will be warm and this call is fast.

  Args:
    model: A fitted ``MarkovRandomField``.
    rows: Number of records to generate.
    seed: Random seed for the JAX PRNG.

  Returns:
    A ``Dataset`` whose marginals closely match the model.
  """
  rows = max(1, int(rows))
  domain = model.domain
  plan = _build_plan(domain, model.cliques)

  # Empirically necessary for large models (574-col, ~60 cliques): without
  # this, XLA compilation is orders of magnitude slower. Exact mechanism TBD.
  potentials = jax.tree.map(jnp.asarray, model.potentials)

  # Clique ordering in model.potentials must match the cliques passed to
  # precompile() for the JIT cache to hit (cliques are pytree metadata).
  _, messages = marginal_oracles.message_passing_implicit(
      potentials,
      1.0,
      jtree=plan.jtree,
      return_messages=True,
  )
  pot_map, msg_map = _build_lookups(
      plan,
      potentials,
      messages,
      domain,
  )

  rng = jax.random.PRNGKey(seed)
  data: dict[str, np.ndarray] = {}

  for step, col in enumerate(plan.order):
    rng, col_rng = jax.random.split(rng)
    cp = plan.columns[col]

    inputs = _gather_inputs(cp, domain, pot_map, msg_map)
    # Parents feed only _ravel_multi_index, which casts them to int32 anyway, so
    # int32 is the real compute dtype. Passing int32 here (and lowering int32 in
    # precompile) keeps the JIT signature a fixed contract, independent of the
    # compact storage dtype (np.min_scalar_type(domain[p]), applied at the end of
    # the loop). A dtype mismatch would miss the JIT cache and recompile every
    # column. The only cost is a slightly larger host->device copy of this
    # O(rows) array; peak HBM is unchanged, since the int32 index is materialized
    # on-device regardless.
    parent_arrays = tuple(
        jnp.asarray(data[p], dtype=jnp.int32) for p in cp.parents
    )
    t0 = time.monotonic()
    data[col] = _generate_column(
        col_rng,
        inputs,
        parent_arrays,
        query=cp.query,
        parent_sizes=cp.parent_sizes,
        total=rows,
    )
    elapsed = time.monotonic() - t0

    if step == 0 and elapsed > 2.0:
      logging.warning(
          'First _generate_column call took %.1fs (expected <0.5s '
          'with a warm JIT cache). This likely means precompile() '
          'was not called, has not finished, or the model potentials '
          'have a different array type than expected (e.g. np.ndarray '
          'vs jax.Array). Subsequent columns will also recompile.',
          elapsed,
      )

    # Offload to host immediately — frees GPU memory for future columns.
    # Async overlap or lazy eviction could save ~20ms/col, but that's <1%
    # of per-column compute time, so we keep this simple.
    column = np.asarray(data[col])
    data[col] = column.astype(np.min_scalar_type(domain[col]))

    if (step + 1) % 10 == 0 or step + 1 == len(plan.order):
      logging.info('Col %d/%d: %s done', step + 1, len(plan.order), col)

  return Dataset(data, domain)


@dataclasses.dataclass(frozen=True)
class _ColumnPlan:
  """Per-column metadata for generation."""

  col: str
  query: tuple[str, ...]
  parents: tuple[str, ...]
  maximal_clique: tuple[str, ...]
  parent_sizes: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class _GenerationPlan:
  """Junction-tree analysis needed by both precompile and generate."""

  order: tuple[str, ...]
  columns: dict[str, _ColumnPlan]
  jtree: Any  # nx.Graph
  maximal_cliques: list[tuple[str, ...]]


def _build_plan(domain, cliques):
  """Analyse the junction tree and build per-column generation metadata."""
  clique_sets = [set(cl) for cl in cliques]
  jtree, elimination_order = junction_tree.make_junction_tree(
      domain,
      clique_sets,
  )
  maximal_cliques = junction_tree.maximal_cliques(jtree)
  order = tuple(reversed(elimination_order))

  # Use maximal cliques (junction tree super-cliques) for parent
  # determination, not the original measurement cliques.  The junction
  # tree merges overlapping cliques into super-cliques that capture
  # the full dependency structure.  Using original cliques misses
  # dependencies between attributes that share a super-clique but
  # not an original clique.
  jtree_clique_sets = [set(cl) for cl in maximal_cliques]

  columns: dict[str, _ColumnPlan] = {}
  used: set[str] = set()
  for col in order:
    relevant = [cl for cl in jtree_clique_sets if col in cl]
    parents = tuple(sorted(used.intersection(set().union(*relevant))))
    used.add(col)

    query = parents + (col,) if parents else (col,)
    mc = _find_maxclique(set(query), maximal_cliques)

    columns[col] = _ColumnPlan(
        col=col,
        query=query,
        parents=parents,
        maximal_clique=mc,
        parent_sizes=tuple(domain[p] for p in parents),
    )

  return _GenerationPlan(
      order=order,
      columns=columns,
      jtree=jtree,
      maximal_cliques=maximal_cliques,
  )


def _build_lookups(plan, potentials, messages, domain):
  """Group potentials and messages by maximal clique."""
  mapping = clique_mapping(
      plan.maximal_cliques,
      potentials.cliques,
      domain=domain,
  )
  pot_map: dict[tuple, list[Factor]] = collections.defaultdict(list)
  for cl in potentials.cliques:
    pot_map[mapping[cl]].append(potentials[cl])

  msg_map: dict[tuple, list[Factor]] = collections.defaultdict(list)
  for (i, j), msg in messages.items():
    msg_map[j].append(msg)

  return pot_map, msg_map


def _gather_inputs(cp, domain, pot_map, msg_map):
  """Collect Factor inputs for a column's marginal computation."""
  inputs = list(pot_map[cp.maximal_clique]) + list(msg_map[cp.maximal_clique])
  query_domain = domain.project(cp.query)
  for attr in query_domain.attributes:
    if not any(attr in inp.domain.attributes for inp in inputs):
      inputs.append(Factor.zeros(domain.project([attr])))
  return inputs


@jax.jit(static_argnames=['query', 'parent_sizes', 'total'])
def _generate_column(
    prng, inputs, parent_arrays, *, query, parent_sizes, total
):
  """Fused per-column program: marginal computation + Gumbel rounding."""
  combined = functools.reduce(lambda a, b: a + b, inputs)
  query_domain = combined.domain.project(query)
  elim_attrs = combined.domain.marginalize(query_domain).attributes
  result = combined.logsumexp(elim_attrs).transpose(query_domain.attributes)
  result = result.normalize(total, log=True).exp()
  marg = result.datavector(flatten=False)

  if parent_arrays:
    marg_2d = marg.reshape(-1, marg.shape[-1])
    parent_idx = _ravel_multi_index(parent_arrays, parent_sizes)
  else:
    marg_2d = marg[jnp.newaxis, :]
    parent_idx = jnp.zeros(total, dtype=jnp.int32)

  return _gumbel_round(prng, marg_2d, parent_idx, total)


def _gumbel_round(prng, marg_2d, flat_parent_idx, total):
  """Assign each record a value matching expected marginals."""
  rng1, rng2 = jax.random.split(prng)

  domain_size = marg_2d.shape[-1]
  parent_product = marg_2d.shape[0]

  marg_parents = marg_2d.sum(axis=-1, keepdims=True)
  cond_probs = jnp.where(marg_parents != 0, marg_2d / marg_parents, 0.0)

  counts_per_parent = jnp.bincount(flat_parent_idx, length=parent_product)

  expected = counts_per_parent[:, None] * cond_probs
  integ = jnp.floor(expected).astype(jnp.int32)
  frac = expected - jnp.floor(expected)
  extra = counts_per_parent - integ.sum(axis=1)

  u = jax.random.uniform(rng1, (parent_product, domain_size))
  scores = jnp.log(frac + 1e-30) - jnp.log(-jnp.log(u + 1e-30))
  scores = jnp.where(frac == 0, -jnp.inf, scores)

  ranked = jnp.argsort(-scores, axis=1)
  positions = jnp.broadcast_to(
      jnp.arange(domain_size)[None, :],
      (parent_product, domain_size),
  )
  roundup_mask = (positions < extra[:, None]).astype(jnp.int32)
  row_indices = jnp.broadcast_to(
      jnp.arange(parent_product)[:, None],
      ranked.shape,
  )
  roundup = jnp.zeros((parent_product, domain_size), dtype=jnp.int32)
  roundup = roundup.at[row_indices, ranked].set(roundup_mask)
  integ = jnp.maximum(integ + roundup, 0)

  value_options = jnp.arange(domain_size, dtype=jnp.int32)
  value_options_tiled = jnp.tile(value_options, parent_product)
  all_values = jnp.repeat(
      value_options_tiled,
      integ.ravel(),
      total_repeat_length=total,
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
  return jnp.empty(total, dtype=jnp.int32).at[sort_order].set(all_values)


def _ravel_multi_index(arrays, sizes):
  """Compute flat indices from a tuple of per-dimension index arrays."""
  result = arrays[0].astype(jnp.int32)
  for arr, size in zip(arrays[1:], sizes[1:]):
    result = result * size + arr.astype(jnp.int32)
  return result


def _find_maxclique(query_vars, maximal_cliques):
  """Return the first maximal clique that contains all query variables."""
  for mc in maximal_cliques:
    if query_vars.issubset(set(mc)):
      return mc
  assert False, f'No maximal clique contains {query_vars}'
