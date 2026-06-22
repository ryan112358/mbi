"""JAX-accelerated synthetic data generation with async precompilation."""

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

_COMPILE_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=1)


def precompile(
    domain: Domain,
    cliques: list[Clique],
    rows: int,
) -> concurrent.futures.Future:
  """Warm the JIT cache for ``synthetic_data`` asynchronously.

  Only requires domain and clique structure — both are available before
  estimation finishes.  Fire and forget; ``synthetic_data`` benefits from
  whatever has compiled so far.

  Args:
    domain: The Domain over which the model is defined.
    cliques: The cliques of the model (known after workload selection).
    rows: Number of records that will be generated.
  """
  rows = max(1, int(rows))
  plan = _build_plan(domain, cliques)

  def _compile_all():
    # Run dummy message passing to get Factor inputs with correct
    # pytree structure for per-column compilation.
    dummy_potentials = _make_dummy_potentials(domain, cliques)
    _, dummy_messages = marginal_oracles.message_passing_implicit(
        dummy_potentials, 1.0, jtree=plan.jtree, return_messages=True,
    )
    pot_map, msg_map = _build_lookups(
        plan, dummy_potentials, dummy_messages, domain,
    )
    for cp in plan.columns.values():
      inputs = _gather_inputs(cp, domain, pot_map, msg_map)
      dummy_parents = tuple(
          jnp.zeros(rows, dtype=jnp.int32) for _ in cp.parents
      )
      _generate_column.lower(
          jax.random.PRNGKey(0), inputs, dummy_parents,
          query=cp.query, parent_sizes=cp.parent_sizes, total=rows,
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

  # Clique ordering in model.potentials must match the cliques passed to
  # precompile() for the JIT cache to hit (cliques are pytree metadata).
  _, messages = marginal_oracles.message_passing_implicit(
      model.potentials, 1.0, jtree=plan.jtree, return_messages=True,
  )
  pot_map, msg_map = _build_lookups(
      plan, model.potentials, messages, domain,
  )

  rng = jax.random.PRNGKey(seed)
  data: dict[str, jax.Array] = {}

  for step, col in enumerate(plan.order):
    rng, col_rng = jax.random.split(rng)
    cp = plan.columns[col]

    inputs = _gather_inputs(cp, domain, pot_map, msg_map)
    parent_arrays = tuple(data[p] for p in cp.parents)
    data[col] = _generate_column(
        col_rng, inputs, parent_arrays,
        query=cp.query, parent_sizes=cp.parent_sizes, total=rows,
    )

    if (step + 1) % 10 == 0 or step + 1 == len(plan.order):
      logging.info('Col %d/%d: %s done', step + 1, len(plan.order), col)

  return Dataset(
      {col: np.asarray(arr) for col, arr in data.items()}, domain,
  )


# ---------------------------------------------------------------------------
# Internal helpers.
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _ColumnPlan:
  col: str
  query: tuple[str, ...]
  parents: tuple[str, ...]
  maximal_clique: tuple[str, ...]
  parent_sizes: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class _GenerationPlan:
  order: tuple[str, ...]
  columns: dict[str, _ColumnPlan]
  jtree: Any  # nx.Graph
  maximal_cliques: list[tuple[str, ...]]


def _build_plan(domain: Domain, cliques) -> _GenerationPlan:
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
    mc = _find_maxclique(set(query), maximal_cliques)

    columns[col] = _ColumnPlan(
        col=col,
        query=query,
        parents=parents,
        maximal_clique=mc,
        parent_sizes=tuple(domain[p] for p in parents),
    )

  return _GenerationPlan(
      order=order, columns=columns,
      jtree=jtree, maximal_cliques=maximal_cliques,
  )


def _make_dummy_potentials(domain, cliques):
  arrays = {}
  for cl in cliques:
    shape = tuple(domain[attr] for attr in cl)
    arrays[cl] = jnp.zeros(shape, dtype=jnp.float32)
  return CliqueVector(domain, list(cliques), arrays)


def _build_lookups(plan, potentials, messages, domain):
  mapping = clique_mapping(
      plan.maximal_cliques, potentials.cliques, domain=domain,
  )
  pot_map: dict[tuple, list[Factor]] = collections.defaultdict(list)
  for cl in potentials.cliques:
    pot_map[mapping[cl]].append(potentials[cl])

  msg_map: dict[tuple, list[Factor]] = collections.defaultdict(list)
  for (i, j), msg in messages.items():
    msg_map[j].append(msg)

  return pot_map, msg_map


def _gather_inputs(cp, domain, pot_map, msg_map):
  inputs = list(pot_map[cp.maximal_clique]) + list(
      msg_map[cp.maximal_clique]
  )
  query_domain = domain.project(cp.query)
  for attr in query_domain.attributes:
    if not any(attr in inp.domain.attributes for inp in inputs):
      inputs.append(Factor.zeros(domain.project([attr])))
  return inputs


@jax.jit(static_argnames=['query', 'parent_sizes', 'total'])
def _generate_column(rng_key, inputs, parent_arrays, *, query, parent_sizes,
                     total):
  """Fused per-column program: marginal computation + Gumbel rounding."""
  # Marginal computation (einsum_materialized, inlined).
  combined = functools.reduce(lambda a, b: a + b, inputs)
  query_domain = combined.domain.project(query)
  elim_attrs = combined.domain.marginalize(query_domain).attributes
  result = combined.logsumexp(elim_attrs).transpose(query_domain.attributes)
  result = result.normalize(total, log=True).exp()
  marg = result.datavector(flatten=False)

  # Parent indexing.
  if parent_arrays:
    marg_2d = marg.reshape(-1, marg.shape[-1])
    parent_idx = _ravel_multi_index_jax(parent_arrays, parent_sizes)
  else:
    marg_2d = marg[jnp.newaxis, :]
    parent_idx = jnp.zeros(total, dtype=jnp.int32)

  # Gumbel rounding.
  return _gumbel_round(rng_key, marg_2d, parent_idx, total)


def _gumbel_round(rng_key, marg_2d, flat_parent_idx, total):
  """Gumbel rounding (called inside JIT, not separately decorated)."""
  rng1, rng2 = jax.random.split(rng_key)

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


def _ravel_multi_index_jax(arrays, sizes):
  result = arrays[0].astype(jnp.int32)
  for arr, size in zip(arrays[1:], sizes[1:]):
    result = result * size + arr.astype(jnp.int32)
  return result


def _find_maxclique(query_vars, maximal_cliques):
  for mc in maximal_cliques:
    if query_vars.issubset(set(mc)):
      return mc
  assert False, f'No maximal clique contains {query_vars}'
