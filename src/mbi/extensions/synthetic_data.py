"""JAX-accelerated synthetic data generation with async precompilation."""

from __future__ import annotations

import collections
from collections.abc import Callable, Sequence
import concurrent.futures
import dataclasses
import functools
import logging
import math
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .. import junction_tree, marginal_oracles
from .._future import RobustFuture
from ..clique_utils import Clique, clique_mapping
from ..clique_vector import CliqueVector
from ..constraint import Constraint
from ..dataset import Dataset
from ..domain import Attribute, Domain
from ..factor import Factor
from ..markov_random_field import MarkovRandomField

_COMPILE_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=1)

_REALIZED_CHUNK_CELLS = 1 << 20


# precompile() and synthetic_data() are coupled: both must build the same
# _GenerationPlan and call _generate_column with matching static args for
# the JIT cache to hit.  Keep their signatures and internal logic in sync.


def precompile(
    domain: Domain,
    cliques: Sequence[Clique],
    rows: int,
    constraints: Sequence[Constraint] = (),
    *,
    contraction: Callable = marginal_oracles.einsum_sharded,
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
    constraints: Structural constraints the model is fit under.  Their
        cliques are added to the plan so the warmed JIT cache matches the
        constraint-aware ``synthetic_data`` call.
    contraction: Contraction function for log-space sum-product.
  """
  rows = max(1, int(rows))
  cliques = _plan_cliques(cliques, constraints, domain)
  plan = _build_plan(domain, cliques)

  def _compile_all():
    # --- Compile message_passing_implicit (abstract, no execution) ---
    abstract_potentials = CliqueVector.abstract(domain, cliques)
    mp_compiled = marginal_oracles.message_passing_implicit.lower(
        abstract_potentials,
        1.0,
        jtree=plan.jtree,
        contraction=contraction,
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
    # Propagate the separator shardings produced by `einsum_sharded` onto the
    # abstract message factors so `_generate_column.lower` sees the exact
    # `NamedSharding` that `synthetic_data` will pass at runtime.
    message_order = junction_tree.message_passing_order(plan.jtree)
    msg_map: dict[tuple, list[Factor]] = collections.defaultdict(list)
    _, msg_shardings = mp_compiled.output_shardings
    for i, j in message_order:
      sep = domain.project([a for a in i if a in j])
      sharding = msg_shardings[(i, j)].values
      if not isinstance(sharding, jax.sharding.NamedSharding):
        sharding = None
      msg_map[j].append(Factor.abstract(sep, sharding=sharding))

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

      # int32 matches _generate_column's ravel dtype (see parent_arrays below).
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
          contraction=contraction,
      ).compile()

  return RobustFuture(_COMPILE_POOL.submit(_compile_all))


def synthetic_data(
    model: MarkovRandomField,
    rows: int,
    seed: int = 0,
    *,
    contraction: Callable = marginal_oracles.einsum_sharded,
) -> Dataset:
  """Generate synthetic data from a fitted model using Gumbel rounding.

  Output marginals match the model within ±2 counts per cell.  If
  ``precompile`` was called earlier with matching domain, cliques, and
  ``rows``, the JIT cache will be warm and this call is fast.

  Args:
    model: A fitted ``MarkovRandomField``.
    rows: Number of records to generate.
    seed: Random seed for the JAX PRNG.
    contraction: Contraction function for log-space sum-product.

  Returns:
    A ``Dataset`` whose marginals closely match the model.
  """
  rows = max(1, int(rows))
  domain = model.domain

  # Empirically necessary for large models (574-col, ~60 cliques): without
  # this, XLA compilation is orders of magnitude slower. Exact mechanism TBD.
  potentials = jax.tree.map(jnp.asarray, model.potentials)

  # Fold constraints in as -inf/0 log-space factors so each column's
  # conditional marginal (reconstructed below from potentials + messages)
  # puts zero mass on forbidden cells.  Without this, generated records can
  # violate the constraints the model was fit under.
  if model.constraints:
    potentials, _ = marginal_oracles._fold_constraints(
        potentials, model.constraints
    )
    # Constraint factors are weak-typed (built from Python -inf scalars),
    # but precompile()'s abstract factors are not.  Strip weak types so the
    # _generate_column JIT signatures match and the warmed cache is reused.
    potentials = jax.tree.map(lambda x: x.astype(x.dtype), potentials)
  plan = _build_plan(domain, potentials.cliques)

  # Clique ordering in potentials must match the cliques passed to
  # precompile() for the JIT cache to hit (cliques are pytree metadata).
  _, messages = marginal_oracles.message_passing_implicit(
      potentials,
      1.0,
      jtree=plan.jtree,
      contraction=contraction,
      return_messages=True,
  )
  pot_map, msg_map = _build_lookups(
      plan,
      potentials,
      messages,
      domain,
  )

  rng = jax.random.PRNGKey(seed)
  data: dict[Attribute, np.ndarray] = {}

  for step, col in enumerate(plan.order):
    rng, col_rng = jax.random.split(rng)
    cp = plan.columns[col]

    inputs = _gather_inputs(cp, domain, pot_map, msg_map)
    # Parents are only fed to _ravel_multi_index (int32); pinning int32 here
    # keeps the JIT signature stable and avoids per-column recompiles.
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
        contraction=contraction,
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

  col: Attribute
  query: Clique
  parents: Clique
  maximal_clique: Clique
  parent_sizes: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class _GenerationPlan:
  """Junction-tree analysis needed by both precompile and generate."""

  order: Clique
  columns: dict[Attribute, _ColumnPlan]
  jtree: Any  # nx.Graph
  maximal_cliques: list[Clique]


def _plan_cliques(cliques, constraints, domain):
  """Model cliques plus any constraint cliques not already present.

  Mirrors the clique construction in ``marginal_oracles._fold_constraints``
  (which ``synthetic_data`` uses) so that ``precompile`` and ``synthetic_data``
  build the same junction tree and JIT signature.
  """
  cliques = list(cliques)
  for c in constraints:
    cl = domain.canonical(c.clique)
    if cl not in cliques:
      cliques.append(cl)
  return cliques


def _build_plan(domain, cliques):
  """Analyse the junction tree and build per-column generation metadata."""
  clique_sets = [tuple(cl) for cl in cliques]
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

  columns: dict[Attribute, _ColumnPlan] = {}
  used: set[Attribute] = set()
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


@jax.jit(
    static_argnames=['query', 'parent_sizes', 'total', 'contraction'],
    # `total` is static, so root-column arrays built from it become large
    # constant subgraphs XLA would fold on the host (slow); defer to device.
    compiler_options={'xla_disable_hlo_passes': 'constant_folding'},
)
def _generate_column(
    prng,
    inputs,
    parent_arrays,
    *,
    query,
    parent_sizes,
    total,
    contraction=marginal_oracles.einsum_materialized,
):
  """Fused per-column program: marginal computation + Gumbel rounding."""
  merged = functools.reduce(
      lambda a, b: a.merge(b.domain), inputs, Domain([], [])
  )
  query_domain = merged.project(query)
  # When the parent state space is at most `total` and the query table fits in a
  # single rounding chunk, contract out unselected clique attributes into a
  # dense `(parent_product, domain_size)` table. Once the parent space exceeds
  # `total`, most parent cells are empty, so evaluate only realized parents.
  use_dense = not parent_sizes or (
      math.prod(parent_sizes) <= total
      and query_domain.size() <= _ROUND_CHUNK_CELLS
  )
  if use_dense:
    marg = contraction(inputs, query_domain)
    marg = marg.normalize(total, log=True).exp().values
    idx = (
        jnp.ravel_multi_index(parent_arrays, parent_sizes, mode='wrap')
        if parent_arrays
        else jnp.zeros(total, dtype=jnp.int32)
    )
    return _gumbel_round(prng, marg.reshape(-1, marg.shape[-1]), idx, total)

  probs, group_id = _realized_marginals(
      inputs, parent_arrays, query, contraction=contraction
  )
  return _gumbel_round(prng, probs, group_id, total)


def _slice_factor(
    f: Factor, batch_p: dict[str, jax.Array], chunk: int
) -> Factor:
  """Slice a Factor along realized parent coordinates for a batch of groups."""
  # Partition the factor's attributes into conditioned parents `p` and
  # unconditioned attributes `r` (a subset of `{col} U elim_attrs`). Advanced
  # indexing with length-`chunk` coordinate arrays `batch_p[a]` replaces all
  # axes in `p` with a single leading `'_batch'` axis of size `chunk`.
  p = [a for a in f.domain.attributes if a in batch_p]
  if not p:
    return f
  r = [a for a in f.domain.attributes if a not in batch_p]
  vals = f.transpose(p + r).values[tuple(batch_p[a] for a in p)]
  dom = Domain(['_batch', *r], [chunk, *(f.domain[a] for a in r)])
  return Factor(dom, vals)


def _realized_marginals(
    inputs,
    parent_arrays,
    query,
    *,
    contraction=marginal_oracles.einsum_materialized,
):
  """Group realized parents and evaluate conditional probabilities per group."""
  # Deduplicate realized parent tuples across the `n` records. `size=n` keeps
  # shapes static for JIT precompilation; unused slots are padded with a repeat
  # of group 0 and receive zero record counts in `_gumbel_round`.
  parents, col = query[:-1], query[-1]
  n = parent_arrays[0].shape[0]
  unique_p, group_id = jnp.unique(
      jnp.stack(parent_arrays, axis=1), axis=0, return_inverse=True, size=n
  )
  # Each group requires contracting over the remaining clique attributes of size
  # `cells = |{col} U elim_attrs|`. Scan over groups in chunks so each step
  # materializes at most `_REALIZED_CHUNK_CELLS` elements.
  merged = functools.reduce(
      lambda a, b: a.merge(b.domain), inputs, Domain([], [])
  )
  domain_size = merged[col]
  cells = merged.marginalize(parents).size()
  chunk = min(n, max(1, _REALIZED_CHUNK_CELLS // max(1, cells)))
  n_chunks = math.ceil(n / chunk)
  pad = n_chunks * chunk - n
  padded_p = jnp.pad(unique_p, ((0, pad), (0, 0))).reshape(n_chunks, chunk, -1)
  batch_dom = Domain(['_batch', col], [chunk, domain_size])

  def _body(_, batch_p_vals):
    # Slice each input factor at the chunk's parent coordinates, include a zero
    # factor over `batch_dom` so `_batch` and `col` are always present, and
    # contract out `elim_attrs` to obtain `(chunk, domain_size)` log-marginals.
    batch_p = dict(zip(parents, batch_p_vals.T))
    sliced = [Factor.zeros(batch_dom)] + [
        _slice_factor(f, batch_p, chunk) for f in inputs
    ]
    marg = contraction(sliced, batch_dom)
    return None, jnp.nan_to_num(jax.nn.softmax(marg.values))

  _, probs = jax.lax.scan(_body, None, padded_p)
  return probs.reshape(n_chunks * chunk, domain_size)[:n], group_id.ravel()


# Chunk the per-parent rounding for large marginals: the dense path allocates
# many (parent_product, domain_size) arrays (~26 GiB near 2^28 x64) and OOMs.
_ROUND_CHUNK_CELLS = 1 << 24  # ~16M cells/chunk (~1 GiB working set under x64)


def _stochastic_round_dense(rng, cond_probs, counts_per_parent):
  """Expected-count rounding: floor + systematic rounding of the remainder."""
  expected = counts_per_parent[:, None] * cond_probs
  integ = jnp.floor(expected).astype(jnp.int32)
  frac = expected - integ
  extra = counts_per_parent[:, None] - integ.sum(axis=1, keepdims=True)
  cum = jnp.cumsum(frac, axis=1)
  total = jnp.where(cum[:, -1:] > 0, cum[:, -1:], 1.0)
  u = jax.random.uniform(rng, (cond_probs.shape[0], 1))
  cum_int = jnp.floor((cum / total) * extra + (1.0 - u)).astype(jnp.int32)
  roundup = jnp.diff(jnp.clip(cum_int, 0, extra), axis=1, prepend=0)
  return jnp.maximum(integ + roundup, 0)


def _stochastic_round_chunked(rng, cond_probs, counts_per_parent):
  """Row-chunked equivalent of ``_stochastic_round_dense`` for large marginals."""
  parent_product, domain_size = cond_probs.shape
  chunk_rows = max(1, _ROUND_CHUNK_CELLS // domain_size)
  n_chunks = -(-parent_product // chunk_rows)  # ceil division
  pad = n_chunks * chunk_rows - parent_product

  cp = jnp.pad(cond_probs, ((0, pad), (0, 0)))
  cp = cp.reshape(n_chunks, chunk_rows, domain_size)
  ct = jnp.pad(counts_per_parent, (0, pad)).reshape(n_chunks, chunk_rows)

  def body(_, xs):
    idx, cp_k, ct_k = xs
    integ_k = _stochastic_round_dense(jax.random.fold_in(rng, idx), cp_k, ct_k)
    return None, integ_k

  _, integ = jax.lax.scan(body, None, (jnp.arange(n_chunks), cp, ct))
  return integ.reshape(n_chunks * chunk_rows, domain_size)[:parent_product]


def _stochastic_round(rng, cond_probs, counts_per_parent):
  """Assign integer counts per (parent, value) matching expected marginals."""
  parent_product, domain_size = cond_probs.shape
  if parent_product * domain_size <= _ROUND_CHUNK_CELLS:
    return _stochastic_round_dense(rng, cond_probs, counts_per_parent)
  return _stochastic_round_chunked(rng, cond_probs, counts_per_parent)


def _gumbel_round(prng, marg_2d, flat_parent_idx, total):
  """Assign each record a value matching expected marginals."""
  rng1, rng2 = jax.random.split(prng)

  domain_size = marg_2d.shape[-1]
  parent_product = marg_2d.shape[0]

  marg_parents = marg_2d.sum(axis=-1, keepdims=True)
  cond_probs = jnp.where(marg_parents != 0, marg_2d / marg_parents, 0.0)

  counts_per_parent = jnp.bincount(flat_parent_idx, length=parent_product)

  integ = _stochastic_round(rng1, cond_probs, counts_per_parent)

  value_options = jnp.arange(domain_size, dtype=jnp.int32)
  value_options_tiled = jnp.tile(value_options, parent_product)
  all_values = jnp.repeat(
      value_options_tiled,
      integ.ravel(),
      total_repeat_length=total,
  )

  group_ids = jnp.repeat(
      jnp.arange(parent_product, dtype=jnp.int32),
      counts_per_parent,
      total_repeat_length=total,
  )
  perm = jnp.lexsort((jax.random.uniform(rng2, (total,)), group_ids))
  sort_order = jnp.argsort(flat_parent_idx)
  return jnp.empty(total, dtype=jnp.int32).at[sort_order].set(all_values[perm])


def _find_maxclique(query_vars, maximal_cliques):
  """Return the first maximal clique that contains all query variables."""
  for mc in maximal_cliques:
    if query_vars.issubset(set(mc)):
      return mc
  assert False, f'No maximal clique contains {query_vars}'
