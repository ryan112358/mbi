"""Precompute all marginals from a dataset using JIT-compiled bincount.

Computes a ``CliqueVector`` of marginals from raw data by JIT-compiling a
``jnp.bincount`` kernel per unique shape.  Three optimizations reduce
compilation overhead for heterogeneous domains:

1. **Power-of-2 bucketing:** Domain sizes are rounded up to the next
   power of 2, so attributes with similar cardinalities share a compiled
   program.
2. **Canonical shape ordering:** Within each clique, attributes are
   sorted so that padded sizes are non-decreasing.  This halves the
   number of unique shapes for 2-way marginals.
3. **Background precompilation:** Unique shapes are sorted by workload
   (descending number of cliques), and all compilations are submitted
   upfront so they overlap with execution.
"""

from __future__ import annotations

import concurrent.futures
import math
from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from ..clique_utils import Clique
from ..clique_vector import CliqueVector
from ..dataset import Dataset, JaxDataset
from ..factor import Factor

_COMPILE_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=1)


def _next_pow2(x):
  """Round up to the next power of 2."""
  return 1 << (x - 1).bit_length()


@jax.jit(static_argnums=(2,))
def _bincount_marginal(cols, weights, padded_shape):
  """JIT-compiled k-way bincount."""
  flat = cols[0].astype(jnp.uint32)
  for col, stride in zip(cols[1:], padded_shape[1:]):
    flat = flat * stride + col.astype(jnp.uint32)
  length = math.prod(padded_shape)
  counts = jnp.bincount(flat, weights=weights, length=length)
  return counts.reshape(padded_shape).astype(jnp.float32)


def _prepare_columns(dataset):
  """Extract JAX column arrays and weights from a Dataset or JaxDataset."""
  domain = dataset.domain
  col_data = {
      a: (
          jnp.asarray(dataset.data[a]).astype(
              np.min_scalar_type(_next_pow2(domain[a]) - 1)
          )
      )
      for a in domain.attributes
  }
  weights = (
      jnp.asarray(dataset.weights) if dataset.weights is not None else None
  )
  return col_data, weights


def _group_cliques_by_shape(cliques, domain):
  """Group cliques by their power-of-2 padded canonical shape.

  Returns ``(sorted_shapes, shape_groups, actual_sizes)`` where
  ``sorted_shapes`` is ordered by descending group size and
  ``shape_groups`` maps each padded shape to its list of work items.
  """
  actual_sizes = {a: domain[a] for a in domain.attributes}
  padded_sizes = {a: _next_pow2(domain[a]) for a in domain.attributes}

  work_items = []
  for cl in cliques:
    sorted_attrs = sorted(cl, key=lambda a: padded_sizes[a])
    padded_shape = tuple(padded_sizes[a] for a in sorted_attrs)
    work_items.append((padded_shape, sorted_attrs, cl))

  shape_groups: dict[tuple[int, ...], list] = {}
  for item in work_items:
    shape_groups.setdefault(item[0], []).append(item)
  sorted_shapes = sorted(shape_groups, key=lambda s: -len(shape_groups[s]))
  return sorted_shapes, shape_groups, actual_sizes


def _col_struct(n_rows, padded_size):
  """Build an abstract column struct for a given padded domain size."""
  dtype = np.min_scalar_type(padded_size - 1)
  return jax.ShapeDtypeStruct((n_rows,), dtype)


def _submit_all_compilations(
    sorted_shapes,
    n_rows,
    weights_struct,
):
  """Submit compilation for every unique shape and return futures."""
  futures: dict[tuple[int, ...], concurrent.futures.Future] = {}
  for shape in sorted_shapes:
    abstract_cols = tuple(_col_struct(n_rows, s) for s in shape)
    futures[shape] = _COMPILE_POOL.submit(
        lambda c, w, s: _bincount_marginal.lower(c, w, s).compile(),
        abstract_cols,
        weights_struct,
        shape,
    )
  return futures


def precompute_marginals(
    dataset: Dataset | JaxDataset,
    cliques: Sequence[Clique],
    *,
    transpose: bool = True,
) -> CliqueVector:
  """Compute all marginals from a dataset using JIT-compiled bincount.

  Compiles one ``jnp.bincount`` program per unique padded shape (after
  power-of-2 bucketing and canonical reordering) and executes it for
  every clique that maps to that shape.  All compilations are submitted
  to a background thread upfront so they overlap with execution.

  Accepts both ``Dataset`` (numpy-backed) and ``JaxDataset``
  (JAX-backed).  Numpy arrays are converted to JAX arrays internally.

  Args:
    dataset: The dataset to compute marginals from.
    cliques: Cliques (tuples of attribute names) to compute.
    transpose: If ``True`` (default), transpose each marginal so its
      axes match the original clique attribute order.  Set to ``False``
      to skip the transpose and keep the canonical sorted order.

  Returns:
    A ``CliqueVector`` mapping each clique to its marginal ``Factor``.
  """
  domain = dataset.domain
  col_data, weights = _prepare_columns(dataset)
  sorted_shapes, shape_groups, actual_sizes = _group_cliques_by_shape(
      cliques, domain
  )

  # Build abstract structs and submit all compilations upfront.
  n_rows = next(iter(col_data.values())).shape[0]
  weights_struct = (
      jax.ShapeDtypeStruct(weights.shape, weights.dtype)
      if weights is not None
      else None
  )
  compile_futures = _submit_all_compilations(
      sorted_shapes,
      n_rows,
      weights_struct,
  )

  arrays: dict[Clique, Factor] = {}

  for shape in sorted_shapes:
    compiled_fn = compile_futures[shape].result()

    for _, sorted_attrs, cl in shape_groups[shape]:
      cols = tuple(col_data[a] for a in sorted_attrs)
      marginal = compiled_fn(cols, weights)

      # Slice off power-of-2 padding.
      actual_shape = tuple(actual_sizes[a] for a in sorted_attrs)
      if actual_shape != shape:
        slices = tuple(slice(0, s) for s in actual_shape)
        marginal = marginal[slices]

      # Transpose back to the original attribute order if needed.
      proj_domain = domain.project(cl)
      if transpose and tuple(sorted_attrs) != proj_domain.attributes:
        perm = tuple(sorted_attrs.index(a) for a in proj_domain.attributes)
        marginal = jnp.transpose(marginal, perm)

      arrays[cl] = Factor(proj_domain, marginal)

  return CliqueVector(domain, list(cliques), arrays)
