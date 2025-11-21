"""Prototype implementation of scan-based einsum implementation."""

from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp


def custom_dot_general(
    lhs: jax.Array,
    rhs: jax.Array,
    dimension_numbers,
    precision = None,
    preferred_element_type = None,
    *,
    combine_fn: Callable[[jax.Array, jax.Array], jax.Array] = jnp.multiply,
    reduce_fn: Callable[[jax.Array, int | Sequence[int]], jax.Array] = jnp.sum,
) -> jax.Array:
  """Computes a generalized dot product of two arrays.

  This function extends the concept of `jax.numpy.dot_general` by allowing
  custom functions for combining elements and reducing along contracting
  dimensions. Specifically, this function works by making lhs and rhs 
  broadcast-compatible, calling combine_fn on them to merge them into a single
  Array, then calling reduce_fn, to marginalize over the contracting dimensions.

  Without JIT compilation, this implementation requires materializing the
  potentially large intermediate obtained after calling combine_fn, but before
  calling reduce_fn.  Under JIT, especially with XLA:GPU or XLA:TPU compilers,
  these operations will often be fused to save memory.

  Examples:
  - combine_fn=jnp.add, reduce_fn=jax.scipy.special.logsumexp gives a
    numerically stable implementations of dot_general that works in logspace.
  - combine_fn=jnp.add, reduce_fn=jnp.max gives a method for finding the maximum
    element along contracting dimensions in a broadcasted sum.

  Args:
    lhs: The left-hand side array-like input.
    rhs: The right-hand side array-like input.
    dimension_numbers: See jax.lax.dot_general for details.
    precision: Currently unused by this implementation.
    preferred_element_type: Optional. This argument is from the JAX API. If
      provided, `lhs` and `rhs` inputs are cast to this `jax.numpy.dtype`.
    combine_fn: A callable (default: `jnp.multiply`) that takes two arrays (the
      broadcast-compatible expanded `lhs` and `rhs`) and returns an array.
      This function is applied element-wise before reduction.
      Example: `jnp.add` for log-space style combination.
    reduce_fn: A callable (default: `jnp.sum`) that takes an array and an `axis`
      argument. It may optionally accept `keepdims`. It's used to reduce results
      along the contracting dimensions. `keepdims=False` behavior is expected.
      Example: `jax.scipy.special.logsumexp` for log-space style reduction.
    out_sharding: Ignored.

  Returns:
    An array containing the result of the generalized dot product.
  """

  del precision, preferred_element_type  # unused

  (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dimension_numbers

  lhs_used_dims = set(list(lhs_contract) + list(lhs_batch))
  lhs_other_dims = sorted([i for i in range(lhs.ndim) if i not in lhs_used_dims])

  rhs_used_dims = set(list(rhs_contract) + list(rhs_batch))
  rhs_other_dims = sorted([i for i in range(rhs.ndim) if i not in rhs_used_dims])

  lhs_permutation = list(lhs_batch) + list(lhs_other_dims) + list(lhs_contract)
  rhs_permutation = list(rhs_batch) + list(rhs_other_dims) + list(rhs_contract)

  lhs_permuted = jnp.transpose(lhs, axes=lhs_permutation)
  rhs_permuted = jnp.transpose(rhs, axes=rhs_permutation)

  batch_dims = len(lhs_batch)
  lhs_other_dims = len(lhs_other_dims)
  rhs_other_dims = len(rhs_other_dims)
  contract_dims = len(lhs_contract)

  # Make broadcast compatible
  new_axes = tuple(batch_dims+lhs_other_dims+i for i in range(rhs_other_dims))
  lhs_expanded = jnp.expand_dims(lhs_permuted, axis=new_axes)

  rhs_axes = tuple(batch_dims+i for i in range(lhs_other_dims))
  rhs_expanded = jnp.expand_dims(rhs_permuted, axis=rhs_axes)
  
  combined = combine_fn(lhs_expanded, rhs_expanded)
  contract_axes = tuple(range(combined.ndim - contract_dims, combined.ndim))
  return reduce_fn(combined, contract_axes)


def _axis_name_to_dim(axis_names: str, axis_name: str) -> int | None:
  """Returns the index of axis_name in axis_names, or None if not found."""
  if axis_name in axis_names:
    return axis_names.index(axis_name)
  else:
    return None


def _get_subarrays(
    arrays: tuple[jax.Array, ...], ax_dims: list[int | None], index: int
) -> tuple[jax.Array, ...]:
  """Return a slice of each array along the given axis at the given index.

  Example:
    >>> A = jnp.arange(6).reshape(2, 3)
    >>> B = jnp.arange(8).reshape(2, 4)
    >>> C = jnp.arange(12).reshape(3, 4)
    >>> X, Y, Z = _get_subarrays([A, B, C], [1, None, 0], 2)
    >>> jnp.allclose(A[:,2], X) & jnp.allclose(B, Y) & jnp.allclose(C[2,:], Z)
    Array(True, dtype=bool)

  Args:
    arrays: A list of arrays.
    ax_dims: A list of axis dimensions to slice along, or None if no slicing is
      needed.
    index: The index to slice at.
  Returns:
    A list of arrays, each of which is a slice of the corresponding input array.
  """
  ans = []
  for array, dim in zip(arrays, ax_dims):
    if dim is None:
      ans.append(array)
    else:
      idx = tuple(slice(None) if i != dim else index for i in range(array.ndim))
      ans.append(array[idx])
  return tuple(ans)


def _infer_shapes(
    input_axes_names: list[str], arrays: tuple[jax.Array]
) -> dict[str, int]:
  """Infer the shapes of the arrays given the input axes names."""
  ans = {}
  for axis_names, arr in zip(input_axes_names, arrays):
    ans.update(**dict(zip(axis_names, arr.shape)))
  return ans


def scan_einsum(
    formula: str, *arrays: jax.Array, sequential: str = '', **kwargs
) -> jax.Array:
  """Einsum implementation that allows sequential execution across any axes.

  Can be more time and memory efficient than jnp.einsum when there are more
  than 2 operands, and even run in settings where jnp.einsum would not.

  Args:
    formula: The einsum formula.
    *arrays: The arrays to einsum.
    sequential: A string of axes to sequentially einsum across.  
    When sequential is empty, jnp.einsum is used.  When sequential is a single
      axis name, the smaller einsums are executed sequentially along that axis,
      and the results are accumulated.  When sequential is a string with
      multiple axis names, sequential einsums are executed along all axis 
      names given.
    **kwargs: Keyword arguments to pass to jnp.einsum in the base case.

  Returns:
    The einsum result.
  """
  if not sequential:
    return jnp.einsum(formula, *arrays, **kwargs)

  ax = sequential[0]
  if ax not in formula:
      raise ValueError(f"Sequential axis '{ax}' not found.")

  input_axes, output_axes = formula.split('->')
  input_axes = input_axes.split(',')
  shapes = _infer_shapes(input_axes, arrays)

  ax_dims = [_axis_name_to_dim(names, ax) for names in input_axes]

  # We compute smaller einsums sequentially by looping over the ax dimension.
  loop = jnp.arange(shapes[ax])
  new_formula = formula.replace(ax, '')

  def small_einsum(i):
    new_arrays = _get_subarrays(arrays, ax_dims, i)
    return scan_einsum(new_formula, *new_arrays, sequential=sequential[1:])

  if ax in output_axes:
    # Each smaller einsum is independent.
    return jax.lax.map(small_einsum, loop).swapaxes(0, output_axes.index(ax))
  else:
    # Each smaller einsum contributes to the global einsum.
    init = jnp.zeros(tuple(shapes[i] for i in output_axes))
    return jax.lax.scan(
        lambda carry, i: (carry + small_einsum(i), ()), init, loop
    )[0]
