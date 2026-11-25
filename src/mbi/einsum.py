"""Semiring-generalized einsum primitives for JAX arrays.

Provides the ``Semiring`` dataclass with built-in ``STANDARD``,
``LOG_SUM_EXP``, and ``MAX_PLUS`` instances, along with ``dot_general``,
``einsum``, and sequential ``scan_einsum`` contractions.
"""

from collections.abc import Callable, Sequence
import dataclasses
import functools
from typing import Any

import jax
from jax._src.core import Literal
import jax.numpy as jnp


@dataclasses.dataclass(frozen=True)
class Semiring:
  """Algebraic semiring for generalized tensor contractions.

  Attributes:
    combine: Binary elementwise product (e.g., ``jnp.multiply`` or ``jnp.add``).
    reduce: Axis reduction (e.g., ``jnp.sum``, ``logsumexp``, or ``jnp.max``).
    add: Binary accumulator matching ``reduce`` for sequential scans.
    zero: Additive identity scalar for ``add``.
  """

  combine: Callable[[jax.Array, jax.Array], jax.Array]
  reduce: Callable[[jax.Array, int | Sequence[int]], jax.Array]
  add: Callable[[jax.Array, jax.Array], jax.Array]
  zero: float


STANDARD = Semiring(combine=jnp.multiply, reduce=jnp.sum, add=jnp.add, zero=0.0)
LOG_SUM_EXP = Semiring(
    combine=jnp.add,
    reduce=jax.scipy.special.logsumexp,
    add=jnp.logaddexp,
    zero=-jnp.inf,
)
MAX_PLUS = Semiring(
    combine=jnp.add, reduce=jnp.max, add=jnp.maximum, zero=-jnp.inf
)


def _semiring_dot_general(
    lhs: jax.Array,
    rhs: jax.Array,
    dimension_numbers: jax.lax.DotDimensionNumbers,
    precision: jax.lax.PrecisionLike = None,
    preferred_element_type: jax.typing.DTypeLike | None = None,
    *,
    combine_fn: Callable[[jax.Array, jax.Array], jax.Array],
    reduce_fn: Callable[[jax.Array, int | Sequence[int]], jax.Array],
) -> jax.Array:
  """Broadcasts lhs and rhs, applies combine_fn, and reduces contracted dims."""
  del precision, preferred_element_type  # unused

  (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dimension_numbers

  lhs_used = set(lhs_contract) | set(lhs_batch)
  lhs_other = [i for i in range(lhs.ndim) if i not in lhs_used]

  rhs_used = set(rhs_contract) | set(rhs_batch)
  rhs_other = [i for i in range(rhs.ndim) if i not in rhs_used]

  lhs_perm = list(lhs_batch) + lhs_other + list(lhs_contract)
  rhs_perm = list(rhs_batch) + rhs_other + list(rhs_contract)

  lhs_permuted = jnp.transpose(lhs, axes=lhs_perm)
  rhs_permuted = jnp.transpose(rhs, axes=rhs_perm)

  n_batch = len(lhs_batch)
  n_lhs_other = len(lhs_other)
  n_rhs_other = len(rhs_other)
  n_contract = len(lhs_contract)

  lhs_new_axes = tuple(n_batch + n_lhs_other + i for i in range(n_rhs_other))
  rhs_new_axes = tuple(n_batch + i for i in range(n_lhs_other))
  lhs_expanded = jnp.expand_dims(lhs_permuted, axis=lhs_new_axes)
  rhs_expanded = jnp.expand_dims(rhs_permuted, axis=rhs_new_axes)

  combined = combine_fn(lhs_expanded, rhs_expanded)
  if not n_contract:
    return combined
  contract_axes = tuple(range(combined.ndim - n_contract, combined.ndim))
  return reduce_fn(combined, contract_axes)


def dot_general(
    lhs: jax.typing.ArrayLike,
    rhs: jax.typing.ArrayLike,
    dimension_numbers: jax.lax.DotDimensionNumbers,
    precision: jax.lax.PrecisionLike = None,
    preferred_element_type: jax.typing.DTypeLike | None = None,
    *,
    semiring: Semiring = STANDARD,
) -> jax.Array:
  """Computes a generalized dot product of two arrays over a semiring.

  Args:
    lhs: Left-hand side array-like input.
    rhs: Right-hand side array-like input.
    dimension_numbers: Contraction and batch dimension numbers.
    precision: Optional precision forwarded on the ``STANDARD`` fast path.
    preferred_element_type: Optional output dtype on the ``STANDARD`` fast path.
    semiring: Semiring specifying the combine and reduce operations.

  Returns:
    Result of the generalized dot product.
  """
  dtype = jnp.result_type(lhs, rhs)
  lhs_arr = jnp.asarray(lhs, dtype=dtype)
  rhs_arr = jnp.asarray(rhs, dtype=dtype)
  if semiring == STANDARD:
    return jax.lax.dot_general(
        lhs_arr,
        rhs_arr,
        dimension_numbers,
        precision=precision,
        preferred_element_type=preferred_element_type,
    )
  return _semiring_dot_general(
      lhs_arr,
      rhs_arr,
      dimension_numbers,
      precision=precision,
      preferred_element_type=preferred_element_type,
      combine_fn=semiring.combine,
      reduce_fn=semiring.reduce,
  )


def _rewrite_reduce_sum(
    fn: Callable[..., jax.Array],
    operands: Sequence[jax.typing.ArrayLike],
    reduce_fn: Callable[[jax.Array, int | Sequence[int]], jax.Array],
) -> jax.Array:
  """Traces fn and replaces every reduce_sum primitive with reduce_fn."""
  closed_jaxpr = jax.make_jaxpr(fn)(*operands)
  env: dict[Any, Any] = dict(zip(closed_jaxpr.jaxpr.invars, operands))
  env.update(zip(closed_jaxpr.jaxpr.constvars, closed_jaxpr.consts))

  for eqn in closed_jaxpr.jaxpr.eqns:
    invals = [
        var.val if isinstance(var, Literal) else env[var] for var in eqn.invars
    ]
    if eqn.primitive.name == "reduce_sum":
      outvals = [reduce_fn(invals[0], eqn.params["axes"])]
    else:
      try:
        bind_params = eqn.primitive.get_bind_params(eqn.params)
        if isinstance(bind_params, tuple) and len(bind_params) == 2:
          subfuns, bind_params = bind_params
        else:
          subfuns = []
      except AttributeError:
        subfuns = []
        bind_params = eqn.params
      ans = eqn.primitive.bind(*subfuns, *invals, **bind_params)
      outvals = ans if eqn.primitive.multiple_results else [ans]
    for outvar, outval in zip(eqn.outvars, outvals):
      env[outvar] = outval

  outvar = closed_jaxpr.jaxpr.outvars[0]
  return outvar.val if isinstance(outvar, Literal) else env[outvar]


def einsum(
    subscripts: str,
    *operands: jax.typing.ArrayLike,
    semiring: Semiring = STANDARD,
    optimize: str | bool | list[tuple[int, ...]] = "auto",
    precision: jax.lax.PrecisionLike = None,
) -> jax.Array:
  """Computes a generalized einsum over the given semiring.

  Args:
    subscripts: Einsum subscripts string.
    *operands: Arrays to contract.
    semiring: Semiring specifying the combine and reduce operations.
    optimize: Contraction order strategy passed to ``jnp.einsum``.
    precision: Optional precision forwarded on the ``STANDARD`` fast path.

  Returns:
    Result of the einsum contraction.
  """
  if semiring == STANDARD:
    return jnp.einsum(
        subscripts, *operands, optimize=optimize, precision=precision
    )

  # Trace with jnp.sum so reduce_sum inside semiring.reduce (e.g. logsumexp)
  # is not rewritten recursively.
  dot_fn = functools.partial(
      _semiring_dot_general, combine_fn=semiring.combine, reduce_fn=jnp.sum
  )

  def f(*args):
    return jnp.einsum(subscripts, *args, optimize=optimize, _dot_general=dot_fn)

  return _rewrite_reduce_sum(f, operands, semiring.reduce)


def _axis_name_to_dim(axis_names: str, axis_name: str) -> int | None:
  """Returns the index of axis_name in axis_names, or None if not found."""
  if axis_name in axis_names:
    return axis_names.index(axis_name)
  return None


def _get_subarrays(
    arrays: Sequence[jax.Array], ax_dims: list[int | None], index: Any
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
    arrays: Sequence of arrays.
    ax_dims: Axis index to slice for each array, or None to keep unchanged.
    index: Index to slice at.

  Returns:
    Tuple of sliced arrays.
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
    input_axes_names: list[str], arrays: Sequence[jax.Array]
) -> dict[str, int]:
  """Infer axis sizes from input axis labels and array shapes."""
  ans = {}
  for axis_names, arr in zip(input_axes_names, arrays):
    ans.update(zip(axis_names, arr.shape))
  return ans


def scan_einsum(
    formula: str,
    *arrays: jax.typing.ArrayLike,
    sequential: str = "",
    semiring: Semiring = STANDARD,
    einsum_fn: Callable[..., jax.Array] | None = None,
    **kwargs,
) -> jax.Array:
  """Executes an einsum sequentially across specified axes to bound memory.

  Args:
    formula: Explicit einsum formula of the form ``"in1,in2->out"``.
    *arrays: Input arrays to contract.
    sequential: Axis labels to loop over sequentially via ``lax.map`` /
      ``lax.scan``.
    semiring: Semiring used for accumulation and the default inner einsum.
    einsum_fn: Inner contraction callable (defaults to ``einsum`` with
      ``semiring``).
    **kwargs: Extra keyword arguments forwarded to ``einsum_fn``.

  Returns:
    The contracted array.
  """
  inner_fn = einsum_fn or functools.partial(einsum, semiring=semiring)
  arrs = tuple(jnp.asarray(a) for a in arrays)
  if not sequential:
    return inner_fn(formula, *arrs, **kwargs)

  ax = sequential[0]
  if ax not in formula:
    raise ValueError(f"Sequential axis '{ax}' not found.")

  input_str, output_axes = formula.split("->")
  input_axes = input_str.split(",")
  shapes = _infer_shapes(input_axes, arrs)
  ax_dims = [_axis_name_to_dim(names, ax) for names in input_axes]

  loop = jnp.arange(shapes[ax])
  new_formula = formula.replace(ax, "")

  def small_einsum(i):
    new_arrays = _get_subarrays(arrs, ax_dims, i)
    return scan_einsum(
        new_formula,
        *new_arrays,
        sequential=sequential[1:],
        semiring=semiring,
        einsum_fn=inner_fn,
        **kwargs,
    )

  if ax in output_axes:
    mapped = jax.lax.map(small_einsum, loop)
    return jnp.moveaxis(mapped, 0, output_axes.index(ax))

  dtype = jnp.result_type(*arrs)
  out_shape = tuple(shapes[i] for i in output_axes)
  init = jnp.full(out_shape, semiring.zero, dtype=dtype)
  return jax.lax.scan(
      lambda carry, i: (semiring.add(carry, small_einsum(i)), ()), init, loop
  )[0]
