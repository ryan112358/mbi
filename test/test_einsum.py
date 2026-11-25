import unittest

import jax
import jax.numpy as jnp
from mbi.einsum import (
    LOG_SUM_EXP,
    MAX_PLUS,
    STANDARD,
    Semiring,
    dot_general,
    einsum,
    scan_einsum,
)
import numpy as np
from parameterized import parameterized

np.random.seed(0)

_MIN_MAX = Semiring(
    combine=jnp.maximum, reduce=jnp.min, add=jnp.minimum, zero=jnp.inf
)
_MAX_PRODUCT = Semiring(
    combine=jnp.multiply, reduce=jnp.max, add=jnp.maximum, zero=0.0
)


class TestDotGeneral(unittest.TestCase):

  def test_standard_vector_vector_inner_product(self):
    lhs = np.array([1.0, 2.0, 3.0])
    rhs = np.array([4.0, 5.0, 6.0])
    dims = (([0], [0]), ([], []))
    np.testing.assert_allclose(dot_general(lhs, rhs, dims), np.array(32.0))

  def test_standard_matrix_vector_product(self):
    lhs = np.array([[1.0, 2.0], [3.0, 4.0]])
    rhs = np.array([5.0, 6.0])
    dims = (([1], [0]), ([], []))
    np.testing.assert_allclose(
        dot_general(lhs, rhs, dims), np.array([17.0, 39.0])
    )

  def test_standard_matrix_matrix_product(self):
    lhs = np.array([[1.0, 2.0], [3.0, 4.0]])
    rhs = np.array([[5.0, 6.0], [7.0, 8.0]])
    dims = (([1], [0]), ([], []))
    np.testing.assert_allclose(
        dot_general(lhs, rhs, dims), np.array([[19.0, 22.0], [43.0, 50.0]])
    )

  def test_standard_batch_matrix_matrix(self):
    lhs = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    rhs = np.array([[[1.0, 0.0], [0.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]])
    expected = np.array(
        [[[1.0, 2.0], [3.0, 4.0]], [[11.0, 11.0], [15.0, 15.0]]]
    )
    dims = (([2], [1]), ([0], [0]))
    np.testing.assert_allclose(dot_general(lhs, rhs, dims), expected)

  def test_log_space_vector_vector(self):
    lhs = np.log(np.array([1.0, 2.0]))
    rhs = np.log(np.array([3.0, 4.0]))
    dims = (([0], [0]), ([], []))
    res = dot_general(lhs, rhs, dims, semiring=LOG_SUM_EXP)
    np.testing.assert_allclose(res, np.log(11.0))

  def test_custom_max_min_product(self):
    lhs = np.array([[1, 6], [5, 2]])
    rhs = np.array([[4, 3], [0, 7]])
    expected = np.array([[4, 3], [2, 5]])
    dims = (([1], [0]), ([], []))
    np.testing.assert_equal(
        dot_general(lhs, rhs, dims, semiring=_MIN_MAX), expected
    )

  def test_outer_product_no_contraction(self):
    lhs = np.array([1, 2])
    rhs = np.array([3, 4, 5])
    expected = np.array([[3, 4, 5], [6, 8, 10]])
    dims = (([], []), ([], []))
    np.testing.assert_equal(dot_general(lhs, rhs, dims), expected)

  def test_scalar_inputs_default_ops(self):
    lhs_s = np.array(2.0)
    rhs_s = np.array(3.0)
    rhs_v = np.array([3.0, 4.0])
    dims_ss = (([], []), ([], []))

    res_ss = dot_general(lhs_s, rhs_s, dims_ss)
    np.testing.assert_allclose(res_ss, np.array(6.0))
    self.assertEqual(res_ss.ndim, 0)

    np.testing.assert_allclose(
        dot_general(lhs_s, rhs_v, dims_ss), np.array([6.0, 8.0])
    )
    np.testing.assert_allclose(
        dot_general(rhs_v, lhs_s, dims_ss), np.array([6.0, 8.0])
    )

    lhs_v = np.array([2.0, 3.0])
    rhs_v2 = np.array([4.0, 5.0])
    res_contract = dot_general(lhs_v, rhs_v2, (([0], [0]), ([], [])))
    np.testing.assert_allclose(res_contract, np.array(23.0))
    self.assertEqual(res_contract.ndim, 0)

  def test_empty_contract_batch_dims(self):
    lhs = np.array([[1, 2], [3, 4]])
    rhs = np.array([[5, 6], [7, 8]])
    expected = np.einsum("ab,cd->abcd", lhs, rhs)
    dims = (([], []), ([], []))
    np.testing.assert_equal(dot_general(lhs, rhs, dims), expected)

  def test_complex_case_multiple_dims(self):
    rng = np.random.default_rng(0)
    lhs = rng.random((2, 3, 2, 2, 3))
    rhs = rng.random((2, 4, 2, 2, 3))
    dims = (((2, 3, 4), (2, 3, 4)), ((0,), (0,)))
    expected = np.einsum("bmcde,bncde->bmn", lhs, rhs)
    np.testing.assert_allclose(dot_general(lhs, rhs, dims), expected, rtol=1e-6)


class TestScanEinsum(unittest.TestCase):

  def assertArraysAllClose(self, arr1, arr2, msg=None, rtol=1e-5, atol=1e-5):
    self.assertTrue(jnp.allclose(arr1, arr2, rtol=rtol, atol=atol), msg=msg)

  def test_no_sequential(self):
    a = jnp.arange(6, dtype=jnp.float64).reshape(2, 3)
    b = jnp.arange(12, dtype=jnp.float64).reshape(3, 4)
    self.assertArraysAllClose(
        jnp.einsum("ij,jk->ik", a, b), scan_einsum("ij,jk->ik", a, b)
    )

    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([4.0, 5.0, 6.0])
    self.assertArraysAllClose(
        jnp.einsum("i,i->", x, y), scan_einsum("i,i->", x, y)
    )
    self.assertArraysAllClose(
        jnp.einsum("i,j->ij", x, y), scan_einsum("i,j->ij", x, y)
    )
    self.assertArraysAllClose(jnp.einsum("ij->ji", a), scan_einsum("ij->ji", a))

    c = jnp.arange(8, dtype=jnp.float64).reshape(2, 4)
    self.assertArraysAllClose(
        jnp.einsum("ij,jk,il->kl", a, b, c),
        scan_einsum("ij,jk,il->kl", a, b, c),
    )

  def test_single_sequential_summed_out(self):
    a = jnp.arange(2 * 3 * 4, dtype=jnp.float64).reshape(2, 3, 4)
    b = jnp.arange(3 * 4 * 5, dtype=jnp.float64).reshape(3, 4, 5)
    expected = jnp.einsum("ijk,jkl->il", a, b)
    for ax in ("j", "k"):
      self.assertArraysAllClose(
          expected, scan_einsum("ijk,jkl->il", a, b, sequential=ax)
      )

    x = jnp.arange(2 * 1 * 3, dtype=jnp.float64).reshape(2, 1, 3)
    y = jnp.arange(1 * 3 * 4, dtype=jnp.float64).reshape(1, 3, 4)
    expected_xy = jnp.einsum("ijk,jkl->il", x, y)
    for ax in ("j", "k"):
      self.assertArraysAllClose(
          expected_xy, scan_einsum("ijk,jkl->il", x, y, sequential=ax)
      )

  def test_single_sequential_in_output(self):
    a = jnp.arange(6, dtype=jnp.float64).reshape(2, 3)
    b = jnp.arange(6, dtype=jnp.float64).reshape(2, 3)
    expected = jnp.einsum("ij,ij->ij", a, b)
    for ax in ("i", "j"):
      self.assertArraysAllClose(
          expected, scan_einsum("ij,ij->ij", a, b, sequential=ax)
      )

    x = jnp.array([1.0, 2.0])
    y = jnp.array([10.0, 20.0, 30.0])
    expected_bc = jnp.einsum("i,j->ij", x, y)
    for ax in ("i", "j"):
      self.assertArraysAllClose(
          expected_bc, scan_einsum("i,j->ij", x, y, sequential=ax)
      )

    p = jnp.arange(24, dtype=jnp.float64).reshape(2, 3, 4)
    q = jnp.arange(40, dtype=jnp.float64).reshape(2, 4, 5)
    self.assertArraysAllClose(
        jnp.einsum("bij,bjk->bik", p, q),
        scan_einsum("bij,bjk->bik", p, q, sequential="b"),
    )

  def test_multiple_sequential_axes(self):
    a = jnp.arange(24, dtype=jnp.float64).reshape(2, 3, 4)
    b = jnp.arange(60, dtype=jnp.float64).reshape(3, 4, 5)
    c = jnp.arange(60, dtype=jnp.float64).reshape(2, 5, 6)
    expected = jnp.einsum("ijk,jkl,ilm->km", a, b, c)
    for seq in ("ij", "ji", "il", "jl"):
      self.assertArraysAllClose(
          expected, scan_einsum("ijk,jkl,ilm->km", a, b, c, sequential=seq)
      )

    x = jnp.arange(24, dtype=jnp.float64).reshape(2, 3, 4)
    y = jnp.arange(3, dtype=jnp.float64)
    self.assertArraysAllClose(
        jnp.einsum("ijk,j->ik", x, y),
        scan_einsum("ijk,j->ik", x, y, sequential="j"),
    )

    t1 = jnp.array(np.random.rand(2, 3, 4), dtype=jnp.float64)
    t2 = jnp.array(np.random.rand(2, 4, 5), dtype=jnp.float64)
    t3 = jnp.array(np.random.rand(2, 5, 3), dtype=jnp.float64)
    expected_abc = jnp.einsum("aij,bjk,ckl->abil", t1, t2, t3)
    for seq in ("jk", "ab", "aj"):
      self.assertArraysAllClose(
          expected_abc,
          scan_einsum("aij,bjk,ckl->abil", t1, t2, t3, sequential=seq),
      )

    d = jnp.arange(8, dtype=jnp.float64).reshape(2, 2, 2)
    expected_all = jnp.einsum("ijk,jkl,kli->", d, d, d)
    for seq in ("ijk", "ikj"):
      self.assertArraysAllClose(
          expected_all, scan_einsum("ijk,jkl,kli->", d, d, d, sequential=seq)
      )

  def test_chain_product_axes(self):
    a = jnp.array(np.random.rand(2, 3, 4), dtype=jnp.float64)
    b = jnp.array(np.random.rand(2, 4, 5), dtype=jnp.float64)
    c = jnp.array(np.random.rand(2, 5, 6), dtype=jnp.float64)
    expected = jnp.einsum("bij,bjk,bkl->bil", a, b, c)
    for seq in ("b", "j", "k", "jk", "bj"):
      self.assertArraysAllClose(
          expected, scan_einsum("bij,bjk,bkl->bil", a, b, c, sequential=seq)
      )

  def test_output_only_formula(self):
    x = jnp.arange(5, dtype=jnp.float64)
    self.assertArraysAllClose(
        jnp.einsum("a->a", x), scan_einsum("a->a", x, sequential="a")
    )
    self.assertArraysAllClose(
        jnp.einsum("a->", x), scan_einsum("a->", x, sequential="a")
    )

  def test_sequential_axis_not_in_all_arrays(self):
    a = jnp.arange(6, dtype=jnp.float64).reshape(2, 3)
    b = jnp.arange(12, dtype=jnp.float64).reshape(3, 4)
    expected = jnp.einsum("ij,jk->ik", a, b)
    for seq in ("i", "j", "k"):
      self.assertArraysAllClose(
          expected, scan_einsum("ij,jk->ik", a, b, sequential=seq)
      )

    x = jnp.arange(6, dtype=jnp.float64).reshape(2, 3)
    y = jnp.arange(12, dtype=jnp.float64).reshape(3, 4)
    z = jnp.arange(20, dtype=jnp.float64).reshape(4, 5)
    expected_xyz = jnp.einsum("ab,bc,cd->ad", x, y, z)
    for seq in ("a", "b", "c", "d", "ac"):
      self.assertArraysAllClose(
          expected_xyz, scan_einsum("ab,bc,cd->ad", x, y, z, sequential=seq)
      )

  def test_output_axis_at_position_ge_2_preserves_axis_order(self):
    rng = np.random.default_rng(0)
    a = jnp.asarray(rng.normal(size=(2, 3)))
    b = jnp.asarray(rng.normal(size=(3, 4)))
    c = jnp.asarray(rng.normal(size=(4, 5)))
    expected = jnp.einsum("ab,bc,cd->acd", a, b, c)
    actual = scan_einsum("ab,bc,cd->acd", a, b, c, sequential="d")
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def _direct_semiring(subscripts, *operands, semiring):
  """Reference broadcast-combine and reduce over the joint domain."""
  in_str, out_str = subscripts.split("->")
  in_terms = in_str.split(",")
  sizes = {
      ch: dim
      for term, arr in zip(in_terms, operands)
      for ch, dim in zip(term, arr.shape)
  }
  all_axes = sorted(set().union(*(set(t) for t in in_terms), set(out_str)))
  pos = {ch: i for i, ch in enumerate(all_axes)}

  combined = None
  for term, arr in zip(in_terms, operands):
    if term:
      order = sorted(range(len(term)), key=lambda i: pos[term[i]])
      aligned = jnp.transpose(arr, order)
      bcast = tuple(sizes[ch] if ch in term else 1 for ch in all_axes)
      expanded = jnp.reshape(aligned, bcast)
    else:
      expanded = jnp.reshape(arr, (1,) * len(all_axes))
    combined = (
        expanded if combined is None else semiring.combine(combined, expanded)
    )

  elim = tuple(i for i, ch in enumerate(all_axes) if ch not in out_str)
  reduced = semiring.reduce(combined, elim) if elim else combined
  kept = [ch for ch in all_axes if ch in out_str]
  return jnp.transpose(reduced, [kept.index(ch) for ch in out_str])


_SEMIRINGS = [
    ("standard", STANDARD),
    ("logsumexp", LOG_SUM_EXP),
    ("max_plus", MAX_PLUS),
    ("max_product", _MAX_PRODUCT),
]

_LOG_SEMIRINGS = [
    ("logsumexp", LOG_SUM_EXP),
    ("max_plus", MAX_PLUS),
]

_TOPOLOGY_CASES = [
    ("chain5", "ab,bc,cd,de->ae", {"a": 3, "b": 4, "c": 5, "d": 3, "e": 4}),
    ("cycle4", "ab,bc,cd,ad->bd", {"a": 3, "b": 4, "c": 5, "d": 3}),
    ("k4", "ab,ac,ad,bc,bd,cd->ab", {"a": 3, "b": 4, "c": 3, "d": 5}),
    ("outer_reduce", "ab,c->ac", {"a": 3, "b": 4, "c": 5}),
    ("broadcast_mul", "ab,b->ab", {"a": 3, "b": 4}),
    ("scalar_mix", ",ab,bc->ac", {"a": 3, "b": 4, "c": 5}),
    ("single_reduce", "abc->a", {"a": 3, "b": 4, "c": 5}),
]


class TestSemiringEinsum(unittest.TestCase):

  def test_semirings_are_hashable(self):
    self.assertEqual(len({STANDARD, LOG_SUM_EXP, MAX_PLUS}), 3)

  def test_vmap_and_jit(self):
    f = jax.jit(
        jax.vmap(lambda x, y: einsum("ij,jk->ik", x, y, semiring=MAX_PLUS))
    )
    x = jnp.array([[[1.0, 2.0], [3.0, 4.0]]])
    y = jnp.array([[[5.0, 6.0], [7.0, 8.0]]])
    expected = jnp.array([[[9.0, 10.0], [11.0, 12.0]]])
    np.testing.assert_allclose(f(x, y), expected)

  @parameterized.expand([
      (f"{t_name}_{s_name}", sub, sizes, sr)
      for t_name, sub, sizes in _TOPOLOGY_CASES
      for s_name, sr in _SEMIRINGS
  ])
  def test_einsum_matches_direct_reference(
      self, _name, subscripts, sizes, semiring
  ):
    rng = np.random.default_rng(0)
    in_terms = subscripts.split("->")[0].split(",")
    operands = [
        jnp.asarray(rng.normal(size=tuple(sizes[c] for c in t)))
        for t in in_terms
    ]
    if semiring is _MAX_PRODUCT:
      # Max-product is only distributive over non-negative values.
      operands = [jnp.abs(x) for x in operands]
    actual = einsum(subscripts, *operands, semiring=semiring)
    expected = _direct_semiring(subscripts, *operands, semiring=semiring)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

  @parameterized.expand(["auto", "greedy", "optimal"])
  def test_optimize_strategies_match(self, optimize):
    rng = np.random.default_rng(1)
    operands = [
        jnp.asarray(rng.normal(size=s))
        for s in [(3, 4), (4, 5), (5, 3), (3, 4)]
    ]
    actual = einsum(
        "ab,bc,cd,de->ae", *operands, semiring=LOG_SUM_EXP, optimize=optimize
    )
    expected = _direct_semiring(
        "ab,bc,cd,de->ae", *operands, semiring=LOG_SUM_EXP
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

  @parameterized.expand(_LOG_SEMIRINGS)
  def test_neg_inf_constraints_match_direct(self, _name, semiring):
    a = jnp.array([[0.0, -jnp.inf, 1.0], [-jnp.inf, -jnp.inf, -jnp.inf]])
    b = jnp.array([[-jnp.inf, 2.0], [3.0, -jnp.inf], [-jnp.inf, 4.0]])
    c = jnp.array([0.5, -jnp.inf])
    actual = einsum("ij,jk,k->ik", a, b, c, semiring=semiring)
    expected = _direct_semiring("ij,jk,k->ik", a, b, c, semiring=semiring)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

  def test_empty_contraction_skips_reduction_and_preserves_neg_inf_grad(self):
    def loss(u):
      out = einsum("a,b->ab", u, jnp.array([1.0, 2.0]), semiring=LOG_SUM_EXP)
      return jnp.sum(jnp.where(jnp.isfinite(out), out, 0.0))

    grad = jax.grad(loss)(jnp.array([0.0, -jnp.inf]))
    np.testing.assert_allclose(grad, jnp.array([2.0, 0.0]))

    jaxpr = jax.make_jaxpr(
        lambda x, y: dot_general(
            x, y, (([], []), ([], [])), semiring=LOG_SUM_EXP
        )
    )(jnp.ones((2,)), jnp.ones((3,)))
    prims = [eqn.primitive.name for eqn in jaxpr.jaxpr.eqns]
    self.assertNotIn("reduce_sum", prims)

  @parameterized.expand([
      ("logsumexp_elim", "b", LOG_SUM_EXP),
      ("logsumexp_kept", "d", LOG_SUM_EXP),
      ("logsumexp_mixed", "bd", LOG_SUM_EXP),
      ("logsumexp_all", "bcad", LOG_SUM_EXP),
      ("max_elim", "b", MAX_PLUS),
      ("max_kept", "d", MAX_PLUS),
      ("max_mixed", "bd", MAX_PLUS),
      ("max_prod_mixed", "bd", _MAX_PRODUCT),
  ])
  def test_scan_einsum_matches_direct(self, _name, sequential, semiring):
    rng = np.random.default_rng(2)
    operands = [
        jnp.asarray(rng.normal(size=s)) for s in [(2, 3), (3, 4), (4, 5)]
    ]
    if semiring is _MAX_PRODUCT:
      operands = [jnp.abs(x) for x in operands]
    actual = scan_einsum(
        "ab,bc,cd->acd", *operands, sequential=sequential, semiring=semiring
    )
    expected = _direct_semiring("ab,bc,cd->acd", *operands, semiring=semiring)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

  @parameterized.expand(_LOG_SEMIRINGS)
  def test_scan_einsum_handles_neg_inf(self, _name, semiring):
    a = jnp.array([[0.0, -jnp.inf, 1.0], [-jnp.inf, -jnp.inf, -jnp.inf]])
    b = jnp.array([[-jnp.inf, 2.0], [3.0, -jnp.inf], [-jnp.inf, 4.0]])
    actual = scan_einsum("ij,jk->ik", a, b, sequential="j", semiring=semiring)
    expected = _direct_semiring("ij,jk->ik", a, b, semiring=semiring)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
  unittest.main(argv=["first-arg-is-ignored"], exit=False)
