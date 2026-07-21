"""Tests for memory-bounded row-chunking in synthetic data generation.

Verifies that ``_stochastic_round`` produces identical rounding *contracts*
whether it runs the dense path or the row-chunked ``lax.scan`` path, and that
end-to-end generation still matches the dense reference when chunking is forced.
"""

import unittest
from unittest import mock

import importlib

import jax
import jax.numpy as jnp
import numpy as np

from mbi import Domain, Factor, CliqueVector, MarkovRandomField, marginal_oracles

# ``mbi.extensions`` re-exports the ``synthetic_data`` *function*, which shadows
# the submodule of the same name; import the module object explicitly.
sd = importlib.import_module('mbi.extensions.synthetic_data')


def _random_cond_probs(parent_product, domain_size, seed=0):
  rng = np.random.default_rng(seed)
  m = rng.random((parent_product, domain_size)) + 1e-6
  return jnp.asarray(m / m.sum(axis=1, keepdims=True))


class TestStochasticRoundChunking(unittest.TestCase):

  def _assert_round_contract(self, integ, cond_probs, counts):
    """The rounding contract that BOTH dense and chunked paths must satisfy."""
    integ = np.asarray(integ)
    counts = np.asarray(counts)
    expected = counts[:, None] * np.asarray(cond_probs)
    floor = np.floor(expected).astype(np.int32)
    # 1. Per-parent totals are preserved exactly.
    np.testing.assert_array_equal(integ.sum(axis=1), counts)
    # 2. Each cell is floor(expected) or floor(expected)+1.
    self.assertTrue(np.all(integ >= floor))
    self.assertTrue(np.all(integ <= floor + 1))
    # 3. Non-negative int32.
    self.assertEqual(integ.dtype, np.int32)
    self.assertTrue(np.all(integ >= 0))

  def test_chunked_matches_contract_and_exercises_scan(self):
    parent_product, domain_size = 5000, 8  # 40k cells
    cond_probs = _random_cond_probs(parent_product, domain_size)
    counts = jnp.asarray(
        np.random.default_rng(1).integers(0, 50, size=parent_product)
    )
    rng = jax.random.PRNGKey(0)

    dense = sd._stochastic_round_dense(rng, cond_probs, counts)
    self._assert_round_contract(dense, cond_probs, counts)

    # Force chunking with a tiny threshold so n_chunks > 1.
    with mock.patch.object(sd, '_ROUND_CHUNK_CELLS', 8 * 64):  # 64 rows/chunk
      chunked = sd._stochastic_round(rng, cond_probs, counts)
    self._assert_round_contract(chunked, cond_probs, counts)

    # Global total identical across both paths.
    self.assertEqual(
        int(np.asarray(dense).sum()), int(np.asarray(chunked).sum())
    )

  def test_dense_path_used_below_threshold(self):
    parent_product, domain_size = 100, 4
    cond_probs = _random_cond_probs(parent_product, domain_size)
    counts = jnp.asarray(np.full(parent_product, 10))
    rng = jax.random.PRNGKey(0)
    # Below threshold -> dense path -> bit-identical to _stochastic_round_dense.
    out = sd._stochastic_round(rng, cond_probs, counts)
    ref = sd._stochastic_round_dense(rng, cond_probs, counts)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(ref))

  def test_end_to_end_generation_with_forced_chunking(self):
    domain = Domain(['a', 'b', 'c'], [12, 10, 8])
    cliques = [('a', 'b'), ('b', 'c')]
    potentials = {}
    rng = np.random.default_rng(0)
    for cl in cliques:
      vals = rng.random(domain.project(cl).shape) + 1e-10
      potentials[cl] = Factor(domain.project(cl), vals).log()
    pv = CliqueVector(domain, cliques, potentials)
    n = 20000
    marginals = marginal_oracles.message_passing_stable(pv, total=n)
    model = MarkovRandomField(potentials=pv, marginals=marginals, total=n)

    dense_df = sd.synthetic_data(model, rows=n, seed=3)
    with mock.patch.object(sd, '_ROUND_CHUNK_CELLS', 16):  # force chunking
      chunk_df = sd.synthetic_data(model, rows=n, seed=3)

    # Both must yield exactly `n` rows over the correct columns.
    self.assertEqual(dense_df.records, n)
    self.assertEqual(chunk_df.records, n)
    # 1-way marginals should agree closely (same rounding contract, diff RNG).
    for col in domain.attributes:
      d = np.asarray(dense_df.project(col).datavector())
      c = np.asarray(chunk_df.project(col).datavector())
      # Total mass identical; distribution close.
      self.assertEqual(d.sum(), c.sum())
      np.testing.assert_allclose(d / d.sum(), c / c.sum(), atol=0.03)


if __name__ == '__main__':
  unittest.main()
