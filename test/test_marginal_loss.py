import dataclasses
import unittest
import jax
import numpy as np
import jax.numpy as jnp
from mbi.domain import Domain
from mbi.marginal_loss import LinearMeasurement, from_linear_measurements, calculate_l2_lipschitz
from mbi.clique_vector import CliqueVector


class TestMarginalLoss(unittest.TestCase):

  def test_l2_lipschitz_disjoint(self):
    """Verify that Lipschitz constant is max(1/stddev^2) for disjoint measurements."""
    domain = Domain.fromdict({'a': 10, 'b': 10, 'c': 10})

    # Create disjoint measurements
    # m1 on ('a',) with stddev 0.5 -> 1/stddev^2 = 4.0
    # m2 on ('b',) with stddev 0.2 -> 1/stddev^2 = 25.0
    # m3 on ('c',) with stddev 1.0 -> 1/stddev^2 = 1.0

    m1 = LinearMeasurement(jnp.zeros(10), ('a',), stddev=0.5)
    m2 = LinearMeasurement(jnp.zeros(10), ('b',), stddev=0.2)
    m3 = LinearMeasurement(jnp.zeros(10), ('c',), stddev=1.0)

    measurements = [m1, m2, m3]

    # Calculate loss function and lipschitz constant
    loss_fn = from_linear_measurements(measurements, domain, norm='l2')

    calculated_L = loss_fn.lipschitz
    expected_L = max(1.0 / m.stddev**2 for m in measurements)

    # We expect calculated_L to be very close to expected_L
    # Using a slightly looser tolerance because power iteration is approximate
    self.assertAlmostEqual(calculated_L, expected_L, delta=1e-3)

  def test_l2_lipschitz_single(self):
    """Verify that Lipschitz constant is 1/stddev^2 for a single measurement."""
    domain = Domain.fromdict({'a': 10})
    m1 = LinearMeasurement(jnp.zeros(10), ('a',), stddev=0.5)
    measurements = [m1]

    loss_fn = from_linear_measurements(measurements, domain, norm='l2')

    calculated_L = loss_fn.lipschitz
    expected_L = 1.0 / m1.stddev**2

    self.assertAlmostEqual(calculated_L, expected_L, delta=1e-3)


class TestCompress(unittest.TestCase):

  def test_single_column(self):
    """Compress a one-way measurement by merging pairs."""
    domain = Domain.fromdict({'a': 6})
    y = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    m = LinearMeasurement(y, ('a',), stddev=1.0)
    mapping = {'a': np.array([0, 0, 1, 1, 2, 2])}
    mc = m.compress(mapping, domain)

    expected = np.array([30, 70, 110]) / np.sqrt(2)
    np.testing.assert_allclose(mc.noisy_measurement, expected)
    self.assertEqual(mc.stddev, 1.0)
    self.assertEqual(mc.clique, ('a',))

  def test_multi_column(self):
    """Compress one attribute of a two-way measurement."""
    domain = Domain.fromdict({'a': 4, 'b': 3})
    y = np.arange(12, dtype=float)  # shape (4, 3) flattened
    m = LinearMeasurement(y, ('a', 'b'), stddev=2.0)
    mapping = {'a': np.array([0, 0, 1, 1])}
    mc = m.compress(mapping, domain)

    y2d = y.reshape(4, 3)
    # Compressed: sum pairs along axis 0, then divide by sqrt(2)
    expected = np.vstack([y2d[0] + y2d[1], y2d[2] + y2d[3]]) / np.sqrt(2)
    np.testing.assert_allclose(mc.noisy_measurement, expected.ravel())
    self.assertEqual(mc.stddev, 2.0)

  def test_irrelevant_mapping_returns_self(self):
    domain = Domain.fromdict({'a': 4})
    m = LinearMeasurement(np.ones(4), ('a',))
    mc = m.compress({'z': np.array([0, 1])}, domain)
    self.assertIs(m, mc)

  def test_query_produces_correct_loss(self):
    """Compressed measurement should recover true compressed marginal."""
    domain = Domain.fromdict({'a': 6})
    true_counts = np.array([100.0, 150.0, 200.0, 250.0, 300.0, 350.0])
    m = LinearMeasurement(true_counts, ('a',), stddev=1.0)
    mapping = {'a': np.array([0, 0, 1, 1, 2, 2])}
    mc = m.compress(mapping, domain)

    compressed_domain = Domain.fromdict({'a': 3})
    loss_fn = from_linear_measurements([mc], compressed_domain)
    # The compressed true counts: [250, 450, 650]
    from mbi import estimation

    model = estimation.MirrorDescent(stepsize=1e-3).estimate(
        compressed_domain, loss_fn, known_total=1350.0, iters=200
    )
    result = np.asarray(model.project(('a',)).datavector())
    np.testing.assert_allclose(result, [250, 450, 650], rtol=0.01)


class TestJitCacheReuse(unittest.TestCase):
  """Regression tests for JIT cache reuse across measurement noise.

  ``stddev`` and ``lipschitz`` must be dynamic (traced) pytree leaves, not
  static aux-data.  When they were static, two otherwise-identical inputs that
  differed only in noise level produced different pytree treedefs, which
  changed the ``jax.jit`` cache key and forced a full recompilation.  This
  silently defeated ``MirrorDescent.precompile`` (its abstract placeholder
  measurements use the default ``stddev`` / ``lipschitz`` and would never match
  the real values), so the precompiled executable was always discarded and
  ``estimate`` recompiled from scratch.
  """

  def _traces_for(self, arg1, arg2):
    """Returns (traces after 1st call, traces after 2nd call).

    The body of a jitted function is executed exactly once per trace, i.e.
    once per compilation.  If ``arg2`` reuses ``arg1``'s compiled program the
    second call adds no trace.
    """
    traces = []

    @jax.jit
    def f(x):
      traces.append(None)
      return jax.tree_util.tree_leaves(x)[0].sum()

    f(arg1)
    n_after_first = len(traces)
    f(arg2)
    n_after_second = len(traces)
    return n_after_first, n_after_second

  def test_stddev_change_does_not_recompile(self):
    """Measurements differing only in ``stddev`` reuse the compiled program."""
    m1 = LinearMeasurement(jnp.zeros(4), ('a',), stddev=1.0)
    m2 = LinearMeasurement(jnp.zeros(4), ('a',), stddev=9.0)
    # Sanity check: the treedef must be identical (stddev is a leaf, not aux).
    self.assertEqual(
        jax.tree_util.tree_structure(m1), jax.tree_util.tree_structure(m2)
    )
    n_after_first, n_after_second = self._traces_for(m1, m2)
    self.assertEqual(n_after_first, n_after_second)

  def test_lipschitz_change_does_not_recompile(self):
    """Losses differing only in ``lipschitz`` reuse the compiled program."""
    domain = Domain.fromdict({'a': 4})
    loss1 = from_linear_measurements(
        [LinearMeasurement(jnp.zeros(4), ('a',), stddev=1.0)], domain
    )
    loss2 = dataclasses.replace(loss1, lipschitz=loss1.lipschitz * 3.0)
    self.assertEqual(
        jax.tree_util.tree_structure(loss1),
        jax.tree_util.tree_structure(loss2),
    )
    n_after_first, n_after_second = self._traces_for(loss1, loss2)
    self.assertEqual(n_after_first, n_after_second)


if __name__ == '__main__':
  unittest.main()
