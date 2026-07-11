import itertools
import unittest

import numpy as np
from parameterized import parameterized

from mbi import Domain, estimation, marginal_loss
from mbi.clique_vector import CliqueVector
from mbi.factor import Factor

np.random.seed(0)  # Avoid flaky tests

_DOMAIN = Domain(["a", "b", "c", "d"], [2, 3, 4, 5])

_CLIQUE_SETS = [
    [("a", "b"), ("b", "c"), ("c", "d")],  # tree
    [("a",), ("a", "b"), ("b", "c"), ("a", "c"), ("b", "d")],  # cyclic
    [("a", "b"), ("d", "a")],  # missing c
    [("a", "b", "c", "d")],  # full materialization
    [("d",)],  # singleton
    [("a", "b", "c"), ("c", "b", "a"), ("b", "d")],  # (permuted) duplicates
    [],  # trivial empty set
]


def fake_measurements(cliques):
  P = Factor.random(_DOMAIN)
  P = P / P.sum()
  measurements = []
  for cl in cliques:
    y = P.project(cl).datavector()
    measurements.append(marginal_loss.LinearMeasurement(y, cl))
  return measurements


class TestEstimation(unittest.TestCase):

  @parameterized.expand(itertools.product(_CLIQUE_SETS))
  def test_total_estimator(self, cliques):
    measurements = fake_measurements(cliques)
    total = estimation.minimum_variance_unbiased_total(measurements)
    np.testing.assert_allclose(total, 1.0, rtol=1e-5)

  @parameterized.expand(itertools.product(_CLIQUE_SETS))
  def test_mirror_descent(self, cliques):
    measurements = fake_measurements(cliques)
    loss_fn = marginal_loss.from_linear_measurements(measurements, _DOMAIN)

    model = estimation.MirrorDescent().estimate(
        _DOMAIN, loss_fn, known_total=1.0, iters=250
    )
    for M in measurements:
      expected = M.noisy_measurement
      actual = model.project(M.clique).datavector()
      np.testing.assert_allclose(actual, expected, atol=1e-2)

  @parameterized.expand(itertools.product(_CLIQUE_SETS))
  def test_mirror_descent_l1(self, cliques):
    measurements = fake_measurements(cliques)
    loss_fn = marginal_loss.from_linear_measurements(
        measurements, _DOMAIN, norm="l1"
    )

    model = estimation.MirrorDescent(stepsize=0.01).estimate(
        _DOMAIN, loss_fn, known_total=1.0, iters=250
    )
    for M in measurements:
      expected = M.noisy_measurement
      actual = model.project(M.clique).datavector()
      np.testing.assert_allclose(actual, expected, atol=1e-2)

  @parameterized.expand(itertools.product(_CLIQUE_SETS))
  def test_multiplicative_weights(self, cliques):
    measurements = fake_measurements(cliques)
    potentials = CliqueVector.zeros(_DOMAIN, [_DOMAIN.attributes])
    loss_fn = marginal_loss.from_linear_measurements(measurements, _DOMAIN)
    model = estimation.MirrorDescent().estimate(
        _DOMAIN, loss_fn, known_total=1.0, potentials=potentials, iters=250
    )

    for M in measurements:
      expected = M.noisy_measurement
      actual = model.project(M.clique).datavector()
      np.testing.assert_allclose(actual, expected, atol=1e-2)

  @parameterized.expand(itertools.product(_CLIQUE_SETS))
  def test_dual_averaging(self, cliques):
    measurements = fake_measurements(cliques)

    model = estimation.DualAveraging().estimate(
        _DOMAIN, measurements, known_total=1.0, iters=250
    )
    for M in measurements:
      expected = M.noisy_measurement
      actual = model.project(M.clique).datavector()
      np.testing.assert_allclose(actual, expected, atol=1e-2)

  @parameterized.expand(itertools.product(_CLIQUE_SETS))
  def test_interior_gradient(self, cliques):
    measurements = fake_measurements(cliques)

    model = estimation.InteriorGradient().estimate(
        _DOMAIN, measurements, known_total=1.0, iters=250
    )
    for M in measurements:
      expected = M.noisy_measurement
      actual = model.project(M.clique).datavector()

  @parameterized.expand(itertools.product(_CLIQUE_SETS))
  def test_mle(self, cliques):
    P = Factor.random(_DOMAIN) * 10
    total = float(P.sum())
    mu = CliqueVector(_DOMAIN, cliques, {cl: P.project(cl) for cl in cliques})

    loss = marginal_loss.mle_loss_fn(mu)
    model = estimation.LBFGS().estimate(_DOMAIN, loss, known_total=total)
    for cl in cliques:
      expected = mu.project(cl).datavector()
      actual = model.project(cl).datavector()
      np.testing.assert_allclose(actual, expected, atol=100 / total)

  def test_precompile(self):
    """precompile() should complete without error."""
    cliques = [("a", "b"), ("b", "c")]
    measurements = fake_measurements(cliques)
    est = estimation.MirrorDescent()
    future = est.precompile(_DOMAIN, measurements)
    future.result()  # Should not raise.

  def test_precompile_with_extra_cliques(self):
    """precompile() with both measurements and extra_cliques."""
    cliques = [("a", "b"), ("b", "c")]
    measurements = fake_measurements(cliques)
    est = estimation.MirrorDescent()
    future = est.precompile(_DOMAIN, measurements, extra_cliques=[("c", "d")])
    future.result()  # Should not raise.

  def test_precompile_extra_cliques_only(self):
    """precompile() with only extra_cliques (no concrete measurements)."""
    est = estimation.MirrorDescent()
    future = est.precompile(_DOMAIN, extra_cliques=[("a", "b"), ("b", "c")])
    future.result()  # Should not raise.

  def test_precompile_cache_hit(self):
    """precompile() warms the cache so estimate() doesn't recompile."""
    import jax

    cliques = [("a", "b"), ("b", "c")]
    measurements = fake_measurements(cliques)
    loss_fn = marginal_loss.from_linear_measurements(measurements, _DOMAIN)
    est = estimation.MirrorDescent()
    future = est.precompile(_DOMAIN, measurements)
    future.result()

    compiled_count = 0
    original_lower = jax.stages.Lowered.compile

    def counting_compile(self_lowered, *args, **kwargs):
      nonlocal compiled_count
      compiled_count += 1
      return original_lower(self_lowered, *args, **kwargs)

    jax.stages.Lowered.compile = counting_compile
    try:
      est.estimate(_DOMAIN, loss_fn, known_total=100.0, iters=10)
    finally:
      jax.stages.Lowered.compile = original_lower
    self.assertEqual(compiled_count, 0, "Expected cache hit after precompile")

  @parameterized.expand([
      estimation.MirrorDescent,
      estimation.DualAveraging,
      estimation.InteriorGradient,
      estimation.LBFGS,
      estimation.UniversalAcceleratedMethod,
  ])
  def test_estimator_with_constraints(self, estimator_cls):
    """Estimator with constraints produces valid marginals."""
    from mbi import Constraint

    domain = Domain(["x", "xp", "y"], [4, 2, 3])
    mapping = np.array([0, 0, 1, 1])
    c = Constraint(domain=domain.project(("x", "xp")), mapping=mapping)

    cliques = [("x", "y")]
    P = Factor.random(domain.project(("x", "y")))
    P = P / P.sum()
    measurements = []
    for cl in cliques:
      x = P.project(cl).datavector()
      measurements.append(marginal_loss.LinearMeasurement(x, cl, 1.0))
    loss_fn = marginal_loss.from_linear_measurements(measurements, domain)

    est = estimator_cls()
    model = est.estimate(
        domain, loss_fn, known_total=1.0, iters=100, constraints=[c]
    )

    # Marginals should only contain the input cliques.
    self.assertEqual(set(model.cliques), set(cliques))
    # Total should be preserved.
    np.testing.assert_allclose(
        model.project(("x",)).datavector().sum(), 1.0, atol=1e-4
    )

  @parameterized.expand([
      estimation.MirrorDescent,
      estimation.DualAveraging,
      estimation.InteriorGradient,
      estimation.LBFGS,
      estimation.UniversalAcceleratedMethod,
  ])
  def test_warm_start_expanding_cliques(self, estimator_cls):
    """Warm-starting with new cliques produces a valid model."""
    cliques1 = [("a", "b")]
    measurements1 = fake_measurements(cliques1)

    est = estimator_cls()
    model1 = est.estimate(_DOMAIN, measurements1, known_total=1.0, iters=50)

    # Round 2 adds a new clique.
    cliques2 = [("a", "b"), ("b", "c")]
    measurements2 = fake_measurements(cliques2)
    model2 = est.estimate(
        _DOMAIN,
        measurements2,
        known_total=1.0,
        iters=50,
        warm_start=model1,
    )

    np.testing.assert_allclose(
        model2.project(("a",)).datavector().sum(), 1.0, atol=1e-4
    )
    # Model 2 should have the expanded clique set.
    self.assertTrue(
        set(cliques2).issubset({_DOMAIN.canonical(c) for c in model2.cliques})
    )

  def test_warm_start_legacy_potentials(self):
    """Legacy potentials= kwarg still works for backwards compatibility."""
    cliques = [("a", "b"), ("b", "c")]
    measurements = fake_measurements(cliques)
    loss_fn = marginal_loss.from_linear_measurements(measurements, _DOMAIN)

    est = estimation.MirrorDescent()
    model1 = est.estimate(_DOMAIN, loss_fn, known_total=1.0, iters=50)

    # Warm-start via legacy potentials= kwarg.
    model2 = est.estimate(
        _DOMAIN,
        loss_fn,
        known_total=1.0,
        iters=50,
        potentials=model1.potentials,
    )

    np.testing.assert_allclose(
        model2.project(("a",)).datavector().sum(), 1.0, atol=1e-4
    )

  def test_warm_start_mixture_of_products(self):
    """MixtureOfProducts warm-start reuses the model directly."""
    from mbi.extensions.mixture_of_products import (
        MixtureOfProductsEstimator,
    )

    cliques = [("a", "b"), ("b", "c")]
    measurements = fake_measurements(cliques)
    loss_fn = marginal_loss.from_linear_measurements(measurements, _DOMAIN)

    est = MixtureOfProductsEstimator(num_components=5)
    model1 = est.estimate(_DOMAIN, loss_fn, known_total=1.0, iters=50)
    loss1 = loss_fn(model1)

    # Warm-starting should improve on the cold-start loss.
    model2 = est.estimate(
        _DOMAIN,
        loss_fn,
        known_total=1.0,
        iters=50,
        warm_start=model1,
    )
    loss2 = loss_fn(model2)
    self.assertLessEqual(float(loss2), float(loss1) + 1e-6)
