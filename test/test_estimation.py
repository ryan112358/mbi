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
        mu = CliqueVector(
            _DOMAIN, cliques, {cl: P.project(cl) for cl in cliques}
        )

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
