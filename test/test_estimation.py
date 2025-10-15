import unittest
from mbi.domain import Domain
from mbi.factor import Factor
from mbi.clique_vector import CliqueVector
from mbi import marginal_loss, estimation, relaxed_projection
import numpy as np
from parameterized import parameterized
import itertools

np.random.seed(0)  # Avoid flaky tests

_DOMAIN = Domain(["a", "b", "c", "d"], [2, 3, 4, 5])

_CLIQUE_SETS = [
    [("a", "b"), ("b", "c"), ("c", "d")],  # tree
    [("a",), ("a", "b"), ("b", "c"), ("a", "c"), ("b", "d")],  # cyclic
    [("a", "b"), ("d", "a")],  # missing c
    [("a", "b", "c", "d")],  # full materialization
    [("d",)],  # singleton
    [("a", "b", "c"), ("c", "b", "a"), ("b", "d")],  # (permuted) duplicates
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
        np.testing.assert_allclose(total, 1.0)

    @parameterized.expand(itertools.product(_CLIQUE_SETS))
    def test_mirror_descent(self, cliques):
        measurements = fake_measurements(cliques)
        loss_fn = marginal_loss.from_linear_measurements(measurements)

        model = estimation.mirror_descent(_DOMAIN, loss_fn, known_total=1.0, iters=250)
        for M in measurements:
            expected = M.noisy_measurement
            actual = model.project(M.clique).datavector()
            np.testing.assert_allclose(actual, expected, atol=1e-2)

    @parameterized.expand(itertools.product(_CLIQUE_SETS))
    def test_universal_accelerated_method(self, cliques):
        measurements = fake_measurements(cliques)
        loss_fn = marginal_loss.from_linear_measurements(measurements)

        model = estimation.universal_accelerated_method(
                _DOMAIN, loss_fn, known_total=1.0, iters=250
        )
        for M in measurements:
            expected = M.noisy_measurement
            actual = model.project(M.clique).datavector()
            np.testing.assert_allclose(actual, expected, atol=1e-2)

    @parameterized.expand(itertools.product(_CLIQUE_SETS))
    def test_mirror_descent_l1(self, cliques):
        measurements = fake_measurements(cliques)
        loss_fn = marginal_loss.from_linear_measurements(measurements, norm="l1")

        model = estimation.mirror_descent(
            _DOMAIN, loss_fn, known_total=1.0, iters=250, stepsize=0.01
        )
        for M in measurements:
            expected = M.noisy_measurement
            actual = model.project(M.clique).datavector()
            np.testing.assert_allclose(actual, expected, atol=1e-2)

    @parameterized.expand(itertools.product(_CLIQUE_SETS))
    def test_multiplicative_weights(self, cliques):
        measurements = fake_measurements(cliques)
        potentials = CliqueVector.zeros(_DOMAIN, [_DOMAIN.attributes])
        loss_fn = marginal_loss.from_linear_measurements(measurements)
        joint_distribution = estimation.mirror_descent(
            _DOMAIN, loss_fn, known_total=1.0, potentials=potentials, iters=250
        )

        for M in measurements:
            expected = M.noisy_measurement
            actual = joint_distribution.project(M.clique).datavector()
            np.testing.assert_allclose(actual, expected, atol=1e-2)

    @parameterized.expand(itertools.product(_CLIQUE_SETS))
    def test_primal_optimization(self, cliques):
        measurements = fake_measurements(cliques)
        loss_fn = marginal_loss.from_linear_measurements(measurements)

        model = estimation.lbfgs(_DOMAIN, loss_fn, known_total=1.0, iters=250)
        for M in measurements:
            expected = M.noisy_measurement
            actual = model.project(M.clique).datavector()
            np.testing.assert_allclose(actual, expected, atol=1e-2)

    @parameterized.expand(itertools.product(_CLIQUE_SETS))
    def test_dual_averaging(self, cliques):
        measurements = fake_measurements(cliques)

        model = estimation.dual_averaging(
            _DOMAIN, measurements, known_total=1.0, iters=250
        )
        for M in measurements:
            expected = M.noisy_measurement
            actual = model.project(M.clique).datavector()
            np.testing.assert_allclose(actual, expected, atol=1e-2)

    @parameterized.expand(itertools.product(_CLIQUE_SETS))
    def test_interior_gradient(self, cliques):
        measurements = fake_measurements(cliques)

        model = estimation.interior_gradient(
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

        model = estimation.mle_from_marginals(mu, known_total=total)
        for cl in cliques:
            expected = mu.project(cl).datavector()
            actual = model.project(cl).datavector()
            np.testing.assert_allclose(actual, expected, atol=100 / total)

    @parameterized.expand(itertools.product(_CLIQUE_SETS))
    def test_relaxed_projection(self, cliques):
        measurements = fake_measurements(cliques)

        # For RP, we often require higher known_total to increase the model representativeness
        # Therefore, we set known_total=1000, which further requires optimization on measurements
        model = relaxed_projection.relaxed_projection_estimation(_DOMAIN, measurements, known_total=1000, iters=1000)
        for M in measurements:
            expected = M.noisy_measurement
            actual = model.project(M.clique).datavector()
            np.testing.assert_allclose(actual, expected, atol=1e-2)