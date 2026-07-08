import unittest
from mbi.domain import Domain
from mbi.factor import Factor
from mbi.clique_vector import CliqueVector
from mbi.marginal_loss import LinearMeasurement
from mbi import approximate_oracles
import numpy as np
from parameterized import parameterized
import itertools

np.random.seed(0)


_ORACLES = [approximate_oracles.convex_generalized_belief_propagation]

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
        measurements.append(LinearMeasurement(y, cl))
    return measurements


class TestApproximateOracles(unittest.TestCase):

    @parameterized.expand(itertools.product(_ORACLES, _CLIQUE_SETS))
    def test_shapes(self, oracle, cliques):
        zeros = CliqueVector.zeros(_DOMAIN, cliques)
        marginals, _ = oracle(zeros)
        self.assertEqual(marginals.domain, _DOMAIN)
        self.assertEqual(marginals.cliques, tuple(cliques))
        self.assertEqual(set(zeros.tables.keys()), set(marginals.tables.keys()))
        for cl in cliques:
            self.assertEqual(marginals[cl].domain.attributes, cl)

    @parameterized.expand(itertools.product(_CLIQUE_SETS))
    def test_mirror_descent(self, cliques):
        # Here we check that the mirror descent algorithm converges to
        # the true marginals even with an approximate marginal oracle.
        measurements = fake_measurements(cliques)

        mu = approximate_oracles.mirror_descent(
            _DOMAIN,
            measurements,
            known_total=1.0,
            iters=250,
            stepsize=1.0,
        )
        for M in measurements:
            expected = M.noisy_measurement
            actual = mu.project(M.clique).datavector()
            np.testing.assert_allclose(actual, expected, atol=1e-2)

    def test_precompile(self):
        """precompile() with concrete measurements should not raise."""
        cliques = [("a", "b"), ("b", "c")]
        measurements = fake_measurements(cliques)
        est = approximate_oracles.ApproxMirrorDescent(stepsize=1.0)
        est.precompile(_DOMAIN, measurements)

    def test_precompile_with_extra_cliques(self):
        """precompile() with both measurements and extra_cliques."""
        cliques = [("a", "b"), ("b", "c")]
        measurements = fake_measurements(cliques)
        est = approximate_oracles.ApproxMirrorDescent(stepsize=1.0)
        est.precompile(_DOMAIN, measurements, extra_cliques=[("c", "d")])

    def test_precompile_extra_cliques_only(self):
        """precompile() with only extra_cliques (no concrete measurements)."""
        est = approximate_oracles.ApproxMirrorDescent(stepsize=1.0)
        est.precompile(_DOMAIN, extra_cliques=[("a", "b"), ("b", "c")])
