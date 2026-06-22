import unittest
from mbi.domain import Domain
from mbi.factor import Factor
from mbi.clique_vector import CliqueVector
from mbi import marginal_oracles
from mbi.marginal_oracles import (
    message_passing_implicit,
    message_passing_hugin,
    message_passing_shafer_shenoy,
    einsum_materialized,
    einsum_fused,
    einsum_semistable,
)
from mbi.extensions.constraints import (
    constrained_shafer_shenoy,
    constrained_implicit,
)
import jax.numpy as jnp
import numpy as np
from parameterized import parameterized
import itertools
import functools

np.random.seed(0)


def _variable_elimination_oracle(
    potentials: CliqueVector, total: float = 1
) -> CliqueVector:
    domain, cliques = potentials.domain, potentials.cliques
    mu = {
        cl: marginal_oracles.variable_elimination(potentials, cl, total)
        for cl in cliques
    }
    return CliqueVector(domain, cliques, mu)


def _calculate_many_oracle(potentials: CliqueVector, total: float = 1):
    return marginal_oracles.calculate_many_marginals(
        potentials, potentials.cliques, total
    )


def _bulk_variable_elimination_oracle(
    potentials: CliqueVector, total: float = 1
):
    return marginal_oracles.bulk_variable_elimination(
        potentials, potentials.cliques, total
    )


_implicit_materialized = functools.partial(
    message_passing_implicit, contraction=einsum_materialized
)
_implicit_fused = functools.partial(
    message_passing_implicit, contraction=einsum_fused
)
_implicit_constraints_materialized = functools.partial(
    constrained_implicit,
    contraction=einsum_materialized,
)
_implicit_constraints_fused = functools.partial(
    constrained_implicit,
    contraction=einsum_fused,
)


_ORACLES = [
    marginal_oracles.brute_force_marginals,
    marginal_oracles.einsum_marginals,
    marginal_oracles.message_passing_stable,
    marginal_oracles.message_passing_shafer_shenoy,
    marginal_oracles.message_passing_fast,
    _implicit_materialized,
    _implicit_fused,
    message_passing_hugin,
    message_passing_shafer_shenoy,
    constrained_shafer_shenoy,
    _implicit_constraints_materialized,
    _implicit_constraints_fused,
    _variable_elimination_oracle,
    _calculate_many_oracle,
    _bulk_variable_elimination_oracle,
]

_STABLE_ORACLES = [
    marginal_oracles.brute_force_marginals,
    marginal_oracles.message_passing_shafer_shenoy,
    message_passing_shafer_shenoy,
    constrained_shafer_shenoy,
]

_DOMAIN = Domain(["a", "b", "c", "d"], [2, 3, 4, 5])

_CLIQUE_SETS = [
    [("a", "b"), ("b", "c"), ("c", "d")],  # tree
    [("a",), ("a", "b"), ("b", "c"), ("a", "c"), ("b", "d")],  # cyclic
    [("a", "b"), ("d", "a")],  # missing c
    [("a", "b", "c", "d")],  # full materialization
    [("d",)],  # singleton
    [("a", "b", "c"), ("c", "b", "a"), ("b", "d")],  # (permuted) duplicates
    [("a", "b"), ("c", "d")],  # disconnected
    [],  # trivial empty set
]

_ALL_CLIQUES = list(
    itertools.chain.from_iterable(
        itertools.combinations(_DOMAIN.attrs, r) for r in range(5)
    )
)

# Contraction functions for the IMPLICIT schedule.
_CONTRACTIONS = [
    einsum_semistable,
    einsum_materialized,
    einsum_fused,
]

# All three schedules for exhaustive schedule testing.
_SCHEDULES = [
    message_passing_implicit,
    message_passing_hugin,
    message_passing_shafer_shenoy,
]


class TestMarginalOracles(unittest.TestCase):

    @parameterized.expand(itertools.product(_ORACLES, _CLIQUE_SETS))
    def test_shapes(self, oracle, cliques):
        zeros = CliqueVector.zeros(_DOMAIN, cliques)
        marginals = oracle(zeros)
        self.assertEqual(marginals.domain, _DOMAIN)
        self.assertEqual(marginals.cliques, tuple(cliques))
        self.assertEqual(set(zeros.arrays.keys()), set(marginals.arrays.keys()))
        for cl in cliques:
            self.assertEqual(marginals[cl].domain.attrs, cl)

    @parameterized.expand(itertools.product(_ORACLES, _CLIQUE_SETS, [1, 100]))
    def test_uniform(self, oracle, cliques, total=1):
        zeros = CliqueVector.zeros(_DOMAIN, cliques)
        marginals = oracle(zeros, total)
        for cl in cliques:
            expected = total / _DOMAIN.size(cl)
            np.testing.assert_allclose(
                marginals[cl].datavector(), expected, rtol=1e-5
            )

    @parameterized.expand(itertools.product(_ORACLES, _CLIQUE_SETS))
    def test_matches_brute_force(self, oracle, cliques, total=10):
        theta = CliqueVector.random(_DOMAIN, cliques)
        mu1 = oracle(theta, total)
        mu2 = marginal_oracles.brute_force_marginals(theta, total)
        for cl in cliques:
            np.testing.assert_allclose(
                mu1[cl].datavector(), mu2[cl].datavector(), atol=1e-5
            )

    @parameterized.expand(itertools.product(_CLIQUE_SETS, _ALL_CLIQUES))
    def test_variable_elimination(self, model_cliques, query_clique):
        theta = CliqueVector.random(_DOMAIN, model_cliques)
        ans = marginal_oracles.variable_elimination(theta, query_clique)
        self.assertEqual(ans.domain.attributes, query_clique)

    @parameterized.expand(itertools.product(_CLIQUE_SETS, _ALL_CLIQUES))
    def test_variable_elimination_evidence(self, model_cliques, query_clique):
        theta = CliqueVector.random(_DOMAIN, model_cliques)
        evidence_attr = _DOMAIN.attributes[0]
        evidence_val = 0
        evidence = {evidence_attr: evidence_val}

        if evidence_attr in query_clique:
            with self.assertRaises(ValueError):
                marginal_oracles.variable_elimination(
                    theta, query_clique, evidence=evidence
                )
            return

        target_clique_full = tuple(set(query_clique) | set(evidence.keys()))

        ans1 = marginal_oracles.variable_elimination(
            theta, query_clique, evidence=evidence
        )

        ans2_full = marginal_oracles.variable_elimination(
            theta, target_clique_full
        )
        ans2 = ans2_full.slice(evidence)

        ans1 = ans1.normalize()
        ans2 = ans2.normalize()

        ans2 = ans2.transpose(ans1.domain.attributes)

        np.testing.assert_allclose(ans1.values, ans2.values, atol=1e-5)
        self.assertEqual(ans1.domain, ans2.domain)

    @parameterized.expand(_STABLE_ORACLES)
    def test_nan_potentials(self, oracle):
        """Test that -inf potentials are handled correctly without NaNs."""
        cliques = [
            ("A", "B", "C"),
            ("A",),
            ("D",),
            ("D", "A"),
            ("D", "A", "C"),
            ("A", "B"),
        ]

        dom = Domain(["A", "B", "C", "D"], [2, 2, 2, 2])
        potentials = CliqueVector.zeros(dom, cliques)

        con = Factor(
            dom.project(["A", "C"]), jnp.array([[0, -np.inf], [-np.inf, 0]])
        )
        potentials = CliqueVector(
            potentials.domain,
            potentials.cliques + (("A", "C"),),
            {**potentials.arrays, ("A", "C"): con},
        )

        marginals = oracle(potentials)

        for cl, factor in marginals.arrays.items():
            self.assertFalse(
                jnp.isnan(factor.values).any(), f"NaNs found in clique {cl}"
            )
            # With normalize(total=1), we expect sums to be 1.0 (or very close)
            # The domain is A,C correlated perfectly (A=C). A=0,C=1 and A=1,C=0 are impossible.
            # This is a valid graphical model configuration.
            self.assertTrue(
                jnp.allclose(factor.sum().values, 1.0),
                f"Marginal for {cl} does not sum to 1",
            )

    # --- Tests for the composable API ---

    @parameterized.expand(itertools.product(_SCHEDULES, _CLIQUE_SETS))
    def test_schedule_matches_brute_force(self, oracle, cliques, total=10):
        """Every schedule produces identical marginals to brute force."""
        theta = CliqueVector.random(_DOMAIN, cliques)
        mu1 = oracle(theta, total)
        mu2 = marginal_oracles.brute_force_marginals(theta, total)
        for cl in cliques:
            np.testing.assert_allclose(
                mu1[cl].datavector(), mu2[cl].datavector(), atol=1e-5
            )

    @parameterized.expand(itertools.product(_CONTRACTIONS, _CLIQUE_SETS))
    def test_contraction_matches_brute_force(
        self, contraction, cliques, total=10
    ):
        """Every contraction function produces identical marginals to brute force."""
        theta = CliqueVector.random(_DOMAIN, cliques)
        mu1 = message_passing_implicit(theta, total, contraction=contraction)
        mu2 = marginal_oracles.brute_force_marginals(theta, total)
        for cl in cliques:
            np.testing.assert_allclose(
                mu1[cl].datavector(), mu2[cl].datavector(), atol=1e-5
            )


class TestDefaultOracle(unittest.TestCase):

    def test_cpu_returns_shafer_shenoy(self):
        oracle = marginal_oracles.default_oracle(backend="cpu")
        self.assertIs(oracle, message_passing_shafer_shenoy)

    def test_gpu_small_returns_shafer_shenoy(self):
        cliques = (("a", "b"), ("b", "c"), ("c", "d"))
        oracle = marginal_oracles.default_oracle(
            cliques, _DOMAIN, backend="gpu"
        )
        self.assertIs(oracle, message_passing_shafer_shenoy)

    def test_gpu_large_returns_implicit(self):
        domain = Domain(["a", "b", "c"], [100, 100, 100])
        cliques = (("a", "b", "c"),)
        oracle = marginal_oracles.default_oracle(cliques, domain, backend="gpu")
        self.assertIs(oracle, message_passing_implicit)

    def test_gpu_no_cliques_returns_shafer_shenoy(self):
        oracle = marginal_oracles.default_oracle(backend="gpu")
        self.assertIs(oracle, message_passing_shafer_shenoy)

    @parameterized.expand([(cs,) for cs in _CLIQUE_SETS])
    def test_correctness(self, cliques):
        """default_oracle produces correct marginals."""
        if not cliques:
            return
        oracle = marginal_oracles.default_oracle(cliques, _DOMAIN)
        theta = CliqueVector.random(_DOMAIN, cliques)
        mu1 = oracle(theta, 1.0)
        mu2 = marginal_oracles.brute_force_marginals(theta, 1.0)
        for cl in cliques:
            np.testing.assert_allclose(
                mu1[cl].datavector(), mu2[cl].datavector(), atol=1e-5
            )
