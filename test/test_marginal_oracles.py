import unittest
from mbi.domain import Domain
from mbi.factor import Factor
from mbi.clique_vector import CliqueVector
from mbi import marginal_oracles
import jax.numpy as jnp
import numpy as np
from parameterized import parameterized
import itertools
import functools


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


def _bulk_variable_elimination_oracle(potentials: CliqueVector, total: float = 1):
    return marginal_oracles.bulk_variable_elimination(
        potentials, potentials.cliques, total
    )

message_passing_fast_v1 = functools.partial(
  marginal_oracles.message_passing_fast,
  logspace_sum_product_fn=marginal_oracles.logspace_sum_product_stable_v1
)


_ORACLES = [
    marginal_oracles.brute_force_marginals,
    marginal_oracles.einsum_marginals,
    marginal_oracles.message_passing_stable,
    marginal_oracles.message_passing_shafer_shenoy,
    marginal_oracles.message_passing_fast,
    message_passing_fast_v1,
    _variable_elimination_oracle,
    _calculate_many_oracle,
    _bulk_variable_elimination_oracle
]

_STABLE_ORACLES = [
    marginal_oracles.brute_force_marginals,
    marginal_oracles.message_passing_shafer_shenoy
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
    # [],  empty is currently not supported
]

_ALL_CLIQUES = list(itertools.chain.from_iterable(
    itertools.combinations(_DOMAIN.attrs, r) for r in range(5)
))


class TestMarginalOracles(unittest.TestCase):

    @parameterized.expand(itertools.product(_ORACLES, _CLIQUE_SETS))
    def test_shapes(self, oracle, cliques):
        zeros = CliqueVector.zeros(_DOMAIN, cliques)
        marginals = oracle(zeros)
        self.assertEqual(marginals.domain, _DOMAIN)
        self.assertEqual(marginals.cliques, cliques)
        self.assertEqual(set(zeros.arrays.keys()), set(marginals.arrays.keys()))
        for cl in cliques:
            self.assertEqual(marginals[cl].domain.attrs, cl)

    @parameterized.expand(itertools.product(_ORACLES, _CLIQUE_SETS, [1, 100]))
    def test_uniform(self, oracle, cliques, total=1):
        zeros = CliqueVector.zeros(_DOMAIN, cliques)
        marginals = oracle(zeros, total)
        for cl in cliques:
            expected = total / _DOMAIN.size(cl)
            np.testing.assert_allclose(marginals[cl].datavector(), expected)

    @parameterized.expand(itertools.product(_ORACLES, _CLIQUE_SETS))
    def test_matches_brute_force(self, oracle, cliques, total=10):
        theta = CliqueVector.random(_DOMAIN, cliques)
        mu1 = oracle(theta, total)
        mu2 = marginal_oracles.brute_force_marginals(theta, total)
        for cl in cliques:
            np.testing.assert_allclose(mu1[cl].datavector(), mu2[cl].datavector())

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
                marginal_oracles.variable_elimination(theta, query_clique, evidence=evidence)
            return

        target_clique_full = tuple(set(query_clique) | set(evidence.keys()))

        ans1 = marginal_oracles.variable_elimination(theta, query_clique, evidence=evidence)

        ans2_full = marginal_oracles.variable_elimination(theta, target_clique_full)
        ans2 = ans2_full.slice(evidence)

        ans1 = ans1.normalize()
        ans2 = ans2.normalize()

        ans2 = ans2.transpose(ans1.domain.attributes)

        np.testing.assert_allclose(ans1.values, ans2.values, atol=1e-12)
        self.assertEqual(ans1.domain, ans2.domain)

    @parameterized.expand(_STABLE_ORACLES)
    def test_nan_potentials(self, oracle):
        """Test that -inf potentials are handled correctly without NaNs."""
        cliques = [('A','B','C'),
         ('A',),
         ('D',),
         ('D', 'A'),
         ('D', 'A', 'C'),
         ('A', 'B')]

        dom = Domain(['A', 'B', 'C', 'D'], [2,2,2,2])
        potentials = CliqueVector.zeros(dom, cliques)

        con = Factor(dom.project(['A', 'C']), jnp.array([[0, -np.inf], [-np.inf, 0]]))
        potentials.arrays[('A', 'C')] = con
        potentials.cliques.append(('A', 'C'))

        marginals = oracle(potentials)

        for cl, factor in marginals.arrays.items():
            self.assertFalse(jnp.isnan(factor.values).any(), f"NaNs found in clique {cl}")
            # With normalize(total=1), we expect sums to be 1.0 (or very close)
            # The domain is A,C correlated perfectly (A=C). A=0,C=1 and A=1,C=0 are impossible.
            # This is a valid graphical model configuration.
            self.assertTrue(jnp.allclose(factor.sum().values, 1.0), f"Marginal for {cl} does not sum to 1")
