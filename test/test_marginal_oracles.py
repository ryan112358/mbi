import unittest
from mbi.domain import Domain
from mbi.factor import Factor
from mbi.clique_vector import CliqueVector
from mbi import marginal_oracles
import numpy as np
from parameterized import parameterized
import itertools


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


_ORACLES = [
    marginal_oracles.brute_force_marginals,
    marginal_oracles.einsum_marginals,
    marginal_oracles.message_passing_stable,
    marginal_oracles.message_passing_fast,
    _variable_elimination_oracle,
    _calculate_many_oracle
]

_DOMAIN = Domain(["a", "b", "c", "d"], [2, 3, 4, 5])

_CLIQUE_SETS = [
    [("a", "b"), ("b", "c"), ("c", "d")],  # tree
    [("a",), ("a", "b"), ("b", "c"), ("a", "c"), ("b", "d")],  # cyclic
    [("a", "b"), ("d", "a")],  # missing c
    [("a", "b", "c", "d")],  # full materialization
    [("d",)],  # singleton
    [("a", "b", "c"), ("c", "b", "a"), ("b", "d")],  # (permuted) duplicates
    # [],  empty is currently not supported
]

_ALL_CLIQUES = itertools.chain.from_iterable(
    itertools.combinations(_DOMAIN.attrs, r) for r in range(5)
)


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
