import unittest
from mbi.domain import Domain
from mbi.factor import Factor
from mbi.clique_vector import CliqueVector
from mbi import marginal_oracles
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
    marginal_oracles.message_passing_fast,
    message_passing_fast_v1,
    _variable_elimination_oracle,
    _calculate_many_oracle,
    _bulk_variable_elimination_oracle
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
        # Select an attribute to be evidence, making sure it's in the domain
        evidence_attr = _DOMAIN.attributes[0]
        evidence_val = 0
        evidence = {evidence_attr: evidence_val}

        # Calculate with evidence
        ans = marginal_oracles.variable_elimination(theta, query_clique, evidence=evidence)

        # Calculate manually by slicing
        # Manually slice the potentials
        # CliqueVector doesn't have a slice method, so we must iterate
        new_arrays = {}
        for cl, factor in theta.arrays.items():
            new_arrays[cl] = factor.slice(evidence)

        # Sliced clique vector might have different domain?
        # slice removes the attribute from the domain.
        new_domain = theta.domain.marginalize([evidence_attr])

        # Cliques also change
        new_cliques = [tuple(a for a in cl if a != evidence_attr) for cl in theta.cliques]

        # Construct sliced CliqueVector
        # Note: keys of arrays must match cliques list. But slicing changes clique tuples.
        # We need to reconstruct the CliqueVector properly.
        # It's tricky because multiple cliques might map to the same sliced clique?
        # Actually CliqueVector needs unique cliques.

        # A simpler way to verify:
        # Calculate marginal without evidence, then slice the result.
        # BUT this is only valid if query_clique contains the evidence variable or we project later?
        # If query_clique contains evidence_attr, then ans should be sliced.
        # If query_clique does not contain evidence_attr, it's a bit more complex.

        # Alternative verification as requested:
        # "Test Case: Compare variable_elimination(..., evidence={'A': 0}) against variable_elimination(...).slice({'A': 0})."
        # Wait, if I run VE without evidence, I get P(Q).
        # If I slice P(Q) at A=0, I get P(Q, A=0) (unnormalized) or P(Q|A=0) depending on normalization?
        # variable_elimination returns unnormalized factor if total is not strictly handled?
        # The docstring says "sums to the input total".

        # If I pass evidence to VE, I get P(Q \ E | E) * P(E)? No, usually just proportional to joint.
        # The result of `variable_elimination` with evidence sums to `total`?
        # Let's check code.
        # It calls `normalize(total, log=True)`.
        # So it normalizes the result over the target domain.

        # So VE(evidence={'A':0}) returns a distribution over Q\A that sums to `total`.

        # If I run VE(...) -> P(Q).
        # Then P(Q).slice({'A':0}) gives P(Q, A=0).
        # This sums to P(A=0).
        # If I normalize this sliced result to `total`, it should match VE(evidence={'A':0}).

        # NOTE: This only works if `evidence_attr` is in `query_clique`.
        # If `evidence_attr` is NOT in `query_clique`, then `VE(...)` marginalizes out `A`.
        # So `VE(...)` gives P(Q). `A` is gone.
        # Slicing P(Q) with `A=0` is impossible/meaningless if A is not in Q.

        # So the test case suggested: "Compare variable_elimination(..., evidence={'A': 0}) against variable_elimination(...).slice({'A': 0})."
        # implies that we should ask for a marginal that INCLUDES 'A', then slice it.
        # And compare to asking for marginal of 'A' (or superset) with evidence 'A=0'?
        # No, if I provide evidence A=0, the result will NOT have A.

        # Correct comparison logic:
        # 1. Compute `res1 = variable_elimination(theta, query_clique, evidence=evidence)`.
        #    `res1` is over `query_clique - evidence`. It sums to `total`.

        # 2. Compute `res2 = variable_elimination(theta, query_clique + evidence_keys)`.
        #    `res2` is over `query_clique + evidence`.
        #    Slice it: `res2_sliced = res2.slice(evidence)`.
        #    Normalize it: `res2_norm = res2_sliced.normalize(total)`.

        #    Then `res1` should equal `res2_norm`.

        target_clique = tuple(set(query_clique) | set(evidence.keys()))

        # Case 1: With evidence parameter
        # Note: variable_elimination with evidence returns factor over query_clique - evidence.
        ans1 = marginal_oracles.variable_elimination(theta, target_clique, evidence=evidence)

        # Case 2: Without evidence parameter, then slice
        ans2_full = marginal_oracles.variable_elimination(theta, target_clique)
        ans2 = ans2_full.slice(evidence)

        # Normalize both to same total (default 1) to compare distributions
        ans1 = ans1.normalize()
        ans2 = ans2.normalize()

        # Compare values
        np.testing.assert_allclose(ans1.values, ans2.values, atol=1e-12)
        self.assertEqual(ans1.domain, ans2.domain)
