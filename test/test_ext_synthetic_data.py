import unittest
import numpy as np
from mbi import (
    Domain,
    Factor,
    CliqueVector,
    MarkovRandomField,
    marginal_oracles,
)
from mbi.extensions.synthetic_data import synthetic_data as ext_synthetic_data


def _create_random_model(domain, cliques, N):
    """Creates a random MRF with given cliques and total count N."""
    potentials = {}
    np.random.seed(0)

    for cl in cliques:
        vals = np.random.rand(*domain.project(cl).shape) + 1e-10
        f = Factor(domain.project(cl), vals)
        potentials[cl] = f.log()

    potential_vector = CliqueVector(domain, cliques, potentials)
    marginals = marginal_oracles.message_passing_stable(
        potential_vector, total=N
    )

    return MarkovRandomField(
        potentials=potential_vector, marginals=marginals, total=N
    )


class TestExtSyntheticDataAccuracy(unittest.TestCase):
    """Tests that extensions.synthetic_data matches MRF.synthetic_data.

    The MRF implementation is the reference.  These tests verify that the
    JAX-based extensions version produces equivalent marginals across
    multiple graph structures, including the ones that triggered the
    clique-conditioning regression (using original measurement cliques
    instead of junction tree maximal cliques for parent determination).
    """

    def _assert_parity(self, model, cliques, N, tol=0.01):
        """Assert extensions synth matches MRF synth within tolerance.

        For each model clique we check that the normalized L1 gap between
        extensions and MRF synthetic marginals is small.  Since both use
        rounding (not sampling), the gap should be near zero at large N.

        Args:
            model: A fitted MarkovRandomField.
            cliques: Cliques to evaluate.
            N: Number of rows to generate.
            tol: Maximum allowed normalized L1/2 gap per clique.
        """
        np.random.seed(0)
        synth_mrf = model.synthetic_data(rows=N, method="round")
        synth_ext = ext_synthetic_data(model, rows=N)

        for cl in cliques:
            mrf_marg = synth_mrf.project(cl).datavector(flatten=True)
            ext_marg = synth_ext.project(cl).datavector(flatten=True)

            mrf_norm = mrf_marg / max(mrf_marg.sum(), 1e-20)
            ext_norm = ext_marg / max(ext_marg.sum(), 1e-20)

            gap = 0.5 * np.sum(np.abs(mrf_norm - ext_norm))
            self.assertLess(
                gap,
                tol,
                f"Extensions gap {gap:.6f} exceeds tolerance {tol} "
                f"for clique {cl}",
            )

    def _test_model_structure(self, domain, cliques, cross_cliques=None):
        """Test a specific model structure for MRF/extensions parity."""
        N = 50000
        model = _create_random_model(domain, cliques, N)
        all_cliques = list(cliques) + (cross_cliques or [])
        self._assert_parity(model, all_cliques, N)

    def test_single_clique(self):
        """Single clique: no conditioning needed."""
        domain = Domain(["A", "B"], [10, 10])
        cliques = [("A", "B")]
        self._test_model_structure(domain, cliques)

    def test_independent_cliques(self):
        """Two disjoint cliques: no shared attributes."""
        domain = Domain(["A", "B", "C", "D"], [5, 5, 5, 5])
        cliques = [("A", "B"), ("C", "D")]
        self._test_model_structure(
            domain, cliques, cross_cliques=[("A", "C"), ("B", "D")]
        )

    def test_chain(self):
        """Chain A-B-C: conditioning through shared attribute B."""
        domain = Domain(["A", "B", "C"], [5, 5, 5])
        cliques = [("A", "B"), ("B", "C")]
        self._test_model_structure(domain, cliques, cross_cliques=[("A", "C")])

    def test_star(self):
        """Star with hub: spokes share a hub but not each other."""
        domain = Domain(["H", "A", "B", "C", "D"], [5, 4, 4, 4, 4])
        cliques = [("H", "A"), ("H", "B"), ("H", "C"), ("H", "D")]
        self._test_model_structure(
            domain,
            cliques,
            cross_cliques=[("A", "B"), ("A", "C"), ("B", "D")],
        )

    def test_two_hubs_shared_separator(self):
        """Two hubs with shared separators — the structure that triggered
        the clique-conditioning bug.

        Hub1 (H1) connects to A, B, C.
        Hub2 (H2) connects to A, B, C.
        The junction tree merges {H1, A, B, C} and {H2, A, B, C} into
        super-cliques with separator {A, B, C}.  Without the fix, A, B, C
        are generated independently instead of from their joint.
        """
        domain = Domain(["H1", "H2", "A", "B", "C"], [4, 4, 3, 3, 3])
        cliques = [
            ("H1", "A"),
            ("H1", "B"),
            ("H1", "C"),
            ("H2", "A"),
            ("H2", "B"),
            ("H2", "C"),
        ]
        self._test_model_structure(
            domain,
            cliques,
            cross_cliques=[
                ("A", "B"),
                ("A", "C"),
                ("B", "C"),
                ("H1", "H2"),
            ],
        )

    def test_overlapping_triples(self):
        """Overlapping 3-way cliques that create 4-way super-cliques."""
        domain = Domain(["A", "B", "C", "D", "E"], [3, 3, 3, 3, 3])
        cliques = [
            ("A", "B", "C"),
            ("B", "C", "D"),
            ("D", "E"),
        ]
        self._test_model_structure(
            domain,
            cliques,
            cross_cliques=[("A", "D"), ("A", "E"), ("C", "E")],
        )

    def test_two_clusters_with_bridge(self):
        """Two dense clusters connected by a single bridge edge.

        Cluster 1: all pairs of {A, B, C} → super-clique (A, B, C)
        Cluster 2: all pairs of {D, E, F} → super-clique (D, E, F)
        Bridge: (C, D)
        """
        domain = Domain(["A", "B", "C", "D", "E", "F"], [3, 3, 3, 3, 3, 3])
        cliques = [
            ("A", "B"),
            ("A", "C"),
            ("B", "C"),
            ("D", "E"),
            ("D", "F"),
            ("E", "F"),
            ("C", "D"),
        ]
        self._test_model_structure(
            domain,
            cliques,
            cross_cliques=[("A", "D"), ("B", "E"), ("A", "F")],
        )

    def test_mixed_arity_cliques(self):
        """Mix of 1-way, 2-way, and 3-way cliques."""
        domain = Domain(["A", "B", "C", "D", "E"], [3, 3, 3, 3, 3])
        cliques = [("A",), ("A", "B"), ("B", "C", "D"), ("D", "E")]
        self._test_model_structure(
            domain,
            cliques,
            cross_cliques=[("A", "C"), ("A", "E"), ("C", "E")],
        )


if __name__ == "__main__":
    unittest.main()
