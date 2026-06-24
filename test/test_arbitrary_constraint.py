"""Tests for ArbitraryConstraint."""

import unittest

import jax.numpy as jnp
from mbi.domain import Domain
from mbi.extensions.constraints import ArbitraryConstraint
from mbi.extensions.constraints import DeterministicConstraint
from mbi.factor import Factor
import numpy as np


class ArbitraryConstraintConstructionTest(unittest.TestCase):
    """Tests for ArbitraryConstraint construction and validation."""

    def test_valid_construction(self):
        c = ArbitraryConstraint(
            attributes=('A', 'B'),
            valid=np.array([[0, 0], [1, 1], [2, 2]]),
        )
        self.assertEqual(c.attributes, ('A', 'B'))
        self.assertEqual(c.valid.shape, (3, 2))
        self.assertIsNone(c.invalid)

    def test_invalid_construction(self):
        c = ArbitraryConstraint(
            attributes=('A', 'B'),
            invalid=np.array([[0, 1], [1, 0]]),
        )
        self.assertEqual(c.attributes, ('A', 'B'))
        self.assertIsNone(c.valid)
        self.assertEqual(c.invalid.shape, (2, 2))

    def test_neither_raises(self):
        with self.assertRaises(ValueError, msg='exactly one'):
            ArbitraryConstraint(attributes=('A', 'B'))

    def test_both_raises(self):
        with self.assertRaises(ValueError, msg='exactly one'):
            ArbitraryConstraint(
                attributes=('A', 'B'),
                valid=np.array([[0, 0]]),
                invalid=np.array([[1, 1]]),
            )

    def test_wrong_shape_raises(self):
        with self.assertRaises(ValueError, msg='shape'):
            ArbitraryConstraint(
                attributes=('A', 'B'),
                valid=np.array([[0, 0, 0]]),  # 3 cols, but 2 attributes
            )

    def test_1d_array_raises(self):
        with self.assertRaises(ValueError, msg='shape'):
            ArbitraryConstraint(
                attributes=('A', 'B'),
                valid=np.array([0, 1]),  # 1D, not 2D
            )

    def test_clique_is_sorted(self):
        c = ArbitraryConstraint(
            attributes=('B', 'A'),
            valid=np.array([[0, 0]]),
        )
        self.assertEqual(c.clique, ('A', 'B'))

    def test_three_attributes(self):
        c = ArbitraryConstraint(
            attributes=('X', 'Y', 'Z'),
            valid=np.array([[0, 0, 0], [1, 1, 1]]),
        )
        self.assertEqual(c.attributes, ('X', 'Y', 'Z'))
        self.assertEqual(c.valid.shape, (2, 3))


class ArbitraryConstraintPotentialTest(unittest.TestCase):
    """Tests for ArbitraryConstraint.as_potential."""

    def test_valid_potential(self):
        domain = Domain(['A', 'B'], [3, 4])
        c = ArbitraryConstraint(
            attributes=('A', 'B'),
            valid=np.array([[0, 0], [1, 1], [2, 2]]),
        )
        potential = c.as_potential(domain)

        self.assertEqual(potential.domain.attributes, ('A', 'B'))
        self.assertEqual(potential.domain.shape, (3, 4))

        vals = np.array(potential.values)
        # Valid entries should be 0
        self.assertEqual(vals[0, 0], 0.0)
        self.assertEqual(vals[1, 1], 0.0)
        self.assertEqual(vals[2, 2], 0.0)
        # Invalid entries should be -inf
        self.assertEqual(vals[0, 1], -np.inf)
        self.assertEqual(vals[1, 0], -np.inf)
        self.assertEqual(vals[2, 3], -np.inf)

    def test_invalid_potential(self):
        domain = Domain(['A', 'B'], [3, 4])
        c = ArbitraryConstraint(
            attributes=('A', 'B'),
            invalid=np.array([[0, 1], [1, 0]]),
        )
        potential = c.as_potential(domain)

        vals = np.array(potential.values)
        # Invalid entries should be -inf
        self.assertEqual(vals[0, 1], -np.inf)
        self.assertEqual(vals[1, 0], -np.inf)
        # Everything else should be 0
        self.assertEqual(vals[0, 0], 0.0)
        self.assertEqual(vals[0, 2], 0.0)
        self.assertEqual(vals[1, 1], 0.0)
        self.assertEqual(vals[2, 3], 0.0)

    def test_valid_and_invalid_are_complementary(self):
        """A constraint from valid combos should produce the same potential
        as one from the complementary invalid combos."""
        domain = Domain(['A', 'B'], [3, 4])

        valid = np.array([[0, 0], [1, 1], [2, 2]])
        # Complement: everything NOT in valid
        all_combos = np.array([[a, b] for a in range(3) for b in range(4)])
        valid_set = {tuple(r) for r in valid}
        invalid = np.array([r for r in all_combos if tuple(r) not in valid_set])

        c_valid = ArbitraryConstraint(attributes=('A', 'B'), valid=valid)
        c_invalid = ArbitraryConstraint(attributes=('A', 'B'), invalid=invalid)

        p_valid = c_valid.as_potential(domain)
        p_invalid = c_invalid.as_potential(domain)

        np.testing.assert_array_equal(
            np.array(p_valid.values), np.array(p_invalid.values)
        )

    def test_three_attribute_potential(self):
        domain = Domain(['X', 'Y', 'Z'], [2, 3, 2])
        c = ArbitraryConstraint(
            attributes=('X', 'Y', 'Z'),
            valid=np.array([[0, 0, 0], [1, 2, 1]]),
        )
        potential = c.as_potential(domain)

        vals = np.array(potential.values)
        self.assertEqual(vals[0, 0, 0], 0.0)
        self.assertEqual(vals[1, 2, 1], 0.0)
        # Everything else is -inf
        n_inf = np.sum(vals == -np.inf)
        self.assertEqual(n_inf, 2 * 3 * 2 - 2)

    def test_potential_with_message_passing(self):
        """ArbitraryConstraint potential integrates with standard inference."""
        from mbi.clique_vector import CliqueVector
        from mbi.marginal_oracles import message_passing_shafer_shenoy

        domain = Domain(['A', 'B', 'C'], [3, 3, 4])

        # Diagonal constraint: A == B
        valid = np.array([[i, i] for i in range(3)])
        c = ArbitraryConstraint(attributes=('A', 'B'), valid=valid)

        # Random potentials on (A, C) and (B, C)
        np.random.seed(42)
        cliques = [('A', 'C'), ('B', 'C')]
        arrays = {
            cl: Factor(
                domain.project(cl),
                jnp.array(np.random.randn(*domain.project(cl).shape)),
            )
            for cl in cliques
        }

        # Add constraint potential
        con_cl = c.clique
        cliques_with_con = list(cliques) + [con_cl]
        arrays[con_cl] = c.as_potential(domain)

        potentials = CliqueVector(domain, cliques_with_con, arrays)
        result = message_passing_shafer_shenoy(potentials, total=1.0)

        # The (A, B) marginal should only have mass on the diagonal
        ab_marginal = result.project(('A', 'B'))
        vals = np.array(ab_marginal.values)
        for a in range(3):
            for b in range(3):
                if a == b:
                    self.assertGreater(vals[a, b], 0)
                else:
                    np.testing.assert_allclose(vals[a, b], 0, atol=1e-6)


class ArbitraryConstraintFromMappingTest(unittest.TestCase):
    """Tests for ArbitraryConstraint.from_mapping."""

    def test_from_mapping(self):
        mapping = np.array([0, 0, 1, 1, 2, 2])
        c = ArbitraryConstraint.from_mapping('A', 'Ap', mapping)

        self.assertEqual(c.attributes, ('A', 'Ap'))
        self.assertIsNotNone(c.valid)
        self.assertEqual(c.valid.shape, (6, 2))

    def test_from_mapping_matches_deterministic(self):
        """from_mapping potential should match DeterministicConstraint's
        implicit -inf/0 factor."""
        mapping = np.array([0, 0, 1, 1, 2, 2])
        domain = Domain(['A', 'Ap'], [6, 3])

        arb = ArbitraryConstraint.from_mapping('A', 'Ap', mapping)
        det = DeterministicConstraint('A', 'Ap', mapping)

        # Build the DeterministicConstraint's equivalent factor
        shape = (det.n_fine, det.n_coarse)
        vals = np.full(shape, -np.inf)
        for a, ap in enumerate(det.mapping):
            vals[a, ap] = 0.0
        det_factor = Factor(Domain(['A', 'Ap'], list(shape)), jnp.array(vals))

        arb_factor = arb.as_potential(domain)

        np.testing.assert_array_equal(
            np.array(arb_factor.values), np.array(det_factor.values)
        )


class ArbitraryConstraintEqualityTest(unittest.TestCase):
    """Tests for __eq__ and __hash__."""

    def test_equal_valid(self):
        c1 = ArbitraryConstraint(
            attributes=('A', 'B'), valid=np.array([[0, 0], [1, 1]])
        )
        c2 = ArbitraryConstraint(
            attributes=('A', 'B'), valid=np.array([[0, 0], [1, 1]])
        )
        self.assertEqual(c1, c2)
        self.assertEqual(hash(c1), hash(c2))

    def test_not_equal_different_data(self):
        c1 = ArbitraryConstraint(
            attributes=('A', 'B'), valid=np.array([[0, 0]])
        )
        c2 = ArbitraryConstraint(
            attributes=('A', 'B'), valid=np.array([[1, 1]])
        )
        self.assertNotEqual(c1, c2)

    def test_not_equal_valid_vs_invalid(self):
        data = np.array([[0, 0], [1, 1]])
        c1 = ArbitraryConstraint(attributes=('A', 'B'), valid=data)
        c2 = ArbitraryConstraint(attributes=('A', 'B'), invalid=data)
        self.assertNotEqual(c1, c2)


class ArbitraryConstraintMessagePassingTest(unittest.TestCase):
    """Tests for plumbing ArbitraryConstraint through constrained_* functions."""

    def test_shafer_shenoy_with_arbitrary_constraint(self):
        """ArbitraryConstraint passed via constraints= should work."""
        from mbi.clique_vector import CliqueVector
        from mbi.extensions.constraints import constrained_shafer_shenoy

        domain = Domain(['A', 'B', 'C'], [3, 3, 4])

        # Diagonal constraint: A == B
        valid = np.array([[i, i] for i in range(3)])
        c = ArbitraryConstraint(attributes=('A', 'B'), valid=valid)

        np.random.seed(42)
        cliques = [('A', 'C'), ('B', 'C')]
        arrays = {
            cl: Factor(
                domain.project(cl),
                jnp.array(np.random.randn(*domain.project(cl).shape)),
            )
            for cl in cliques
        }
        potentials = CliqueVector(domain, cliques, arrays)

        result = constrained_shafer_shenoy(
            potentials, total=1.0, constraints=(c,)
        )

        # The (A, B) constraint should be enforced
        for cl in cliques:
            marginal = result[cl]
            self.assertEqual(marginal.domain.shape, domain.project(cl).shape)
            # Marginal should sum to approximately 1
            np.testing.assert_allclose(float(marginal.sum()), 1.0, atol=1e-5)

    def test_shafer_shenoy_matches_manual_potential(self):
        """Passing ArbitraryConstraint via constraints= should produce the
        same result as manually adding the factor to potentials."""
        from mbi.clique_vector import CliqueVector
        from mbi.extensions.constraints import constrained_shafer_shenoy
        from mbi.marginal_oracles import message_passing_shafer_shenoy

        domain = Domain(['A', 'B', 'C'], [3, 3, 4])
        valid = np.array([[i, i] for i in range(3)])
        c = ArbitraryConstraint(attributes=('A', 'B'), valid=valid)

        np.random.seed(42)
        cliques = [('A', 'C'), ('B', 'C')]
        arrays = {
            cl: Factor(
                domain.project(cl),
                jnp.array(np.random.randn(*domain.project(cl).shape)),
            )
            for cl in cliques
        }
        potentials = CliqueVector(domain, cliques, arrays)

        # Method 1: via constraints= parameter
        result1 = constrained_shafer_shenoy(
            potentials, total=1.0, constraints=(c,)
        )

        # Method 2: manually add factor to potentials
        con_cl = domain.canonical(c.clique)
        arrays2 = dict(arrays)
        arrays2[con_cl] = c.as_potential(domain)
        cliques2 = list(cliques) + [con_cl]
        potentials2 = CliqueVector(domain, cliques2, arrays2)
        result2 = message_passing_shafer_shenoy(potentials2, total=1.0)

        for cl in cliques:
            np.testing.assert_allclose(
                np.array(result1[cl].values),
                np.array(result2.project(cl).values),
                atol=1e-5,
            )

    def test_mixed_constraints(self):
        """A tuple with both DeterministicConstraint and ArbitraryConstraint."""
        from mbi.clique_vector import CliqueVector
        from mbi.extensions.constraints import constrained_shafer_shenoy

        domain = Domain(['A', 'Ap', 'B'], [6, 3, 4])
        mapping = np.array([0, 0, 1, 1, 2, 2])
        det_c = DeterministicConstraint('A', 'Ap', mapping)

        # Constraint on (Ap, B): only certain combos are valid.
        valid = np.array([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
            [1, 2],
            [2, 0],
            [2, 1],
            [2, 2],
            [2, 3],
        ])
        arb_c = ArbitraryConstraint(attributes=('Ap', 'B'), valid=valid)

        np.random.seed(123)
        # Use cliques that cover all variables properly.
        cliques = [('A', 'B'), ('A', 'Ap'), ('Ap', 'B')]
        arrays = {
            cl: Factor(
                domain.project(cl),
                jnp.array(np.random.randn(*domain.project(cl).shape)),
            )
            for cl in cliques
        }
        potentials = CliqueVector(domain, cliques, arrays)

        # Should not raise
        result = constrained_shafer_shenoy(
            potentials, total=1.0, constraints=(det_c, arb_c)
        )

        for cl in cliques:
            self.assertEqual(result[cl].domain.shape, domain.project(cl).shape)


if __name__ == '__main__':
    unittest.main()
