import unittest

import jax.numpy as jnp
import numpy as np
from mbi import Constraint, Domain, Factor


class TestConstraintConstruction(unittest.TestCase):

    def test_valid(self):
        domain = Domain(["a", "b"], [3, 4])
        valid = np.array([[0, 0], [1, 2], [2, 3]])
        c = Constraint(domain=domain, valid=valid)
        self.assertIsNotNone(c.valid)
        self.assertIsNone(c.invalid)
        self.assertIsNone(c.mapping)

    def test_invalid(self):
        domain = Domain(["a", "b"], [3, 4])
        invalid = np.array([[0, 1]])
        c = Constraint(domain=domain, invalid=invalid)
        self.assertIsNone(c.valid)
        self.assertIsNotNone(c.invalid)

    def test_mapping(self):
        domain = Domain(["fine", "coarse"], [6, 3])
        mapping = np.array([0, 0, 1, 1, 2, 2])
        c = Constraint(domain=domain, mapping=mapping)
        self.assertIsNone(c.valid)
        self.assertTrue(c.is_deterministic)

    def test_none_specified_raises(self):
        domain = Domain(["a"], [3])
        with self.assertRaises(ValueError):
            Constraint(domain=domain)

    def test_multiple_specified_raises(self):
        domain = Domain(["a", "b"], [3, 4])
        with self.assertRaises(ValueError):
            Constraint(
                domain=domain,
                valid=np.array([[0, 0]]),
                invalid=np.array([[1, 1]]),
            )

    def test_valid_wrong_columns_raises(self):
        domain = Domain(["a", "b"], [3, 4])
        with self.assertRaises(ValueError):
            Constraint(domain=domain, valid=np.array([[0, 0, 0]]))

    def test_mapping_wrong_size_raises(self):
        domain = Domain(["fine", "coarse"], [6, 3])
        with self.assertRaises(ValueError):
            Constraint(domain=domain, mapping=np.array([0, 0, 1]))

    def test_mapping_requires_two_attributes(self):
        domain = Domain(["a", "b", "c"], [3, 3, 3])
        with self.assertRaises(ValueError):
            Constraint(domain=domain, mapping=np.array([0, 1, 2]))


class TestConstraintProperties(unittest.TestCase):

    def test_clique(self):
        domain = Domain(["b", "a"], [3, 4])
        c = Constraint(domain=domain, valid=np.array([[0, 0]]))
        self.assertEqual(c.clique, ("a", "b"))

    def test_is_deterministic(self):
        domain = Domain(["fine", "coarse"], [4, 2])
        c_map = Constraint(domain=domain, mapping=np.array([0, 0, 1, 1]))
        c_valid = Constraint(domain=domain, valid=np.array([[0, 0], [1, 0]]))
        self.assertTrue(c_map.is_deterministic)
        self.assertFalse(c_valid.is_deterministic)


class TestAsPotential(unittest.TestCase):

    def test_valid_potential(self):
        domain = Domain(["a", "b"], [3, 3])
        valid = np.array([[0, 0], [1, 1], [2, 2]])
        c = Constraint(domain=domain, valid=valid)
        f = c.potential
        self.assertEqual(f.domain, domain)
        vals = np.asarray(f.values)
        # Diagonal should be 0, off-diagonal -inf
        for i in range(3):
            for j in range(3):
                if i == j:
                    self.assertEqual(vals[i, j], 0.0)
                else:
                    self.assertEqual(vals[i, j], -np.inf)

    def test_invalid_potential(self):
        domain = Domain(["a", "b"], [2, 2])
        invalid = np.array([[0, 1]])
        c = Constraint(domain=domain, invalid=invalid)
        f = c.potential
        vals = np.asarray(f.values)
        self.assertEqual(vals[0, 1], -np.inf)
        self.assertEqual(vals[0, 0], 0.0)
        self.assertEqual(vals[1, 0], 0.0)
        self.assertEqual(vals[1, 1], 0.0)

    def test_mapping_potential(self):
        domain = Domain(["fine", "coarse"], [4, 2])
        mapping = np.array([0, 0, 1, 1])
        c = Constraint(domain=domain, mapping=mapping)
        f = c.potential
        vals = np.asarray(f.values)
        # (0,0), (1,0), (2,1), (3,1) should be 0; rest -inf
        self.assertEqual(vals[0, 0], 0.0)
        self.assertEqual(vals[1, 0], 0.0)
        self.assertEqual(vals[2, 1], 0.0)
        self.assertEqual(vals[3, 1], 0.0)
        self.assertEqual(vals[0, 1], -np.inf)
        self.assertEqual(vals[2, 0], -np.inf)

    def test_valid_and_mapping_produce_same_potential(self):
        """A mapping constraint should produce the same factor as the
        equivalent valid-combos constraint."""
        domain = Domain(["fine", "coarse"], [6, 3])
        mapping = np.array([0, 0, 1, 1, 2, 2])
        valid = np.column_stack([np.arange(6), mapping])

        c_map = Constraint(domain=domain, mapping=mapping)
        c_valid = Constraint(domain=domain, valid=valid)

        np.testing.assert_array_equal(
            c_map.potential.values, c_valid.potential.values
        )

    def test_three_way_valid(self):
        """Constraints over 3+ attributes should work."""
        domain = Domain(["a", "b", "c"], [2, 2, 2])
        valid = np.array([[0, 0, 0], [1, 1, 1]])
        c = Constraint(domain=domain, valid=valid)
        f = c.potential
        vals = np.asarray(f.values)
        self.assertEqual(vals[0, 0, 0], 0.0)
        self.assertEqual(vals[1, 1, 1], 0.0)
        self.assertEqual(vals[0, 0, 1], -np.inf)
        self.assertEqual(vals[1, 0, 0], -np.inf)


class TestOracleIntegration(unittest.TestCase):
    """Test that constraints flow through marginal oracles correctly."""

    def _make_model(self):
        domain = Domain(["A", "B", "C"], [3, 3, 4])
        cliques = [("A", "C"), ("B", "C")]
        np.random.seed(42)
        from mbi import CliqueVector

        arrays = {
            cl: Factor(
                domain.project(cl),
                jnp.array(np.random.randn(*domain.project(cl).shape)),
            )
            for cl in cliques
        }
        return domain, cliques, CliqueVector(domain, cliques, arrays)

    def _diagonal_constraint(self, domain):
        """A=B constraint (diagonal only)."""
        valid = np.array([[i, i] for i in range(3)])
        return Constraint(domain=domain.project(("A", "B")), valid=valid)

    def test_shafer_shenoy_with_constraints(self):
        from mbi import marginal_oracles

        domain, cliques, potentials = self._make_model()
        c = self._diagonal_constraint(domain)

        result = marginal_oracles.message_passing_shafer_shenoy(
            potentials, total=1.0, constraints=(c,)
        )
        # Marginals should sum to 1.
        for cl in cliques:
            np.testing.assert_allclose(float(result[cl].sum()), 1.0, atol=1e-5)

    def test_shafer_shenoy_constraints_match_manual(self):
        """constraints= should match manually adding the factor."""
        from mbi import CliqueVector, marginal_oracles

        domain, cliques, potentials = self._make_model()
        c = self._diagonal_constraint(domain)

        result1 = marginal_oracles.message_passing_shafer_shenoy(
            potentials, total=1.0, constraints=(c,)
        )

        # Manually add the constraint factor.
        con_cl = domain.canonical(c.clique)
        arrays = {cl: potentials[cl] for cl in cliques}
        arrays[con_cl] = c.potential
        potentials2 = CliqueVector(domain, list(cliques) + [con_cl], arrays)
        result2 = marginal_oracles.message_passing_shafer_shenoy(
            potentials2, total=1.0
        )

        for cl in cliques:
            np.testing.assert_allclose(
                np.array(result1[cl].values),
                np.array(result2.project(cl).values),
                atol=1e-5,
            )

    def test_implicit_with_constraints(self):
        from mbi import marginal_oracles

        domain, cliques, potentials = self._make_model()
        c = self._diagonal_constraint(domain)

        result = marginal_oracles.message_passing_implicit(
            potentials, total=1.0, constraints=(c,)
        )
        for cl in cliques:
            np.testing.assert_allclose(float(result[cl].sum()), 1.0, atol=1e-5)

    def test_implicit_einsum_semistable_raises(self):
        from mbi import marginal_oracles

        domain, cliques, potentials = self._make_model()
        c = self._diagonal_constraint(domain)

        with self.assertRaises(ValueError):
            marginal_oracles.message_passing_implicit(
                potentials,
                total=1.0,
                constraints=(c,),
                contraction=marginal_oracles.einsum_semistable,
            )

    def test_hugin_warns(self):
        import warnings as w
        from mbi import marginal_oracles

        domain, cliques, potentials = self._make_model()
        c = self._diagonal_constraint(domain)

        with w.catch_warnings(record=True) as caught:
            w.simplefilter("always")
            marginal_oracles.message_passing_hugin(
                potentials, total=1.0, constraints=(c,)
            )
        self.assertTrue(
            any("HUGIN" in str(warning.message) for warning in caught)
        )

    def test_mapping_constraint_through_oracle(self):
        """Mapping constraints work through oracles."""
        from mbi import marginal_oracles

        domain = Domain(["fine", "coarse", "X"], [6, 3, 4])
        mapping = np.array([0, 0, 1, 1, 2, 2])
        c = Constraint(
            domain=domain.project(("fine", "coarse")), mapping=mapping
        )

        np.random.seed(0)
        from mbi import CliqueVector

        cliques = [("fine", "X"), ("coarse", "X")]
        arrays = {
            cl: Factor(
                domain.project(cl),
                jnp.array(np.random.randn(*domain.project(cl).shape)),
            )
            for cl in cliques
        }
        potentials = CliqueVector(domain, cliques, arrays)

        result = marginal_oracles.message_passing_shafer_shenoy(
            potentials, total=1.0, constraints=(c,)
        )
        for cl in cliques:
            np.testing.assert_allclose(float(result[cl].sum()), 1.0, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
