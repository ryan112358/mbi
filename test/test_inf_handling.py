import unittest
import jax.numpy as jnp
from mbi.domain import Domain
from mbi.factor import Factor
from mbi.clique_vector import CliqueVector
from mbi.marginal_oracles import calculate_many_marginals, message_passing_shafer_shenoy

class TestInfHandling(unittest.TestCase):
    def test_factor_dot_nan(self):
        """Test that Factor.dot handles 0 * -inf = 0."""
        domain = Domain(["A"], [2])
        f1 = Factor(domain, jnp.array([1.0, 0.0]))
        f2 = Factor(domain, jnp.array([2.0, -jnp.inf]))

        # f1.dot(f2) should be 1*2 + 0*-inf = 2 + 0 = 2
        result = f1.dot(f2)

        self.assertFalse(jnp.isnan(result), "Result should not be NaN")
        self.assertEqual(result, 2.0, "Result should be 2.0")

    def test_calculate_many_marginals_zero_division(self):
        """Test that calculate_many_marginals handles 0/0 division."""
        domain = Domain(['A', 'B', 'C'], [2, 2, 2])
        cliques = [('A', 'B'), ('B', 'C')]
        potentials = CliqueVector.zeros(domain, cliques)

        # Make P(B=0) = 0 by setting potentials to -inf where B=0
        fAB = potentials[('A', 'B')]
        # B is axis 1 in (A, B)
        valsAB = fAB.values.at[:, 0].set(-jnp.inf)
        potentials[('A', 'B')] = Factor(fAB.domain, valsAB)

        fBC = potentials[('B', 'C')]
        # B is axis 0 in (B, C)
        valsBC = fBC.values.at[0, :].set(-jnp.inf)
        potentials[('B', 'C')] = Factor(fBC.domain, valsBC)

        # This should not raise or produce NaNs.
        # We use message_passing_shafer_shenoy to avoid NaNs from the oracle itself.
        try:
            res = calculate_many_marginals(potentials, cliques, belief_propagation_oracle=message_passing_shafer_shenoy)
        except Exception as e:
            self.fail(f"calculate_many_marginals raised exception: {e}")

        for cl in res.cliques:
            self.assertFalse(jnp.isnan(res[cl].values).any(), f"NaN found in clique {cl}")

if __name__ == "__main__":
    unittest.main()
