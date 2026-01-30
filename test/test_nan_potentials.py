
import unittest
import jax.numpy as jnp
import numpy as np
import mbi
from mbi import Domain, CliqueVector, Factor

class TestNanPotentials(unittest.TestCase):
    def test_shafer_shenoy_reproduction(self):
        """Test that message_passing_shafer_shenoy handles -inf potentials correctly."""
        cliques = [('A','B','C'),
         ('A',),
         ('D',),
         ('D', 'A'),
         ('D', 'A', 'C'),
         ('A', 'B')]

        dom = mbi.Domain(['A', 'B', 'C', 'D'], [2,2,2,2])
        potentials = mbi.CliqueVector.zeros(dom, cliques)

        con = mbi.Factor(dom.project(['A', 'C']), jnp.array([[0, -np.inf], [-np.inf, 0]]))
        potentials.arrays[('A', 'C')] = con
        potentials.cliques.append(('A', 'C'))

        marginals = mbi.message_passing_shafer_shenoy(potentials)

        has_nan = False
        for cl, factor in marginals.arrays.items():
            if jnp.isnan(factor.values).any():
                has_nan = True
                break

        self.assertFalse(has_nan, "message_passing_shafer_shenoy produced NaNs")

        # Verify basic properties (e.g., sums to 1)
        # Note: Since there are -inf potentials, the total mass might be less than expected if normalized incorrectly,
        # but normalize(total=1) should force it to 1 unless everything is -inf (impossible).
        # In this case, [0, -inf] and [-inf, 0] are consistent with A=C.
        # So we expect valid distributions.
        for cl, factor in marginals.arrays.items():
            self.assertTrue(jnp.allclose(factor.sum().values, 1.0), f"Marginal for {cl} does not sum to 1")

    def test_stable_produces_nans(self):
        """Confirm that message_passing_stable DOES produce NaNs on this input (regression check)."""
        # This test ensures we are indeed testing a problematic case.
        cliques = [('A','B','C'),
         ('A',),
         ('D',),
         ('D', 'A'),
         ('D', 'A', 'C'),
         ('A', 'B')]

        dom = mbi.Domain(['A', 'B', 'C', 'D'], [2,2,2,2])
        potentials = mbi.CliqueVector.zeros(dom, cliques)

        con = mbi.Factor(dom.project(['A', 'C']), jnp.array([[0, -np.inf], [-np.inf, 0]]))
        potentials.arrays[('A', 'C')] = con
        potentials.cliques.append(('A', 'C'))

        marginals = mbi.marginal_oracles.message_passing_stable(potentials)

        has_nan = False
        for cl, factor in marginals.arrays.items():
            if jnp.isnan(factor.values).any():
                has_nan = True
                break

        self.assertTrue(has_nan, "message_passing_stable should produce NaNs on this input (unless it was fixed elsewhere)")

if __name__ == '__main__':
    unittest.main()
