
import unittest
import numpy as np
import jax.numpy as jnp
from mbi import Domain, Factor, CliqueVector
from mbi.marginal_oracles import variable_elimination

class TestVariableEliminationVectorEvidence(unittest.TestCase):
    def test_vector_evidence_single_factor(self):
        dom = Domain(['X', 'Y'], [2, 2])
        # Values X (row), Y (col)
        # [[1, 2], [3, 4]]
        # X=0 -> [1, 2]
        # X=1 -> [3, 4]
        # Wait. In test I used [[1, 2], [3, 4]].
        # And evidence Y=[0, 1, 0].
        # Y=0: X=0(1), X=1(3). Col 0.
        # Y=1: X=0(2), X=1(4). Col 1.

        linear_values = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        f = Factor(dom, jnp.log(linear_values))

        potentials = CliqueVector(dom, [('X', 'Y')], {('X', 'Y'): f})

        evidence = {'Y': np.array([0, 1, 0])}

        # Expected:
        # Row 0 (Y=0): [1, 3] -> 1/14, 3/14
        # Row 1 (Y=1): [2, 4] -> 2/14, 4/14
        # Row 2 (Y=0): [1, 3] -> 1/14, 3/14

        expected = np.array([
            [1.0/14.0, 3.0/14.0],
            [2.0/14.0, 4.0/14.0],
            [1.0/14.0, 3.0/14.0]
        ])

        res = variable_elimination(potentials, ('X',), evidence=evidence)

        self.assertEqual(res.domain.attributes[0], '_mbi_evidence')
        self.assertEqual(res.domain.attributes[1], 'X')

        np.testing.assert_array_almost_equal(res.values, expected)

    def test_vector_evidence_multiple_factors(self):
        dom = Domain(['X', 'Y', 'Z'], [2, 2, 2])
        v1 = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        v2 = jnp.array([[5.0, 6.0], [7.0, 8.0]])
        f1 = Factor(dom.project(('X', 'Y')), jnp.log(v1))
        f2 = Factor(dom.project(('Y', 'Z')), jnp.log(v2))

        potentials = CliqueVector(dom, [('X', 'Y'), ('Y', 'Z')], {('X', 'Y'): f1, ('Y', 'Z'): f2})

        evidence = {'Y': np.array([0, 1])}

        # Y=0: f1 col 0 [1, 3]. f2 row 0 [5, 6] -> sum 11.
        # Joint [1*11, 3*11] = [11, 33]. Sum 44.
        # Y=1: f1 col 1 [2, 4]. f2 row 1 [7, 8] -> sum 15.
        # Joint [2*15, 4*15] = [30, 60]. Sum 90.
        # Total 134.

        expected = np.array([
            [11.0/134.0, 33.0/134.0],
            [30.0/134.0, 60.0/134.0]
        ])
        res = variable_elimination(potentials, ('X',), evidence=evidence)
        np.testing.assert_array_almost_equal(res.values, expected)

    def test_multiple_evidence_arrays(self):
        dom = Domain(['X', 'Y', 'Z'], [2, 2, 2])
        val_list = [1,2,3,4,5,6,7,8]
        # X varies slowest. (0,0,0)->1. (1,0,0)->5.
        values = jnp.array(val_list).reshape((2,2,2))
        f = Factor(dom, jnp.log(values))

        potentials = CliqueVector(dom, [('X', 'Y', 'Z')], {('X', 'Y', 'Z'): f})

        evidence = {'Y': np.array([0, 1]), 'Z': np.array([0, 1])}

        # Y=0, Z=0 -> X values at [:, 0, 0]. -> [1, 5]. Sum 6.
        # Y=1, Z=1 -> X values at [:, 1, 1]. -> [4, 8]. Sum 12.
        # Total 18.

        res = variable_elimination(potentials, ('X',), evidence=evidence)

        expected = np.array([
            [1.0/18.0, 5.0/18.0],
            [4.0/18.0, 8.0/18.0]
        ])

        np.testing.assert_array_almost_equal(res.values, expected)

if __name__ == '__main__':
    unittest.main()
