import unittest
import numpy as np
from mbi import Domain, Factor, CliqueVector, MarkovRandomField

class TestMarkovRandomField(unittest.TestCase):
    def test_synthetic_data_accuracy(self):
        domain = Domain(['A', 'B'], [10, 10])
        factor = Factor.random(domain).normalize(1.0)
        marginals = CliqueVector(domain, [('A', 'B')], {('A', 'B'): factor})
        N = 1000
        model = MarkovRandomField(potentials=marginals, marginals=marginals, total=N)
        synthetic = model.synthetic_data(rows=N, method="round")
        syn_factor = synthetic.project(('A', 'B'))
        syn_counts = syn_factor.datavector(flatten=True)
        exp_factor = marginals.project(('A', 'B'))
        exp_counts = exp_factor.datavector(flatten=True) * N
        diff = np.abs(syn_counts - exp_counts)
        max_diff = np.max(diff)
        self.assertTrue(np.all(diff <= 2.0000001), f"Counts deviated by {max_diff}, expected <= 2")

    def test_linear_indexing_strides(self):
        """
        Regression test for linear indexing bug where strides were calculated in Fortran order
        instead of C order, causing mismatch with C-ordered conditional probability arrays.
        """
        # Domain: A(2), B(3), C(2).
        # Generation order is C, B, A (based on previous run).
        # So A is child of B, C.
        # Parents of A: B(3), C(2). Sizes: [3, 2].
        # C-strides: [2, 1]. Index = 2*B + 1*C.
        # F-strides: [1, 3]. Index = 1*B + 3*C.
        
        # We want to distinguish (B=1, C=0) from (B=0, C=1).
        # (B=1, C=0):
        #   C-index: 1*2 + 0*1 = 2.
        #   F-index: 1*1 + 0*3 = 1.
        # Index 1 corresponds to (B=0, C=1) in C-order (0*2 + 1*1 = 1).
        
        # Setup:
        #   P(A=1 | B=1, C=0) = 1.0  (Target case)
        #   P(A=1 | B=0, C=1) = 0.0  (Trap case)
        #
        # If bug is present:
        #   Row with (B=1, C=0) calculates index 1.
        #   It looks up cond_probs[1], which corresponds to (B=0, C=1).
        #   P(A=1) = 0.0.
        #   Generates A=0.
        #
        # If correct:
        #   Row with (B=1, C=0) calculates index 2.
        #   P(A=1) = 1.0.
        #   Generates A=1.

        domain = Domain(['A', 'B', 'C'], [2, 3, 2])
        
        # Initialize data
        data = np.ones((2, 3, 2))
        
        # Target: (B=1, C=0) -> A=1
        data[1, 1, 0] = 1000
        data[0, 1, 0] = 0
        
        # Trap: (B=0, C=1) -> A=0
        data[1, 0, 1] = 0
        data[0, 0, 1] = 1000
        
        # Ensure we generate B=1, C=0 frequently
        # We can just rely on the fact that (B=1, C=0) has high weight (1000)
        # compared to others (1).
        
        factor = Factor(domain, data)
        marginals = CliqueVector(domain, [('A', 'B', 'C')], {('A', 'B', 'C'): factor})
        
        N = 1000
        model = MarkovRandomField(potentials=marginals, marginals=marginals, total=N)
        
        synthetic = model.synthetic_data(rows=N, method="round")
        data = synthetic.to_dict()
        
        # Filter rows where B=1, C=0
        mask = (data['B'] == 1) & (data['C'] == 0)
        subset_A = data['A'][mask]
        
        self.assertGreater(len(subset_A), 0, "Should have generated rows with B=1, C=0")
        
        # Check A values
        # Should be all 1s
        a_ones = subset_A.sum()
        self.assertEqual(a_ones, len(subset_A),
                         f"Expected all A=1 for B=1, C=0. Found {a_ones}/{len(subset_A)} A=1s. "
                         "This indicates linear indexing stride mismatch.")

if __name__ == '__main__':
    unittest.main()
