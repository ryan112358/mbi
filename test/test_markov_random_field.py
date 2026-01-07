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

if __name__ == '__main__':
    unittest.main()
