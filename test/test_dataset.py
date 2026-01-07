import unittest
import numpy as np
from mbi.domain import Domain
from mbi.dataset import Dataset


class TestDomain(unittest.TestCase):
    def setUp(self):
        attrs = ["a", "b", "c", "d"]
        shape = [3, 4, 5, 6]
        domain = Domain(attrs, shape)
        self.data = Dataset.synthetic(domain, 100)

    def test_project(self):
        proj = self.data.project(["a", "b"])
        ans = Domain(["a", "b"], [3, 4])
        self.assertEqual(proj.domain, ans)
        proj = self.data.project(("a", "b"))
        self.assertEqual(proj.domain, ans)
        proj = self.data.project("c")
        self.assertEqual(proj.domain, Domain(["c"], [5]))

    def test_datavector(self):
        vec = self.data.datavector()
        self.assertTrue(vec.size, 3 * 4 * 5 * 6)


class TestDatasetDeterministic(unittest.TestCase):
    def setUp(self):
        attrs = ['A', 'B']
        shape = [2, 3]
        self.domain = Domain(attrs, shape)

        # Data: (0, 0), (0, 1), (1, 2), (0, 0)
        self.data_dict = {
            'A': np.array([0, 0, 1, 0]),
            'B': np.array([0, 1, 2, 0])
        }
        self.dataset = Dataset(self.data_dict, self.domain)

        self.weights = np.array([1.0, 2.0, 0.5, 1.0])
        self.weighted_dataset = Dataset(self.data_dict, self.domain, self.weights)

    def test_datavector_unweighted(self):
        # Expected: [2, 1, 0, 0, 0, 1]
        expected = np.array([2, 1, 0, 0, 0, 1])
        result = self.dataset.datavector(flatten=True)
        np.testing.assert_array_equal(result, expected)

        # Unflattened: [[2, 1, 0], [0, 0, 1]]
        expected_unflat = np.array([[2, 1, 0], [0, 0, 1]])
        result_unflat = self.dataset.datavector(flatten=False)
        np.testing.assert_array_equal(result_unflat, expected_unflat)

    def test_project_unweighted(self):
        # Project A
        # Expected: [3, 1]
        expected_A = np.array([3, 1])
        proj_A = self.dataset.project('A')
        np.testing.assert_array_equal(proj_A.datavector(), expected_A)
        self.assertEqual(proj_A.domain.attrs, ('A',))

        # Project B
        # Expected: [2, 1, 1]
        expected_B = np.array([2, 1, 1])
        proj_B = self.dataset.project('B')
        np.testing.assert_array_equal(proj_B.datavector(), expected_B)
        self.assertEqual(proj_B.domain.attrs, ('B',))

    def test_datavector_weighted(self):
        # Weights: [1.0, 2.0, 0.5, 1.0] for indices [0, 1, 5, 0]
        # Expected: [2.0, 2.0, 0, 0, 0, 0.5]
        expected = np.array([2.0, 2.0, 0, 0, 0, 0.5])
        result = self.weighted_dataset.datavector(flatten=True)
        np.testing.assert_array_equal(result, expected)

    def test_project_weighted(self):
        # Project A
        # 0: 1.0 + 2.0 + 1.0 = 4.0
        # 1: 0.5
        expected_A = np.array([4.0, 0.5])
        proj_A = self.weighted_dataset.project('A')
        np.testing.assert_array_equal(proj_A.datavector(), expected_A)

        # Project B
        # 0: 1.0 + 1.0 = 2.0
        # 1: 2.0
        # 2: 0.5
        expected_B = np.array([2.0, 2.0, 0.5])
        proj_B = self.weighted_dataset.project('B')
        np.testing.assert_array_equal(proj_B.datavector(), expected_B)

    def test_project_multiple_columns(self):
        # Project (A, B) - should be same as full datavector
        expected = np.array([2, 1, 0, 0, 0, 1])
        proj = self.dataset.project(['A', 'B'])
        np.testing.assert_array_equal(proj.datavector(), expected)


class TestDatasetCompression(unittest.TestCase):
    def test_compress(self):
        # Domain: a:3, b:2
        domain = Domain(['a', 'b'], [3, 2])
        # Data
        # a: 0, 1, 2, 0, 1
        # b: 0, 1, 0, 1, 0
        data = {
            'a': np.array([0, 1, 2, 0, 1]),
            'b': np.array([0, 1, 0, 1, 0])
        }
        dataset = Dataset(data, domain)

        # Mapping for 'a': 0->0, 1->1, 2->1.  New domain size for a is 2.
        # 'b' is untouched.
        mapping = {
            'a': np.array([0, 1, 1])
        }

        compressed_dataset = dataset.compress(mapping)

        # Check new domain
        self.assertEqual(compressed_dataset.domain['a'], 2)
        self.assertEqual(compressed_dataset.domain['b'], 2)

        # Check data
        expected_a = np.array([0, 1, 1, 0, 1])
        expected_b = np.array([0, 1, 0, 1, 0])

        np.testing.assert_array_equal(compressed_dataset.to_dict()['a'], expected_a)
        np.testing.assert_array_equal(compressed_dataset.to_dict()['b'], expected_b)

        # Check weights are preserved
        np.testing.assert_array_equal(compressed_dataset.weights, dataset.weights)

    def test_decompress(self):
        # Initial Domain: a:3
        # Compressed Domain: a:2 (via mapping 0->0, 1->1, 2->1)

        # Create a compressed dataset directly
        domain = Domain(['a', 'b'], [2, 2])
        data = {
            'a': np.array([0, 1, 1, 0, 1]), # Corresponds to original 0, (1 or 2), (1 or 2), 0, (1 or 2)
            'b': np.array([0, 1, 0, 1, 0])
        }
        compressed_dataset = Dataset(data, domain)

        mapping = {
            'a': np.array([0, 1, 1])
        }

        # Decompress
        # For 'a':
        # 0 should map to 0.
        # 1 should map to 1 or 2 with equal prob.

        decompressed_dataset = compressed_dataset.decompress(mapping)

        # Check domain size restoration
        self.assertEqual(decompressed_dataset.domain['a'], 3)
        self.assertEqual(decompressed_dataset.domain['b'], 2)

        # Check values
        new_a = decompressed_dataset.to_dict()['a']

        # Indices where original was 0 (idx 0 and 3) must be 0
        self.assertEqual(new_a[0], 0)
        self.assertEqual(new_a[3], 0)

        # Indices where original was 1 (idx 1, 2, 4) must be 1 or 2
        for i in [1, 2, 4]:
            self.assertIn(new_a[i], [1, 2])

        # Statistical check? Maybe overkill for unit test but good for verification
        # Let's run a large batch to check uniform distribution
        large_N = 10000
        large_domain = Domain(['a'], [2])
        large_data = {'a': np.ones(large_N, dtype=int)} # All 1s
        large_dataset = Dataset(large_data, large_domain)

        decompressed_large = large_dataset.decompress(mapping)
        vals = decompressed_large.to_dict()['a']
        count_1 = np.sum(vals == 1)
        count_2 = np.sum(vals == 2)

        # Should be roughly equal
        diff = abs(count_1 - count_2)
        self.assertTrue(diff < large_N * 0.1, f"Difference {diff} is too large for uniform distribution")

    def test_validation(self):
        domain = Domain(['a'], [3])
        data = {'a': np.array([0, 1, 2])}
        dataset = Dataset(data, domain)

        # Wrong size mapping
        with self.assertRaises(ValueError):
            dataset.compress({'a': np.array([0, 1])}) # Size 2 instead of 3

        # Negative values
        with self.assertRaises(ValueError):
            dataset.compress({'a': np.array([0, -1, 1])})

        # Non-integer
        with self.assertRaises(ValueError):
            dataset.compress({'a': np.array([0, 1.5, 1])})

class TestDatasetEmpty(unittest.TestCase):
    def test_compress_empty(self):
        domain = Domain(['a'], [3])
        # Empty data
        data = {'a': np.array([], dtype=int)}
        dataset = Dataset(data, domain, weights=np.array([]))

        mapping = {'a': np.array([0, 1, 1])}

        # Should not raise
        compressed = dataset.compress(mapping)
        self.assertEqual(compressed.domain['a'], 2)
        self.assertEqual(len(compressed.to_dict()['a']), 0)

    def test_decompress_empty(self):
        domain = Domain(['a'], [2])
        data = {'a': np.array([], dtype=int)}
        dataset = Dataset(data, domain, weights=np.array([]))

        mapping = {'a': np.array([0, 1, 1])}

        decompressed = dataset.decompress(mapping)
        self.assertEqual(decompressed.domain['a'], 3)
        self.assertEqual(len(decompressed.to_dict()['a']), 0)

if __name__ == "__main__":
    unittest.main()
