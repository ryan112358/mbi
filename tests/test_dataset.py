
import unittest
import numpy as np
from mbi.dataset import Dataset
from mbi.domain import Domain

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

if __name__ == '__main__':
    unittest.main()
