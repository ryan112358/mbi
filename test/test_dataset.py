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

        self.data_dict = {
            'A': np.array([0, 0, 1, 0]),
            'B': np.array([0, 1, 2, 0])
        }
        self.dataset = Dataset(self.data_dict, self.domain)

        self.weights = np.array([1.0, 2.0, 0.5, 1.0])
        self.weighted_dataset = Dataset(self.data_dict, self.domain, self.weights)

    def test_datavector_unweighted(self):
        expected = np.array([2, 1, 0, 0, 0, 1])
        result = self.dataset.datavector(flatten=True)
        np.testing.assert_array_equal(result, expected)

        expected_unflat = np.array([[2, 1, 0], [0, 0, 1]])
        result_unflat = self.dataset.datavector(flatten=False)
        np.testing.assert_array_equal(result_unflat, expected_unflat)

    def test_project_unweighted(self):
        expected_A = np.array([3, 1])
        proj_A = self.dataset.project('A')
        np.testing.assert_array_equal(proj_A.datavector(), expected_A)
        self.assertEqual(proj_A.domain.attrs, ('A',))

        expected_B = np.array([2, 1, 1])
        proj_B = self.dataset.project('B')
        np.testing.assert_array_equal(proj_B.datavector(), expected_B)
        self.assertEqual(proj_B.domain.attrs, ('B',))

    def test_datavector_weighted(self):
        expected = np.array([2.0, 2.0, 0, 0, 0, 0.5])
        result = self.weighted_dataset.datavector(flatten=True)
        np.testing.assert_array_equal(result, expected)

    def test_project_weighted(self):
        expected_A = np.array([4.0, 0.5])
        proj_A = self.weighted_dataset.project('A')
        np.testing.assert_array_equal(proj_A.datavector(), expected_A)

        expected_B = np.array([2.0, 2.0, 0.5])
        proj_B = self.weighted_dataset.project('B')
        np.testing.assert_array_equal(proj_B.datavector(), expected_B)

    def test_project_multiple_columns(self):
        expected = np.array([2, 1, 0, 0, 0, 1])
        proj = self.dataset.project(['A', 'B'])
        np.testing.assert_array_equal(proj.datavector(), expected)

    def test_compress(self):
        domain = Domain(['a', 'b'], [3, 2])
        data = {
            'a': np.array([0, 1, 2, 0, 1]),
            'b': np.array([0, 1, 0, 1, 0])
        }
        dataset = Dataset(data, domain)

        mapping = {
            'a': np.array([0, 1, 1])
        }

        compressed_dataset = dataset.compress(mapping)

        self.assertEqual(compressed_dataset.domain['a'], 2)
        self.assertEqual(compressed_dataset.domain['b'], 2)

        expected_a = np.array([0, 1, 1, 0, 1])
        expected_b = np.array([0, 1, 0, 1, 0])

        np.testing.assert_array_equal(compressed_dataset.to_dict()['a'], expected_a)
        np.testing.assert_array_equal(compressed_dataset.to_dict()['b'], expected_b)

        np.testing.assert_array_equal(compressed_dataset.weights, dataset.weights)

    def test_decompress(self):
        domain = Domain(['a', 'b'], [2, 2])
        data = {
            'a': np.array([0, 1, 1, 0, 1]),
            'b': np.array([0, 1, 0, 1, 0])
        }
        compressed_dataset = Dataset(data, domain)

        mapping = {
            'a': np.array([0, 1, 1])
        }

        decompressed_dataset = compressed_dataset.decompress(mapping)

        self.assertEqual(decompressed_dataset.domain['a'], 3)
        self.assertEqual(decompressed_dataset.domain['b'], 2)

        new_a = decompressed_dataset.to_dict()['a']

        self.assertEqual(new_a[0], 0)
        self.assertEqual(new_a[3], 0)

        for i in [1, 2, 4]:
            self.assertIn(new_a[i], [1, 2])

        large_N = 10000
        large_domain = Domain(['a'], [2])
        large_data = {'a': np.ones(large_N, dtype=int)}
        large_dataset = Dataset(large_data, large_domain)

        decompressed_large = large_dataset.decompress(mapping)
        vals = decompressed_large.to_dict()['a']
        count_1 = np.sum(vals == 1)
        count_2 = np.sum(vals == 2)

        diff = abs(count_1 - count_2)
        self.assertTrue(diff < large_N * 0.1, f"Difference {diff} is too large for uniform distribution")

    def test_validation(self):
        domain = Domain(['a'], [3])
        data = {'a': np.array([0, 1, 2])}
        dataset = Dataset(data, domain)

        with self.assertRaises(ValueError):
            dataset.compress({'a': np.array([0, 1])})

        with self.assertRaises(ValueError):
            dataset.compress({'a': np.array([0, -1, 1])})

        with self.assertRaises(ValueError):
            dataset.compress({'a': np.array([0, 1.5, 1])})

    def test_compress_empty(self):
        domain = Domain(['a'], [3])
        data = {'a': np.array([], dtype=int)}
        dataset = Dataset(data, domain, weights=np.array([]))

        mapping = {'a': np.array([0, 1, 1])}

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
