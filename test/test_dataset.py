import unittest
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
        import numpy as np
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
        import numpy as np
        # Expected: [2, 1, 0, 0, 0, 1]
        expected = np.array([2, 1, 0, 0, 0, 1])
        result = self.dataset.datavector(flatten=True)
        np.testing.assert_array_equal(result, expected)

        # Unflattened: [[2, 1, 0], [0, 0, 1]]
        expected_unflat = np.array([[2, 1, 0], [0, 0, 1]])
        result_unflat = self.dataset.datavector(flatten=False)
        np.testing.assert_array_equal(result_unflat, expected_unflat)

    def test_project_unweighted(self):
        import numpy as np
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
        import numpy as np
        # Weights: [1.0, 2.0, 0.5, 1.0] for indices [0, 1, 5, 0]
        # Expected: [2.0, 2.0, 0, 0, 0, 0.5]
        expected = np.array([2.0, 2.0, 0, 0, 0, 0.5])
        result = self.weighted_dataset.datavector(flatten=True)
        np.testing.assert_array_equal(result, expected)

    def test_project_weighted(self):
        import numpy as np
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
        import numpy as np
        # Project (A, B) - should be same as full datavector
        expected = np.array([2, 1, 0, 0, 0, 1])
        proj = self.dataset.project(['A', 'B'])
        np.testing.assert_array_equal(proj.datavector(), expected)


if __name__ == "__main__":
    unittest.main()
