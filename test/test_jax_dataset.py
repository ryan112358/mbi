
import unittest
import jax.numpy as jnp
import numpy as np
import jax
from mbi.domain import Domain
from mbi.dataset import JaxDataset

class TestJaxDataset(unittest.TestCase):
    def setUp(self):
        self.attrs = ["a", "b"]
        self.shape = [3, 4]
        self.domain = Domain(self.attrs, self.shape)

        # Dict input
        self.data_dict = {
            "a": jnp.array([0, 1, 2]),
            "b": jnp.array([0, 2, 3])
        }
        self.dataset = JaxDataset(self.data_dict, self.domain)

    def test_init(self):
        # Test with dict input
        self.assertEqual(self.dataset.records, 3)
        self.assertTrue(isinstance(self.dataset.data, dict))
        self.assertEqual(len(self.dataset.data), 2)
        self.assertTrue("a" in self.dataset.data)
        self.assertTrue("b" in self.dataset.data)

        # Test 2D array failure
        with self.assertRaises(ValueError):
            JaxDataset(jnp.array([[0,0]]), self.domain)

    def test_project(self):
        # Project on "a"
        proj = self.dataset.project(["a"])
        self.assertEqual(proj.domain.attrs, ("a",))
        # Factor uses flattened datavector
        self.assertEqual(proj.datavector().size, 3)
        np.testing.assert_array_equal(proj.datavector(), np.array([1, 1, 1]))

        # Project on "b"
        proj_b = self.dataset.project(["b"])
        # b domain size is 4. Data has 0, 2, 3.
        # counts: 0->1, 1->0, 2->1, 3->1
        np.testing.assert_array_equal(proj_b.datavector(), np.array([1, 0, 1, 1]))

    def test_datavector(self):
        vec = self.dataset.datavector(flatten=True)
        self.assertEqual(vec.size, 12)
        # data: (0,0), (1,2), (2,3)
        # index: 0*4+0=0, 1*4+2=6, 2*4+3=11
        # counts should be 1 at 0, 6, 11
        expected = np.zeros(12)
        expected[0] = 1
        expected[6] = 1
        expected[11] = 1
        np.testing.assert_array_equal(vec, expected)

    def test_synthetic(self):
        syn = JaxDataset.synthetic(self.domain, 10)
        self.assertEqual(syn.records, 10)
        self.assertTrue(isinstance(syn.data, dict))
        self.assertEqual(len(syn.data), 2)

    def test_weights(self):
        weights = jnp.array([2.0, 1.0, 0.5])
        w_dataset = JaxDataset(self.data_dict, self.domain, weights=weights)
        vec = w_dataset.datavector(flatten=True)
        # index 0 (0,0) -> weight 2.0
        # index 6 (1,2) -> weight 1.0
        # index 11 (2,3) -> weight 0.5
        self.assertEqual(vec[0], 2.0)
        self.assertEqual(vec[6], 1.0)
        self.assertEqual(vec[11], 0.5)

if __name__ == "__main__":
    unittest.main()
