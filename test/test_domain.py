import unittest
from mbi.domain import Domain


class TestDomain(unittest.TestCase):
    def setUp(self):
        attrs = ["a", "b", "c", "d"]
        shape = [10, 20, 30, 40]
        self.domain = Domain(attrs, shape)

    def test_eq(self):
        attrs = ["a", "b", "c", "d"]
        shape = [10, 20, 30, 40]
        ans = Domain(attrs, shape)
        self.assertEqual(self.domain, ans)

        attrs = ["b", "a", "c", "d"]
        ans = Domain(attrs, shape)
        self.assertNotEqual(self.domain, ans)

    def test_project(self):
        ans = Domain(["a", "b"], [10, 20])
        res = self.domain.project(["a", "b"])
        self.assertEqual(ans, res)

        ans = Domain(["c", "b"], [30, 20])
        res = self.domain.project(["c", "b"])
        self.assertEqual(ans, res)

    def test_marginalize(self):
        ans = Domain(["a", "b"], [10, 20])
        res = self.domain.marginalize(["c", "d"])
        self.assertEqual(ans, res)

        res = self.domain.marginalize(["c", "d", "e"])
        self.assertEqual(ans, res)

    def test_axes(self):
        ans = (1, 3)
        res = self.domain.axes(["b", "d"])
        self.assertEqual(ans, res)

    def test_transpose(self):
        ans = Domain(["b", "d", "a", "c"], [20, 40, 10, 30])
        res = self.domain.project(["b", "d", "a", "c"])
        self.assertEqual(ans, res)

    def test_merge(self):
        ans = Domain(["a", "b", "c", "d", "e", "f"], [10, 20, 30, 40, 50, 60])
        new = Domain(["b", "d", "e", "f"], [20, 40, 50, 60])
        res = self.domain.merge(new)
        self.assertEqual(ans, res)

    def test_contains(self):
        new = Domain(["b", "d"], [20, 40])
        self.assertTrue(self.domain.contains(new))

        new = Domain(["b", "e"], [20, 50])
        self.assertFalse(self.domain.contains(new))

    def test_iter(self):
        self.assertEqual(len(self.domain), 4)
        for a, b, c in zip(self.domain, ["a", "b", "c", "d"], [10, 20, 30, 40]):
            self.assertEqual(a, b)
            self.assertEqual(self.domain[a], c)

    def test_repeated_attributes(self):
        with self.assertRaises(ValueError):
            Domain(["a", "a"], [10, 20])

    def test_intersect_disjoint_domains(self):
        other_domain = Domain(["e", "f"], [5, 6])
        intersection = self.domain.intersect(other_domain)
        self.assertEqual(intersection.attributes, ())
        self.assertEqual(intersection.shape, ())

    def test_labels(self):
        attrs = ["a", "b"]
        shape = [2, 3]
        labels = [["yes", "no"], ["small", "medium", "large"]]
        dom = Domain(attrs, shape, labels=labels)
        self.assertIsNotNone(dom.labels)
        self.assertEqual(dom.labels, tuple(tuple(l) for l in labels))

        # Test project
        proj = dom.project(["b"])
        self.assertEqual(proj.labels, (("small", "medium", "large"),))

        # Test merge
        dom2 = Domain(["b", "c"], [3, 2], labels=[["small", "medium", "large"], ["up", "down"]])
        merged = dom.merge(dom2)
        # expected: a, b, c
        self.assertEqual(merged.attributes, ("a", "b", "c"))
        # labels should be preserved
        expected_labels = (("yes", "no"), ("small", "medium", "large"), ("up", "down"))
        self.assertEqual(merged.labels, expected_labels)

        # Test merge mismatch
        dom3 = Domain(["c"], [2]) # no labels
        merged_mismatch = dom.merge(dom3)
        self.assertIsNone(merged_mismatch.labels)

        # Test invalid labels
        with self.assertRaises(ValueError):
            Domain(attrs, shape, labels=[["yes"], ["small", "medium", "large"]]) # wrong length for 'a'

        with self.assertRaises(ValueError):
            Domain(attrs, shape, labels=[["yes", "no"]]) # wrong number of label lists


if __name__ == "__main__":
    unittest.main()
