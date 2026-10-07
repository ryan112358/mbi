import unittest

import mbi
import numpy as np


class TestCliqueVectorParent(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.domain = mbi.Domain.fromdict({'a': 2, 'b': 3, 'c': 4})

  def test_prefers_smallest_domain_superset(self):
    # (a,) fits in both (a,b) [size 6] and (a,c) [size 8]; pick the smaller.
    cv = mbi.CliqueVector.zeros(self.domain, [('a', 'c'), ('a', 'b')])
    self.assertEqual(cv.parent(('a',)), ('a', 'b'))

  def test_order_independent(self):
    cv1 = mbi.CliqueVector.zeros(self.domain, [('a', 'c'), ('a', 'b')])
    cv2 = mbi.CliqueVector.zeros(self.domain, [('a', 'b'), ('a', 'c')])
    self.assertEqual(cv1.parent(('a',)), cv2.parent(('a',)))

  def test_returns_self_when_present(self):
    cv = mbi.CliqueVector.zeros(self.domain, [('a', 'b'), ('a', 'c')])
    self.assertEqual(cv.parent(('a', 'b')), ('a', 'b'))

  def test_none_when_no_superset(self):
    cv = mbi.CliqueVector.zeros(self.domain, [('a', 'b')])
    self.assertIsNone(cv.parent(('c',)))


class TestCliqueVectorSlice(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.domain = mbi.Domain.fromdict({'a': 2, 'b': 3, 'c': 4})

  def test_slice_empty_evidence(self):
    cv = mbi.CliqueVector.ones(self.domain, [('a', 'b'), ('b', 'c')])
    sliced = cv.slice({})
    self.assertIs(sliced, cv)

  def test_slice_drops_contained_and_sums_colliding_cliques(self):
    f_a = mbi.Factor(self.domain.project(('a',)), np.array([1.0, 2.0]))
    f_ab = mbi.Factor(
        self.domain.project(('a', 'b')),
        np.arange(6, dtype=float).reshape(2, 3),
    )
    f_b = mbi.Factor(self.domain.project(('b',)), np.array([10.0, 20.0, 30.0]))
    f_bc = mbi.Factor.ones(self.domain.project(('b', 'c')))
    cv = mbi.CliqueVector(
        self.domain,
        [('a',), ('a', 'b'), ('b',), ('b', 'c')],
        {('a',): f_a, ('a', 'b'): f_ab, ('b',): f_b, ('b', 'c'): f_bc},
    )

    sliced = cv.slice({'a': 1})
    self.assertEqual(sliced.domain, self.domain.project(('b', 'c')))
    self.assertEqual(sliced.cliques, (('b',), ('b', 'c')))
    expected_b = f_ab.slice({'a': 1}).values + f_b.values
    np.testing.assert_allclose(sliced[('b',)].values, expected_b)
    np.testing.assert_allclose(sliced[('b', 'c')].values, f_bc.values)

  def test_slice_all_attributes(self):
    cv = mbi.CliqueVector.ones(self.domain, [('a', 'b'), ('b', 'c')])
    sliced = cv.slice({'a': 0, 'b': 1, 'c': 2})
    self.assertEqual(len(sliced.domain), 0)
    self.assertEqual(sliced.cliques, ())


if __name__ == '__main__':
  unittest.main()
