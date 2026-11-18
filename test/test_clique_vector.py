import unittest

import numpy as np

import mbi


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

  def test_projectable_preserves_64bit_numpy(self):
    cols = {'a': np.array([0, 1]), 'b': np.array([0, 1]), 'c': np.array([0, 1])}
    w1 = np.array([2**26 + 4.0, 1000.0], dtype=np.float64)
    w2 = np.array([2**26 + 5.0, 1000.0], dtype=np.float64)
    cv1 = mbi.CliqueVector.from_projectable(
        mbi.Dataset(cols, self.domain, weights=w1), [('a', 'b'), ('b', 'c')]
    )
    cv2 = mbi.CliqueVector.from_projectable(
        mbi.Dataset(cols, self.domain, weights=w2), [('a', 'b'), ('b', 'c')]
    )

    for cl in [('a', 'b'), ('a',), ()]:
      f1 = cv1.project(cl)
      f2 = cv2.project(cl)
      self.assertIsInstance(f1.values, np.ndarray)
      diff = f2.datavector() - f1.datavector()
      self.assertEqual(diff.dtype, np.float64)
      self.assertEqual(float(diff[0]), 1.0)

    mrf = mbi.mrf_from_marginals(cv1, iters=5)
    self.assertIsInstance(mrf, mbi.MarkovRandomField)


if __name__ == '__main__':
  unittest.main()
