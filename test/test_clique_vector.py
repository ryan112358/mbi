import unittest

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


if __name__ == '__main__':
  unittest.main()
