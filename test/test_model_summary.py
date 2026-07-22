"""Tests for model summary formatting, especially huge-domain safety."""

import unittest

from mbi import Domain
from mbi._model_summary import _fmt_bytes, _fmt_size, summarize


class TestFormatHelpers(unittest.TestCase):

  def test_fmt_size_regular(self):
    self.assertEqual(_fmt_size(500), "500")
    self.assertEqual(_fmt_size(2500), "2.50K")
    self.assertEqual(_fmt_size(3.4e6), "3.40M")
    self.assertEqual(_fmt_size(7e9), "7.00B")

  def test_fmt_size_huge_int_does_not_overflow(self):
    # 2**1100 (~1e331) exceeds float range; float(n) would raise OverflowError.
    self.assertEqual(_fmt_size(10**331), "1.00e331")
    self.assertEqual(_fmt_size(2**1100), "1.36e331")

  def test_fmt_bytes_regular(self):
    self.assertEqual(_fmt_bytes(1500), "1.46 KiB")
    self.assertEqual(_fmt_bytes(5 * 2**30), "5.00 GiB")

  def test_fmt_bytes_huge_int_does_not_overflow(self):
    self.assertEqual(_fmt_bytes(10**400), "1.00e400 B")


class TestSummarizeHugeDomain(unittest.TestCase):

  def test_astronomical_domain_size(self):
    # 1100 binary attributes -> domain size 2**1100, which overflows float.
    # summarize / str(summary) must not raise OverflowError.
    domain = Domain([f"x{i}" for i in range(1100)], [2] * 1100)
    cliques = [(f"x{i}",) for i in range(1100)]
    summary = summarize(domain, cliques)
    self.assertIn("1.36e331 total cells", str(summary))


if __name__ == "__main__":
  unittest.main()
