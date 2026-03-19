import os
import sys
import unittest

import numpy as np


MECHANISMS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "mechanisms")
)
if MECHANISMS_DIR not in sys.path:
    sys.path.insert(0, MECHANISMS_DIR)

from workload_utils import subsample_candidates


class TestWorkloadUtils(unittest.TestCase):
    def test_subsample_candidates_clamps_when_request_exceeds_available(self):
        candidates = [("a", "b"), ("a", "c"), ("b", "c")]
        prng = np.random.RandomState(0)

        selected = subsample_candidates(candidates, 99, prng)

        self.assertEqual(selected, candidates)

    def test_subsample_candidates_uses_prng_for_true_subsampling(self):
        candidates = [("a", "b"), ("a", "c"), ("b", "c"), ("a", "d")]
        prng = np.random.RandomState(0)

        selected = subsample_candidates(candidates, 2, prng)

        self.assertEqual(len(selected), 2)
        self.assertEqual(selected, [("b", "c"), ("a", "d")])


if __name__ == "__main__":
    unittest.main()
