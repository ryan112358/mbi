"""Tests for the logging Callback step numbering."""

import unittest

from mbi import callbacks, estimation
import numpy as np


class TestCallbackStepNumbering(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self._lines = []
    callbacks.set_log_fn(lambda *a, **k: self._lines.append(a))

  def tearDown(self):
    callbacks.set_log_fn(print)
    super().tearDown()

  def test_steps_are_multiples_of_cadence(self):
    # Each call represents estimation.CALLBACK_EVERY optimization steps, so the
    # reported step should advance by that cadence, not by one.
    cb = callbacks.Callback(loss_fns={})
    for _ in range(4):
      cb(None)
    steps = [row[0] for row in cb.summary["data"]]
    every = estimation.CALLBACK_EVERY
    self.assertEqual(steps, [0, every, 2 * every, 3 * every])

  def test_default_include_primal_feasibility(self):
    domain = callbacks.Domain(["a", "b"], [2, 2])
    measurements = [
        callbacks.LinearMeasurement(np.array([10.0, 10.0]), ("a",), 1.0)
    ]
    cb = callbacks.default(measurements, domain)
    self.assertIn("Primal Feas", cb.loss_fns)

    cb_no_pf = callbacks.default(
        measurements, domain, include_primal_feasibility=False
    )
    self.assertNotIn("Primal Feas", cb_no_pf.loss_fns)

    large_measurements = measurements * 51
    cb_large = callbacks.default(large_measurements, domain)
    self.assertNotIn("Primal Feas", cb_large.loss_fns)


if __name__ == "__main__":
  unittest.main()
