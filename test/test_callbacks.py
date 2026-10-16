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
    self.assertIn("Dual Gap", cb.loss_fns)
    self.assertIn("Primal Feas", cb.loss_fns)

    cb_no_pf = callbacks.default(
        measurements, domain, include_primal_feasibility=False
    )
    self.assertIn("Dual Gap", cb_no_pf.loss_fns)
    self.assertNotIn("Primal Feas", cb_no_pf.loss_fns)

    large_measurements = measurements * 51
    cb_large = callbacks.default(large_measurements, domain)
    self.assertIn("Dual Gap", cb_large.loss_fns)
    self.assertNotIn("Primal Feas", cb_large.loss_fns)

  def test_default_logs_primal_dual_gap(self):
    domain = callbacks.Domain(["a", "b"], [2, 2])
    measurements = [
        callbacks.LinearMeasurement(np.array([15.0, 5.0]), ("a",), 1.0),
        callbacks.LinearMeasurement(np.array([8.0, 12.0]), ("b",), 1.0),
    ]
    cb = callbacks.default(measurements, domain)
    marginals = callbacks.CliqueVector.ones(domain, [("a",), ("b",)]) * 10.0
    cb(marginals)

    cols = cb.summary["columns"]
    row = cb.summary["data"][0]
    gap_idx = cols.index("Dual Gap")
    loss_fn = callbacks.marginal_loss.from_linear_measurements(
        measurements, domain
    )
    expected_gap = float(estimation.primal_dual_gap(loss_fn, marginals))
    np.testing.assert_allclose(float(row[gap_idx]), expected_gap, rtol=1e-5)


if __name__ == "__main__":
  unittest.main()
