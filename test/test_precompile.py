"""End-to-end tests for ahead-of-time precompilation across mbi.

These tests pin down the JIT-cache-reuse *contract* that the two ``precompile``
entry points rely on:

  * ``mbi.estimation.Estimator.precompile`` (warms ``_multi_step`` for
    ``estimate``), and
  * ``mbi.extensions.synthetic_data.precompile`` (warms ``_generate_column``
    for ``synthetic_data``).

They are intentionally more thorough than the per-module unit tests because a
regression here is *silent*: precompile would stop saving compile time (a full
recompile happens at ``estimate``/``synthetic_data`` time) without changing any
numerical output. The only observable symptom is a large latency spike, so we
assert the reuse directly by counting XLA traces.

The tests also exercise the dtype-stability invariants that make reuse possible,
under both ``jax_enable_x64=False`` (float32) and ``True`` (float64). Dtype
mismatches between the abstract values used for precompilation and the concrete
values used at run time are the most common way to accidentally defeat the JIT
cache; several such bugs motivated this file:

  * ``Factor.abstract`` must track ``Factor.zeros``' dtype (``result_type(float)``)
    rather than pinning float32/float64.
  * ``minimum_variance_unbiased_total`` must return a Python ``float`` even in
    the ``<= 1`` fallback branch; returning an ``int`` makes ``known_total`` an
    int leaf that mismatches the float total lowered by ``precompile`` and forces
    a recompile.
  * synthetic-data parent-index arrays must be pinned to int32 on both the
    abstract and concrete paths.
"""

import contextlib
import importlib
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np

from mbi import (
    CliqueVector,
    Domain,
    Factor,
    MarkovRandomField,
    estimation,
    marginal_loss,
    marginal_oracles,
)
# ``mbi.extensions.__init__`` re-exports the ``synthetic_data`` *function*, which
# shadows the submodule of the same name under normal ``import ... as`` (that
# resolves via the parent package attribute). Fetch the module object directly so
# we can reach module-level helpers like ``_gumbel_round``.
synthetic_data = importlib.import_module("mbi.extensions.synthetic_data")

np.random.seed(0)  # Avoid flaky tests.

_DOMAIN = Domain(["a", "b", "c", "d"], [2, 3, 4, 5])
_CLIQUES = [("a", "b"), ("b", "c"), ("c", "d")]


@contextlib.contextmanager
def _x64(enabled: bool):
  """Temporarily set ``jax_enable_x64``, clearing caches around the switch.

  Changing x64 changes the default float/int dtypes, so any cached traces from
  the other mode must be dropped; otherwise reuse counters would be polluted by
  executables compiled under the previous dtype regime.
  """
  prev = bool(jax.config.jax_enable_x64)  # pylint: disable=no-member
  if prev == enabled:
    # Still clear so each test starts from a cold, deterministic cache.
    jax.clear_caches()
    yield
    return
  jax.clear_caches()
  jax.config.update("jax_enable_x64", enabled)
  try:
    yield
  finally:
    jax.config.update("jax_enable_x64", prev)
    jax.clear_caches()


@contextlib.contextmanager
def _count_multi_step_traces():
  """Count how many times ``MirrorDescent._multi_step`` is traced.

  ``_multi_step`` wraps its body in ``lax.scan(step, ...)`` and ``step`` calls
  ``self._step`` exactly once per trace. A jitted function only re-runs its
  Python body when it (re)compiles, so the number of ``_step`` invocations
  equals the number of ``_multi_step`` compilations. Reusing a precompiled
  executable performs zero additional traces.
  """
  counter = {"n": 0}
  orig = estimation.MirrorDescent._step

  def wrapper(self, *args, **kwargs):
    counter["n"] += 1
    return orig(self, *args, **kwargs)

  with mock.patch.object(estimation.MirrorDescent, "_step", wrapper):
    yield counter


@contextlib.contextmanager
def _count_generate_column_traces():
  """Count how many times ``_generate_column`` is traced.

  ``_generate_column`` calls the module-level ``_gumbel_round`` exactly once per
  trace, so counting ``_gumbel_round`` invocations counts column (re)compiles
  summed over all column signatures.
  """
  counter = {"n": 0}
  orig = synthetic_data._gumbel_round

  def wrapper(*args, **kwargs):
    counter["n"] += 1
    return orig(*args, **kwargs)

  with mock.patch.object(synthetic_data, "_gumbel_round", wrapper):
    yield counter


def _measurements(total: float = 1.0):
  """Identity measurements over ``_CLIQUES`` for a model of the given total."""
  p = Factor.random(_DOMAIN)
  p = total * p / p.sum()
  return [
      marginal_loss.LinearMeasurement(p.project(cl).datavector(), cl)
      for cl in _CLIQUES
  ]


def _random_model(total: float):
  """Build a fitted ``MarkovRandomField`` directly (no estimation loop)."""
  potentials = {}
  for cl in _CLIQUES:
    vals = np.random.rand(*_DOMAIN.project(cl).shape) + 1e-10
    potentials[cl] = Factor(_DOMAIN.project(cl), vals).log()
  pv = CliqueVector(_DOMAIN, _CLIQUES, potentials)
  marginals = marginal_oracles.message_passing_stable(pv, total=total)
  return MarkovRandomField(potentials=pv, marginals=marginals, total=total)


class TestEstimatorPrecompileReuse(unittest.TestCase):
  """``precompile`` must make ``estimate`` reuse the compiled ``_multi_step``."""

  def _check_reuse(self):
    jax.clear_caches()
    est = estimation.MirrorDescent()
    meas = _measurements(total=100.0)
    with _count_multi_step_traces() as counter:
      est.precompile(_DOMAIN, meas).result()
      after_precompile = counter["n"]
      self.assertGreaterEqual(
          after_precompile, 1, "precompile should trace _multi_step"
      )
      model = est.estimate(_DOMAIN, meas, known_total=100.0, iters=100)
      self.assertEqual(
          counter["n"],
          after_precompile,
          "estimate must reuse the precompiled executable (no recompile)",
      )
    self.assertIsNotNone(model)

  def test_reuse_float32(self):
    with _x64(False):
      self._check_reuse()

  def test_reuse_float64(self):
    with _x64(True):
      self._check_reuse()

  def test_estimate_without_precompile_compiles_once(self):
    # Sanity: even across multiple CALLBACK_EVERY rounds a single estimate call
    # compiles _multi_step exactly once (the scan is reused round-to-round).
    with _x64(True):
      jax.clear_caches()
      est = estimation.MirrorDescent()
      meas = _measurements(total=100.0)
      with _count_multi_step_traces() as counter:
        # iters spanning >1 CALLBACK_EVERY round.
        est.estimate(
            _DOMAIN,
            meas,
            known_total=100.0,
            iters=2 * estimation.CALLBACK_EVERY + 1,
        )
      self.assertEqual(counter["n"], 1)


class TestKnownTotalDtype(unittest.TestCase):
  """The known_total float floor must not defeat the precompile cache."""

  def test_mvue_returns_float_for_small_totals(self):
    # Normalized measurements => estimated total == 1.0 (the fallback branch).
    meas = _measurements(total=1.0)
    total = estimation.minimum_variance_unbiased_total(meas)
    self.assertIsInstance(total, float)
    np.testing.assert_allclose(total, 1.0, rtol=1e-5)

  def test_small_total_does_not_trigger_recompile(self):
    # Regression: precompile lowers _multi_step with a float total (1.0). If the
    # auto-computed known_total came back as int 1, estimate would pass an int
    # leaf, miss the cache, and recompile. With the float floor it reuses.
    with _x64(True):
      jax.clear_caches()
      est = estimation.MirrorDescent()
      meas = _measurements(total=1.0)
      with _count_multi_step_traces() as counter:
        est.precompile(_DOMAIN, meas).result()
        after_precompile = counter["n"]
        # known_total omitted => auto MVUE, which must be float 1.0 (not int 1).
        est.estimate(_DOMAIN, meas, iters=100)
        self.assertEqual(
            counter["n"],
            after_precompile,
            "int known_total would force a recompile here",
        )


class TestAbstractDtypeStability(unittest.TestCase):
  """Abstract constructors must track the active float dtype (x32 vs x64)."""

  def _check(self, expected):
    self.assertEqual(
        Factor.abstract(_DOMAIN).values.dtype,
        Factor.zeros(_DOMAIN).values.dtype,
    )
    self.assertEqual(Factor.abstract(_DOMAIN).values.dtype, expected)
    cv_abstract = CliqueVector.abstract(_DOMAIN, _CLIQUES)
    cv_zeros = CliqueVector.zeros(_DOMAIN, _CLIQUES)
    for cl in _CLIQUES:
      self.assertEqual(cv_abstract[cl].values.dtype, cv_zeros[cl].values.dtype)
      self.assertEqual(cv_abstract[cl].values.dtype, expected)

  def test_float32(self):
    with _x64(False):
      self._check(jnp.float32)

  def test_float64(self):
    with _x64(True):
      self._check(jnp.float64)


class TestSyntheticDataPrecompile(unittest.TestCase):
  """``precompile`` must make ``synthetic_data`` reuse ``_generate_column``."""

  def _check_reuse(self, rows):
    jax.clear_caches()
    model = _random_model(total=float(rows))
    with _count_generate_column_traces() as counter:
      synthetic_data.precompile(_DOMAIN, model.cliques, rows).result()
      after_precompile = counter["n"]
      self.assertGreaterEqual(
          after_precompile, 1, "precompile should trace _generate_column"
      )
      data = synthetic_data.synthetic_data(model, rows, seed=0)
      self.assertEqual(
          counter["n"],
          after_precompile,
          "synthetic_data must reuse precompiled columns (no recompile)",
      )
    self.assertEqual(data.records, rows)

  def test_reuse_float64(self):
    with _x64(True):
      self._check_reuse(rows=200)

  def test_generate_float32(self):
    # Generation should at least run correctly in float32 mode.
    with _x64(False):
      jax.clear_caches()
      model = _random_model(total=200.0)
      data = synthetic_data.synthetic_data(model, rows=200, seed=0)
      self.assertEqual(data.records, 200)
      for col in _DOMAIN.attributes:
        vals = data.project((col,)).datavector()
        self.assertEqual(vals.sum(), 200)


if __name__ == "__main__":
  unittest.main()
