import importlib
import logging
import unittest
import jax
import numpy as np
from mbi import (
    Domain,
    Factor,
    CliqueVector,
    MarkovRandomField,
    marginal_oracles,
)
from mbi.extensions.synthetic_data import synthetic_data as ext_synthetic_data


def _create_random_model(domain, cliques, N):
  """Creates a random MRF with given cliques and total count N."""
  potentials = {}
  np.random.seed(0)

  for cl in cliques:
    vals = np.random.rand(*domain.project(cl).shape) + 1e-10
    f = Factor(domain.project(cl), vals)
    potentials[cl] = f.log()

  potential_vector = CliqueVector(domain, cliques, potentials)
  marginals = marginal_oracles.message_passing_stable(potential_vector, total=N)

  return MarkovRandomField(
      potentials=potential_vector, marginals=marginals, total=N
  )


class TestExtSyntheticDataAccuracy(unittest.TestCase):
  """Tests that extensions.synthetic_data matches MRF.synthetic_data.

  The MRF implementation is the reference.  These tests verify that the
  JAX-based extensions version produces equivalent marginals across
  multiple graph structures, including the ones that triggered the
  clique-conditioning regression (using original measurement cliques
  instead of junction tree maximal cliques for parent determination).
  """

  def _assert_parity(self, model, cliques, N, tol=0.01):
    """Assert extensions synth matches MRF synth within tolerance.

    For each model clique we check that the normalized L1 gap between
    extensions and MRF synthetic marginals is small.  Since both use
    rounding (not sampling), the gap should be near zero at large N.

    Args:
        model: A fitted MarkovRandomField.
        cliques: Cliques to evaluate.
        N: Number of rows to generate.
        tol: Maximum allowed normalized L1/2 gap per clique.
    """
    np.random.seed(0)
    synth_mrf = model.synthetic_data(rows=N, method="round")
    synth_ext = ext_synthetic_data(model, rows=N)

    for cl in cliques:
      mrf_marg = synth_mrf.project(cl).datavector(flatten=True)
      ext_marg = synth_ext.project(cl).datavector(flatten=True)

      mrf_norm = mrf_marg / max(mrf_marg.sum(), 1e-20)
      ext_norm = ext_marg / max(ext_marg.sum(), 1e-20)

      gap = 0.5 * np.sum(np.abs(mrf_norm - ext_norm))
      self.assertLess(
          gap,
          tol,
          f"Extensions gap {gap:.6f} exceeds tolerance {tol} for clique {cl}",
      )

  def _test_model_structure(self, domain, cliques, cross_cliques=None):
    """Test a specific model structure for MRF/extensions parity."""
    N = 50000
    model = _create_random_model(domain, cliques, N)
    all_cliques = list(cliques) + (cross_cliques or [])
    self._assert_parity(model, all_cliques, N)

  def test_single_clique(self):
    """Single clique: no conditioning needed."""
    domain = Domain(["A", "B"], [10, 10])
    cliques = [("A", "B")]
    self._test_model_structure(domain, cliques)

  def test_independent_cliques(self):
    """Two disjoint cliques: no shared attributes."""
    domain = Domain(["A", "B", "C", "D"], [5, 5, 5, 5])
    cliques = [("A", "B"), ("C", "D")]
    self._test_model_structure(
        domain, cliques, cross_cliques=[("A", "C"), ("B", "D")]
    )

  def test_chain(self):
    """Chain A-B-C: conditioning through shared attribute B."""
    domain = Domain(["A", "B", "C"], [5, 5, 5])
    cliques = [("A", "B"), ("B", "C")]
    self._test_model_structure(domain, cliques, cross_cliques=[("A", "C")])

  def test_star(self):
    """Star with hub: spokes share a hub but not each other."""
    domain = Domain(["H", "A", "B", "C", "D"], [5, 4, 4, 4, 4])
    cliques = [("H", "A"), ("H", "B"), ("H", "C"), ("H", "D")]
    self._test_model_structure(
        domain,
        cliques,
        cross_cliques=[("A", "B"), ("A", "C"), ("B", "D")],
    )

  def test_two_hubs_shared_separator(self):
    """Two hubs with shared separators — the structure that triggered
    the clique-conditioning bug.

    Hub1 (H1) connects to A, B, C.
    Hub2 (H2) connects to A, B, C.
    The junction tree merges {H1, A, B, C} and {H2, A, B, C} into
    super-cliques with separator {A, B, C}.  Without the fix, A, B, C
    are generated independently instead of from their joint.
    """
    domain = Domain(["H1", "H2", "A", "B", "C"], [4, 4, 3, 3, 3])
    cliques = [
        ("H1", "A"),
        ("H1", "B"),
        ("H1", "C"),
        ("H2", "A"),
        ("H2", "B"),
        ("H2", "C"),
    ]
    self._test_model_structure(
        domain,
        cliques,
        cross_cliques=[
            ("A", "B"),
            ("A", "C"),
            ("B", "C"),
            ("H1", "H2"),
        ],
    )

  def test_overlapping_triples(self):
    """Overlapping 3-way cliques that create 4-way super-cliques."""
    domain = Domain(["A", "B", "C", "D", "E"], [3, 3, 3, 3, 3])
    cliques = [
        ("A", "B", "C"),
        ("B", "C", "D"),
        ("D", "E"),
    ]
    self._test_model_structure(
        domain,
        cliques,
        cross_cliques=[("A", "D"), ("A", "E"), ("C", "E")],
    )

  def test_two_clusters_with_bridge(self):
    """Two dense clusters connected by a single bridge edge.

    Cluster 1: all pairs of {A, B, C} → super-clique (A, B, C)
    Cluster 2: all pairs of {D, E, F} → super-clique (D, E, F)
    Bridge: (C, D)
    """
    domain = Domain(["A", "B", "C", "D", "E", "F"], [3, 3, 3, 3, 3, 3])
    cliques = [
        ("A", "B"),
        ("A", "C"),
        ("B", "C"),
        ("D", "E"),
        ("D", "F"),
        ("E", "F"),
        ("C", "D"),
    ]
    self._test_model_structure(
        domain,
        cliques,
        cross_cliques=[("A", "D"), ("B", "E"), ("A", "F")],
    )

  def test_mixed_arity_cliques(self):
    """Mix of 1-way, 2-way, and 3-way cliques."""
    domain = Domain(["A", "B", "C", "D", "E"], [3, 3, 3, 3, 3])
    cliques = [("A",), ("A", "B"), ("B", "C", "D"), ("D", "E")]
    self._test_model_structure(
        domain,
        cliques,
        cross_cliques=[("A", "C"), ("A", "E"), ("C", "E")],
    )


class TestExtSyntheticDataPrecompile(unittest.TestCase):
  """Tests that precompile() lowers against the signatures generation uses.

  precompile() ahead-of-time lowers ``_generate_column`` for each column; for
  the resulting compiled executables to be reused, the argument avals (shape +
  dtype) it lowers against must exactly match those ``synthetic_data`` passes
  at generation.  Two dtype mismatches previously defeated this whenever
  ``jax_enable_x64`` was set, forcing a recompile on every one of the columns:

    * ``Factor.abstract`` emitted float32 while real factors were float64, and
    * parent arrays were passed as ``np.min_scalar_type`` (uint8/uint16/...)
      instead of the int32 signature precompile lowered against.

  This test records the per-column avals from both paths and asserts they are
  identical, which is the precise cache-hit precondition.
  """

  def setUp(self):
    super().setUp()
    # precompile() compiles in a background thread that does not inherit a
    # thread-local ``jax.enable_x64()`` context, so enable x64 globally here.
    self._prev_x64 = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)

  def tearDown(self):
    jax.config.update("jax_enable_x64", self._prev_x64)
    super().tearDown()

  def _leaf_avals(self, *args):
    return tuple(
        (tuple(leaf.shape), str(leaf.dtype))
        for leaf in jax.tree_util.tree_leaves(args)
        if hasattr(leaf, "shape") and hasattr(leaf, "dtype")
    )

  def test_precompile_signature_matches_generation(self):
    import importlib

    # The __init__ re-exports the ``synthetic_data`` function, shadowing the
    # submodule of the same name, so import it explicitly from sys.modules.
    sd = importlib.import_module("mbi.extensions.synthetic_data")
    real = sd._generate_column

    # Use a chain so at least one column is generated with parents, exercising
    # the parent-array dtype path.  The bugs only manifest under x64, where the
    # default float dtype is float64 (enabled globally in setUp).
    domain = Domain(["A", "B", "C"], [5, 5, 5])
    cliques = [("A", "B"), ("B", "C")]
    model = _create_random_model(domain, cliques, N=1000)
    rows = 1000

    precompile_sigs = {}
    generation_sigs = {}

    class _PrecompileRecorder:

      def lower(inner, prng, inputs, parents, *, query, **kwargs):  # pylint: disable=no-self-argument
        precompile_sigs[query] = self._leaf_avals(prng, inputs, parents)
        return real.lower(prng, inputs, parents, query=query, **kwargs)

    class _GenerationRecorder:

      def __call__(inner, prng, inputs, parents, *, query, **kwargs):  # pylint: disable=no-self-argument
        generation_sigs[query] = self._leaf_avals(prng, inputs, parents)
        return real(prng, inputs, parents, query=query, **kwargs)

    try:
      sd._generate_column = _PrecompileRecorder()
      sd.precompile(domain, list(model.cliques), rows).result()
      sd._generate_column = _GenerationRecorder()
      sd.synthetic_data(model, rows)
    finally:
      sd._generate_column = real

    self.assertTrue(precompile_sigs, "precompile lowered no columns")
    self.assertEqual(
        set(precompile_sigs), set(generation_sigs), "column sets differ"
    )
    # At least one column must be generated with a parent (the chain
    # guarantees it): a parent appears as an int32 array of shape (rows,).
    self.assertTrue(
        any(((rows,), "int32") in sig for sig in generation_sigs.values()),
        "expected at least one column generated with an int32 parent array",
    )
    for query, gen_sig in generation_sigs.items():
      self.assertEqual(
          precompile_sigs[query],
          gen_sig,
          f"aval mismatch for column {query}: precompile lowered "
          f"{precompile_sigs[query]} but generation passed {gen_sig}; "
          "the JIT cache would miss and recompile.",
      )


class TestExtSyntheticDataCacheHit(unittest.TestCase):
  """Behavioral test: synthetic_data() reuses precompile()'s JIT cache.

  Complements the aval-signature test above by asserting the *observable*
  consequence: after precompile() warms the cache, generation triggers zero
  new compilations of the per-column kernel ``_generate_column``.  Checked in
  both the default (float32) and ``jax_enable_x64`` (float64) regimes, since
  the dtype mismatches that used to defeat reuse only surfaced under x64.
  """

  class _CompileCounter(logging.Handler):
    """Counts JAX 'Compiling ... _generate_column' log records."""

    def __init__(self):
      super().__init__()
      self.n = 0

    def emit(self, record):
      try:
        msg = record.getMessage()
      except Exception:  # pylint: disable=broad-exception-caught
        return
      if "Compiling" in msg and "_generate_column" in msg:
        self.n += 1

  def _run_regime(self, enable_x64):
    sd = importlib.import_module("mbi.extensions.synthetic_data")
    # ``jax_log_compiles`` / ``jax_enable_compilation_cache`` are contextmanager
    # flags, which must be read via attribute access rather than config.read().
    prev_x64 = jax.config.jax_enable_x64
    prev_log = jax.config.jax_log_compiles
    prev_cache = jax.config.jax_enable_compilation_cache
    counter = self._CompileCounter()
    loggers = [logging.getLogger(name) for name in ("", "jax", "jax._src")]
    prev_levels = [(lg, lg.level) for lg in loggers]
    try:
      jax.config.update("jax_enable_x64", enable_x64)
      # Disable the persistent on-disk cache and clear the in-memory cache so
      # that precompile() genuinely compiles (emitting the log records we
      # count) rather than loading a warm executable from a previous run.
      jax.config.update("jax_enable_compilation_cache", False)
      jax.clear_caches()
      jax.config.update("jax_log_compiles", True)
      for lg in loggers:
        lg.setLevel(logging.DEBUG)
        lg.addHandler(counter)

      # A chain guarantees columns generated with parents, exercising the
      # parent-array dtype path that broke reuse under x64.
      domain = Domain(["A", "B", "C"], [5, 5, 5])
      cliques = [("A", "B"), ("B", "C")]
      model = _create_random_model(domain, cliques, N=1000)
      rows = 1000

      sd.precompile(domain, list(model.cliques), rows).result()
      n_after_precompile = counter.n
      sd.synthetic_data(model, rows)
      n_after_generate = counter.n
    finally:
      for lg in loggers:
        lg.removeHandler(counter)
      for lg, level in prev_levels:
        lg.setLevel(level)
      jax.config.update("jax_log_compiles", prev_log)
      jax.config.update("jax_enable_compilation_cache", prev_cache)
      jax.config.update("jax_enable_x64", prev_x64)

    # precompile() must actually have compiled the kernel, otherwise the test
    # is vacuous; then generation must add no further compilations.
    self.assertGreater(
        n_after_precompile,
        0,
        "precompile() compiled no _generate_column; test is vacuous",
    )
    self.assertEqual(
        n_after_generate,
        n_after_precompile,
        "synthetic_data() recompiled _generate_column "
        f"({n_after_generate - n_after_precompile} new compiles) with "
        f"jax_enable_x64={enable_x64}: the precompile() cache missed.",
    )

  def test_cache_hit_x64(self):
    """Under jax_enable_x64 (float64), generation reuses the precompiled cache."""
    self._run_regime(enable_x64=True)

  def test_cache_hit_default(self):
    """Under the default (float32) regime, generation reuses the cache."""
    self._run_regime(enable_x64=False)


if __name__ == "__main__":
  unittest.main()
