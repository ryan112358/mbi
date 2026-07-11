"""Tests for the Constraint dataclass and constraint-aware inference.

Covers construction, validation, potential generation, coarsen/refine
primitives, and constrained message passing against brute-force baselines.
"""

import unittest

import jax.numpy as jnp
import numpy as np
from mbi import CliqueVector, Constraint, Domain, Factor
from mbi.extensions.message_passing import coarsen
from mbi.extensions.message_passing import implicit as ext_implicit
from mbi.extensions.message_passing import project_to_coarse
from mbi.extensions.message_passing import refine
from mbi.extensions.message_passing import shafer_shenoy as ext_shafer_shenoy
from mbi.marginal_oracles import message_passing_shafer_shenoy
from parameterized import parameterized


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mapping_constraint(fine, coarse, mapping):
  """Shorthand: build a Constraint from variable names and a mapping array."""
  n_fine = len(mapping)
  n_coarse = int(np.max(mapping)) + 1
  return Constraint(
      domain=Domain([fine, coarse], [n_fine, n_coarse]), mapping=mapping
  )


def _make_constraint_factor(domain, constraint):
  """Build the explicit 0/-inf constraint factor for baseline comparison."""
  fine, coarse = constraint.domain.attributes
  n_fine, n_coarse = constraint.domain.shape
  shape = (n_fine, n_coarse)
  vals = np.full(shape, -np.inf)
  for a, a_prime in enumerate(constraint.mapping):
    vals[a, a_prime] = 0.0
  dom = Domain([fine, coarse], list(shape))
  return Factor(dom, jnp.array(vals)).transpose(
      domain.canonical(constraint.clique)
  )


def _baseline_marginals(domain, cliques, potentials, constraints, total=10.0):
  """Marginals via standard Shafer-Shenoy with explicit -inf constraints."""
  if isinstance(constraints, Constraint):
    constraints = [constraints]
  arrays = {cl: potentials[cl] for cl in cliques}
  all_cliques = list(cliques)
  for c in constraints:
    con_cl = domain.canonical(c.clique)
    con_factor = _make_constraint_factor(domain, c)
    if con_cl in arrays:
      arrays[con_cl] = arrays[con_cl] + con_factor
    else:
      arrays[con_cl] = con_factor
      all_cliques.append(con_cl)
  return message_passing_shafer_shenoy(
      CliqueVector(domain, all_cliques, arrays), total
  )


def _random_surjection(n_domain, n_range, rng):
  """Generate a random surjective mapping from [n_domain] to [n_range]."""
  mapping = np.zeros(n_domain, dtype=int)
  mapping[:n_range] = np.arange(n_range)
  mapping[n_range:] = rng.integers(0, n_range, size=n_domain - n_range)
  rng.shuffle(mapping)
  return mapping


_EXTENSION_ORACLES = [ext_shafer_shenoy, ext_implicit]


def _assert_matches_baseline(
    test, domain, cliques, constraints, total=10.0, seed=None, oracle=None
):
  """Assert constraint-aware message passing matches the -inf baseline."""
  if seed is not None:
    np.random.seed(seed)
  potentials = CliqueVector.random(domain, cliques)
  cons = tuple(
      constraints if isinstance(constraints, (list, tuple)) else [constraints]
  )
  baseline = _baseline_marginals(
      domain, cliques, potentials, constraints, total
  )
  for fn in [oracle] if oracle else _EXTENSION_ORACLES:
    result = fn(potentials, total, constraints=cons)
    for cl in cliques:
      np.testing.assert_allclose(
          result[cl].datavector(),
          baseline[cl].datavector(),
          atol=1e-4,
          err_msg=f'Mismatch at {cl} with {fn.__name__}',
      )


# ---------------------------------------------------------------------------
# Construction and validation
# ---------------------------------------------------------------------------


class TestConstraintConstruction(unittest.TestCase):

  def test_valid(self):
    domain = Domain(['a', 'b'], [3, 4])
    valid = np.array([[0, 0], [1, 2], [2, 3]])
    c = Constraint(domain=domain, valid=valid)
    self.assertIsNotNone(c.valid)
    self.assertIsNone(c.invalid)
    self.assertIsNone(c.mapping)

  def test_invalid(self):
    domain = Domain(['a', 'b'], [3, 4])
    invalid = np.array([[0, 1]])
    c = Constraint(domain=domain, invalid=invalid)
    self.assertIsNone(c.valid)
    self.assertIsNotNone(c.invalid)

  def test_mapping(self):
    domain = Domain(['fine', 'coarse'], [6, 3])
    mapping = np.array([0, 0, 1, 1, 2, 2])
    c = Constraint(domain=domain, mapping=mapping)
    self.assertIsNone(c.valid)
    self.assertTrue(c.is_deterministic)

  def test_none_specified_raises(self):
    domain = Domain(['a'], [3])
    with self.assertRaises(ValueError):
      Constraint(domain=domain)

  def test_multiple_specified_raises(self):
    domain = Domain(['a', 'b'], [3, 4])
    with self.assertRaises(ValueError):
      Constraint(
          domain=domain,
          valid=np.array([[0, 0]]),
          invalid=np.array([[1, 1]]),
      )

  def test_valid_wrong_columns_raises(self):
    domain = Domain(['a', 'b'], [3, 4])
    with self.assertRaises(ValueError):
      Constraint(domain=domain, valid=np.array([[0, 0, 0]]))

  def test_mapping_wrong_size_raises(self):
    domain = Domain(['fine', 'coarse'], [6, 3])
    with self.assertRaises(ValueError):
      Constraint(domain=domain, mapping=np.array([0, 0, 1]))

  def test_mapping_requires_two_attributes(self):
    domain = Domain(['a', 'b', 'c'], [3, 3, 3])
    with self.assertRaises(ValueError):
      Constraint(domain=domain, mapping=np.array([0, 1, 2]))


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestConstraintProperties(unittest.TestCase):

  def test_clique(self):
    domain = Domain(['b', 'a'], [3, 4])
    c = Constraint(domain=domain, valid=np.array([[0, 0]]))
    self.assertEqual(c.clique, ('a', 'b'))

  def test_is_deterministic(self):
    domain = Domain(['fine', 'coarse'], [4, 2])
    c_map = Constraint(domain=domain, mapping=np.array([0, 0, 1, 1]))
    c_valid = Constraint(domain=domain, valid=np.array([[0, 0], [1, 0]]))
    self.assertTrue(c_map.is_deterministic)
    self.assertFalse(c_valid.is_deterministic)

  def test_mapping_sizes(self):
    c = _mapping_constraint('A', 'Ap', np.array([0, 0, 1, 1, 2, 2]))
    self.assertEqual(c.domain.shape[0], 6)
    self.assertEqual(c.domain.shape[1], 3)
    self.assertEqual(c.clique, ('A', 'Ap'))
    self.assertTrue(c.is_deterministic)


# ---------------------------------------------------------------------------
# Potential generation
# ---------------------------------------------------------------------------


class TestAsPotential(unittest.TestCase):

  def test_valid_potential(self):
    domain = Domain(['a', 'b'], [3, 3])
    valid = np.array([[0, 0], [1, 1], [2, 2]])
    c = Constraint(domain=domain, valid=valid)
    f = c.potential
    self.assertEqual(f.domain, domain)
    vals = np.asarray(f.values)
    for i in range(3):
      for j in range(3):
        if i == j:
          self.assertEqual(vals[i, j], 0.0)
        else:
          self.assertEqual(vals[i, j], -np.inf)

  def test_invalid_potential(self):
    domain = Domain(['a', 'b'], [2, 2])
    invalid = np.array([[0, 1]])
    c = Constraint(domain=domain, invalid=invalid)
    f = c.potential
    vals = np.asarray(f.values)
    self.assertEqual(vals[0, 1], -np.inf)
    self.assertEqual(vals[0, 0], 0.0)
    self.assertEqual(vals[1, 0], 0.0)
    self.assertEqual(vals[1, 1], 0.0)

  def test_mapping_potential(self):
    domain = Domain(['fine', 'coarse'], [4, 2])
    mapping = np.array([0, 0, 1, 1])
    c = Constraint(domain=domain, mapping=mapping)
    f = c.potential
    vals = np.asarray(f.values)
    self.assertEqual(vals[0, 0], 0.0)
    self.assertEqual(vals[1, 0], 0.0)
    self.assertEqual(vals[2, 1], 0.0)
    self.assertEqual(vals[3, 1], 0.0)
    self.assertEqual(vals[0, 1], -np.inf)
    self.assertEqual(vals[2, 0], -np.inf)

  def test_valid_and_mapping_produce_same_potential(self):
    """A mapping constraint should produce the same factor as the
    equivalent valid-combos constraint."""
    domain = Domain(['fine', 'coarse'], [6, 3])
    mapping = np.array([0, 0, 1, 1, 2, 2])
    valid = np.column_stack([np.arange(6), mapping])

    c_map = Constraint(domain=domain, mapping=mapping)
    c_valid = Constraint(domain=domain, valid=valid)

    np.testing.assert_array_equal(
        c_map.potential.values, c_valid.potential.values
    )

  def test_three_way_valid(self):
    """Constraints over 3+ attributes should work."""
    domain = Domain(['a', 'b', 'c'], [2, 2, 2])
    valid = np.array([[0, 0, 0], [1, 1, 1]])
    c = Constraint(domain=domain, valid=valid)
    f = c.potential
    vals = np.asarray(f.values)
    self.assertEqual(vals[0, 0, 0], 0.0)
    self.assertEqual(vals[1, 1, 1], 0.0)
    self.assertEqual(vals[0, 0, 1], -np.inf)
    self.assertEqual(vals[1, 0, 0], -np.inf)


# ---------------------------------------------------------------------------
# Core oracle integration
# ---------------------------------------------------------------------------


class TestOracleIntegration(unittest.TestCase):
  """Test that constraints flow through core marginal oracles correctly."""

  def _make_model(self):
    domain = Domain(['A', 'B', 'C'], [3, 3, 4])
    cliques = [('A', 'C'), ('B', 'C')]
    np.random.seed(42)
    arrays = {
        cl: Factor(
            domain.project(cl),
            jnp.array(np.random.randn(*domain.project(cl).shape)),
        )
        for cl in cliques
    }
    return domain, cliques, CliqueVector(domain, cliques, arrays)

  def _diagonal_constraint(self, domain):
    """A=B constraint (diagonal only)."""
    valid = np.array([[i, i] for i in range(3)])
    return Constraint(domain=domain.project(('A', 'B')), valid=valid)

  def test_shafer_shenoy_with_constraints(self):
    from mbi import marginal_oracles

    domain, cliques, potentials = self._make_model()
    c = self._diagonal_constraint(domain)

    result = marginal_oracles.message_passing_shafer_shenoy(
        potentials, total=1.0, constraints=(c,)
    )
    for cl in cliques:
      np.testing.assert_allclose(float(result[cl].sum()), 1.0, atol=1e-5)

  def test_shafer_shenoy_constraints_match_manual(self):
    """constraints= should match manually adding the factor."""
    from mbi import marginal_oracles

    domain, cliques, potentials = self._make_model()
    c = self._diagonal_constraint(domain)

    result1 = marginal_oracles.message_passing_shafer_shenoy(
        potentials, total=1.0, constraints=(c,)
    )

    # Manually add the constraint factor.
    con_cl = domain.canonical(c.clique)
    arrays = {cl: potentials[cl] for cl in cliques}
    arrays[con_cl] = c.potential
    potentials2 = CliqueVector(domain, list(cliques) + [con_cl], arrays)
    result2 = marginal_oracles.message_passing_shafer_shenoy(
        potentials2, total=1.0
    )

    for cl in cliques:
      np.testing.assert_allclose(
          np.array(result1[cl].values),
          np.array(result2.project(cl).values),
          atol=1e-5,
      )

  def test_implicit_with_constraints(self):
    from mbi import marginal_oracles

    domain, cliques, potentials = self._make_model()
    c = self._diagonal_constraint(domain)

    result = marginal_oracles.message_passing_implicit(
        potentials, total=1.0, constraints=(c,)
    )
    for cl in cliques:
      np.testing.assert_allclose(float(result[cl].sum()), 1.0, atol=1e-5)

  def test_implicit_einsum_semistable_raises(self):
    from mbi import marginal_oracles

    domain, cliques, potentials = self._make_model()
    c = self._diagonal_constraint(domain)

    with self.assertRaises(ValueError):
      marginal_oracles.message_passing_implicit(
          potentials,
          total=1.0,
          constraints=(c,),
          contraction=marginal_oracles.einsum_semistable,
      )

  def test_hugin_warns(self):
    import warnings as w
    from mbi import marginal_oracles

    domain, cliques, potentials = self._make_model()
    c = self._diagonal_constraint(domain)

    with w.catch_warnings(record=True) as caught:
      w.simplefilter('always')
      marginal_oracles.message_passing_hugin(
          potentials, total=1.0, constraints=(c,)
      )
    self.assertTrue(any('HUGIN' in str(warning.message) for warning in caught))

  def test_mapping_constraint_through_oracle(self):
    """Mapping constraints work through oracles."""
    from mbi import marginal_oracles

    domain = Domain(['fine', 'coarse', 'X'], [6, 3, 4])
    mapping = np.array([0, 0, 1, 1, 2, 2])
    c = Constraint(domain=domain.project(('fine', 'coarse')), mapping=mapping)

    np.random.seed(0)
    cliques = [('fine', 'X'), ('coarse', 'X')]
    arrays = {
        cl: Factor(
            domain.project(cl),
            jnp.array(np.random.randn(*domain.project(cl).shape)),
        )
        for cl in cliques
    }
    potentials = CliqueVector(domain, cliques, arrays)

    result = marginal_oracles.message_passing_shafer_shenoy(
        potentials, total=1.0, constraints=(c,)
    )
    for cl in cliques:
      np.testing.assert_allclose(float(result[cl].sum()), 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# Coarsen / refine primitives
# ---------------------------------------------------------------------------


class TestCoarsenRefine(unittest.TestCase):

  def setUp(self):
    self.mapping = np.array([0, 0, 1, 1, 2, 2])
    self.constraint = _mapping_constraint('A', 'Ap', self.mapping)

  def test_coarsen_1d(self):
    dom = Domain(['A'], [6])
    vals = jnp.log(jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    result = coarsen(Factor(dom, vals), self.constraint)
    self.assertEqual(result.domain.attributes, ('Ap',))
    np.testing.assert_allclose(jnp.exp(result.values), [3.0, 7.0, 11.0])

  def test_refine_1d(self):
    dom = Domain(['Ap'], [3])
    result = refine(Factor(dom, jnp.array([10.0, 20.0, 30.0])), self.constraint)
    self.assertEqual(result.domain.attributes, ('A',))
    np.testing.assert_allclose(result.values, [10, 10, 20, 20, 30, 30])

  def test_coarsen_2d(self):
    dom = Domain(['A', 'B'], [6, 2])
    result = coarsen(Factor(dom, jnp.log(jnp.ones((6, 2)))), self.constraint)
    self.assertEqual(result.domain.attributes, ('Ap', 'B'))
    np.testing.assert_allclose(jnp.exp(result.values), 2.0 * jnp.ones((3, 2)))

  def test_refine_2d(self):
    dom = Domain(['Ap', 'C'], [3, 4])
    result = refine(
        Factor(dom, jnp.arange(12.0).reshape(3, 4)), self.constraint
    )
    self.assertEqual(result.domain.attributes, ('A', 'C'))
    # Rows 0,1 match Ap=0; rows 2,3 match Ap=1; rows 4,5 match Ap=2
    np.testing.assert_allclose(result.values[0], result.values[1])
    np.testing.assert_allclose(result.values[2], result.values[3])
    np.testing.assert_allclose(result.values[4], result.values[5])

  def test_roundtrip_coarsen_refine(self):
    dom = Domain(['Ap'], [3])
    vals = jnp.array([1.0, 2.0, 3.0])
    recovered = coarsen(
        refine(Factor(dom, vals), self.constraint), self.constraint
    )
    np.testing.assert_allclose(recovered.values, vals + jnp.log(2.0), atol=1e-6)

  def test_project_to_coarse(self):
    dom = Domain(['A', 'B'], [6, 2])
    result = project_to_coarse(Factor(dom, jnp.ones((6, 2))), self.constraint)
    self.assertEqual(result.domain.attributes, ('Ap', 'B'))
    np.testing.assert_allclose(result.values, 2.0 * jnp.ones((3, 2)))


class TestCoarsenRefineRandomized(unittest.TestCase):
  """Property-based tests: coarsen matches explicit constraint + logsumexp."""

  @parameterized.expand(range(10))
  def test_coarsen_matches_explicit(self, seed):
    rng = np.random.default_rng(seed)
    n_fine = rng.integers(4, 20)
    n_coarse = rng.integers(2, n_fine)
    mapping = _random_surjection(n_fine, n_coarse, rng)
    constraint = _mapping_constraint('X', 'Y', mapping)

    n_other = rng.integers(2, 6)
    dom = Domain(['X', 'Z'], [n_fine, n_other])
    factor = Factor(dom, jnp.array(rng.standard_normal((n_fine, n_other))))

    result = coarsen(factor, constraint)

    # Reference: explicit constraint factor + logsumexp
    con_vals = np.full((n_fine, n_coarse), -np.inf)
    for a in range(n_fine):
      con_vals[a, mapping[a]] = 0.0
    joint_dom = Domain(['X', 'Y', 'Z'], [n_fine, n_coarse, n_other])
    joint = factor.expand(joint_dom) + Factor(
        Domain(['X', 'Y'], [n_fine, n_coarse]), jnp.array(con_vals)
    ).expand(joint_dom)
    reference = joint.logsumexp(['X'])

    np.testing.assert_allclose(
        result.values,
        reference.transpose(result.domain.attributes).values,
        atol=1e-5,
    )


# ---------------------------------------------------------------------------
# Extension message passing with constraints
# ---------------------------------------------------------------------------

_SIMPLE_DOMAIN = Domain(['A', 'Ap', 'B', 'C'], [6, 3, 4, 5])
_SIMPLE_CONSTRAINT = _mapping_constraint(
    'A', 'Ap', np.array([0, 0, 1, 1, 2, 2])
)
_CLIQUE_CONFIGS = [
    [('A', 'B'), ('Ap', 'C')],
    [('A', 'B'), ('A',), ('Ap', 'C')],
    [('A', 'B'), ('Ap', 'C'), ('Ap',)],
    [('A', 'B'), ('A',), ('Ap', 'C'), ('Ap',)],
]


class TestExtensionMessagePassing(unittest.TestCase):
  """Extension oracles match the -inf baseline."""

  @parameterized.expand([
      (fn, cliques, total)
      for fn in _EXTENSION_ORACLES
      for cliques in _CLIQUE_CONFIGS
      for total in [1.0, 10.0, 100.0]
  ])
  def test_uniform_sums_to_total(self, fn, cliques, total):
    potentials = CliqueVector.zeros(_SIMPLE_DOMAIN, cliques)
    result = fn(potentials, total, constraints=(_SIMPLE_CONSTRAINT,))
    for cl in cliques:
      np.testing.assert_allclose(result[cl].values.sum(), total, atol=1e-4)

  @parameterized.expand([(c,) for c in _CLIQUE_CONFIGS])
  def test_matches_baseline(self, cliques):
    _assert_matches_baseline(
        self, _SIMPLE_DOMAIN, cliques, _SIMPLE_CONSTRAINT, total=10.0
    )

  @parameterized.expand([
      (fn, cliques) for fn in _EXTENSION_ORACLES for cliques in _CLIQUE_CONFIGS
  ])
  def test_no_nans(self, fn, cliques):
    potentials = CliqueVector.random(_SIMPLE_DOMAIN, cliques)
    result = fn(potentials, 10.0, constraints=(_SIMPLE_CONSTRAINT,))
    for cl in cliques:
      self.assertFalse(jnp.isnan(result[cl].values).any())

  @parameterized.expand([
      (fn, cliques) for fn in _EXTENSION_ORACLES for cliques in _CLIQUE_CONFIGS
  ])
  def test_non_negative(self, fn, cliques):
    potentials = CliqueVector.random(_SIMPLE_DOMAIN, cliques)
    result = fn(potentials, 10.0, constraints=(_SIMPLE_CONSTRAINT,))
    for cl in cliques:
      self.assertTrue((result[cl].values >= -1e-10).all())


# ---------------------------------------------------------------------------
# Multiple constraints
# ---------------------------------------------------------------------------


class TestMultipleConstraints(unittest.TestCase):
  """Tests with two deterministic constraints."""

  def _two_constraint_setup(self):
    domain = Domain(['A', 'Ap', 'B', 'Bp', 'C'], [6, 3, 8, 4, 5])
    c1 = _mapping_constraint('A', 'Ap', np.array([0, 0, 1, 1, 2, 2]))
    c2 = _mapping_constraint('B', 'Bp', np.array([0, 0, 1, 1, 2, 2, 3, 3]))
    return domain, (c1, c2)

  def test_two_constraints_basic(self):
    domain, constraints = self._two_constraint_setup()
    _assert_matches_baseline(
        self, domain, [('A', 'C'), ('Ap', 'Bp'), ('B',)], constraints
    )

  def test_two_constraints_shared_factor(self):
    domain, constraints = self._two_constraint_setup()
    _assert_matches_baseline(
        self, domain, [('A',), ('B',), ('Ap', 'Bp')], constraints
    )

  def test_two_constraints_fine_and_coarse_mixed(self):
    domain, constraints = self._two_constraint_setup()
    _assert_matches_baseline(
        self, domain, [('A', 'Bp'), ('B', 'Ap')], constraints
    )

  @parameterized.expand(
      [(fn, seed) for fn in _EXTENSION_ORACLES for seed in range(10)]
  )
  def test_two_constraints_randomized(self, fn, seed):
    rng = np.random.default_rng(seed)
    n_a, n_ap = rng.integers(4, 12), rng.integers(2, 4)
    n_b, n_bp = rng.integers(4, 12), rng.integers(2, 4)
    n_c = rng.integers(2, 6)

    domain = Domain(['A', 'Ap', 'B', 'Bp', 'C'], [n_a, n_ap, n_b, n_bp, n_c])
    c1 = _mapping_constraint('A', 'Ap', _random_surjection(n_a, n_ap, rng))
    c2 = _mapping_constraint('B', 'Bp', _random_surjection(n_b, n_bp, rng))
    cliques = [('A', 'C'), ('Bp',)]
    total = 10.0

    potentials = CliqueVector.random(domain, cliques)
    baseline = _baseline_marginals(domain, cliques, potentials, [c1, c2], total)
    result = fn(potentials, total, constraints=(c1, c2))
    for cl in cliques:
      np.testing.assert_allclose(
          result[cl].datavector(),
          baseline[cl].datavector(),
          atol=1e-4,
          err_msg=f'Seed {seed}, clique {cl}, {fn.__name__}',
      )


# ---------------------------------------------------------------------------
# Complex topologies
# ---------------------------------------------------------------------------


class TestComplexTopologies(unittest.TestCase):
  """Tests with more complex graph structures."""

  _CONSTRAINT = _mapping_constraint('A', 'Ap', np.array([0, 0, 1, 1, 2, 2]))

  def _check(self, domain, cliques, constraint=None):
    _assert_matches_baseline(
        self, domain, cliques, constraint or self._CONSTRAINT
    )

  def test_many_neighbors(self):
    domain = Domain(['A', 'Ap', 'B', 'C', 'D'], [6, 3, 4, 5, 3])
    self._check(domain, [('A', 'B'), ('A', 'C'), ('Ap', 'D')])

  def test_singletons(self):
    domain = Domain(['A', 'Ap', 'B'], [6, 3, 4])
    self._check(domain, [('A',), ('Ap',), ('B',)])

  def test_fine_in_multiple_factors(self):
    domain = Domain(['A', 'Ap', 'B', 'C', 'D'], [10, 3, 4, 5, 3])
    constraint = _mapping_constraint(
        'A', 'Ap', np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2])
    )
    self._check(
        domain, [('A', 'B'), ('A', 'C'), ('A', 'D'), ('Ap',)], constraint
    )

  def test_coarse_in_multiple_factors(self):
    domain = Domain(['A', 'Ap', 'B', 'C', 'D'], [10, 3, 4, 5, 3])
    constraint = _mapping_constraint(
        'A', 'Ap', np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2])
    )
    self._check(
        domain, [('A',), ('Ap', 'B'), ('Ap', 'C'), ('Ap', 'D')], constraint
    )

  def test_uneven_groups(self):
    domain = Domain(['A', 'Ap', 'B'], [10, 3, 4])
    constraint = _mapping_constraint(
        'A', 'Ap', np.array([0, 1, 2, 2, 2, 2, 2, 2, 2, 2])
    )
    self._check(domain, [('A', 'B'), ('Ap',)], constraint)

  def test_large_ratio(self):
    n_fine, n_coarse = 50, 5
    mapping = np.repeat(np.arange(n_coarse), n_fine // n_coarse)
    domain = Domain(['A', 'Ap', 'B'], [n_fine, n_coarse, 8])
    constraint = _mapping_constraint('A', 'Ap', mapping)
    self._check(domain, [('A', 'B'), ('Ap',)], constraint)

  def test_disconnected_components(self):
    domain = Domain(['A', 'Ap', 'B', 'C', 'D'], [6, 3, 4, 5, 3])
    self._check(domain, [('A', 'B'), ('Ap',), ('C', 'D')])

  def test_user_clique_spans_constraint_pair(self):
    """User provides a clique (A, Ap) that matches the constraint pair."""
    domain = Domain(['A', 'Ap', 'B'], [6, 3, 4])
    self._check(domain, [('A', 'Ap'), ('A', 'B')])

  def test_user_clique_is_fine_coarse_plus_extra(self):
    """User clique (A, Ap, B) embeds constraint in a 3-way factor."""
    domain = Domain(['A', 'Ap', 'B'], [6, 3, 4])
    self._check(domain, [('A', 'Ap', 'B')])

  def test_identity_mapping(self):
    """1:1 mapping (permutation) — degenerate constraint."""
    domain = Domain(['A', 'Ap', 'B'], [4, 4, 5])
    constraint = _mapping_constraint('A', 'Ap', np.array([0, 1, 2, 3]))
    self._check(domain, [('A', 'B'), ('Ap',)], constraint)

  def test_orphan_coarse_variable(self):
    """Coarse variable appears only via constraint, not in any user clique."""
    domain = Domain(['A', 'Ap', 'B'], [6, 3, 4])
    self._check(domain, [('A', 'B')])

  def test_orphan_fine_variable(self):
    """Fine variable appears only via constraint, not in any user clique."""
    domain = Domain(['A', 'Ap', 'B'], [6, 3, 4])
    self._check(domain, [('Ap', 'B')])

  def test_cyclic_model(self):
    """Cyclic model structure (requires triangulation)."""
    domain = Domain(['A', 'Ap', 'B', 'C'], [6, 3, 4, 5])
    self._check(domain, [('A', 'B'), ('B', 'C'), ('Ap', 'C')])

  def test_only_constraint_variables(self):
    """All cliques reference only constraint variables."""
    domain = Domain(['A', 'Ap'], [6, 3])
    self._check(domain, [('A',), ('Ap',)])

  def test_fine_and_coarse_singletons(self):
    """Both fine and coarse as singleton user cliques."""
    domain = Domain(['A', 'Ap', 'B', 'C'], [6, 3, 4, 5])
    self._check(domain, [('A',), ('Ap',), ('A', 'B'), ('Ap', 'C')])

  def test_all_variables_in_one_clique(self):
    """All variables including constraint pair in a single clique."""
    domain = Domain(['A', 'Ap', 'B', 'C'], [6, 3, 4, 5])
    self._check(domain, [('A', 'Ap', 'B', 'C')])

  def test_shared_constraint_separator(self):
    """Two embedded cliques sharing (fine, coarse) as the separator."""
    domain = Domain(['A', 'Ap', 'B', 'C'], [6, 3, 4, 5])
    self._check(domain, [('A', 'Ap', 'B'), ('A', 'Ap', 'C')])


if __name__ == '__main__':
  unittest.main()
