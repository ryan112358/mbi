"""Tests for deterministic constraint handling.

Tests the coarsen/refine primitives and the constraint-aware message passing
against standard Shafer-Shenoy baselines with explicit -inf constraint factors.
"""

import unittest

import jax.numpy as jnp
from mbi.clique_vector import CliqueVector
from mbi.domain import Domain
from mbi.extensions.constraints import coarsen
from mbi.extensions.constraints import DeterministicConstraint
from mbi.extensions.constraints import message_passing_with_constraints
from mbi.extensions.constraints import project_to_coarse
from mbi.extensions.constraints import refine
from mbi.factor import Factor
from mbi.marginal_oracles import message_passing_shafer_shenoy
import numpy as np
from parameterized import parameterized


def _make_constraint_factor(domain, constraint):
    """Build the explicit 0/-inf constraint factor for baseline comparison."""
    shape = (constraint.n_fine, constraint.n_coarse)
    vals = np.full(shape, -np.inf)
    for a, a_prime in enumerate(constraint.mapping):
        vals[a, a_prime] = 0.0
    dom = Domain([constraint.fine, constraint.coarse], list(shape))
    return Factor(dom, jnp.array(vals)).transpose(
        domain.canonical(constraint.clique)
    )


def _baseline_marginals(domain, cliques, potentials, constraints, total=10.0):
    """Marginals via standard Shafer-Shenoy with explicit -inf constraints."""
    if isinstance(constraints, DeterministicConstraint):
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


def _assert_matches_baseline(
    test, domain, cliques, constraints, total=10.0, seed=None
):
    """Assert constraint-aware message passing matches the -inf baseline."""
    if seed is not None:
        np.random.seed(seed)
    potentials = CliqueVector.random(domain, cliques)
    result = message_passing_with_constraints(
        potentials,
        total,
        constraints=tuple(
            constraints
            if isinstance(constraints, (list, tuple))
            else [constraints]
        ),
    )
    baseline = _baseline_marginals(
        domain, cliques, potentials, constraints, total
    )
    for cl in cliques:
        np.testing.assert_allclose(
            result[cl].datavector(),
            baseline[cl].datavector(),
            atol=1e-4,
            err_msg=f'Mismatch at {cl}',
        )
    return result


def _random_surjection(n_domain, n_range, rng):
    """Generate a random surjective mapping from [n_domain] to [n_range]."""
    mapping = np.zeros(n_domain, dtype=int)
    mapping[:n_range] = np.arange(n_range)
    mapping[n_range:] = rng.integers(0, n_range, size=n_domain - n_range)
    rng.shuffle(mapping)
    return mapping


_SIMPLE_DOMAIN = Domain(['A', 'Ap', 'B', 'C'], [6, 3, 4, 5])
_SIMPLE_CONSTRAINT = DeterministicConstraint(
    'A', 'Ap', np.array([0, 0, 1, 1, 2, 2])
)
_CLIQUE_CONFIGS = [
    [('A', 'B'), ('Ap', 'C')],
    [('A', 'B'), ('A',), ('Ap', 'C')],
    [('A', 'B'), ('Ap', 'C'), ('Ap',)],
    [('A', 'B'), ('A',), ('Ap', 'C'), ('Ap',)],
]


class TestCoarsenRefine(unittest.TestCase):

    def setUp(self):
        self.mapping = np.array([0, 0, 1, 1, 2, 2])
        self.constraint = DeterministicConstraint('A', 'Ap', self.mapping)

    def test_coarsen_1d(self):
        dom = Domain(['A'], [6])
        vals = jnp.log(jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
        result = coarsen(Factor(dom, vals), self.constraint)
        self.assertEqual(result.domain.attributes, ('Ap',))
        np.testing.assert_allclose(jnp.exp(result.values), [3.0, 7.0, 11.0])

    def test_refine_1d(self):
        dom = Domain(['Ap'], [3])
        result = refine(
            Factor(dom, jnp.array([10.0, 20.0, 30.0])), self.constraint
        )
        self.assertEqual(result.domain.attributes, ('A',))
        np.testing.assert_allclose(result.values, [10, 10, 20, 20, 30, 30])

    def test_coarsen_2d(self):
        dom = Domain(['A', 'B'], [6, 2])
        result = coarsen(
            Factor(dom, jnp.log(jnp.ones((6, 2)))), self.constraint
        )
        self.assertEqual(result.domain.attributes, ('Ap', 'B'))
        np.testing.assert_allclose(
            jnp.exp(result.values), 2.0 * jnp.ones((3, 2))
        )

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
        np.testing.assert_allclose(
            recovered.values, vals + jnp.log(2.0), atol=1e-6
        )

    def test_project_to_coarse(self):
        dom = Domain(['A', 'B'], [6, 2])
        result = project_to_coarse(
            Factor(dom, jnp.ones((6, 2))), self.constraint
        )
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
        constraint = DeterministicConstraint('X', 'Y', mapping)

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


class TestMessagePassingWithConstraints(unittest.TestCase):
    """Constraint-aware message passing matches the -inf baseline."""

    @parameterized.expand([
        (cliques, total)
        for cliques in _CLIQUE_CONFIGS
        for total in [1.0, 10.0, 100.0]
    ])
    def test_uniform_sums_to_total(self, cliques, total):
        potentials = CliqueVector.zeros(_SIMPLE_DOMAIN, cliques)
        result = message_passing_with_constraints(
            potentials, total, constraints=(_SIMPLE_CONSTRAINT,)
        )
        for cl in cliques:
            np.testing.assert_allclose(
                result[cl].values.sum(), total, atol=1e-4
            )

    @parameterized.expand([(c,) for c in _CLIQUE_CONFIGS])
    def test_matches_baseline(self, cliques):
        _assert_matches_baseline(
            self, _SIMPLE_DOMAIN, cliques, _SIMPLE_CONSTRAINT, total=10.0
        )

    @parameterized.expand([(c,) for c in _CLIQUE_CONFIGS])
    def test_no_nans(self, cliques):
        potentials = CliqueVector.random(_SIMPLE_DOMAIN, cliques)
        result = message_passing_with_constraints(
            potentials, 10.0, constraints=(_SIMPLE_CONSTRAINT,)
        )
        for cl in cliques:
            self.assertFalse(jnp.isnan(result[cl].values).any())

    @parameterized.expand([(c,) for c in _CLIQUE_CONFIGS])
    def test_non_negative(self, cliques):
        potentials = CliqueVector.random(_SIMPLE_DOMAIN, cliques)
        result = message_passing_with_constraints(
            potentials, 10.0, constraints=(_SIMPLE_CONSTRAINT,)
        )
        for cl in cliques:
            self.assertTrue((result[cl].values >= -1e-10).all())


class TestMultipleConstraints(unittest.TestCase):
    """Tests with two deterministic constraints."""

    def _two_constraint_setup(self):
        domain = Domain(['A', 'Ap', 'B', 'Bp', 'C'], [6, 3, 8, 4, 5])
        c1 = DeterministicConstraint('A', 'Ap', np.array([0, 0, 1, 1, 2, 2]))
        c2 = DeterministicConstraint(
            'B', 'Bp', np.array([0, 0, 1, 1, 2, 2, 3, 3])
        )
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

    @parameterized.expand(range(10))
    def test_two_constraints_randomized(self, seed):
        rng = np.random.default_rng(seed)
        n_a, n_ap = rng.integers(4, 12), rng.integers(2, 4)
        n_b, n_bp = rng.integers(4, 12), rng.integers(2, 4)
        n_c = rng.integers(2, 6)

        domain = Domain(
            ['A', 'Ap', 'B', 'Bp', 'C'], [n_a, n_ap, n_b, n_bp, n_c]
        )
        c1 = DeterministicConstraint(
            'A', 'Ap', _random_surjection(n_a, n_ap, rng)
        )
        c2 = DeterministicConstraint(
            'B', 'Bp', _random_surjection(n_b, n_bp, rng)
        )
        cliques = [('A', 'C'), ('Bp',)]
        total = 10.0

        potentials = CliqueVector.random(domain, cliques)
        result = message_passing_with_constraints(
            potentials, total, constraints=(c1, c2)
        )
        baseline = _baseline_marginals(
            domain, cliques, potentials, [c1, c2], total
        )

        for cl in cliques:
            np.testing.assert_allclose(
                result[cl].datavector(),
                baseline[cl].datavector(),
                atol=1e-4,
                err_msg=f'Seed {seed}, clique {cl}',
            )


class TestComplexTopologies(unittest.TestCase):
    """Tests with more complex graph structures."""

    _CONSTRAINT = DeterministicConstraint(
        'A', 'Ap', np.array([0, 0, 1, 1, 2, 2])
    )

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
        constraint = DeterministicConstraint(
            'A', 'Ap', np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2])
        )
        self._check(
            domain, [('A', 'B'), ('A', 'C'), ('A', 'D'), ('Ap',)], constraint
        )

    def test_coarse_in_multiple_factors(self):
        domain = Domain(['A', 'Ap', 'B', 'C', 'D'], [10, 3, 4, 5, 3])
        constraint = DeterministicConstraint(
            'A', 'Ap', np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2])
        )
        self._check(
            domain, [('A',), ('Ap', 'B'), ('Ap', 'C'), ('Ap', 'D')], constraint
        )

    def test_uneven_groups(self):
        domain = Domain(['A', 'Ap', 'B'], [10, 3, 4])
        constraint = DeterministicConstraint(
            'A', 'Ap', np.array([0, 1, 2, 2, 2, 2, 2, 2, 2, 2])
        )
        self._check(domain, [('A', 'B'), ('Ap',)], constraint)

    def test_large_ratio(self):
        n_fine, n_coarse = 50, 5
        mapping = np.repeat(np.arange(n_coarse), n_fine // n_coarse)
        domain = Domain(['A', 'Ap', 'B'], [n_fine, n_coarse, 8])
        constraint = DeterministicConstraint('A', 'Ap', mapping)
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
        constraint = DeterministicConstraint('A', 'Ap', np.array([0, 1, 2, 3]))
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


class TestDeterministicConstraint(unittest.TestCase):

    def test_properties(self):
        c = DeterministicConstraint('A', 'Ap', np.array([0, 0, 1, 1, 2, 2]))
        self.assertEqual(c.n_fine, 6)
        self.assertEqual(c.n_coarse, 3)
        self.assertEqual(c.clique, ('A', 'Ap'))

    def test_hash_eq(self):
        c1 = DeterministicConstraint('A', 'Ap', np.array([0, 0, 1]))
        c2 = DeterministicConstraint('A', 'Ap', np.array([0, 0, 1]))
        c3 = DeterministicConstraint('A', 'Ap', np.array([0, 1, 1]))
        self.assertEqual(c1, c2)
        self.assertNotEqual(c1, c3)
        self.assertEqual(hash(c1), hash(c2))


if __name__ == '__main__':
    unittest.main()
