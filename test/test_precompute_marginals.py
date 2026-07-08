"""Tests for mbi.extensions.precompute_marginals."""

import itertools
import unittest

import jax.numpy as jnp
import numpy as np

from mbi import Dataset, Domain
from mbi.dataset import JaxDataset
from mbi.extensions.precompute_marginals import precompute_marginals


def _make_domain(sizes):
    """Create a domain with attributes a0, a1, ... with given sizes."""
    attrs = [f'a{i}' for i in range(len(sizes))]
    return Domain(attrs, sizes)


def _make_dataset(domain, n, seed=0, weights=None):
    """Create a random dataset with n records."""
    rng = np.random.default_rng(seed)
    data = {
        a: (
            rng.integers(0, domain[a], size=n).astype(
                np.min_scalar_type(domain[a] - 1)
            )
        )
        for a in domain.attributes
    }
    return Dataset(data, domain, weights=weights)


def _reference_marginal(dataset, cl):
    """Compute a marginal using Dataset.project (numpy reference)."""
    return dataset.project(cl)


class TestPrecomputeMarginals(unittest.TestCase):
    """Tests correctness of precompute_marginals against Dataset.project."""

    def _assert_marginals_match(self, dataset, cliques):
        """Assert precomputed marginals match the reference for all cliques."""
        result = precompute_marginals(dataset, cliques)
        for cl in cliques:
            expected = _reference_marginal(dataset, cl)
            actual = result[cl]
            np.testing.assert_array_almost_equal(
                np.asarray(actual.datavector(flatten=True)),
                np.asarray(expected.datavector(flatten=True)),
                decimal=4,
                err_msg=f'Marginal mismatch for clique {cl}',
            )

    def test_pairwise_homogeneous(self):
        """All attributes have the same domain size."""
        domain = _make_domain([8, 8, 8, 8])
        dataset = _make_dataset(domain, 1000)
        cliques = list(itertools.combinations(domain.attributes, 2))
        self._assert_marginals_match(dataset, cliques)

    def test_pairwise_heterogeneous(self):
        """Attributes have different domain sizes triggering bucketing."""
        domain = _make_domain([2, 5, 16, 64])
        dataset = _make_dataset(domain, 5000)
        cliques = list(itertools.combinations(domain.attributes, 2))
        self._assert_marginals_match(dataset, cliques)

    def test_three_way_marginals(self):
        """Cliques of size 3."""
        domain = _make_domain([4, 8, 3, 6])
        dataset = _make_dataset(domain, 2000)
        cliques = list(itertools.combinations(domain.attributes, 3))
        self._assert_marginals_match(dataset, cliques)

    def test_mixed_clique_sizes(self):
        """Mix of 1-way, 2-way, and 3-way cliques."""
        domain = _make_domain([3, 7, 5])
        dataset = _make_dataset(domain, 3000)
        cliques = [
            ('a0',),
            ('a1',),
            ('a2',),
            ('a0', 'a1'),
            ('a1', 'a2'),
            ('a0', 'a1', 'a2'),
        ]
        self._assert_marginals_match(dataset, cliques)

    def test_single_attribute_cliques(self):
        """Univariate marginals."""
        domain = _make_domain([2, 100, 32])
        dataset = _make_dataset(domain, 1000)
        cliques = [('a0',), ('a1',), ('a2',)]
        self._assert_marginals_match(dataset, cliques)

    def test_jax_dataset_input(self):
        """Accepts JaxDataset directly."""
        domain = _make_domain([4, 8, 16])
        np_dataset = _make_dataset(domain, 2000)
        jax_data = {
            a: jnp.asarray(np_dataset.to_dict()[a]) for a in domain.attributes
        }
        jax_dataset = JaxDataset(jax_data, domain)
        cliques = list(itertools.combinations(domain.attributes, 2))
        result = precompute_marginals(jax_dataset, cliques)
        for cl in cliques:
            expected = _reference_marginal(np_dataset, cl)
            actual = result[cl]
            np.testing.assert_array_almost_equal(
                np.asarray(actual.datavector(flatten=True)),
                np.asarray(expected.datavector(flatten=True)),
                decimal=4,
            )

    def test_attribute_order_preserved(self):
        """Clique attribute order is respected in the output Factor."""
        domain = _make_domain([3, 5, 7])
        dataset = _make_dataset(domain, 1000)
        cl = ('a2', 'a0')
        result = precompute_marginals(dataset, [cl])
        self.assertEqual(result[cl].domain.attributes, ('a2', 'a0'))
        expected = _reference_marginal(dataset, cl)
        np.testing.assert_array_almost_equal(
            np.asarray(result[cl].datavector(flatten=True)),
            np.asarray(expected.datavector(flatten=True)),
            decimal=4,
        )

    def test_large_domain_sizes(self):
        """Domain sizes exceeding uint8 range."""
        domain = _make_domain([256, 512])
        dataset = _make_dataset(domain, 5000)
        cliques = [('a0', 'a1')]
        self._assert_marginals_match(dataset, cliques)

    def test_random_heterogeneous_sweep(self):
        """Randomized sweep over various domain configurations."""
        rng = np.random.default_rng(42)
        for trial in range(10):
            d = rng.integers(3, 8)
            sizes = [
                int(rng.choice([2, 3, 4, 8, 16, 32, 64])) for _ in range(d)
            ]
            domain = _make_domain(sizes)
            dataset = _make_dataset(domain, 500, seed=trial)
            cliques = list(itertools.combinations(domain.attributes, 2))
            self._assert_marginals_match(dataset, cliques)

    def test_weighted_dataset(self):
        """Weighted datasets produce correct marginals."""
        domain = _make_domain([4, 8])
        rng = np.random.default_rng(99)
        weights = rng.exponential(size=2000)
        dataset = _make_dataset(domain, 2000, seed=99, weights=weights)
        cliques = [('a0',), ('a1',), ('a0', 'a1')]
        self._assert_marginals_match(dataset, cliques)

    def test_accepts_tuple_of_cliques(self):
        """Accepts a tuple (Sequence) of cliques, not just a list."""
        domain = _make_domain([4, 8])
        dataset = _make_dataset(domain, 1000)
        cliques = (('a0', 'a1'),)
        result = precompute_marginals(dataset, cliques)
        expected = _reference_marginal(dataset, ('a0', 'a1'))
        np.testing.assert_array_almost_equal(
            np.asarray(result[('a0', 'a1')].datavector(flatten=True)),
            np.asarray(expected.datavector(flatten=True)),
            decimal=4,
        )


if __name__ == '__main__':
    unittest.main()
