"""Tests for mbi.save / mbi.load serialization."""

import io

import jax.numpy as jnp
import numpy as np
import pytest

from mbi import (
    CliqueVector,
    DatavectorQuery,
    Domain,
    Factor,
    LinearMeasurement,
    MarkovRandomField,
    NormalizedQuery,
    SlicedQuery,
    WeightedQuery,
    load,
    marginal_oracles,
    save,
)


def _roundtrip(obj):
    """Save and load a pytree through an in-memory buffer."""
    buf = io.BytesIO()
    save(obj, buf)
    buf.seek(0)
    return load(buf)


@pytest.fixture
def domain():
    return Domain(["A", "B", "C"], [3, 4, 5])


@pytest.fixture
def cliques():
    return [("A", "B"), ("B", "C")]


class TestCliqueVector:

    def test_roundtrip(self, domain, cliques):
        cv = CliqueVector.random(domain, cliques)
        loaded = _roundtrip(cv)

        assert loaded.domain == cv.domain
        assert loaded.cliques == cv.cliques
        for cl in cv.cliques:
            np.testing.assert_allclose(
                np.asarray(loaded[cl].values),
                np.asarray(cv[cl].values),
                atol=1e-6,
            )


class TestMarkovRandomField:

    def test_roundtrip(self, domain, cliques):
        potentials = CliqueVector.random(domain, cliques).log()
        marginals = marginal_oracles.message_passing_hugin(
            potentials, total=100.0
        )
        mrf = MarkovRandomField(
            potentials=potentials, marginals=marginals, total=100.0
        )
        loaded = _roundtrip(mrf)

        assert loaded.domain == mrf.domain
        np.testing.assert_allclose(float(loaded.total), float(mrf.total))
        for cl in mrf.potentials.cliques:
            np.testing.assert_allclose(
                np.asarray(loaded.potentials[cl].values),
                np.asarray(mrf.potentials[cl].values),
                atol=1e-6,
            )

    def test_project_after_load(self, domain, cliques):
        """Loaded model should support the same queries as the original."""
        potentials = CliqueVector.random(domain, cliques).log()
        marginals = marginal_oracles.message_passing_hugin(
            potentials, total=100.0
        )
        mrf = MarkovRandomField(
            potentials=potentials, marginals=marginals, total=100.0
        )
        loaded = _roundtrip(mrf)

        for attr in mrf.domain.attrs:
            np.testing.assert_allclose(
                np.asarray(loaded.project((attr,)).values),
                np.asarray(mrf.project((attr,)).values),
                atol=1e-5,
            )


class TestLinearMeasurements:

    def test_roundtrip(self, domain, cliques):
        ms = [
            LinearMeasurement(jnp.asarray(np.random.randn(domain.size(cl))), cl)
            for cl in cliques
        ]
        loaded = _roundtrip(ms)

        assert len(loaded) == len(ms)
        for orig, ld in zip(ms, loaded):
            assert ld.clique == orig.clique
            np.testing.assert_allclose(ld.stddev, orig.stddev)
            np.testing.assert_allclose(
                np.asarray(ld.noisy_measurement),
                np.asarray(orig.noisy_measurement),
                atol=1e-6,
            )

    def test_mixed_query_types(self, domain, cliques):
        """All query types roundtrip correctly in a single list."""
        cl = cliques[0]
        size = domain.size(cl)
        weights = np.random.rand(size)
        ms = [
            LinearMeasurement(jnp.ones(size), cl),
            LinearMeasurement(jnp.ones(size), cl, query=WeightedQuery(weights)),
            LinearMeasurement(jnp.ones(size), cl, query=NormalizedQuery()),
            LinearMeasurement(
                jnp.ones(size - 1), cl, query=SlicedQuery(start=1)
            ),
        ]
        loaded = _roundtrip(ms)

        assert isinstance(loaded[0].query, DatavectorQuery)
        assert isinstance(loaded[1].query, WeightedQuery)
        np.testing.assert_allclose(loaded[1].query.weights, weights, atol=1e-7)
        assert isinstance(loaded[2].query, NormalizedQuery)
        assert isinstance(loaded[3].query, SlicedQuery)
        assert loaded[3].query.start == 1

        # Functional equivalence for the weighted query.
        f = Factor(domain.project(cl), jnp.ones(size))
        np.testing.assert_allclose(
            np.asarray(loaded[1].query(f)),
            np.asarray(ms[1].query(f)),
            atol=1e-7,
        )
