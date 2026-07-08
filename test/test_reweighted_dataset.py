"""Tests for mbi.extensions.reweighted_dataset."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np
from parameterized import parameterized

from mbi import Domain, Model, Projectable, marginal_loss
from mbi.clique_vector import CliqueVector
from mbi.dataset import Dataset, JaxDataset
from mbi.extensions.reweighted_dataset import ReweightedDatasetEstimator
from mbi.factor import Factor


np.random.seed(42)  # Avoid flaky tests


_DOMAIN = Domain(["a", "b", "c", "d"], [2, 3, 4, 5])

_CLIQUE_SETS = [
    [("a", "b"), ("b", "c"), ("c", "d")],  # tree
    [("a",), ("a", "b"), ("b", "c"), ("a", "c"), ("b", "d")],  # cyclic/dense
    [("a", "b"), ("d", "a")],  # missing c
    [("d",)],  # singleton
]


def _make_seed_data(domain, n=5000):
    """Create a random Dataset over the domain."""
    data = {
        col: np.random.randint(0, domain[col], n) for col in domain.attributes
    }
    return Dataset(data, domain)


def _fake_measurements(domain, cliques):
    """Create noise-free measurements from a random distribution."""
    P = Factor.random(domain)
    P = P / P.sum()
    measurements = []
    for cl in cliques:
        y = P.project(cl).datavector()
        measurements.append(marginal_loss.LinearMeasurement(y, cl))
    return measurements, P


class TestJaxDatasetBasics(unittest.TestCase):
    """Tests for JaxDataset used as the reweighted model."""

    def setUp(self):
        self.domain = Domain(["x", "y"], [3, 4])
        data = {"x": np.array([0, 1, 2, 0, 1]), "y": np.array([0, 1, 2, 3, 0])}
        self.dataset = Dataset(data, self.domain)
        weights = jax.nn.softmax(jnp.zeros(5)) * 100.0
        self.model = JaxDataset(
            {col: jnp.array(data[col]) for col in self.domain.attributes},
            self.domain,
            weights,
        )

    def test_num_records(self):
        self.assertEqual(self.model.records, 5)

    def test_weights_sum_to_total(self):
        np.testing.assert_allclose(
            float(self.model.weights.sum()), 100.0, atol=1e-5
        )

    def test_uniform_weights(self):
        """Initial weights should be uniform."""
        weights = np.asarray(self.model.weights)
        np.testing.assert_allclose(weights, 20.0, atol=1e-5)

    def test_project_single_attr(self):
        marg = self.model.project("x")
        self.assertEqual(marg.domain, self.model.domain.project(("x",)))
        np.testing.assert_allclose(float(marg.sum()), 100.0, atol=1e-5)

    def test_project_multiple_attrs(self):
        marg = self.model.project(("x", "y"))
        self.assertEqual(marg.domain.shape, (3, 4))
        np.testing.assert_allclose(float(marg.sum()), 100.0, atol=1e-5)

    def test_project_consistency(self):
        """Marginal of the joint should equal the direct marginal."""
        joint = self.model.project(("x", "y"))
        marg_x_from_joint = joint.project("x")
        marg_x_direct = self.model.project("x")
        np.testing.assert_allclose(
            marg_x_from_joint.datavector(),
            marg_x_direct.datavector(),
            atol=1e-6,
        )

    def test_supports(self):
        self.assertTrue(self.model.supports("x"))
        self.assertTrue(self.model.supports(("x", "y")))
        self.assertFalse(self.model.supports("z"))

    def test_synthetic_data(self):
        data = self.model.synthetic_data(rows=500)
        self.assertEqual(data.records, 500)
        self.assertEqual(data.domain, self.model.domain)

    def test_synthetic_data_default_rows(self):
        data = self.model.synthetic_data()
        self.assertEqual(data.records, 100)

    def test_synthetic_data_values_in_range(self):
        data = self.model.synthetic_data(rows=200)
        data_dict = data.to_dict()
        for col in self.model.domain.attributes:
            col_data = data_dict[col]
            self.assertTrue(np.all(col_data >= 0))
            self.assertTrue(np.all(col_data < self.model.domain[col]))

    def test_pytree_roundtrip(self):
        """Model should survive JAX pytree flatten/unflatten."""
        leaves, treedef = jax.tree_util.tree_flatten(self.model)
        reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)
        np.testing.assert_allclose(
            np.asarray(reconstructed.weights), np.asarray(self.model.weights)
        )


class TestEstimation(unittest.TestCase):
    """Tests for the reweighted_dataset.estimate function."""

    @parameterized.expand([(cliques,) for cliques in _CLIQUE_SETS])
    def test_noiseless_recovery(self, cliques):
        """With noise-free measurements, the model should fit them closely."""
        measurements, P = _fake_measurements(_DOMAIN, cliques)
        seed_data = _make_seed_data(_DOMAIN, n=5000)
        model = ReweightedDatasetEstimator(
            seed_data=seed_data,
            learning_rate=0.1,
        ).estimate(
            _DOMAIN,
            measurements,
            iters=500,
        )

        for M in measurements:
            expected = M.noisy_measurement
            actual = model.project(M.clique).datavector()
            np.testing.assert_allclose(actual, expected, atol=5e-2)

    def test_l1_loss(self):
        """Test that L1 loss works without crashing and converges."""
        cliques = [("a", "b"), ("b", "c")]
        measurements, P = _fake_measurements(_DOMAIN, cliques)
        loss_fn = marginal_loss.from_linear_measurements(
            measurements, _DOMAIN, norm="l1"
        )
        seed_data = _make_seed_data(_DOMAIN)

        model = ReweightedDatasetEstimator(
            seed_data=seed_data,
            learning_rate=0.01,
        ).estimate(
            _DOMAIN,
            loss_fn,
            known_total=1.0,
            iters=500,
        )

        for M in measurements:
            expected = M.noisy_measurement
            actual = model.project(M.clique).datavector()
            np.testing.assert_allclose(actual, expected, atol=0.1)

    def test_l2_loss(self):
        """Test explicit L2 loss construction."""
        cliques = [("a", "b"), ("b", "c")]
        measurements, P = _fake_measurements(_DOMAIN, cliques)
        loss_fn = marginal_loss.from_linear_measurements(
            measurements, _DOMAIN, norm="l2"
        )
        seed_data = _make_seed_data(_DOMAIN)

        model = ReweightedDatasetEstimator(
            seed_data=seed_data,
        ).estimate(
            _DOMAIN,
            loss_fn,
            known_total=1.0,
            iters=500,
        )

        for M in measurements:
            expected = M.noisy_measurement
            actual = model.project(M.clique).datavector()
            np.testing.assert_allclose(actual, expected, atol=5e-2)

    def test_returns_jax_dataset(self):
        """estimate() should return a JaxDataset."""
        cliques = [("a", "b"), ("c", "d")]
        measurements, _ = _fake_measurements(_DOMAIN, cliques)
        seed_data = _make_seed_data(_DOMAIN)
        model = ReweightedDatasetEstimator(
            seed_data=seed_data,
        ).estimate(
            _DOMAIN,
            measurements,
            iters=50,
        )

        self.assertIsInstance(model, JaxDataset)
        self.assertEqual(model.domain, _DOMAIN)
        self.assertTrue(model.supports(("a", "b")))

    def test_conforms_to_model_protocol(self):
        """JaxDataset should satisfy the Model protocol."""
        cliques = [("a", "b"), ("c", "d")]
        measurements, _ = _fake_measurements(_DOMAIN, cliques)
        seed_data = _make_seed_data(_DOMAIN)
        model = ReweightedDatasetEstimator(
            seed_data=seed_data,
        ).estimate(
            _DOMAIN,
            measurements,
            iters=50,
        )

        # Model extends Projectable with synthetic_data
        self.assertTrue(hasattr(model, "domain"))
        self.assertTrue(hasattr(model, "project"))
        self.assertTrue(hasattr(model, "supports"))
        self.assertTrue(hasattr(model, "synthetic_data"))

    def test_synthetic_data_from_estimation(self):
        """Synthetic data should be generatable from an estimated model."""
        cliques = [("a", "b"), ("b", "c")]
        measurements, _ = _fake_measurements(_DOMAIN, cliques)
        seed_data = _make_seed_data(_DOMAIN)
        model = ReweightedDatasetEstimator(
            seed_data=seed_data,
        ).estimate(
            _DOMAIN,
            measurements,
            iters=100,
        )

        data = model.synthetic_data(rows=1000)
        self.assertEqual(data.records, 1000)
        self.assertEqual(data.domain, _DOMAIN)

    def test_callback(self):
        """Callback should be invoked during estimation."""
        cliques = [("a", "b")]
        measurements, _ = _fake_measurements(_DOMAIN, cliques)
        seed_data = _make_seed_data(_DOMAIN)

        callback_count = [0]

        def callback(model):
            callback_count[0] += 1

        ReweightedDatasetEstimator(
            seed_data=seed_data,
        ).estimate(
            _DOMAIN,
            measurements,
            iters=100,
            callback_fn=callback,
        )
        self.assertGreater(callback_count[0], 0)

    def test_custom_optimizer(self):
        """Should accept a custom optax optimizer."""
        import optax

        cliques = [("a", "b"), ("b", "c")]
        measurements, _ = _fake_measurements(_DOMAIN, cliques)
        seed_data = _make_seed_data(_DOMAIN)
        model = ReweightedDatasetEstimator(
            seed_data=seed_data,
            optimizer=optax.sgd(0.01),
        ).estimate(
            _DOMAIN,
            measurements,
            iters=100,
        )
        self.assertIsInstance(model, JaxDataset)


class TestNonNegativity(unittest.TestCase):
    """Test that the softmax parameterization guarantees non-negativity."""

    def test_marginals_nonnegative(self):
        """All marginals should be non-negative."""
        cliques = [("a", "b"), ("b", "c"), ("c", "d")]
        measurements, _ = _fake_measurements(_DOMAIN, cliques)
        seed_data = _make_seed_data(_DOMAIN)
        model = ReweightedDatasetEstimator(
            seed_data=seed_data,
        ).estimate(
            _DOMAIN,
            measurements,
            iters=100,
        )

        for cl in cliques:
            marg = model.project(cl)
            self.assertTrue(
                jnp.all(marg.values >= 0),
                f"Negative values in marginal for clique {cl}",
            )

    def test_marginals_sum_to_total(self):
        """Each marginal should sum to total."""
        cliques = [("a", "b"), ("b", "c")]
        measurements, _ = _fake_measurements(_DOMAIN, cliques)
        seed_data = _make_seed_data(_DOMAIN)
        model = ReweightedDatasetEstimator(
            seed_data=seed_data,
        ).estimate(
            _DOMAIN,
            measurements,
            iters=100,
        )

        for cl in cliques:
            marg = model.project(cl)
            np.testing.assert_allclose(
                float(marg.sum()), float(model.weights.sum()), atol=1e-5
            )


if __name__ == "__main__":
    unittest.main()
