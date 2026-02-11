
import unittest
import jax.numpy as jnp
from mbi.domain import Domain
from mbi.marginal_loss import LinearMeasurement, from_linear_measurements, calculate_l2_lipschitz
from mbi.clique_vector import CliqueVector

class TestMarginalLoss(unittest.TestCase):
    def test_l2_lipschitz_disjoint(self):
        """Verify that Lipschitz constant is max(1/stddev) for disjoint measurements."""
        domain = Domain.fromdict({'a': 10, 'b': 10, 'c': 10})

        # Create disjoint measurements
        # m1 on ('a',) with stddev 0.5 -> 1/stddev = 2.0
        # m2 on ('b',) with stddev 0.2 -> 1/stddev = 5.0
        # m3 on ('c',) with stddev 1.0 -> 1/stddev = 1.0

        m1 = LinearMeasurement(jnp.zeros(10), ('a',), stddev=0.5)
        m2 = LinearMeasurement(jnp.zeros(10), ('b',), stddev=0.2)
        m3 = LinearMeasurement(jnp.zeros(10), ('c',), stddev=1.0)

        measurements = [m1, m2, m3]

        # Calculate loss function and lipschitz constant
        loss_fn = from_linear_measurements(measurements, norm='l2', domain=domain)

        calculated_L = loss_fn.lipschitz
        expected_L = max(1.0 / m.stddev for m in measurements)

        # We expect calculated_L to be very close to expected_L
        # Using a slightly looser tolerance because power iteration is approximate
        self.assertAlmostEqual(calculated_L, expected_L, delta=1e-3)

    def test_l2_lipschitz_single(self):
        """Verify that Lipschitz constant is 1/stddev for a single measurement."""
        domain = Domain.fromdict({'a': 10})
        m1 = LinearMeasurement(jnp.zeros(10), ('a',), stddev=0.5)
        measurements = [m1]

        loss_fn = from_linear_measurements(measurements, norm='l2', domain=domain)

        calculated_L = loss_fn.lipschitz
        expected_L = 1.0 / m1.stddev

        self.assertAlmostEqual(calculated_L, expected_L, delta=1e-3)

if __name__ == '__main__':
    unittest.main()
