"""Defines loss functions based on linear measurements of marginals.

This module provides structures and functions for defining and calculating loss
based on potentially noisy linear measurements of marginal distributions. Key
components include the `LinearMeasurement` class to represent individual
measurements and the `MarginalLossFn` class to define loss functions over
`CliqueVector` objects, enabling the evaluation of model fit against observed
or noisy data. Utilities for clique manipulation and feasibility checks are also
included.
"""
import functools
from collections.abc import Callable

import attr
import chex
import jax
import jax.numpy as jnp
import optax

from .clique_utils import Clique, maximal_subset
from .clique_vector import CliqueVector
from .domain import Domain
from .factor import Factor


@functools.partial(
    jax.tree_util.register_dataclass,
    meta_fields=["clique", "stddev", "query"],
    data_fields=["values"],
)
@attr.dataclass(frozen=True)
class LinearMeasurement:
    """A class for representing a private linear measurement of a marginal.

    Attributes:
        noisy_measurement: The noisy measurement of the marginal.
        clique: The clique (a tuple of attribute names) defining the marginal.
        stddev: The standard deviation of the noise added to the measurement.
        query: A linear function that, when applied to a Factor, extracts a
        a vector with the same shape and interpretation as `noisy_measurement`.
    """

    noisy_measurement: jax.Array
    clique: Clique = attr.field(converter=tuple)
    stddev: float = 1.0
    query: Callable[[Factor], jax.Array] = Factor.datavector


# this class might need to be refactored so that loss_fn consumes measurements
# that way measurements can be included as an input to the jitted.
# Or it can be a pytree where the measurements are one node in the PyTree.
@attr.dataclass(frozen=True)
class MarginalLossFn:
    """A Loss function over the concatenated vector of marginals.

    Attributes:
        cliques: A list of cliques (tuples of attribute names) that define the
            scope of the marginals used in the loss function.
        loss_fn: A callable that takes a `CliqueVector` (representing the
            marginals) and returns a numeric loss value.
        lipschitz: An optional float representing the Lipschitz constant of the
            gradient of the loss function. This is used for optimization algorithms.
    """

    cliques: list[Clique]
    loss_fn: Callable[[CliqueVector], chex.Numeric]
    lipschitz: float | None = None

    def __call__(self, marginals: CliqueVector) -> chex.Numeric:
        return self.loss_fn(marginals)


def calculate_l2_lipschitz(domain: Domain, cliques: list[Clique], loss_fn: Callable[[CliqueVector], chex.Numeric]) -> float:
    """Estimate the Lipschitz constant of L(x) = || f(x) - y ||_2^2 where f is a linear function.

    The Lipschitz constant can usually be obtained via the largest eigenvalue of the Hessian, which
    for linear functions represented in matrix form is A^T A.  This function computes the same
    value without materializing this n x n matrix by using power iteration and leveraging jax.jvp.

    Args:
        domain: The domain over which the loss_fn is defined.
        loss_fn: The loss function, assumed to be of the form || f(x) - y ||_2^2 where f is linear.

    Returns:
        An estimate of the Lipschitz constant of the grad(L).
    """
    x0 = CliqueVector.zeros(domain, cliques)
    @jax.jit
    def compute_Hv(v: CliqueVector) -> CliqueVector:
        return jax.jvp(jax.grad(loss_fn), (x0,), (v,))[1]
    v = CliqueVector.ones(domain, cliques)
    v = v / optax.global_norm(v)
    for _ in range(50):
        Hv = compute_Hv(v)
        estimate = optax.global_norm(Hv)
        v = Hv / (estimate + 1e-12)
    return estimate


def from_linear_measurements(
    measurements: list[LinearMeasurement], norm: str = "l2", normalize: bool = False, domain: Domain | None = None,
) -> MarginalLossFn:
    """Construct a MarginalLossFn from a list of LinearMeasurements.

    Args:
        measurements: A list of LinearMeasurements.
        norm: Either "l1" or "l2".
        normalize: Flag determining if the loss function should be normalized
            by the length of linear measurements and estimated total.
        domain: The domain over which the measurements were made, necessary for calcualting the Lipschitz parameter.

    Returns:
        The MarginalLossFn L(mu) = sum_{c} || Q_c mu_c - y_c || (possibly squared or normalized).
    """
    if norm not in ["l1", "l2"]:
        raise ValueError(f"Unknown norm {norm}.")
    cliques = [m.clique for m in measurements]
    maximal_cliques = maximal_subset(cliques)

    def loss_fn(marginals: CliqueVector) -> chex.Numeric:
        loss = 0.0
        for M in measurements:
            mu = marginals.project(M.clique)
            diff = M.query(mu) - M.noisy_measurement
            if norm == "l2":
                loss += (diff @ diff) / (2 * M.stddev)
            elif norm == "l1":
                loss += jnp.sum(jnp.abs(diff)) / M.stddev

        if normalize:
            total = marginals.project([]).datavector(flatten=False)
            loss = loss / len(measurements) / total
            if norm == "l2":
                loss = jnp.sqrt(loss)
        return loss

    if norm == "l2" and not normalize and domain is not None:
        lipschitz = calculate_l2_lipschitz(domain, maximal_cliques, loss_fn)
        return MarginalLossFn(maximal_cliques, loss_fn, lipschitz)

    return MarginalLossFn(maximal_cliques, loss_fn)


def primal_feasibility(mu: CliqueVector) -> chex.Numeric:
    """Calculates the average L1 distance between overlapping marginals in `mu` (consistency)."""
    ans = 0
    count = 0
    for r in mu.cliques:
        for s in mu.cliques:
            if r == s:
                break
            d = tuple(set(r) & set(s))
            if len(d) > 0:
                x = mu[r].project(d).datavector()
                y = mu[s].project(d).datavector()
                denom = 0.5 * x.sum() + 0.5 * y.sum()
                err = jnp.linalg.norm(x - y, 1) / denom
                ans += err
                count += 1
    try:
        return ans / count
    except:
        return 0
