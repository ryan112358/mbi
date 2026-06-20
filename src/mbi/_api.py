"""Public protocol definitions for the mbi package.

This module defines the core protocols (interfaces) that mbi types implement.
Protocols are grouped from most general to most specific:

- ``Projectable``: anything that can compute marginals.
- ``Model``: a fitted distribution that can also generate synthetic data.
- ``Estimator``: a function that fits a Model from noisy measurements.
- ``PGMEstimator``: an Estimator backed by a graphical model with potentials
  and a marginal oracle.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Protocol

import jax

if TYPE_CHECKING:
    from .clique_vector import CliqueVector
    from .dataset import Dataset
    from .domain import Domain
    from .factor import Factor
    from .marginal_loss import LinearMeasurement, MarginalLossFn
    from .marginal_oracles import MarginalOracle
    from .markov_random_field import MarkovRandomField


class Projectable(Protocol):
    """An object whose marginals can be computed over subsets of attributes.

    Example projectables:
        * Dataset
        * Factor
        * CliqueVector
        * MarkovRandomField
        * MixtureOfProducts
        * JaxDataset
    """

    @property
    def domain(self) -> Domain:
        """Returns the domain over which this projectable is defined."""

    def project(self, attrs: str | Sequence[str]) -> Factor:
        """Projection onto a subset of attributes."""

    def supports(self, attrs: str | Sequence[str]) -> bool:
        """Returns true if the given attributes can be projected onto."""


class Model(Projectable, Protocol):
    """A fitted distribution that supports marginal queries and synthetic data.

    All estimation functions return a Model.  Unlike a bare Projectable,
    a Model can also generate synthetic tabular data that is consistent
    with the learned distribution.

    Example models:
        * MarkovRandomField
        * MixtureOfProducts
        * JaxDataset (with weights)
    """

    def synthetic_data(self, rows: int | None = None) -> Dataset:
        """Generate synthetic tabular data from the model."""


class Estimator(Protocol):
    """Callable that estimates a Model from a marginal-based loss function.

    Any function conforming to this protocol can be used interchangeably
    as an estimation algorithm, regardless of the underlying model class.

    Examples of conforming functions:
        * ``estimation.mirror_descent``
        * ``estimation.lbfgs``
        * ``estimation.dual_averaging``
        * ``mixture_of_products.estimate``
        * ``reweighted_dataset.estimate``
    """

    def __call__(
        self,
        domain: Domain,
        loss_fn: MarginalLossFn | list[LinearMeasurement],
        *,
        known_total: float | None = None,
        **kwargs: Any,
    ) -> Projectable:
        """Estimate a Projectable from noisy marginal measurements."""


class PGMEstimator(Protocol):
    """An Estimator backed by a Markov Random Field.

    PGM estimators optimize potentials using a marginal oracle and
    return a MarkovRandomField.

    Examples of conforming functions:
        * ``estimation.mirror_descent``
        * ``estimation.lbfgs``
        * ``estimation.dual_averaging``
        * ``estimation.interior_gradient``
        * ``estimation.universal_accelerated_method``
    """

    def __call__(
        self,
        domain: Domain,
        loss_fn: MarginalLossFn | list[LinearMeasurement],
        *,
        known_total: float | None = None,
        potentials: CliqueVector | None = None,
        marginal_oracle: MarginalOracle = ...,
        iters: int = ...,
        callback_fn: Callable[[CliqueVector], None] = ...,
        mesh: jax.sharding.Mesh | None = None,
        **kwargs: Any,
    ) -> MarkovRandomField:
        """Estimate a MarkovRandomField from noisy marginal measurements."""
