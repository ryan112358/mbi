"""Public type definitions for the mbi package.

This module defines the core types (interfaces) that mbi types implement:

- ``Projectable``: anything that can compute marginals.
- ``Model``: a fitted distribution that can also generate synthetic data.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
  from .dataset import Dataset
  from .domain import Domain
  from .factor import Factor


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

  @property
  def total(self) -> float:
    """The total count (number of records) represented by the model."""

  def synthetic_data(self, rows: int | None = None) -> Dataset:
    """Generate synthetic tabular data from the model."""
