"""Reweighted dataset estimation for discrete distributions.

Inspired by PMW^Pub (https://arxiv.org/abs/2102.08598).  Given a seed set of
records, the distribution is represented by learning a weight for each record
to minimize the marginal loss.  The weights are parameterized in log-space
with softmax normalization to guarantee non-negativity.
"""

from __future__ import annotations

from typing import NamedTuple

import dataclasses
import jax
import jax.numpy as jnp
import optax


from ..estimation import Estimator
from ..clique_vector import CliqueVector
from ..dataset import Dataset, JaxDataset
from ..domain import Attribute


class ReweightedDatasetState(NamedTuple):
  """Optimization state for ReweightedDatasetEstimator."""

  log_weights: jax.Array
  opt_state: optax.OptState


@dataclasses.dataclass(frozen=True)
class ReweightedDatasetEstimator(Estimator):
  """Estimates a distribution by reweighting seed records.

  Inspired by PMW^Pub (https://arxiv.org/abs/2102.08598).  Given a seed
  set of records, the distribution is represented by learning a weight for
  each record to minimize the marginal loss.

  Attributes:
      seed_data: A Dataset of seed records to reweight.
      learning_rate: Learning rate for the optimizer.
      optimizer: An optax optimizer.  Defaults to ``optax.adam``.
  """

  seed_data: Dataset
  learning_rate: float = 0.1
  optimizer: optax.GradientTransformation | None = None
  _jax_data: dict[Attribute, jax.Array] = dataclasses.field(
      init=False, repr=False, compare=False
  )

  def __post_init__(self):
    if self.optimizer is None:
      object.__setattr__(self, "optimizer", optax.adam(self.learning_rate))
    # Precompute JAX arrays from seed data once.
    data_dict = self.seed_data.to_dict()
    jax_data = {
        col: jnp.asarray(data_dict[col])
        for col in self.seed_data.domain.attributes
    }
    object.__setattr__(self, "_jax_data", jax_data)

  def _init(self, domain, loss_fn, known_total, *, warm_start=None, **kwargs):
    """Initialize log-weights and optimizer state."""
    log_weights = jnp.zeros(self.seed_data.records)
    opt_state = self.optimizer.init(log_weights)  # pyrefly: ignore[missing-attribute]
    return ReweightedDatasetState(log_weights, opt_state)

  def _step(self, state, loss_fn, known_total, constraints=()):
    """Run one gradient step."""
    domain = self.seed_data.domain
    jax_data = self._jax_data

    def params_loss(lw):
      weights = jax.nn.softmax(lw) * known_total
      weighted = JaxDataset(jax_data, domain, weights)
      arrays = {cl: weighted.project(cl) for cl in loss_fn.cliques}
      mu = CliqueVector(domain, loss_fn.cliques, arrays)
      return loss_fn(mu)

    _, grad = jax.value_and_grad(params_loss)(state.log_weights)
    updates, opt_state = self.optimizer.update(  # pyrefly: ignore[missing-attribute]
        grad, state.opt_state, state.log_weights
    )
    log_weights = optax.apply_updates(state.log_weights, updates)
    return ReweightedDatasetState(log_weights, opt_state)  # pyrefly: ignore[bad-argument-type]

  def _callback_value(self, state, known_total, constraints=()):
    weights = jax.nn.softmax(state.log_weights) * known_total
    return JaxDataset(self._jax_data, self.seed_data.domain, weights)

  def _finalize(self, state, known_total, constraints=()):  # pyrefly: ignore[bad-override]  # returns a JaxDataset Model
    weights = jax.nn.softmax(state.log_weights) * known_total
    return JaxDataset(self._jax_data, self.seed_data.domain, weights)
