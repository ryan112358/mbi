"""Reweighted dataset estimation for discrete distributions.

Inspired by PMW^Pub (https://arxiv.org/abs/2102.08598).  Given a seed set of
records, the distribution is represented by learning a weight for each record
to minimize the marginal loss.  The weights are parameterized in log-space
with softmax normalization to guarantee non-negativity.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .. import estimation, marginal_loss
from ..clique_vector import CliqueVector
from ..dataset import Dataset
from ..domain import Domain
from ..factor import Factor
from ..marginal_loss import LinearMeasurement


@jax.tree_util.register_dataclass
@dataclass
class ReweightedDataset:
    """A discrete distribution as a reweighted set of seed records."""

    _log_weights: jax.Array
    _data: dict[str, jax.Array]
    domain: Domain = field(metadata={"static": True})
    total: float = field(metadata={"static": True})

    @classmethod
    def from_dataset(cls, dataset: Dataset, total: float | None = None):
        """Initialize from a Dataset with uniform weights."""
        data_dict = dataset.to_dict()
        data = {col: jnp.array(data_dict[col]) for col in dataset.domain.attrs}
        n = dataset.records
        total = total if total is not None else float(n)
        return cls(jnp.zeros(n), data, dataset.domain, total)

    @property
    def weights(self) -> jax.Array:
        """Non-negative weights summing to total (softmax of log-weights)."""
        return jax.nn.softmax(self._log_weights) * self.total

    @property
    def num_records(self) -> int:
        """Number of seed records."""
        return next(iter(self._data.values())).shape[0]

    def project(self, attrs) -> Factor:
        """Compute the weighted marginal over ``attrs``."""
        if isinstance(attrs, str):
            attrs = (attrs,)
        attrs = tuple(attrs)
        proj_domain = self.domain.project(attrs)
        weights = self.weights

        indices = tuple(self._data[a] for a in attrs)
        values = jnp.zeros(proj_domain.shape)
        values = values.at[indices].add(weights)
        return Factor(proj_domain, values)

    def supports(self, attrs) -> bool:
        """Any subset of domain attributes is supported."""
        if isinstance(attrs, str):
            attrs = (attrs,)
        return all(a in self.domain.attrs for a in attrs)

    def synthetic_data(self, rows=None) -> Dataset:
        """Generate synthetic data via randomized rounding of weights."""
        total = max(1, int(rows or self.total))
        rng = np.random.default_rng()
        weights = np.asarray(self.weights)
        counts = weights * total / weights.sum()
        frac, integ = np.modf(counts)
        integ = integ.astype(int)
        extra = total - integ.sum()
        if extra > 0:
            p = frac / frac.sum()
            idx = rng.choice(len(counts), extra, replace=False, p=p)
            integ[idx] += 1
        row_indices = np.repeat(np.arange(len(counts)), integ)
        rng.shuffle(row_indices)
        row_indices = row_indices[:total]
        data = {
            col: np.asarray(self._data[col])[row_indices]
            for col in self.domain.attrs
        }
        return Dataset(data, self.domain)


def estimate(
    domain: Domain,
    loss_fn: marginal_loss.MarginalLossFn | list[LinearMeasurement],
    seed_data: Dataset,
    *,
    known_total: float | None = None,
    iters: int = 2500,
    learning_rate: float = 0.1,
    optimizer: optax.GradientTransformation | None = None,
    callback_fn: Callable[[ReweightedDataset], None] | None = None,
    callback_every: int = 100,
) -> ReweightedDataset:
    """Estimate a distribution by reweighting seed records.

    Args:
        domain: The discrete domain over which the distribution is defined.
        loss_fn: A MarginalLossFn or list of LinearMeasurement objects.
        seed_data: A Dataset of seed records to reweight.
        known_total: Known or estimated number of records.  If None and
            ``loss_fn`` is a list, estimated automatically.
        iters: Number of optimization iterations.
        learning_rate: Learning rate for the optimizer.
        optimizer: An optax optimizer.  Defaults to ``optax.adam``.
        callback_fn: Called every ``callback_every`` iterations with the
            current model.
        callback_every: Controls callback frequency and ``lax.scan`` chunk size.

    Returns:
        A ReweightedDataset fitted to the measurements.
    """
    loss_fn, known_total, _ = estimation._initialize(
        domain, loss_fn, known_total, None
    )

    if optimizer is None:
        optimizer = optax.adam(learning_rate)

    model = ReweightedDataset.from_dataset(seed_data, known_total)
    cliques = loss_fn.cliques

    # Optimize only _log_weights.  model._data (JAX arrays) is closed over
    # and passed as implicit arguments to JIT, not baked as HLO constants.
    def params_loss(log_weights: jax.Array) -> float:
        m = ReweightedDataset(log_weights, model._data, domain, known_total)
        arrays = {cl: m.project(cl) for cl in cliques}
        mu = CliqueVector(domain, cliques, arrays)
        return loss_fn(mu)

    params_loss_and_grad = jax.jit(jax.value_and_grad(params_loss))
    log_weights = model._log_weights
    opt_state = optimizer.init(log_weights)

    def step(carry, _):
        log_weights, opt_state = carry
        loss, grad = params_loss_and_grad(log_weights)
        updates, opt_state = optimizer.update(grad, opt_state, log_weights)
        log_weights = optax.apply_updates(log_weights, updates)
        return (log_weights, opt_state), loss

    scan_block = jax.jit(
        lambda carry: jax.lax.scan(step, carry, None, length=callback_every)
    )

    def make_model(log_weights):
        return ReweightedDataset(log_weights, model._data, domain, known_total)

    num_blocks = -(-iters // callback_every)  # ceil division
    for _ in range(num_blocks):
        (log_weights, opt_state), _ = scan_block((log_weights, opt_state))
        if callback_fn is not None:
            callback_fn(make_model(log_weights))

    return make_model(log_weights)
