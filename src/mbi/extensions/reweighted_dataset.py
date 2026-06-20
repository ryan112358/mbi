"""Reweighted dataset estimation for discrete distributions.

Inspired by PMW^Pub (https://arxiv.org/abs/2102.08598).  Given a seed set of
records, the distribution is represented by learning a weight for each record
to minimize the marginal loss.  The weights are parameterized in log-space
with softmax normalization to guarantee non-negativity.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import optax

from .. import estimation, marginal_loss
from ..clique_vector import CliqueVector
from ..dataset import Dataset, JaxDataset
from ..domain import Domain
from ..marginal_loss import LinearMeasurement


def estimate(
    domain: Domain,
    loss_fn: marginal_loss.MarginalLossFn | list[LinearMeasurement],
    *,
    seed_data: Dataset,
    known_total: float | None = None,
    iters: int = 2500,
    learning_rate: float = 0.1,
    optimizer: optax.GradientTransformation | None = None,
    callback_fn: Callable[[JaxDataset], None] | None = None,
    callback_every: int = 100,
) -> JaxDataset:
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
            current JaxDataset.
        callback_every: Controls callback frequency and ``lax.scan`` chunk size.

    Returns:
        A JaxDataset with learned weights fitted to the measurements.
    """
    loss_fn, known_total, _ = estimation._initialize(
        domain, loss_fn, known_total, None
    )

    if optimizer is None:
        optimizer = optax.adam(learning_rate)

    data_dict = seed_data.to_dict()
    jax_data = {col: jnp.array(data_dict[col]) for col in domain.attrs}
    log_weights = jnp.zeros(seed_data.records)
    cliques = loss_fn.cliques

    # jax_data values (JAX arrays) are closed over and passed as implicit
    # arguments to JIT — not baked as HLO constants.
    def params_loss(log_weights: jax.Array) -> float:
        weights = jax.nn.softmax(log_weights) * known_total
        weighted = JaxDataset(jax_data, domain, weights)
        arrays = {cl: weighted.project(cl) for cl in cliques}
        mu = CliqueVector(domain, cliques, arrays)
        return loss_fn(mu)

    params_loss_and_grad = jax.jit(jax.value_and_grad(params_loss))
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
        weights = jax.nn.softmax(log_weights) * known_total
        return JaxDataset(jax_data, domain, weights)

    num_blocks = -(-iters // callback_every)  # ceil division
    for _ in range(num_blocks):
        (log_weights, opt_state), _ = scan_block((log_weights, opt_state))
        if callback_fn is not None:
            callback_fn(make_model(log_weights))

    return make_model(log_weights)
