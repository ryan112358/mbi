"""Mixture-of-products estimation for discrete distributions.

Inspired by RAP (https://arxiv.org/abs/2103.06641) and RAP^{softmax}
(https://arxiv.org/abs/2106.07153).  Instead of parameterizing via a graphical
model with potentials, the distribution is represented as a mixture of K
product distributions optimized via gradient descent with optax.

This is a cleaned-up, optimized replacement for
``mbi.experimental.mixture_of_products``.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.nn
import jax.numpy as jnp
import numpy as np
import optax

from .. import estimation, marginal_loss
from ..clique_vector import CliqueVector
from ..dataset import Dataset
from ..domain import Domain
from ..factor import Factor
from ..marginal_loss import LinearMeasurement


class MixtureOfProducts:
    """A discrete distribution as a mixture of product distributions."""

    def __init__(
        self, products: dict[str, jax.Array], domain: Domain, total: float
    ):
        """Initialize from per-attribute product arrays, domain, and total."""
        self.products = products
        self.domain = domain
        self.total = total
        self.num_components = next(iter(products.values())).shape[0]

    def project(self, attrs) -> Factor:
        """Compute the marginal over ``attrs`` via einsum."""
        if isinstance(attrs, str):
            attrs = (attrs,)
        attrs = tuple(attrs)
        d = len(attrs)
        letters = "bcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"[:d]
        formula = ",".join(f"a{l}" for l in letters) + "->" + "".join(letters)
        components = [self.products[col] for col in attrs]
        values = (
            jnp.einsum(formula, *components) * self.total / self.num_components
        )
        return Factor(self.domain.project(attrs), values)

    def supports(self, attrs) -> bool:
        """Any subset of domain attributes is supported."""
        if isinstance(attrs, str):
            attrs = (attrs,)
        return all(a in self.domain.attrs for a in attrs)

    def synthetic_data(self, rows=None) -> Dataset:
        """Generate synthetic data via randomized rounding."""
        total = max(1, int(rows or self.total))
        rng = np.random.default_rng()
        subtotal = total // self.num_components + 1

        blocks = []
        for k in range(self.num_components):
            comp_data = {}
            for col in self.domain.attrs:
                counts = np.asarray(self.products[col][k])
                counts = counts * subtotal / counts.sum()
                frac, integ = np.modf(counts)
                integ = integ.astype(int)
                extra = subtotal - integ.sum()
                if extra > 0:
                    p = frac / frac.sum()
                    idx = rng.choice(len(counts), extra, replace=False, p=p)
                    integ[idx] += 1
                vals = np.repeat(np.arange(len(counts)), integ)
                rng.shuffle(vals)
                comp_data[col] = vals
            blocks.append(
                np.stack([comp_data[col] for col in self.domain.attrs], axis=1)
            )

        full_data = np.concatenate(blocks, axis=0)
        rng.shuffle(full_data)
        full_data = full_data[:total]

        data = {col: full_data[:, i] for i, col in enumerate(self.domain.attrs)}
        return Dataset(data, self.domain)


def estimate(
    domain: Domain,
    loss_fn: marginal_loss.MarginalLossFn | list[LinearMeasurement],
    *,
    known_total: float | None = None,
    num_components: int = 100,
    iters: int = 2500,
    learning_rate: float = 0.1,
    optimizer: optax.GradientTransformation | None = None,
    seed: int = 0,
    callback_fn: Callable[[MixtureOfProducts], None] | None = None,
    callback_every: int = 100,
) -> MixtureOfProducts:
    """Estimate a distribution as a mixture of product distributions.

    Args:
        domain: The discrete domain over which the distribution is defined.
        loss_fn: A MarginalLossFn or list of LinearMeasurement objects.
        known_total: Known or estimated number of records.  If None and
            ``loss_fn`` is a list, estimated automatically.
        num_components: Number of mixture components K.
        iters: Number of optimization iterations.
        learning_rate: Learning rate for the optimizer.
        optimizer: An optax optimizer.  Defaults to ``optax.adam``.
        seed: Random seed for parameter initialization.
        callback_fn: Called every ``callback_every`` iterations with the
            current model.
        callback_every: Controls callback frequency and ``lax.scan`` chunk size.

    Returns:
        A MixtureOfProducts fitted to the measurements.
    """
    loss_fn, known_total, _ = estimation._initialize(
        domain, loss_fn, known_total, None
    )

    if optimizer is None:
        optimizer = optax.adam(learning_rate)

    key = jax.random.PRNGKey(seed)
    one_hot_features = sum(domain.shape)
    params = jax.random.normal(key, (num_components, one_hot_features)) * 0.25

    cliques = loss_fn.cliques
    letters = "bcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

    def get_products(params):
        products = {}
        idx = 0
        for col in domain:
            n = domain[col]
            products[col] = jax.nn.softmax(params[:, idx : idx + n], axis=1)
            idx += n
        return products

    def marginals_from_params(params):
        products = get_products(params)
        arrays = {}
        for cl in cliques:
            let = letters[: len(cl)]
            formula = ",".join(f"a{l}" for l in let) + "->" + "".join(let)
            components = [products[col] for col in cl]
            ans = (
                jnp.einsum(formula, *components) * known_total / num_components
            )
            arrays[cl] = Factor(domain.project(cl), ans)
        return CliqueVector(domain, cliques, arrays)

    def params_loss(params: jax.Array) -> float:
        return loss_fn(marginals_from_params(params))

    params_loss_and_grad = jax.jit(jax.value_and_grad(params_loss))
    opt_state = optimizer.init(params)

    def step(carry, _):
        params, opt_state = carry
        loss, grad = params_loss_and_grad(params)
        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), loss

    scan_block = jax.jit(
        lambda carry: jax.lax.scan(step, carry, None, length=callback_every)
    )

    num_blocks = -(-iters // callback_every)  # ceil division
    for _ in range(num_blocks):
        (params, opt_state), _ = scan_block((params, opt_state))
        if callback_fn is not None:
            callback_fn(
                MixtureOfProducts(get_products(params), domain, known_total)
            )

    return MixtureOfProducts(get_products(params), domain, known_total)
