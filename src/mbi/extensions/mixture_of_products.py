"""Mixture-of-products estimation for discrete distributions.

Inspired by RAP (https://arxiv.org/abs/2103.06641) and RAP^{softmax}
(https://arxiv.org/abs/2106.07153).  Instead of parameterizing via a graphical
model with potentials, the distribution is represented as a mixture of K
product distributions optimized via gradient descent with optax.

This is a cleaned-up, optimized replacement for
``mbi.experimental.mixture_of_products``.
"""

from __future__ import annotations


from dataclasses import dataclass, field
from typing import NamedTuple

import jax
import jax.nn
import jax.numpy as jnp
import numpy as np
import optax

import dataclasses


from ..estimation import Estimator
from ..clique_vector import CliqueVector
from ..dataset import Dataset
from ..domain import Domain
from ..factor import Factor


@jax.tree_util.register_dataclass
@dataclass
class MixtureOfProducts:
  """A discrete distribution as a mixture of product distributions."""

  _logits: dict[str, jax.Array]
  domain: Domain = field(metadata={"static": True})
  total: float = field(metadata={"static": True})

  @classmethod
  def random(
      cls,
      domain: Domain,
      total: float,
      num_components: int = 100,
      seed: int = 0,
      scale: float = 0.25,
  ) -> MixtureOfProducts:
    """Initialize with random logits."""
    key = jax.random.PRNGKey(seed)
    logits = {}
    for col in domain:
      key, subkey = jax.random.split(key)
      logits[col] = (
          jax.random.normal(subkey, (num_components, domain[col])) * scale
      )
    return cls(logits, domain, total)

  @property
  def num_components(self) -> int:
    """Number of mixture components K."""
    return next(iter(self._logits.values())).shape[0]

  @property
  def products(self) -> dict[str, jax.Array]:
    """Per-attribute categorical probabilities (softmax of logits)."""
    return {
        col: jax.nn.softmax(self._logits[col], axis=1) for col in self._logits
    }

  def project(self, attrs) -> Factor:
    """Compute the marginal over ``attrs`` via einsum."""
    if isinstance(attrs, str):
      attrs = (attrs,)
    attrs = tuple(attrs)
    d = len(attrs)
    letters = "bcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"[:d]
    formula = ",".join(f"a{l}" for l in letters) + "->" + "".join(letters)
    products = self.products
    components = [products[col] for col in attrs]
    values = jnp.einsum(formula, *components) * self.total / self.num_components
    return Factor(self.domain.project(attrs), values)

  def supports(self, attrs) -> bool:
    """Any subset of domain attributes is supported."""
    if isinstance(attrs, str):
      attrs = (attrs,)
    return all(a in self.domain.attributes for a in attrs)

  def synthetic_data(self, rows=None) -> Dataset:
    """Generate synthetic data via randomized rounding."""
    total = max(1, int(rows or self.total))
    rng = np.random.default_rng()
    products = self.products
    subtotal = total // self.num_components + 1

    blocks = []
    for k in range(self.num_components):
      comp_data = {}
      for col in self.domain.attributes:
        counts = np.asarray(products[col][k])
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
          np.stack([comp_data[col] for col in self.domain.attributes], axis=1)
      )

    full_data = np.concatenate(blocks, axis=0)
    rng.shuffle(full_data)
    full_data = full_data[:total]

    data = {
        col: full_data[:, i] for i, col in enumerate(self.domain.attributes)
    }
    return Dataset(data, self.domain)


class MixtureOfProductsState(NamedTuple):
  """Optimization state for MixtureOfProductsEstimator."""

  model: MixtureOfProducts
  opt_state: optax.OptState


@dataclasses.dataclass(frozen=True)
class MixtureOfProductsEstimator(Estimator):
  """Estimates a distribution as a mixture of product distributions.

  Inspired by RAP (https://arxiv.org/abs/2103.06641) and RAP^{softmax}
  (https://arxiv.org/abs/2106.07153).  Instead of parameterizing via a
  graphical model with potentials, the distribution is represented as a
  mixture of K product distributions optimized via gradient descent.

  Attributes:
      num_components: Number of mixture components K.
      learning_rate: Learning rate for the optimizer.
      optimizer: An optax optimizer.  Defaults to ``optax.adam``.
      seed: Random seed for parameter initialization.
  """

  num_components: int = 100
  learning_rate: float = 0.1
  optimizer: optax.GradientTransformation | None = None
  seed: int = 0

  def __post_init__(self):
    if self.optimizer is None:
      object.__setattr__(self, "optimizer", optax.adam(self.learning_rate))

  def _init(self, domain, loss_fn, known_total, *, warm_start=None, **kwargs):
    """Initialize model and optimizer state."""
    model = MixtureOfProducts.random(
        domain, known_total, self.num_components, self.seed
    )
    if warm_start is not None:
      model = warm_start
    opt_state = self.optimizer.init(model)
    return MixtureOfProductsState(model, opt_state)

  def _step(self, state, loss_fn, known_total, constraints=()):
    """Run one gradient step."""

    def model_loss(m):
      arrays = {cl: m.project(cl) for cl in loss_fn.cliques}
      mu = CliqueVector(m.domain, loss_fn.cliques, arrays)
      return loss_fn(mu)

    _, grad = jax.value_and_grad(model_loss)(state.model)
    updates, opt_state = self.optimizer.update(
        grad, state.opt_state, state.model
    )
    model = optax.apply_updates(state.model, updates)
    return MixtureOfProductsState(model, opt_state)

  def _finalize(self, state, known_total, constraints=()):
    return state.model
