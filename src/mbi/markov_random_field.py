"""Defines the MarkovRandomField class representing learned graphical models.

This module provides the `MarkovRandomField` class, which encapsulates the
results of learning a graphical model. It stores the learned potentials,
the resulting marginal distributions, and the associated total count (e.g.,
number of records). It also offers methods for querying marginals and
generating synthetic data.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses

import chex
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
import numpy as np

from . import junction_tree, marginal_oracles
from .clique_utils import Clique
from .clique_vector import CliqueVector
from .constraint import Constraint
from .dataset import Dataset
from .domain import Attribute
from .domain import Domain
from .factor import Factor


@jax.jit
def _condition_jit(
    potentials: CliqueVector,
    evidence: dict[Attribute, jax.Array],
    total: jax.Array,
    parent_marginal: Factor | None,
    constraints: tuple[Constraint, ...],
) -> tuple[CliqueVector, CliqueVector, jax.Array]:
  # Slice log-potentials on the evidence and run belief propagation over the
  # reduced domain inside a single JIT boundary so varying evidence values
  # reuse the compiled executable.
  if parent_marginal is not None:
    ev_attrs = tuple(evidence.keys())
    total = parent_marginal.project(ev_attrs).slice(evidence).values
  sliced_constraints = tuple(
      c for c in constraints if set(c.domain.attributes) & set(evidence)
  )
  rem_constraints = tuple(
      c for c in constraints if not (set(c.domain.attributes) & set(evidence))
  )
  potentials, _ = marginal_oracles._fold_constraints(
      potentials, sliced_constraints
  )
  sliced = potentials.slice(evidence)
  if constraints:
    marginals = marginal_oracles.message_passing_shafer_shenoy(
        sliced, 1.0, constraints=rem_constraints
    )
  else:
    marginals = marginal_oracles.message_passing_stable(sliced, 1.0)
  return sliced, marginals * total, total


@chex.dataclass(frozen=True, kw_only=False)
class MarkovRandomField:
  """Represents a learned graphical model.

  This class encapsulates the components of a Markov Random Field that has been
  learned from data. It stores the learned potentials, the resulting marginal
  distributions over specified cliques, and the total count (e.g., number of
  records or equivalent sample size) associated with the model.

  Attributes:
      potentials (CliqueVector): A `CliqueVector` containing the learned
          potential functions for the cliques in the model.
      marginals (CliqueVector): A `CliqueVector` containing the marginal
          distributions for a set of cliques, derived from the potentials.
      total (ArrayLike): The total count or effective sample size
          represented by the model. This is often used for scaling or
          interpreting the marginals.
      constraints (tuple[Constraint, ...]): Structural constraints the model
          was fit under, used by ``synthetic_data`` so generated records
          respect them. Empty if the model was fit without constraints.
  """

  potentials: CliqueVector
  marginals: CliqueVector
  total: ArrayLike = 1
  constraints: tuple[Constraint, ...] = ()

  def project(self, attrs: Attribute | Sequence[Attribute]) -> Factor:
    if isinstance(attrs, (str, int)):
      attrs = (attrs,)
    attrs = tuple(attrs)
    if self.marginals.supports(attrs):
      return self.marginals.project(attrs)
    return marginal_oracles.variable_elimination(
        self.potentials,
        attrs,
        float(self.total),  # pyrefly: ignore[bad-argument-type]
        constraints=self.constraints,
    )

  def supports(self, attrs: Attribute | Sequence[Attribute]) -> bool:
    return self.marginals.domain.supports(attrs)

  def condition(
      self,
      evidence: Mapping[Attribute, int | jax.Array],
      total: ArrayLike | None = None,
  ) -> MarkovRandomField:
    """Conditions the model on observed scalar attribute values.

    Args:
        evidence: Mapping from attribute names to observed integer values.
        total: Total count for the conditioned model. If None, defaults to the
          expected conditional count ``self.total * P(evidence)``.

    Returns:
        A new MarkovRandomField defined over the reduced domain.
    """
    unknown = set(evidence) - set(self.domain.attributes)
    if unknown:
      raise ValueError(f"Unknown evidence attributes: {unknown}.")
    if not evidence:
      if total is None:
        return self
      scale = jnp.asarray(total, dtype=float) / jnp.asarray(
          self.total, dtype=float
      )
      return dataclasses.replace(
          self, marginals=self.marginals * scale, total=total
      )
    ev_jax = {k: jnp.asarray(v, dtype=jnp.int32) for k, v in evidence.items()}
    ev_attrs = tuple(ev_jax.keys())
    parent_marginal = None
    if total is not None:
      cond_total = jnp.asarray(total, dtype=float)
    elif self.marginals.supports(ev_attrs):
      parent_cl = self.marginals.parent(ev_attrs)
      assert parent_cl is not None
      parent_marginal = self.marginals[parent_cl]
      cond_total = jnp.asarray(0.0)
    else:
      cond_total = self.project(ev_attrs).slice(ev_jax).values
    sliced_potentials, cond_marginals, cond_total = _condition_jit(
        self.potentials, ev_jax, cond_total, parent_marginal, self.constraints
    )
    rem_constraints = tuple(
        c
        for c in self.constraints
        if not (set(c.domain.attributes) & set(evidence))
    )
    return MarkovRandomField(
        potentials=sliced_potentials,
        marginals=cond_marginals,
        total=cond_total if total is None else total,
        constraints=rem_constraints,
    )

  def synthetic_data(
      self,
      rows: int | None = None,
      method: str = "round",
      evidence: Mapping[Attribute, int | jax.Array] | None = None,
  ) -> Dataset:
    """Generates synthetic data based on the learned model's marginals.

    Args:
        rows: The number of rows to generate. If not provided, uses the model
          total, which is usually estimated automatically.
        method: Specification for strategy to use to generate records. - "round"
          for randomized rounding - "sample" for i.i.d sampling
        evidence: Optional mapping from attribute names to fixed integer values
          to condition on before generating the remaining attributes.

    Returns:
        A synthetic dataset whose marginals should closely match those of the
        model.
    """
    if evidence:
      cond_model = self.condition(evidence, total=rows)
      data = (
          cond_model.synthetic_data(rows=rows, method=method).to_dict()
          if len(cond_model.domain) > 0
          else {}
      )
      n_rows = max(1, int(rows or cond_model.total))  # pyrefly: ignore[bad-argument-type]
      for attr, val in evidence.items():
        dtype = np.min_scalar_type(self.domain[attr])
        data[attr] = np.full(n_rows, int(val), dtype=dtype)
      return Dataset(data, self.domain)

    total = max(1, int(rows or self.total))  # pyrefly: ignore[bad-argument-type]
    domain = self.domain
    jtree, elimination_order = junction_tree.make_junction_tree(
        domain, [tuple(cl) for cl in self.cliques]
    )

    # Use maximal cliques from the junction tree for conditioning
    # decisions (not the original measurement cliques).  The junction
    # tree merges overlapping cliques into super-cliques that capture
    # the full dependency structure of the model.
    cliques = [set(cl) for cl in jtree.nodes]

    potentials = self.potentials.expand(list(jtree.nodes))
    marginals = marginal_oracles.message_passing_shafer_shenoy(
        potentials, self.total, jtree=jtree, constraints=self.constraints
    )

    def synthetic_col(counts, total):
      """Generates a synthetic column by sampling or rounding based on counts and total."""
      counts = np.asarray(counts, dtype=np.float64)
      dtype = np.min_scalar_type(counts.size)
      options = np.arange(counts.size, dtype=dtype)
      if total == 0:
        return np.array([], dtype=int)
      if method == "sample":
        probas = counts / counts.sum()
        return np.random.choice(options, total, True, probas)
      counts = counts * (total / counts.sum())
      frac, integ = np.modf(counts)
      integ = integ.astype(int)
      extra = total - integ.sum()
      if extra > 0:
        idx = np.random.choice(options, extra, False, frac / frac.sum())
        integ[idx] += 1
      vals = np.repeat(options, integ)
      np.random.shuffle(vals)
      return vals

    data = {}
    order = elimination_order[::-1]
    if not order:
      return Dataset(data, domain)
    col = order[0]
    marg = marginals.project((col,)).datavector(flatten=False)
    data[col] = synthetic_col(marg, total)
    used = {col}

    for col in order[1:]:
      relevant = [cl for cl in cliques if col in cl]
      relevant = used.intersection(set().union(*relevant))
      proj = tuple(relevant)
      used.add(col)

      if len(proj) >= 1:
        current_proj_data = np.stack(tuple(data[col] for col in proj), -1)

        marg = np.asarray(
            marginals.project(proj + (col,)).datavector(flatten=False)
        )

        marg_parents = marg.sum(axis=-1, keepdims=True)
        cond_probs = np.divide(
            marg,
            marg_parents,
            out=np.zeros_like(marg),
            where=marg_parents != 0,
        )
        cond_cdfs = cond_probs.cumsum(axis=-1)

        uniques, inverse, counts = np.unique(
            current_proj_data,
            axis=0,
            return_inverse=True,
            return_counts=True,
        )

        perm = np.argsort(inverse, kind="stable")
        if method == "sample":
          u = np.random.rand(total)
        else:
          group_starts = np.zeros(len(counts), dtype=int)
          np.cumsum(counts[:-1], out=group_starts[1:])

          # Shuffle within each parent group to break
          # spurious correlations with previously generated
          # columns that shared the same parent set.
          for gi in range(len(counts)):
            s = group_starts[gi]
            e = s + counts[gi]
            np.random.shuffle(perm[s:e])

          inverse_sorted = inverse[perm]
          sorted_indices = np.arange(total)

          ranks_sorted = sorted_indices - group_starts[inverse_sorted]

          ranks = np.empty(total, dtype=int)
          ranks[perm] = ranks_sorted

          noise = np.random.rand(total)
          u = (ranks + noise) / counts[inverse]

        indices = tuple(uniques.T)
        unique_cdfs = cond_cdfs[indices]

        choices = np.empty(total, dtype=np.min_scalar_type(self.domain[col]))
        domain_size = self.domain[col]

        u_sorted = u[perm]

        start = 0
        for i, count in enumerate(counts):
          end = start + count
          cdf = unique_cdfs[i]
          indices_chunk = np.searchsorted(
              cdf, u_sorted[start:end], side="right"
          )
          if len(indices_chunk) > 0:
            np.minimum(indices_chunk, domain_size - 1, out=indices_chunk)
            choices[perm[start:end]] = indices_chunk
          start = end

        data[col] = choices

      else:
        marg = marginals.project((col,)).datavector(flatten=False)
        data[col] = synthetic_col(marg, total)

    return Dataset(data, domain)

  @property
  def domain(self) -> Domain:
    """Returns the Domain object associated with this graphical model."""
    return self.potentials.domain

  @property
  def cliques(self) -> Sequence[Clique]:
    """Returns the list of cliques the model's potentials are defined over."""
    return self.potentials.cliques
