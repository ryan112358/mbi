"""Defines the MarkovRandomField class representing learned graphical models.

This module provides the `MarkovRandomField` class, which encapsulates the
results of learning a graphical model. It stores the learned potentials,
the resulting marginal distributions, and the associated total count (e.g.,
number of records). It also offers methods for querying marginals and
generating synthetic data.
"""
from collections.abc import Sequence
import chex
import numpy as np

from . import junction_tree, marginal_oracles
from .clique_vector import CliqueVector
from .dataset import Dataset
from .factor import Factor


@chex.dataclass(frozen=True, kw_only=False)
class MarkovRandomField:
    """Represents a learned graphical model.

    This class encapsulates the components of a Markov Random Field that has been
    learned from data. It stores the learned potentials, the resulting marginal
    distributions over specified cliques, and the resulting total count (e.g., number of
    records or equivalent sample size) associated with the model.

    Attributes:
        potentials (CliqueVector): A `CliqueVector` containing the learned
            potential functions for the cliques in the model.
        marginals (CliqueVector): A `CliqueVector` containing the marginal
            distributions for a set of cliques, derived from the potentials.
        total (chex.Numeric): The total count or effective sample size
            represented by the model. This is often used for scaling or
            interpreting the marginals.
    """
    potentials: CliqueVector
    marginals: CliqueVector
    total: chex.Numeric = 1

    def project(self, attrs: str | Sequence[str]) -> Factor:
        if isinstance(attrs, str):
            attrs = (attrs,)
        attrs = tuple(attrs)
        if self.marginals.supports(attrs):
            return self.marginals.project(attrs)
        return marginal_oracles.variable_elimination(
            self.potentials, attrs, self.total
        )

    def supports(self, attrs: str | Sequence[str]) -> bool:
        return self.marginals.domain.supports(attrs)

    def synthetic_data(self, rows: int | None = None, method: str = "round"):
        """Generates synthetic data based on the learned model's marginals."""
        total = max(1, int(rows or self.total))
        domain = self.domain
        cols = domain.attrs
        col_to_idx = {col: i for i, col in enumerate(cols)}
        data = np.zeros((total, len(cols)), dtype=int)
        cliques = [set(cl) for cl in self.cliques]
        jtree, elimination_order = junction_tree.make_junction_tree(domain, cliques)

        # Optimization Phase 1: Pre-compute marginals on maximal cliques
        # This avoids repeated JAX compilation/inference in variable_elimination
        max_cliques = junction_tree.maximal_cliques(jtree)
        expanded_potentials = self.potentials.expand(max_cliques)
        beliefs = marginal_oracles.message_passing_stable(expanded_potentials, self.total)

        def synthetic_col(counts, total):
            """Generates a synthetic column by sampling or rounding based on counts and total."""
            if total == 0:
                return np.array([], dtype=int)
            if method == "sample":
                probas = counts / counts.sum()
                return np.random.choice(counts.size, total, True, probas)
            counts *= total / counts.sum()
            frac, integ = np.modf(counts)
            integ = integ.astype(int)
            extra = total - integ.sum()
            if extra > 0:
                idx = np.random.choice(counts.size, extra, False, frac / frac.sum())
                integ[idx] += 1
            vals = np.repeat(np.arange(counts.size), integ)
            np.random.shuffle(vals)
            return vals

        order = elimination_order[::-1]
        col = order[0]
        col_idx = col_to_idx[col]

        marg = beliefs.project((col,)).datavector(flatten=False)
        marg = np.array(marg)
        data[:, col_idx] = synthetic_col(marg, total)
        used = {col}

        for col in order[1:]:
            relevant = [cl for cl in cliques if col in cl]
            relevant = used.intersection(set().union(*relevant))
            proj = tuple(relevant)
            used.add(col)
            col_idx = col_to_idx[col]

            # Use beliefs to get marginal P(proj, col)
            marg = beliefs.project(proj + (col,)).datavector(flatten=False)
            marg = np.array(marg)

            if len(proj) >= 1:
                proj_idxs = [col_to_idx[c] for c in proj]
                current_proj_data = data[:, proj_idxs]

                unique_rows, inverse, counts = np.unique(
                    current_proj_data, axis=0, return_inverse=True, return_counts=True
                )

                # Fetch marginals for all unique configurations
                indexer = tuple(unique_rows[:, i] for i in range(unique_rows.shape[1]))
                W = marg[indexer] # Shape (G, col_dim)

                row_sums = W.sum(axis=1, keepdims=True)
                probas = np.divide(W, row_sums, out=np.zeros_like(W, dtype=float), where=row_sums > 0)

                if method == "sample":
                    # Fallback to loop for sample method
                    for i in range(len(unique_rows)):
                         mask = (inverse == i)
                         count = counts[i]
                         if count > 0:
                             p = probas[i]
                             if p.sum() > 0:
                                 p = p / p.sum()
                                 vals = np.random.choice(p.size, count, True, p)
                                 data[mask, col_idx] = vals
                else:
                    # Vectorized round
                    target_float = probas * counts[:, None]
                    target_integ = target_float.astype(int)
                    frac = target_float - target_integ
                    extra = counts - target_integ.sum(axis=1) # (G,)

                    # Distribute extra
                    scores = frac + np.random.uniform(0, 1e-6, size=frac.shape)
                    max_extra = extra.max()
                    sorted_indices = np.argsort(-scores, axis=1) # Descending

                    for k in range(max_extra):
                        mask = extra > k
                        rows = np.where(mask)[0]
                        cols = sorted_indices[rows, k]
                        target_integ[rows, cols] += 1

                    # Construct result column
                    # Shuffle rows within groups using lexsort
                    perm = np.lexsort((np.random.rand(len(inverse)), inverse))

                    repeats = target_integ.flatten()
                    values = np.tile(np.arange(W.shape[1]), (W.shape[0], 1)).flatten()

                    generated_col_ordered = np.repeat(values, repeats)

                    data[perm, col_idx] = generated_col_ordered

            else:
                data[:, col_idx] = synthetic_col(marg, total)

        return Dataset(data, domain)


    @property
    def domain(self):
        """Returns the Domain object associated with this graphical model."""
        return self.potentials.domain

    @property
    def cliques(self):
        """Returns the list of cliques the model's potentials are defined over."""
        return self.potentials.cliques
