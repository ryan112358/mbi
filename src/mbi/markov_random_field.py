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


def _deterministic_round(counts):
    """Rounds counts to integers while preserving the sum using largest remainder method."""
    # This handles the case where sum(counts) is not an integer by rounding the sum first.
    total = int(round(counts.sum()))
    
    floor_counts = np.floor(counts).astype(int)
    remainder = counts - floor_counts
    
    current_sum = floor_counts.sum()
    diff = total - current_sum
    
    if diff > 0:
        indices = np.argsort(-remainder, kind='stable')
        floor_counts[indices[:diff]] += 1

    return floor_counts


def _round_matrix(row_sums, col_probs, col_target):
    """
    Rounds a matrix of counts to integers such that:
    1. Row i sums to row_sums[i]
    2. Col j sums to col_target[j] (approximately/exactly via residual carryover)
    3. Entries are close to row_sums[i] * col_probs[i, j]
    """
    total = row_sums.sum()
    empirical_expected = row_sums.dot(col_probs)
    global_adjustment_vector = (col_target - empirical_expected) / total
    
    residual = np.zeros_like(col_target)
    
    for i in range(len(row_sums)):
        n_i = row_sums[i]
        p_i = col_probs[i]
        
        target_i = n_i * (p_i + global_adjustment_vector)
        target_with_residual = target_i + residual
        
        # Project to non-negative simplex to ensure valid counts
        projected_target = np.maximum(0, target_with_residual)
        current_sum = projected_target.sum()
        if current_sum > 0:
            projected_target *= n_i / current_sum
        
        c_i = _deterministic_round(projected_target)
        residual = target_with_residual - c_i
        yield c_i


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
        cliques = [set(cl) for cl in self.cliques]
        jtree, elimination_order = junction_tree.make_junction_tree(domain, cliques)

        def synthetic_col(counts, total):
            """Generates a synthetic column by sampling or rounding based on counts and total."""
            dtype = np.min_scalar_type(counts.size)
            options = np.arange(counts.size, dtype=dtype)
            if total == 0:
                return np.array([], dtype=int)
            if method == "sample":
                probas = counts / counts.sum()
                return np.random.choice(options, total, True, probas)
            
            counts *= total / counts.sum()
            integ = _deterministic_round(counts)
            vals = np.repeat(options, integ)
            return vals

        data = {}
        order = elimination_order[::-1]
        col = order[0]
        marg = self.project((col,)).datavector(flatten=False)
        data[col] = synthetic_col(marg, total)
        used = {col}

        for col in order[1:]:
            relevant = [cl for cl in cliques if col in cl]
            relevant = used.intersection(set().union(*relevant))
            proj = tuple(relevant)
            used.add(col)

            if len(proj) >= 1:
                current_proj_data = np.stack(tuple(data[col] for col in proj), -1)

                marg = self.project(proj + (col,)).datavector(flatten=False)
                marg_parents = marg.sum(axis=-1, keepdims=True)
                cond_probs = np.divide(marg, marg_parents, out=np.zeros_like(marg), where=marg_parents!=0)

                if method == "sample":
                    cond_cdfs = cond_probs.cumsum(axis=-1)
                    indices = tuple(current_proj_data.T)
                    rows_cdfs = cond_cdfs[indices]
                    u = np.random.rand(total, 1)
                    choices = (rows_cdfs > u).argmax(axis=1)
                    data[col] = choices.astype(np.min_scalar_type(self.domain[col]))
                else:
                    target_global = self.project((col,)).datavector(flatten=False)
                    target_global = _deterministic_round(target_global * total / target_global.sum())
                    
                    # Group rows by parent configuration
                    _, inverse, counts = np.unique(current_proj_data, axis=0, return_inverse=True, return_counts=True)
                    perm = np.argsort(inverse, kind='stable')

                    global_error = target_global.astype(float)
                    unique_parents = current_proj_data[perm[np.cumsum(counts) - 1]]

                    parent_indices = tuple(unique_parents.T)
                    group_cond_probs = cond_probs[parent_indices] # (N_groups, domain[col])
                    
                    output_col = np.zeros(total, dtype=int)
                    
                    rounded_rows = _round_matrix(counts, group_cond_probs, target_global)
                    
                    start_idx = 0
                    for c_i in rounded_rows:
                        n_i = c_i.sum()
                        group_vals = np.repeat(np.arange(len(c_i)), c_i)
                        group_indices = perm[start_idx : start_idx + n_i]
                        output_col[group_indices] = group_vals

                        start_idx += n_i
                    
                    data[col] = output_col

            else:
                marg = self.project((col,)).datavector(flatten=False)
                data[col] = synthetic_col(marg, total)

        return Dataset(data, domain)

    @property
    def domain(self):
        """Returns the Domain object associated with this graphical model."""
        return self.potentials.domain

    @property
    def cliques(self):
        """Returns the list of cliques the model's potentials are defined over."""
        return self.potentials.cliques
