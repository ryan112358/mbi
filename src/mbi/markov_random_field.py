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
    # Handle 1D case
    if counts.ndim == 1:
        total = int(round(counts.sum()))
        floor_counts = np.floor(counts).astype(int)
        remainder = counts - floor_counts
        diff = total - floor_counts.sum()
        if diff > 0:
            indices = np.argsort(-remainder, kind='stable')
            floor_counts[indices[:diff]] += 1
        return floor_counts

    # Handle 2D case (axis=1)
    total = np.round(counts.sum(axis=1)).astype(int)
    floor_counts = np.floor(counts).astype(int)
    remainder = counts - floor_counts
    diff = total - floor_counts.sum(axis=1)
    
    indices = np.argsort(-remainder, axis=1, kind='stable')
    rows = np.arange(counts.shape[0])
    cols = np.arange(counts.shape[1])
    mask = cols < diff[:, None]

    to_add = np.zeros_like(floor_counts)
    np.put_along_axis(to_add, indices, mask.astype(int), axis=1)
    
    return floor_counts + to_add


def _round_matrix(row_sums, col_probs, col_target):
    """
    Rounds a matrix of counts to integers such that:
    1. Row i sums to row_sums[i]
    2. Col j sums to col_target[j] (approximately/exactly via cumulative rounding)
    3. Entries are close to row_sums[i] * col_probs[i, j]
    """
    expected = row_sums[:, None] * col_probs
    total = row_sums.sum()
    empirical_col_sums = expected.sum(axis=0)
    adjustment = (col_target - empirical_col_sums) / total
    
    targets = row_sums[:, None] * (col_probs + adjustment)
    targets = np.maximum(0, targets)
    
    current_row_sums = targets.sum(axis=1)
    mask = current_row_sums > 0
    targets[mask] *= (row_sums[mask] / current_row_sums[mask])[:, None]
    
    cum_targets = np.cumsum(targets, axis=0)
    cum_rounded = _deterministic_round(cum_targets)    
    counts = np.diff(cum_rounded, axis=0, prepend=0)
    
    if np.any(counts < 0):
        counts = np.maximum(0, counts)
        
    current_sum = counts.sum()
    diff = int(total - current_sum)
    
    if diff != 0:
        rows_cnt, cols_cnt = counts.shape
        row_idx = 0

        while diff != 0:
            r = row_idx % rows_cnt
            
            if diff > 0:
                c = np.argmax(counts[r])
                counts[r, c] += 1
                diff -= 1
            elif diff < 0:
                c = np.argmax(counts[r])
                if counts[r, c] > 0:
                    counts[r, c] -= 1
                    diff += 1
            
            row_idx += 1
    
    return counts.astype(int)


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
            dtype = np.min_scalar_type(self.domain[col]-1)
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
                    data[col] = choices.astype(dtype)
                else:
                    target_global = self.project((col,)).datavector(flatten=False)
                    target_global = _deterministic_round(target_global * total / target_global.sum())
                    
                    # Group rows by parent configuration
                    parent_domain_sizes = [self.domain[p] for p in proj]
                    parent_domain_size = np.prod(parent_domain_sizes, dtype=np.int64)
                    assert parent_domain_size < 2**63, "Parent domain size too large for linear indexing"
                    
                    strides = np.cumprod([1] + list(parent_domain_sizes[::-1])[:-1])[::-1]
                    flat_parents = np.dot(current_proj_data, strides)
                    
                    # Shuffle within groups to maximize entropy for out-of-model marginals
                    random_perm = np.random.permutation(total)
                    shuffled_parents = flat_parents[random_perm]
                    
                    sort_idx = np.argsort(shuffled_parents)
                    perm = random_perm[sort_idx]
                    sorted_parents = shuffled_parents[sort_idx]
                    
                    mask = np.r_[True, sorted_parents[1:] != sorted_parents[:-1]]
                    
                    unique_flat = sorted_parents[mask]
                    counts = np.diff(np.flatnonzero(np.concatenate((mask, [True]))))
                    
                    # Get conditional probs using flat index
                    cond_probs_flat = cond_probs.reshape(-1, cond_probs.shape[-1])
                    group_cond_probs = cond_probs_flat[unique_flat]
                    
                    counts_matrix = _round_matrix(counts, group_cond_probs, target_global)
                    repeats = counts_matrix.flatten()

                    num_groups = counts_matrix.shape[0]
                    domain_size = counts_matrix.shape[1]
                    data[col] = np.zeros(total, dtype=dtype)
                    values = np.tile(np.arange(domain_size, dtype=dtype), num_groups)
                    data[col][perm] = np.repeat(values, repeats)

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
