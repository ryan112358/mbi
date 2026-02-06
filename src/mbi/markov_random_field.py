"""Defines the MarkovRandomField class representing learned graphical models.

This module provides the `MarkovRandomField` class, which encapsulates the
results of learning a graphical model. It stores the learned potentials,
the resulting marginal distributions, and the associated total count (e.g.,
number of records). It also offers methods for querying marginals and
generating synthetic data.
"""

from collections.abc import Sequence
import chex
import math
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
            indices = np.argsort(-remainder, kind="stable")
            floor_counts[indices[:diff]] += 1
        return floor_counts

    # Handle 2D case (axis=1)
    total = np.round(counts.sum(axis=1)).astype(int)
    floor_counts = np.floor(counts).astype(int)
    remainder = counts - floor_counts
    diff = total - floor_counts.sum(axis=1)

    indices = np.argsort(-remainder, axis=1, kind="stable")
    rows = np.arange(counts.shape[0])
    cols = np.arange(counts.shape[1])
    mask = cols < diff[:, None]

    to_add = np.zeros_like(floor_counts)
    np.put_along_axis(to_add, indices, mask.astype(int), axis=1)

    return floor_counts + to_add


def _controlled_round(counts):
    """Rounds counts to integers using stochastic rounding (controlled rounding)."""
    # Handle 1D case
    if counts.ndim == 1:
        total = int(round(counts.sum()))
        frac, integ = np.modf(counts)
        integ = integ.astype(int)
        extra = total - integ.sum()
        if extra > 0:
            # We must normalize frac to use as probabilities
            probs = frac / frac.sum()
            idx = np.random.choice(counts.size, extra, replace=False, p=probs)
            integ[idx] += 1
        return integ

    # Handle 2D case (axis=1)
    total = np.round(counts.sum(axis=1)).astype(int)
    frac, integ = np.modf(counts)
    integ = integ.astype(int)
    extra = total - integ.sum(axis=1)

    # We need to perform the random choice per row.
    # Vectorizing np.random.choice with variable p per row is tricky.
    # However, since we are doing controlled rounding, we can implement it
    # by adding a random number to the fractional part and thresholding,
    # or by using Gumbel-max trick for 'replace=False' sampling?
    # Actually, simpler: for each row, we need to pick 'extra' indices based on 'frac'.
    # A vectorized way to do "weighted sampling without replacement" is
    # to sort by (log(p) + Gumbel_noise).
    # Here p = frac.

    # Add small epsilon to avoid log(0)
    noise = np.random.rand(*counts.shape)
    # The reference implementation uses `frac` as weights.
    # Sampling k items with prob p_i is not exactly trivial to vectorize perfectly
    # if we want to match np.random.choice(..., replace=False, p=p) exactly.
    # However, "systematic sampling" or sorting by (frac + noise) might be close enough.

    # Let's verify the reference implementation logic:
    # idx = np.random.choice(counts.size, extra, False, frac / frac.sum())

    # We can use the same logic as _deterministic_round but with randomized keys
    # Instead of sorting by -remainder, we sort by something else.
    # We want to select indices with probability proportional to remainder.
    # Efroymson's method / weighted sampling without replacement.

    # A simple approximation for controlled rounding:
    # Sort by (remainder + random_noise) ? No.
    # Sort by (remainder / random_exponential) ? (E.g. top-k Gumbel)
    # Yes, Gumbel-max trick gives weighted sampling without replacement.
    # keys = log(probs) + Gumbel(0,1)
    # probs = frac / sum(frac)

    # Avoid div by zero
    sum_frac = frac.sum(axis=1, keepdims=True)
    probs = np.divide(frac, sum_frac, out=np.zeros_like(frac), where=sum_frac!=0)

    # Gumbel noise: -log(-log(u))
    u = np.random.rand(*counts.shape)
    gumbel = -np.log(-np.log(u))

    # keys = log(probs) + gumbel = log(probs) - log(-log(u)) = -log(-log(u)/probs)
    # We want to select the top 'extra' indices.
    # Note: if prob is 0, key should be -inf.

    with np.errstate(divide='ignore'):
        keys = np.log(probs) + gumbel

    # Where extra is 0, it doesn't matter.
    # Sort indices by keys descending.
    indices = np.argsort(-keys, axis=1, kind="stable")

    rows = np.arange(counts.shape[0])
    cols = np.arange(counts.shape[1])
    mask = cols < extra[:, None]

    to_add = np.zeros_like(integ)
    np.put_along_axis(to_add, indices, mask.astype(int), axis=1)

    return integ + to_add


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
        return marginal_oracles.variable_elimination(self.potentials, attrs, self.total)

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
            integ = _controlled_round(counts)
            vals = np.repeat(options, integ)
            return vals

        data = {}
        order = elimination_order[::-1]
        col = order[0]
        marg = self.project((col,)).datavector(flatten=False)
        data[col] = synthetic_col(marg, total)
        used = {col}

        for col in order[1:]:
            dtype = np.min_scalar_type(self.domain[col] - 1)
            relevant = [cl for cl in cliques if col in cl]
            relevant = used.intersection(set().union(*relevant))
            proj = tuple(relevant)
            used.add(col)

            if len(proj) >= 1:
                # Below we sample data for "col", conditioned on "relevant"
                # For each unique setting of values for the relevant columns,
                # we compute the conditional distribution of "col" and use
                # sampling or rounding to materialize the values.
                # We implement this as a single vectorized operation for efficiency,
                # also avoiding the usage of np.unique which was the bottleneck
                # in an earlier version of this function.
                current_proj_data = np.stack(tuple(data[col] for col in proj), -1)

                marg = self.project(proj + (col,)).datavector(flatten=False)
                marg_parents = marg.sum(axis=-1, keepdims=True)
                cond_probs = np.divide(
                    marg, marg_parents, out=np.zeros_like(marg), where=marg_parents != 0
                )

                if method == "sample":
                    cond_cdfs = cond_probs.cumsum(axis=-1)
                    indices = tuple(current_proj_data.T)
                    rows_cdfs = cond_cdfs[indices]
                    u = np.random.rand(total, 1)
                    choices = (rows_cdfs > u).argmax(axis=1)
                    data[col] = choices.astype(dtype)
                else:
                    # Group rows by parent configuration
                    parent_sizes = [self.domain[p] for p in proj]
                    assert math.prod(parent_sizes) < 2**63, "Parent domain size too large."

                    strides = np.cumprod([1] + list(parent_sizes[::-1])[:-1])[::-1]
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

                    counts_matrix = _controlled_round(
                        counts[:, None] * group_cond_probs
                    )
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
