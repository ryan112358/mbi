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
        col = order[0]
        marg = self.project((col,)).datavector(flatten=False)
        data[col] = synthetic_col(marg, total)
        used = {col}

        for col in order[1:]:
            # we only care about relevant columns for generating col
            relevant = [cl for cl in cliques if col in cl]
            relevant = used.intersection(set().union(*relevant))
            proj = tuple(relevant)
            used.add(col)

            if len(proj) >= 1:
                current_proj_data = np.stack(tuple(data[col] for col in proj), -1)

                # Get the joint marginal for proj + col
                # Ensure correct ordering: proj attributes, then col
                # Note: self.project returns factor with attributes in the order requested.
                marg = self.project(proj + (col,)).datavector(flatten=False)

                # Precompute conditional CDFs
                # marg has shape (d_p1, d_p2, ..., d_col)
                # Sum over the last axis (col) to get marginal of parents
                marg_parents = marg.sum(axis=-1, keepdims=True)

                # Compute conditional probabilities: P(col | parents)
                # Handle division by zero where marginal of parents is 0
                cond_probs = np.divide(marg, marg_parents, out=np.zeros_like(marg), where=marg_parents!=0)

                # Compute CDFs along the last axis
                cond_cdfs = cond_probs.cumsum(axis=-1)

                # Verify that the last element of CDF is 1 (or close to it)
                # This is naturally true due to normalization.

                # Select CDFs corresponding to the parent configurations of the current rows
                # indices is a tuple of arrays, one for each dimension of proj
                indices = tuple(current_proj_data.T)

                # vectorized lookup of CDFs for each row
                # rows_cdfs has shape (N, d_col)
                rows_cdfs = cond_cdfs[indices]

                # Sample uniformly
                u = np.random.rand(total, 1)

                # Find indices where CDF > u
                # argmax returns the first index where condition is true
                # This corresponds to inverse CDF sampling
                choices = (rows_cdfs > u).argmax(axis=1)

                data[col] = choices.astype(np.min_scalar_type(self.domain[col]))
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
