"""Defines the MarkovRandomField class representing learned graphical models.

This module provides the `MarkovRandomField` class, which encapsulates the
results of learning a graphical model. It stores the learned potentials,
the resulting marginal distributions, and the associated total count (e.g.,
number of records). It also offers methods for querying marginals and
generating synthetic data.
"""
from collections.abc import Sequence
from typing import Union
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

    def project(self, attrs: Union[str, Sequence[str]]) -> Factor:
        if isinstance(attrs, str):
            attrs = (attrs,)
        attrs = tuple(attrs)
        if self.marginals.supports(attrs):
            return self.marginals.project(attrs)
        return marginal_oracles.variable_elimination(
            self.potentials, attrs, self.total
        )

    def supports(self, attrs: Union[str, Sequence[str]]) -> bool:
        return self.marginals.domain.supports(attrs)

    def synthetic_data(self, rows: Union[int, None] = None, method: str = "round"):
        """Generates synthetic data based on the learned model's marginals."""
        total = max(1, int(rows or self.total))
        domain = self.domain
        cols = domain.attrs
        col_to_idx = {col: i for i, col in enumerate(cols)}
        data = np.zeros((total, len(cols)), dtype=int)
        cliques = [set(clique) for clique in self.cliques]
        _, elimination_order = junction_tree.make_junction_tree(domain, cliques)

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
        marg = self.project((col,)).datavector(flatten=False)
        data[:, col_idx] = synthetic_col(marg, total)
        used = {col}

        for col in order[1:]:
            relevant = [clique for clique in cliques if col in clique]
            relevant = used.intersection(set().union(*relevant))
            proj = tuple(relevant)
            used.add(col)
            col_idx = col_to_idx[col]

            # Will this work without having the maximal cliques of the junction tree?
            marg = self.project(proj + (col,)).datavector(flatten=False)

            if len(proj) >= 1:
                proj_idxs = [col_to_idx[c] for c in proj]
                # Get unique configurations of the projected columns in the current data
                # We only care about the columns in 'proj'
                current_proj_data = data[:, proj_idxs]

                # Find unique rows and the inverse mapping (which row belongs to which unique config)
                unique_rows, inverse = np.unique(current_proj_data, axis=0, return_inverse=True)

                # For each unique configuration, sample the new column
                for i, _ in enumerate(unique_rows):
                    # Identify rows matching this configuration
                    mask = (inverse == i)
                    count = np.sum(mask)

                    if count > 0:
                        # Get the conditional marginal for this configuration
                        # unique_rows[i] corresponds to the values of 'proj'
                        # marg is indexed by (val_proj_1, val_proj_2, ..., val_col)
                        # So marg[tuple(unique_rows[i])] gives the vector for 'col'
                        idx = tuple(unique_rows[i])
                        vals = synthetic_col(marg[idx], count)
                        data[mask, col_idx] = vals
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
