"""Provides the Dataset class for representing and manipulating tabular data.

This module defines the `Dataset` class, which serves as a wrapper around a
numpy array, associating it with a `Domain` object. It allows for
structured representation of data, facilitating operations like projection onto
subsets of attributes and conversion into a data vector format suitable for
various statistical and machine learning tasks.
"""
from __future__ import annotations

import csv
import functools
import json
from collections.abc import Sequence

import attr
import jax
import jax.numpy as jnp
import numpy as np

from .domain import Domain
from .factor import Factor


class Dataset:
    def __init__(self, data, domain, weights=None):
        """create a Dataset object

        :param data: a numpy array (n x d) OR a dictionary of 1d arrays (length n), keyed by attribute
        :param domain: a domain object
        :param weight: weight for each row
        """
        self.domain = domain

        if isinstance(data, dict):
            # Validate keys
            if not set(data.keys()) == set(domain.attrs):
                raise ValueError("Keys in data dictionary must match domain attributes")

            # Validate lengths and convert to numpy arrays
            n = None
            self._data = {}
            for attr in domain.attrs:
                col = np.array(data[attr])
                if col.ndim != 1:
                    raise ValueError(f"Data for attribute {attr} must be 1D array")
                if n is None:
                    n = col.size
                elif col.size != n:
                    raise ValueError(f"All columns must have the same length. Attribute {attr} has {col.size}, expected {n}")
                self._data[attr] = col
            # Handle empty dictionary (0 attributes) case, though usually handled by not entering loop if attrs empty?
            # If attrs is empty, n remains None.
            if n is None:
                 # If dict is empty, we don't know N unless implied by something else?
                 # But typical usage with empty domain and dict input is likely rare or assumes N=0?
                 # However, if passed data={}, we can't infer N.
                 # If domain is empty, the loop over attrs doesn't run.
                 # We should probably default N to 0 or check if user provided some indication?
                 # But wait, earlier I said if domain is empty, we might lose N.
                 # If data={} and domain is empty, we assume N=0 unless weights is provided?
                 # But weights is checked against n.
                 # Let's assume n=0 if attrs is empty and data is empty.
                 # But if data was a 2D array of (N, 0), we get N.
                 # With dict input {}, we don't have N.
                 # For now, let's assume N=0 if no attributes in dict.
                 n = 0
        else:
            # Assume array-like (n x d)
            data_arr = np.array(data)
            if data_arr.ndim != 2:
                 raise ValueError(f"Data must be 2d array or dictionary, got {data_arr.shape}")

            if data_arr.shape[1] != len(domain):
                raise ValueError("data columns must match domain attributes")

            n = data_arr.shape[0]
            self._data = {attr: data_arr[:, i] for i, attr in enumerate(domain.attrs)}

        assert weights is None or n == weights.size
        self.weights = weights
        self._n = n

    @property
    def data(self):
        """Returns the data as a 2D numpy array for backward compatibility."""
        return np.column_stack([self._data[attr] for attr in self.domain.attrs])

    @staticmethod
    def synthetic(domain, N):
        """Generate synthetic data conforming to the given domain

        :param domain: The domain object
        :param N: the number of individuals
        """
        arr = [np.random.randint(low=0, high=n, size=N) for n in domain.shape]
        values = np.array(arr).T
        return Dataset(values, domain)

    @staticmethod
    def load(path, domain):
        """Load data into a dataset object

        :param path: path to csv file
        :param domain: path to json file encoding the domain information
        """
        with open(domain, "r", encoding="utf-8") as f:
            config = json.load(f)
        domain = Domain(config.keys(), config.values())

        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            header_map = {name: i for i, name in enumerate(header)}

            if not set(domain.attrs) <= set(header):
                 raise ValueError("data must contain domain attributes")

            indices = [header_map[attr] for attr in domain.attrs]

            data = []
            for row in reader:
                # Convert to int, handling potential float strings like '1.0'
                try:
                    mapped_row = [int(float(row[i])) for i in indices]
                except ValueError:
                    # Fallback or error if data is not numeric
                    # Assuming domain implies discrete/integer data
                     mapped_row = [int(row[i]) for i in indices]
                data.append(mapped_row)

        return Dataset(np.array(data), domain)

    def project(self, cols):
        """project dataset onto a subset of columns"""
        if type(cols) in [str, int]:
            cols = [cols]

        if not cols:
            # If projecting to empty set, return a factor with empty domain
            # The value should be the total count (N)
            # datavector() handles empty domain correctly now by using self.records
            domain = Domain([], [])
            data = Dataset({}, domain, self.weights)
            # We need to manually set internal n because empty dict doesn't convey it
            # But we passed weights, so init will check n == weights.size
            # If weights is None, we need to ensure n is set correctly.
            # But Dataset.__init__ sets self._n = 0 if data is empty dict.
            # So data.records will be 0, unless we pass weights.
            # If we want to support unweighted case, we might need to be careful.
            # Actually, if we project onto empty set, we want a Factor representing the count.
            # If self.weights is None, the sum is N.
            # If we create a new Dataset({}, domain, weights), records will be 0 or weights.size.
            # If weights is None, records is 0. That's WRONG if original N > 0.
            # So we should pass a dummy array of length N? Or handle it?

            # Better approach:
            # Factor.datavector works by histogramdd.
            # For empty domain, bins=[], sample=(N,0).
            # histogramdd returns sum of weights (or count).
            # So we need a Dataset that knows it has N records.
            # But our Dataset handles empty dict as N=0.

            # Let's fix this by allowing explicit N in init? Or just special case here.

            # If cols is empty, we just want the total count.
            total = self.records if self.weights is None else self.weights.sum()
            return Factor(Domain([], []), total)

        # Handle integer indexing
        if all(isinstance(c, int) for c in cols):
             cols = [self.domain.attrs[c] for c in cols]

        domain = self.domain.project(cols)
        proj_data = {col: self._data[col] for col in domain.attrs}
        data = Dataset(proj_data, domain, self.weights)
        return Factor(data.domain, data.datavector(flatten=False))

    def supports(self, cols: str | Sequence[str]) -> bool:
        return self.domain.supports(cols)

    def drop(self, cols):
        """Returns a new Dataset with the specified columns removed."""
        proj = [c for c in self.domain if c not in cols]
        return self.project(proj)

    @property
    def records(self):
        """Returns the number of records (rows) in the dataset."""
        return self._n

    def datavector(self, flatten=True):
        """return the database in vector-of-counts form"""
        bins = [range(n + 1) for n in self.domain.shape]
        if self._data:
            sample = np.column_stack([self._data[attr] for attr in self.domain.attrs])
        else:
             # Handle empty domain case (should not happen usually given domain shape, but to be safe)
             # If domain has 0 attributes, bins is [], sample should be (N, 0) array.
             # N is records.
             N = self.records
             sample = np.zeros((N, 0))

        ans = np.histogramdd(sample, bins, weights=self.weights)[0]
        return ans.flatten() if flatten else ans


@functools.partial(
    jax.tree_util.register_dataclass,
    meta_fields=["domain"],
    data_fields=["data", "weights"]
)
@attr.dataclass(frozen=True)
class JaxDataset:
    """Represents a discrete dataset backed by a JAX Array.

    Attributes:
        data (jax.Array): A 2D JAX array where rows represent records and columns
            represent attributes. The data should be integral.
        domain (Domain): A `Domain` object describing the attributes and their
            possible discrete values.
        weights (jax.Array | None): An optional 1D JAX array representing the
            weight for each record in the dataset. If None, all records are
            assumed to have a weight of 1.
    """
    data: jax.Array = attr.field(converter=jnp.asarray)
    domain: Domain
    weights: jax.Array | None = None

    def __post_init__(self):
        if not jnp.issubdtype(self.data.dtype, jnp.integer):
             raise ValueError(f"Data must be integral, got {self.data.dtype}.")

        if self.data.ndim != 2:
            raise ValueError(f"Data must be 2d aray, got {self.data.shape}")
        if self.data.shape[1] != len(self.domain):
            raise ValueError("Number of columns of data must equal the number of attributes in the domain.")
        # This will not work in a jitted context, but not sure if this will be called from one normally.
        for i, ax in enumerate(self.domain):
            if self.data[:, i].min() < 0:
                raise ValueError("Data must be non-negative.")
            if self.data[:, i].max() >= self.domain[ax]:
                raise ValueError("Data must be within the bounds of the domain.")

    @staticmethod
    def synthetic(domain: Domain, records: int) -> JaxDataset:
        """Generate synthetic data conforming to the given domain

        :param domain: The domain object
        :param N: the number of individuals
        """
        arr = [np.random.randint(low=0, high=n, size=records) for n in domain.shape]
        data = np.array(arr).T
        return JaxDataset(data, domain)

    def project(self, cols: str | Sequence[str]) -> Factor:
        """project dataset onto a subset of columns"""
        if type(cols) in [str, int]:
            cols = [cols]
        idx = self.domain.axes(cols)
        data = self.data[:, idx]
        domain = self.domain.project(cols)
        data = JaxDataset(data, domain, self.weights)
        return Factor(data.domain, data.datavector(flatten=False))

    def supports(self, cols: str | Sequence[str]) -> bool:
        return self.domain.supports(cols)

    @property
    def records(self) -> int:
        """Returns the number of records (rows) in the dataset."""
        return self.data.shape[0]

    def datavector(self, flatten: bool=True) -> jax.Array:
        """return the database in vector-of-counts form"""
        bins = [range(n + 1) for n in self.domain.shape]
        ans = jnp.histogramdd(self.data, bins, weights=self.weights)[0]
        return ans.flatten() if flatten else ans

    def apply_sharding(self, mesh: jax.sharding.Mesh) -> JaxDataset:
        # Not sure if this function makes sense.  This sharding strategy is what we want,
        # but we will most likely have to read the data in sharded, so I don't
        # know if this will actually be used.
        pspec = jax.sharding.PartitionSpec(mesh.axis_names)
        sharding = jax.sharding.NamedSharding(mesh, pspec)
        data = jax.lax.with_sharding_constraint(self.data, sharding)
        weights = self.weights if self.weights is None else jax.lax.with_sharding_constraint(self.weights, sharding)
        return JaxDataset(data, self.domain, weights)
