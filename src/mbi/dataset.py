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
            if not set(data.keys()) == set(domain.attrs):
                raise ValueError("Keys in data dictionary must match domain attributes")

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
        else:
            data_arr = np.array(data)
            if data_arr.ndim != 2:
                 raise ValueError(f"Data must be 2d array or dictionary, got {data_arr.shape}")

            if data_arr.shape[1] != len(domain):
                raise ValueError("data columns must match domain attributes")

            n = data_arr.shape[0]
            self._data = {attr: data_arr[:, i] for i, attr in enumerate(domain.attrs)}

        if n is None:
             if weights is None:
                 raise ValueError("Weights must be provided if data is empty (cannot infer N)")
             n = weights.size

        if weights is None:
            weights = np.ones(n)

        assert n == weights.size
        self.weights = weights
        self._n = n

    @property
    def data(self):
        """Returns the data as a 2D numpy array for backward compatibility."""
        if not self._data:
            return np.zeros((self.records, 0))
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
            ans = np.histogramdd(sample, bins, weights=self.weights)[0]
        else:
            ans = np.array(self.weights.sum())

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
