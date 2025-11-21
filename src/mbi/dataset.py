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
from typing import Union, Optional

import attr
import jax
import jax.numpy as jnp
import numpy as np

from .domain import Domain
from .factor import Factor


class Dataset:
    """Represents a discrete dataset backed by a NumPy array.

    Attributes:
        data (np.ndarray): A 2D numpy array where rows represent records and columns
            represent attributes. The data should be integral.
        domain (Domain): A `Domain` object describing the attributes and their
            possible discrete values.
        weights (np.ndarray | None): An optional 1D numpy array representing the
            weight for each record in the dataset. If None, all records are
            assumed to have a weight of 1.
    """
    def __init__(self, data, domain, weights=None):
        """create a Dataset object

        :param data: a numpy array, columns aligned with domain.attrs
        :param domain: a domain object
        :param weight: weight for each row
        """
        self.domain = domain
        self.data = np.array(data)

        if self.data.ndim != 2:
             raise ValueError(f"Data must be 2d array, got {self.data.shape}")

        assert self.data.shape[1] == len(domain), "data columns must match domain attributes"
        assert weights is None or self.data.shape[0] == weights.size
        self.weights = weights

    @staticmethod
    def synthetic(domain, N): # pylint: disable=invalid-name
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
        with open(domain, 'r', encoding='utf-8') as f_domain:
            config = json.load(f_domain)
        domain = Domain.fromdict(config)

        with open(path, 'r', encoding='utf-8') as f_csv:
            reader = csv.reader(f_csv)
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
        if isinstance(cols, (str, int)):
            cols = [cols]
        indices = self.domain.axes(cols)
        data = self.data[:, indices]
        domain = self.domain.project(cols)
        data = Dataset(data, domain, self.weights)
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
        return self.data.shape[0]

    def datavector(self, flatten=True):
        """return the database in vector-of-counts form"""
        bins = [range(n + 1) for n in self.domain.shape]
        ans = np.histogramdd(self.data, bins, weights=self.weights)[0]
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
             raise ValueError(f'Data must be integral, got {self.data.dtype}.')

        if self.data.ndim != 2:
            raise ValueError(f'Data must be 2d aray, got {self.data.shape}')
        if self.data.shape[1] != len(self.domain):
            raise ValueError('Number of columns of data must equal the number of attributes in the domain.')
        # This will not work in a jitted context, but not sure if this will be called from one normally.
        for i, ax in enumerate(self.domain):
            if self.data[:, i].min() < 0:
                raise ValueError('Data must be non-negative.')
            if self.data[:, i].max() >= self.domain[ax]:
                raise ValueError('Data must be within the bounds of the domain.')

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
        if isinstance(cols, (str, int)):
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
