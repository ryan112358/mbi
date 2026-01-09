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
from typing import Any

import attr
import jax
import jax.numpy as jnp
import math
import numpy as np
from numpy.typing import ArrayLike, NDArray

from .domain import Domain
from .factor import Factor
import warnings


def _validate_column(data: np.ndarray, size: int):
    if data.ndim != 1:
        raise ValueError(f"Expected column data to be 1D, found shape {data.shape}")
    if not np.issubdtype(data.dtype, np.integer):
        raise ValueError(f"Expected integer data, got {data.dtype}")
    if not np.all((data >= 0) & (data < size)):
        raise ValueError(f"Expected data in range [0, {size})")


def _validate_data(data: dict[str, np.ndarray], domain: Domain):
    if set(data.keys()) != set(domain.attrs):
        raise ValueError("Keys in data dictionary must match domain attributes")
    n = None
    for col in data:
        _validate_column(data[col], domain[col])
        if n is None:
            n = data[col].shape[0]
        if n != data[col].shape[0]:
            raise ValueError("Expected data to have same size for each record.")


def _validate_mapping(map_array: np.ndarray, attr: str):
    if map_array.ndim != 1:
        raise ValueError(f"Mapping for {attr} must be 1D array")
    if not np.issubdtype(map_array.dtype, np.integer):
        raise ValueError(f"Mapping for {attr} must be integers")
    if np.any(map_array < 0):
        raise ValueError(f"Mapping for {attr} must be non-negative")


class Dataset:
    def __init__(
        self,
        data: ArrayLike | dict[str, ArrayLike],
        domain: Domain,
        weights: np.ndarray | None = None,
    ):
        """create a Dataset object

        :param data: a numpy array (n x d) or a dictionary of 1d arrays (length n), keyed by attribute.
        :param domain: a domain object
        :param weight: weight for each row
        """

        if isinstance(data, np.ndarray):
            if data.shape[1] != len(domain.attrs):
                raise ValueError("Shape of data does not match shape of domain")
            n = data.shape[0]
            data = {attr: data[:, i] for i, attr in enumerate(domain.attrs)}

        elif isinstance(data, dict):
            if len(data) > 0:
                n = list(data.values())[0].shape[0]
            else:
                n = None

        elif hasattr(data, "values"):  # Pandas DataFrame
            warnings.warn(
                "Pandas dataframe inputs are deprecated, please pass in a dictionary of numpy arrays instead."
            )
            n = data.shape[0]
            data = {attr: data[attr].values for attr in domain.attrs}

        else:
            raise ValueError(f"Unrecognized data type {type(data)}")

        _validate_data(data, domain)

        if n == None:
            if weights is None:
                raise ValueError(
                    "Weights must be provided if data is empty (cannot infer N)"
                )
            n = weights.size

        if weights is None:
            weights = np.ones(n)

        assert n == weights.size

        self.domain = domain
        self._data = data
        self.weights = weights
        self._n = n

    def to_dict(self) -> dict[str, np.ndarray]:
        return self._data

    @property
    def df(self):
        import pandas

        return pandas.DataFrame(self._data)

    @staticmethod
    def synthetic(domain: Domain, N: int) -> Dataset:
        """Generate synthetic data conforming to the given domain

        :param domain: The domain object
        :param N: the number of individuals
        """
        arr = [np.random.randint(low=0, high=n, size=N) for n in domain.shape]
        values = np.array(arr).T
        return Dataset(values, domain)

    @staticmethod
    def load(path: str, domain: str | Domain) -> Dataset:
        """Load data into a dataset object

        :param path: path to csv file
        :param domain: path to json file encoding the domain information
        """
        if isinstance(domain, str):
            with open(domain, "r", encoding="utf-8") as f:
                config = json.load(f)
            domain_obj = Domain(config.keys(), config.values())
        else:
            domain_obj = domain

        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            header_map = {name: i for i, name in enumerate(header)}

            if not set(domain_obj.attrs) <= set(header):
                raise ValueError("data must contain domain attributes")

            indices = [header_map[attr] for attr in domain_obj.attrs]

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

        return Dataset(np.array(data), domain_obj)

    def project(self, cols: int | str | Sequence[str] | Sequence[int]) -> Factor:
        """project dataset onto a subset of columns"""
        if isinstance(cols, (str, int)):
            cols = [cols]

        domain = self.domain.project(cols)
        data = {col: self._data[col] for col in domain.attrs}
        data = Dataset(data, domain, self.weights)
        return Factor(data.domain, jnp.asarray(data.datavector(flatten=False)))

    def supports(self, cols: str | Sequence[str]) -> bool:
        return self.domain.supports(cols)

    def drop(self, cols: Sequence[str]) -> Factor:
        """Returns a new Dataset with the specified columns removed."""
        proj = [c for c in self.domain if c not in cols]
        return self.project(proj)

    @property
    def records(self) -> int:
        """Returns the number of records (rows) in the dataset."""
        return self._n

    def datavector(self, flatten: bool = True) -> NDArray:
        """return the database in vector-of-counts form"""
        dims = self.domain.shape
        if len(dims) == 0:
            result = self.weights.sum()
            return np.array([result]) if flatten else result
        multi_index = tuple(self._data[a] for a in self.domain.attrs)
        linear_indices = np.ravel_multi_index(multi_index, dims, order='C')
        counts = np.bincount(linear_indices, minlength=math.prod(dims), weights=self.weights)
        return counts if flatten else counts.reshape(dims)

    def compress(self, mapping: dict[str, np.ndarray]) -> Dataset:
        """
        Compresses the dataset by mapping domain elements to a smaller domain.

        Args:
            mapping: A dictionary where keys are attribute names and values are 1D arrays.
                     mapping[attr][i] gives the new value for original value i.

        Returns:
            A new Dataset with transformed values and updated domain.
        """
        new_data = dict(self._data)
        new_domain_config = self.domain.config.copy()

        for attr, map_array in mapping.items():
            if attr not in self.domain:
                continue

            _validate_mapping(map_array, attr)
            if map_array.shape[0] != self.domain[attr]:
                raise ValueError(f"Mapping size {map_array.shape[0]} does not match domain size {self.domain[attr]} for attribute {attr}")

            new_col = map_array[self._data[attr]]
            new_data[attr] = new_col.astype(np.min_scalar_type(np.max(map_array)))

            new_size = int(np.max(map_array) + 1)
            new_domain_config[attr] = new_size

        new_domain = Domain(new_domain_config.keys(), new_domain_config.values())
        return Dataset(new_data, new_domain, self.weights)

    def decompress(self, mapping: dict[str, np.ndarray]) -> Dataset:
        """
        Decompresses the dataset by reversing the mapping.
        Since the mapping is surjective, the reverse mapping is one-to-many.
        We sample uniformly from the possible original values.

        Args:
            mapping: The same mapping dictionary used for compression.

        Returns:
            A new Dataset with restored domain size and sampled values.
        """
        new_data = dict(self._data)
        new_domain_config = self.domain.config.copy()

        for attr, map_array in mapping.items():
            if attr not in self.domain:
                continue

            _validate_mapping(map_array, attr)

            permutation = np.argsort(map_array)
            sorted_map = map_array[permutation]

            compressed_domain_size = int(np.max(map_array) + 1)
            counts = np.bincount(sorted_map, minlength=compressed_domain_size)

            starts = np.zeros(compressed_domain_size + 1, dtype=int)
            starts[1:] = np.cumsum(counts)
            starts = starts[:-1]

            current_col = self._data[attr]

            col_counts = counts[current_col]
            if np.any(col_counts == 0):
                 raise ValueError(f"Data contains values for {attr} that have no preimage in the mapping.")

            random_offsets = np.floor(np.random.rand(len(current_col)) * col_counts).astype(int)

            lookup_indices = starts[current_col] + random_offsets

            new_col = permutation[lookup_indices]
            new_data[attr] = new_col.astype(np.min_scalar_type(len(map_array) - 1))

            new_domain_config[attr] = len(map_array)

        new_domain = Domain(new_domain_config.keys(), new_domain_config.values())
        return Dataset(new_data, new_domain, self.weights)


@functools.partial(
    jax.tree_util.register_dataclass,
    meta_fields=["domain"],
    data_fields=["data", "weights"],
)
@attr.dataclass(frozen=True)
class JaxDataset:
    """Represents a discrete dataset backed by JAX Arrays.

    Attributes:
        data (dict[str, jax.Array]): A dictionary of 1D JAX arrays where keys are attributes
            and values are columns of data.
        domain (Domain): A `Domain` object describing the attributes and their
            possible discrete values.
        weights (jax.Array | None): An optional 1D JAX array representing the
            weight for each record in the dataset. If None, all records are
            assumed to have a weight of 1.
    """

    data: dict[str, jax.Array]
    domain: Domain
    weights: jax.Array | None = None

    @staticmethod
    def synthetic(domain: Domain, records: int) -> JaxDataset:
        """Generate synthetic data conforming to the given domain

        :param domain: The domain object
        :param records: the number of individuals
        """
        data = {}
        for attr, n in zip(domain.attrs, domain.shape):
             data[attr] = jnp.array(np.random.randint(low=0, high=n, size=records))

        return JaxDataset(data, domain)

    def project(self, cols: str | Sequence[str]) -> Factor:
        """project dataset onto a subset of columns"""
        if isinstance(cols, (str, int)):
            cols = [cols]

        domain = self.domain.project(cols)

        dims = domain.shape
        if len(dims) == 0:
             w = self.weights if self.weights is not None else jnp.ones(self.records)
             result = w.sum()
             return Factor(domain, jnp.array([result]))

        multi_index = tuple(self.data[a] for a in domain.attrs)
        linear_indices = jnp.ravel_multi_index(multi_index, dims, mode='wrap', order='C')

        length = math.prod(dims)

        counts = jnp.bincount(linear_indices, weights=self.weights, minlength=length)

        return Factor(domain, counts.reshape(dims))

    def supports(self, cols: str | Sequence[str]) -> bool:
        return self.domain.supports(cols)

    @property
    def records(self) -> int:
        """Returns the number of records (rows) in the dataset."""
        if not self.data:
             raise ValueError("Dataset is empty (no columns).")
        return list(self.data.values())[0].shape[0]

    def apply_sharding(self, mesh: jax.sharding.Mesh) -> JaxDataset:
        pspec = jax.sharding.PartitionSpec(mesh.axis_names)
        sharding = jax.sharding.NamedSharding(mesh, pspec)

        new_data = {}
        for k, v in self.data.items():
             new_data[k] = jax.lax.with_sharding_constraint(v, sharding)

        weights = (
            self.weights
            if self.weights is None
            else jax.lax.with_sharding_constraint(self.weights, sharding)
        )
        return JaxDataset(new_data, self.domain, weights)
