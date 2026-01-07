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
        return Factor(data.domain, data.datavector(flatten=False))

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
        multi_index = tuple(self.df[a].values for a in self.domain.attrs)
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

            # Validation
            if map_array.ndim != 1:
                raise ValueError(f"Mapping for {attr} must be 1D array")
            if map_array.shape[0] != self.domain[attr]:
                raise ValueError(f"Mapping size {map_array.shape[0]} does not match domain size {self.domain[attr]} for attribute {attr}")
            if not np.issubdtype(map_array.dtype, np.integer):
                raise ValueError(f"Mapping for {attr} must be integers")
            if np.any(map_array < 0):
                raise ValueError(f"Mapping for {attr} must be non-negative")

            # Update data
            # Use the mapping to transform the column
            original_col = self._data[attr]
            new_col = map_array[original_col]
            new_data[attr] = new_col

            # Update domain
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

            # Validation (same as compress)
            if map_array.ndim != 1:
                raise ValueError(f"Mapping for {attr} must be 1D array")
            # For decompress, map_array length is the target domain size (original domain size)
            # The current domain size should match the range of map_array (roughly)
            # But strictly, we are restoring TO the domain implied by len(map_array).

            if not np.issubdtype(map_array.dtype, np.integer):
                raise ValueError(f"Mapping for {attr} must be integers")
            if np.any(map_array < 0):
                raise ValueError(f"Mapping for {attr} must be non-negative")

            # Efficient Inversion using argsort
            # Sort the mapping to group indices by their target value
            permutation = np.argsort(map_array)
            sorted_map = map_array[permutation]

            # Count occurrences of each target value
            compressed_domain_size = int(np.max(map_array) + 1)
            counts = np.bincount(sorted_map, minlength=compressed_domain_size)

            # Calculate starting indices for each group in the sorted array
            starts = np.zeros(compressed_domain_size + 1, dtype=int)
            starts[1:] = np.cumsum(counts)
            starts = starts[:-1] # Remove the last element which is total sum

            # Transform the data column
            current_col = self._data[attr]

            # For each value in current_col, we need to pick a random index from its group
            # We generate a random offset for each element
            # offset[i] ~ Uniform(0, counts[current_col[i]])

            # Check for invalid values in data (values that have no preimage in mapping)
            # If counts[val] == 0, it's an error if that val appears in data
            col_counts = counts[current_col]
            if np.any(col_counts == 0):
                 raise ValueError(f"Data contains values for {attr} that have no preimage in the mapping.")

            random_offsets = np.floor(np.random.rand(self._n) * col_counts).astype(int)

            # Calculate indices into the permutation array
            lookup_indices = starts[current_col] + random_offsets

            # Retrieve original values
            new_col = permutation[lookup_indices]
            new_data[attr] = new_col

            # Update domain to original size
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
            raise ValueError(
                "Number of columns of data must equal the number of attributes in the domain."
            )
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

    def datavector(self, flatten: bool = True) -> jax.Array:
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
        weights = (
            self.weights
            if self.weights is None
            else jax.lax.with_sharding_constraint(self.weights, sharding)
        )
        return JaxDataset(data, self.domain, weights)
