"""Provides the Dataset class for representing and manipulating tabular data.

This module defines the ``Dataset`` class, which wraps a dictionary of
1D numpy arrays (one per attribute) and a :class:`Domain` object.  It
supports projection, data vector computation, compression / decompression,
and weighted records.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import math
import warnings
from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike, NDArray

from .domain import Domain
from .factor import Factor


def _validate_column_meta(data: np.ndarray, attr: str):
    """Validate shape and dtype of a single column (no value checks)."""
    if data.ndim != 1:
        raise ValueError(f"Column '{attr}' must be 1D, got shape {data.shape}")
    if not np.issubdtype(data.dtype, np.integer):
        raise ValueError(
            f"Column '{attr}' must have integer dtype, got {data.dtype}"
        )


def _validate_data(data: dict[str, np.ndarray], domain: Domain):
    if set(data.keys()) != set(domain.attrs):
        raise ValueError("Keys in data dictionary must match domain attributes")
    n = None
    for col in data:
        _validate_column_meta(data[col], col)
        if n is None:
            n = data[col].shape[0]
        if n != data[col].shape[0]:
            raise ValueError("All columns must have the same length.")


def _validate_mapping(map_array: np.ndarray, attr: str):
    if map_array.ndim != 1:
        raise ValueError(f"Mapping for {attr} must be 1D array")
    if not np.issubdtype(map_array.dtype, np.integer):
        raise ValueError(f"Mapping for {attr} must be integers")
    if np.any(map_array < 0):
        raise ValueError(f"Mapping for {attr} must be non-negative")


def _group_labels(orig_labels, mapping):
    """Group labels into tuples according to a compression mapping."""
    grouped = [[] for _ in range(int(mapping.max()) + 1)]
    for i, b in enumerate(mapping):
        grouped[b].append(orig_labels[i])
    return tuple(tuple(g) for g in grouped)


def _compress_labels(domain, mapping, new_domain_config, labels=None):
    """Build compressed label tuples from a domain and mapping."""
    if domain.labels is None:
        return None
    lc = dict(domain.labels_config)
    for attr, m in mapping.items():
        if attr not in lc:
            continue
        if labels and attr in labels:
            lc[attr] = labels[attr]
        else:
            lc[attr] = _group_labels(lc[attr], m)
    return tuple(lc[a] for a in new_domain_config)


@dataclasses.dataclass(frozen=True, eq=False)
class Dataset:
    """A discrete tabular dataset backed by a dictionary of 1D numpy arrays.

    Args:
        data: Dictionary mapping attribute names to 1D integer arrays.
        domain: A Domain describing the attributes and their sizes.
        weights: Optional per-row weights (defaults to all ones).
    """

    data: dict[str, ArrayLike]
    domain: Domain
    weights: ArrayLike | None = dataclasses.field(default=None)

    def __post_init__(self):
        object.__setattr__(
            self,
            "data",
            {k: np.asarray(v) for k, v in self.data.items()},
        )
        if self.weights is not None:
            object.__setattr__(self, "weights", np.asarray(self.weights))

        _validate_data(self.data, self.domain)

        if self.data:
            n = next(iter(self.data.values())).shape[0]
        elif self.weights is not None:
            n = self.weights.size
        else:
            raise ValueError(
                "Weights must be provided if data is empty (cannot infer N)"
            )

        if self.weights is None:
            object.__setattr__(self, "weights", np.ones(n))
        elif self.weights.size != n:
            raise ValueError(
                f"Weights length ({self.weights.size}) does not match "
                f"data length ({n})"
            )

    def to_dict(self) -> dict[str, np.ndarray]:
        return self.data

    @staticmethod
    def synthetic(domain: Domain, N: int) -> Dataset:
        """Generate random data conforming to the given domain.

        Args:
            domain: The domain object.
            N: The number of rows.
        """
        data = {
            attr: np.random.randint(low=0, high=n, size=N)
            for attr, n in zip(domain.attrs, domain.shape)
        }
        return Dataset(data, domain)

    @staticmethod
    def load(path: str, domain: str | Domain) -> Dataset:
        """Load data from a CSV file.

        .. deprecated::
            ``Dataset.load`` will be removed in a future release.
            Load the CSV yourself, convert columns to a
            ``dict[str, np.ndarray]``, and pass it to ``Dataset`` directly.

        Args:
            path: Path to csv file.
            domain: Path to json file encoding the domain, or a Domain.
        """
        warnings.warn(
            "Dataset.load is deprecated. Load your data, convert columns "
            "to a dict of numpy arrays, and instantiate Dataset directly.",
            DeprecationWarning,
            stacklevel=2,
        )
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
            rows = []
            for row in reader:
                try:
                    mapped_row = [int(float(row[i])) for i in indices]
                except ValueError:
                    mapped_row = [int(row[i]) for i in indices]
                rows.append(mapped_row)

        arr = np.array(rows)
        data = {attr: arr[:, i] for i, attr in enumerate(domain_obj.attrs)}
        return Dataset(data, domain_obj)

    def project(
        self, cols: int | str | Sequence[str] | Sequence[int]
    ) -> Factor:
        """Project dataset onto a subset of columns."""
        if isinstance(cols, (str, int)):
            cols = [cols]

        domain = self.domain.project(cols)
        data = {col: self.data[col] for col in domain.attrs}
        sub = Dataset(data, domain, self.weights)
        return Factor(sub.domain, jnp.asarray(sub.datavector(flatten=False)))

    def supports(self, cols: str | Sequence[str]) -> bool:
        return self.domain.supports(cols)

    def drop(self, cols: Sequence[str]) -> Factor:
        """Returns a Factor with the specified columns marginalized out."""
        proj = [c for c in self.domain if c not in cols]
        return self.project(proj)

    @property
    def records(self) -> int:
        """Returns the number of records (rows) in the dataset."""
        if not self.data:
            return self.weights.size
        return next(iter(self.data.values())).shape[0]

    def datavector(self, flatten: bool = True) -> NDArray:
        """Return the database in vector-of-counts form."""
        dims = self.domain.shape
        if len(dims) == 0:
            result = self.weights.sum()
            return np.array([result]) if flatten else result
        multi_index = tuple(self.data[a] for a in self.domain.attrs)
        linear_indices = np.ravel_multi_index(multi_index, dims, order="C")
        counts = np.bincount(
            linear_indices, minlength=math.prod(dims), weights=self.weights
        )
        return counts if flatten else counts.reshape(dims)

    def compress(
        self,
        mapping: dict[str, np.ndarray],
        labels: dict[str, tuple] | None = None,
    ) -> Dataset:
        """Compress the dataset by mapping domain elements to a smaller domain.

        Args:
            mapping: A dictionary where keys are attribute names and values
                are 1D arrays.  ``mapping[attr][i]`` gives the new value
                for original value ``i``.
            labels: Optional explicit labels for the compressed domain.
                If not provided and the domain has labels, the original
                labels are grouped into tuples (e.g. compressing
                ``("cat", "dog", "bird")`` with mapping ``[0, 0, 1]``
                gives ``(("cat", "dog"), ("bird",))``).

        Returns:
            A new Dataset with transformed values and updated domain.
        """
        new_data = dict(self.data)
        new_domain_config = self.domain.config.copy()

        for attr, map_array in mapping.items():
            if attr not in self.domain:
                continue

            _validate_mapping(map_array, attr)
            if map_array.shape[0] != self.domain[attr]:
                raise ValueError(
                    f"Mapping size {map_array.shape[0]} does not match domain"
                    f" size {self.domain[attr]} for attribute {attr}"
                )

            new_col = map_array[self.data[attr]]
            new_data[attr] = new_col.astype(
                np.min_scalar_type(np.max(map_array))
            )
            new_domain_config[attr] = int(np.max(map_array) + 1)

        new_domain = Domain(
            new_domain_config.keys(),
            new_domain_config.values(),
            labels=_compress_labels(
                self.domain, mapping, new_domain_config, labels
            ),
        )
        return Dataset(new_data, new_domain, self.weights)

    def decompress(self, mapping: dict[str, np.ndarray]) -> Dataset:
        """Decompress the dataset by reversing the mapping.

        Since the mapping is surjective, the reverse mapping is one-to-many.
        We sample uniformly from the possible original values.

        Args:
            mapping: The same mapping dictionary used for compression.

        Returns:
            A new Dataset with restored domain size and sampled values.
        """
        new_data = dict(self.data)
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

            current_col = self.data[attr]

            col_counts = counts[current_col]
            if np.any(col_counts == 0):
                raise ValueError(
                    f"Data contains values for {attr} that have no preimage"
                    " in the mapping."
                )

            random_offsets = np.floor(
                np.random.rand(len(current_col)) * col_counts
            ).astype(int)

            lookup_indices = starts[current_col] + random_offsets

            new_col = permutation[lookup_indices]
            new_data[attr] = new_col.astype(
                np.min_scalar_type(len(map_array) - 1)
            )

            new_domain_config[attr] = len(map_array)

        new_domain = Domain(
            new_domain_config.keys(), new_domain_config.values()
        )
        return Dataset(new_data, new_domain, self.weights)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
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
    domain: Domain = jax.tree.static()
    weights: jax.Array | None = None

    @staticmethod
    def synthetic(domain: Domain, records: int) -> JaxDataset:
        """Generate synthetic data conforming to the given domain.

        Args:
            domain: The domain object.
            records: The number of individuals.
        """
        data = {}
        for attr, n in zip(domain.attrs, domain.shape):
            data[attr] = jnp.array(
                np.random.randint(low=0, high=n, size=records)
            )

        return JaxDataset(data, domain)

    def project(self, cols: str | Sequence[str]) -> Factor:
        """Project dataset onto a subset of columns."""
        if isinstance(cols, (str, int)):
            cols = [cols]

        domain = self.domain.project(cols)

        dims = domain.shape
        if not dims:
            w = (
                self.weights
                if self.weights is not None
                else jnp.ones(self.records)
            )
            result = w.sum()
            return Factor(domain, jnp.array([result]))

        length = math.prod(dims)
        dtype = np.min_scalar_type(length - 1)
        multi_index = [self.data[a] for a in domain.attrs]
        multi_index[0] = multi_index[0].astype(dtype)
        linear_indices = jnp.ravel_multi_index(
            tuple(multi_index), dims, mode="wrap", order="C"
        )

        counts = jnp.bincount(
            linear_indices, weights=self.weights, length=length
        )

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

    def synthetic_data(self, rows: int | None = None) -> Dataset:
        """Generate synthetic data via randomized rounding of weights.

        Args:
            rows: Number of rows to generate.  Defaults to the sum of weights.

        Returns:
            A Dataset with integer-valued rows.
        """
        weights = np.asarray(
            self.weights if self.weights is not None else np.ones(self.records)
        )
        total = max(1, int(rows or weights.sum()))
        rng = np.random.default_rng()
        counts = weights * total / weights.sum()
        frac, integ = np.modf(counts)
        integ = integ.astype(int)
        extra = total - integ.sum()
        if extra > 0:
            p = frac / frac.sum()
            idx = rng.choice(len(counts), extra, replace=False, p=p)
            integ[idx] += 1
        row_indices = np.repeat(np.arange(len(counts)), integ)
        rng.shuffle(row_indices)
        row_indices = row_indices[:total]
        data = {
            col: np.asarray(self.data[col])[row_indices]
            for col in self.domain.attrs
        }
        return Dataset(data, self.domain)
