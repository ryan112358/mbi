"""Provides the Dataset class for representing and manipulating tabular data.

This module defines the `Dataset` class, which serves as a wrapper around a
Pandas DataFrame, associating it with a `Domain` object. It allows for
structured representation of data, facilitating operations like projection onto
subsets of attributes and conversion into a data vector format suitable for
various statistical and machine learning tasks.
"""
from __future__ import annotations
import json
import os

import numpy as np
import pandas as pd
import chex
import jax
import attr
import functools
import jax.numpy as jnp
from typing import Sequence

from .domain import Domain
from .factor import Factor


class Dataset:
    def __init__(self, df, domain, weights=None):
        """create a Dataset object

        :param df: a pandas dataframe
        :param domain: a domain object
        :param weight: weight for each row
        """
        assert set(domain.attrs) <= set(
            df.columns
        ), "data must contain domain attributes"
        assert weights is None or df.shape[0] == weights.size
        self.domain = domain
        self.df = df.loc[:, domain.attrs]
        self.weights = weights

    @staticmethod
    def synthetic(domain, N):
        """Generate synthetic data conforming to the given domain

        :param domain: The domain object
        :param N: the number of individuals
        """
        arr = [np.random.randint(low=0, high=n, size=N) for n in domain.shape]
        values = np.array(arr).T
        df = pd.DataFrame(values, columns=domain.attrs)
        return Dataset(df, domain)

    @staticmethod
    def load(path, domain):
        """Load data into a dataset object

        :param path: path to csv file
        :param domain: path to json file encoding the domain information
        """
        df = pd.read_csv(path)
        config = json.load(open(domain))
        domain = Domain(config.keys(), config.values())
        return Dataset(df, domain)

    def project(self, cols):
        """project dataset onto a subset of columns"""
        if type(cols) in [str, int]:
            cols = [cols]
        data = self.df.loc[:, cols]
        domain = self.domain.project(cols)
        return Dataset(data, domain, self.weights)

    def drop(self, cols):
        """Returns a new Dataset with the specified columns removed."""
        proj = [c for c in self.domain if c not in cols]
        return self.project(proj)

    @property
    def records(self):
        """Returns the number of records (rows) in the dataset."""
        return self.df.shape[0]

    def datavector(self, flatten=True):
        """return the database in vector-of-counts form"""
        bins = [range(n + 1) for n in self.domain.shape]
        ans = np.histogramdd(self.df.values, bins, weights=self.weights)[0]
        return ans.flatten() if flatten else ans


@functools.partial(
    jax.tree_util.register_dataclass, 
    meta_fields=["domain"], 
    data_fields=["data", "weights"]
)
@attr.dataclass(frozen=True)
class JaxDataset:
    data: jax.Array = attr.field(converter=jnp.asarray)
    domain: Domain
    weights: jax.Array | None = None

    def __post_init__(self):
        if self.data.dtype != 'int':
            raise ValueError(f'Data must be integral, got {self.data.dtype}.')
        if self.data.ndim != 2:
            raise ValueError(f'Data must be 2d aray, got {self.data.shape}')
        if self.data.shape[1] != len(self.domain):
            raise ValueError('Number of columns of data must equal the number of attributes in the domain.')
        # This will not work in a jitted context, but not sure if this will be called from one normally.
        for i, attr in enumerate(self.domain):
            if self.data[:, i].min() < 0:
                raise ValueError('Data must be non-negative.')
            if self.data[:, i].max() >= self.domain[attr]:
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

    def project(self, cols: str | Sequence[str]) -> JaxDataset:
        """project dataset onto a subset of columns"""
        if type(cols) in [str, int]:
            cols = [cols]
        idx = self.domain.axes(cols)
        data = self.data.loc[:, idx]
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
