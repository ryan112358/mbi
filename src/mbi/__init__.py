"""Main entry point for the mbi package.

This module exposes the core classes and submodules of the mbi library,
making them available for direct import. It simplifies access to functionalities
like data representation (Domain, Dataset), factor manipulation (Factor),
and various estimation and oracle modules.
"""

import warnings

import jax

if not jax.config.jax_enable_x64:  # pylint: disable=no-member
    warnings.warn(
        "JAX is running in float32 mode. For large datasets (N > 100K),"
        " this can cause estimation algorithms to stall or diverge due to"
        " insufficient floating-point precision. Enable float64 with:"
        ' jax.config.update("jax_enable_x64", True)',
        stacklevel=1,
    )

if jax.config.jax_enable_compilation_cache:  # pylint: disable=no-member
    warnings.warn(
        "JAX persistent compilation cache is enabled. MBI generates many"
        " small compiled programs, which makes the cache counterproductive"
        " — especially when running sweeps of experiments that all write"
        " to the same cache location. Consider disabling it with:"
        ' jax.config.update("jax_enable_compilation_cache", False)',
        stacklevel=1,
    )

from . import callbacks, estimation, extensions, junction_tree, marginal_oracles
from ._model_summary import summarize
from ._api import Model, Projectable
from .estimation import Estimator
from .extensions import constraints
from .extensions.constraints import DeterministicConstraint
from .clique_vector import CliqueVector
from .dataset import Dataset
from .domain import Domain
from .factor import Factor
from .marginal_loss import LinearMeasurement, MarginalLossFn
from .marginal_oracles import MarginalOracle
from .markov_random_field import MarkovRandomField

Clique = tuple[str, ...]

__all__ = [
    "DeterministicConstraint",
    "Domain",
    "Dataset",
    "Factor",
    "Clique",
    "CliqueVector",
    "LinearMeasurement",
    "MarginalLossFn",
    "MarkovRandomField",
    "Projectable",
    "Model",
    "MarginalOracle",
    "Estimator",
    "summarize",
    "estimation",
    "extensions",
    "callbacks",
    "junction_tree",
    "marginal_oracles",
]
