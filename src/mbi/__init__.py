"""Main entry point for the mbi package.

This module exposes the core classes and submodules of the mbi library,
making them available for direct import. It simplifies access to functionalities
like data representation (Domain, Dataset), factor manipulation (Factor),
and various estimation and oracle modules.
"""

from . import callbacks, estimation, extensions, junction_tree, marginal_oracles
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
    "estimation",
    "extensions",
    "callbacks",
    "junction_tree",
    "marginal_oracles",
]
