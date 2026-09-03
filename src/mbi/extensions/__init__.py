"""Extensions for mbi providing alternative estimation approaches."""

from .mixture_of_products import MixtureOfProducts, MixtureOfProductsEstimator
from .precompute_marginals import precompute_marginals
from .reweighted_dataset import ReweightedDatasetEstimator
from .synthetic_data import precompile, synthetic_data

__all__ = [
    'MixtureOfProducts',
    'MixtureOfProductsEstimator',
    'precompute_marginals',
    'ReweightedDatasetEstimator',
    'precompile',
    'synthetic_data',
]
