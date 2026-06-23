"""Extensions for mbi providing alternative estimation approaches."""

from .constraints import ArbitraryConstraint

from .mixture_of_products import MixtureOfProducts, MixtureOfProductsEstimator
from .reweighted_dataset import ReweightedDatasetEstimator
from .synthetic_data import precompile, synthetic_data
