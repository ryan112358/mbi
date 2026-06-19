"""Extensions for mbi that provide alternative estimation approaches.

This subpackage contains estimation methods that go beyond the standard
graphical model (PGM) approach. These methods parameterize the distribution
directly rather than using a Markov random field with belief propagation.

Available extensions:
    - mixture_of_products: RAP-style relaxed tabular estimation using a
      mixture of product distributions with softmax parameterization.
"""

from .mixture_of_products import MixtureOfProducts, mixture_of_products

__all__ = [
    "MixtureOfProducts",
    "mixture_of_products",
]
