# Contributing

Contributions to this repository are welcome and encouraged.

* Adding new functionality (new estimators, synthetic data generators, etc.).
* Filing bugs if you run into any errors or unexpected performance characteristics.
* Improving documentation.
* Checking in mechanisms you built on top of this library.
* Code that got cut during the migration might be nice to add back in under the new design.
>* Calculate many marginals (theoretically more efficient than calling "project" multiple times)
>* Answering Kronecker product queries
>* Some approximate marginal inference oracles that were previously implemented did not get moved over, only the one we presented in our paper.
>* Some examples were deleted rather than ported, since they took dependencies on other github repositories.  
>* Calculation of Lipschitz constant for L2 losses
>* We did not refactor other marginal-based estimation algorithms in terms of the current design.  [PublicInference](https://ppai21.github.io/files/26-paper.pdf) and [MixtureInference](https://arxiv.org/abs/2103.06641), are still implemented in terms of the old API, and are in the experimental subpackage.  

