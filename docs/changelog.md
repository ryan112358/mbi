# Changelog

This page documents the history of changes for each version of `mbi`.

## Version 2.0.0 - 09/2026

* New unified Estimator API & optimizers. MBI introduced an Estimator class replacing legacy ad-hoc estimation functions. Optimizers including MirrorDescent, DualAveraging, InteriorGradient, and LBFGS now share a common estimation lifecycle and benefit from shared components and a unified API. Key features include:
    * Warm-starting (warm_start=model): Initialize optimization from a previously estimated model, automatically expanding potentials to cover newly observed cliques.
    * Tolerance-based early stopping (tol, patience): Terminate estimation early when relative loss improvement plateaus.
    * Analytic Lipschitz computation: Automatically computes exact L2 Lipschitz constants directly from measurement metadata to tune step sizes without manual parameter sweeps.
    * New estimator models: Added MixtureOfProductsEstimator (i.e., [Relaxed Projection](https://arxiv.org/html/2103.06641v2)) and ReweightedDatasetEstimator (from [PMW^{pub}](https://proceedings.mlr.press/v139/liu21w/liu21w.pdf)) in mbi.extensions.
* Structural constraints (possible/impossible values & functional dependencies). A new first-class Constraint dataclass allows declaring exact domain validity rules in three flexible ways: valid / invalid possible combinations or functional dependencies. Constraints are surfaced throughout the API and automatically converted into log-space potentials (0.0 for valid, -inf for invalid) where needed.
* Library-wide ahead-of-time precompilation. Ahead-of-time compilation has been systematically introduced across the core library (estimation, inference, data generation, and marginal computation) to mitigate JIT tracing latency. Some mechanisms (e.g., AIM) built on top of MBI require compiling a large number of different programs, and compilation time can be a significant overhead.
* Serialization (save / load). Added native numpy-based serialization for MarkovRandomField, LinearMeasurement, and CliqueVector, ensuring models and intermediates can be checkpointed and restored reliably.
* Domain compression: Added LinearMeasurement.compress() and Dataset.compress() for domain reduction.
* Modernization & code quality.
    * Migrated internal structures from attrs to standard Python dataclasses.
    * Deprecated Domain.attrs in favor of Domain.attributes.
    * Migrated from pytype to pyrefly for static analysis checks, and improved type safety across the library.

## Version 1.0 - 11/2024

This library recently underwent major refactorings.  Below are a list of notable changes.

* Library has been modernized to use more recent python features.  Therefore, we require Python>=3.9.
* JAX is now used as the numerical backend rather than numpy.  This means the code now natively supports running on GPUs, although the scalability advantages have not yet been tested.
* To more naturally support JAX with JIT compilation, the code has been reorganized into a more functional design.  This design follows closely from how we describe the approach mathematically in our papers.
* Classes now have more narrowly defined scope, and have been removed where they did not provide significant utility.  Dataclasses are used liberally throughout the code.
* A new belief propagation algorithm is used which is more space efficient than the prior one in settings where the maximal cliques in the Junction Tree are larger than the measured cliques.  This new algorithm is also significantly faster when running on GPUs in some cases.
* Expanded test coverage for marginal inference, we correctly handle a number of tricky edge cases.
* Added type information to most functions for better documentation.  Also added example usages to some functions in the form of doctests.  More will be added in the future.
* Setup continuous integration tests on GitHub.

Currently, not all functionality that was previously supported been integrated into the new design.  However, the core features that are used by the majority of the use cases have been.  These left out functionalities are discussed in the end of this document. 

## Version 0.1.0 (Initial Release)

* Initial release of `mbi`.
