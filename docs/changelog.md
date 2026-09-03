# Changelog

This page documents the history of changes for each version of `mbi`.

## Version 2.0.0 - 09/2026

This release introduces significant optimizations by decoupling the primary data engine from pandas, alongside several new algorithmic extensions, extensive API modernization, and quality-of-life updates.

**New Features & Algorithms**
* [**Unified Estimator API & optimizers**](mbi.html#mbi.Estimator): Relaxed and generalized the `Estimator` Protocol, providing a single, unified `estimate(loss_fn, ...)` signature across all implementations (`MirrorDescent`, `DualAveraging`, `InteriorGradient`, `LBFGS`, `MixtureOfProductsEstimator`, `ReweightedDatasetEstimator`).
    * **Warm-starting (`warm_start=model`)**: Initialize optimization from a previously estimated model, automatically expanding potentials to cover newly observed cliques.
    * **Tolerance-based early stopping (`tol`, `patience`)**: Terminate estimation early when relative loss improvement plateaus.
    * **Analytic Lipschitz computation**: Automatically computes exact L2 Lipschitz constants directly from measurement metadata to tune step sizes without manual parameter sweeps.
* [**Structural constraints**](mbi.html#mbi.Constraint): A new first-class `Constraint` dataclass allows declaring exact domain validity rules in three flexible ways: valid / invalid possible combinations or functional dependencies.
* [**Library-wide ahead-of-time precompilation**](mbi.extensions.html#mbi.extensions.precompile): Ahead-of-time compilation has been systematically introduced across the core library (estimation, inference, data generation, and marginal computation) to mitigate JIT tracing latency via `synthetic_data.precompile`.
* [**Serialization (`save` / `load`)**](mbi.html#mbi.save): Added native numpy-based serialization for `MarkovRandomField`, `LinearMeasurement`, and `CliqueVector`, ensuring models and intermediates can be checkpointed and restored reliably.
* [**Domain compression & query filtering**](mbi.html#mbi.Dataset): Added `LinearMeasurement.compress()` and `Dataset.compress()` for domain reduction, plus `DatavectorQuery.use_for_total_estimation` to exclude auxiliary queries from total count estimation.
* **JAM Mechanism:** Introduced the Joint Adaptive Measurements (`JAM`) mechanism for DP synthetic data generation, available in the codebase as an example workflow.
* **Belief Propagation:** Bolstered belief propagation with support for vector-valued and single-row evidence in `variable_elimination`, and structural graph unit tests for cliques in `junction_tree`.
* **Factor Slicing:** Implemented multidimensional `Factor.slice()` optimized with native tuple indexing.

**Performance & Modernization**
* **Removed pandas dependency:** Removed the pandas dependency internally from `src/mbi/`, completely overhauling `Dataset` and `synthetic_data` logic to leverage highly optimized NumPy 1-D arrays and `bincount` operations natively.
* **Migration to dataclasses:** Migrated internal structures from `attrs` to standard Python `dataclasses`.
* **Pyrefly Type Safety:** Migrated from pytype to pyrefly for static analysis checks, and improved type safety across the library.
* **API Cleanup:** Deprecated and dropped `**kwargs` and stale parameters across the library. Deprecated `Domain.attrs` in favor of `Domain.attributes`. Stripped out outdated backward compatibility properties (`CliqueVector.arrays`) and the deprecated `gaussian_noise_scale` method.
* **Documentation:** Fully configured scalable Sphinx API documentation generation, integrating native `autosummary` for the new `extensions` and `junction_tree` modules. Formalized continuous integration checking Pyre typing, Pyink PEP-8 formatting, and `docs/` HTML compilation tests.

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
