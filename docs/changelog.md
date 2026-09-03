# Changelog

This page documents the history of changes for each version of `mbi`.

## Version 2.0.0 - 09/2026

* Major version bump reflecting the removal of deprecated APIs and parameters.
* Standardized the `warm_start` parameter in all estimators (now natively accepts `MarkovRandomField`, `MixtureOfProducts`, or raw `CliqueVector`).
* Removed stale compatibility properties (`CliqueVector.arrays` and `Domain.attrs`) and the `gaussian_noise_scale` function.
* Added `extensions` and `junction_tree` modules to the API documentation.

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
