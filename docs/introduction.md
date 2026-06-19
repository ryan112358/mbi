# Introduction

This is a general-purpose library for estimating discrete distributions from noisy
observations of their marginals.  To that end, we provide scalable algorithms for solving convex optimization problems over the marginal polytope.  From that, we obtain an undirected graphical model (i.e., a Markov random field) which can be used to (1) obtain more accurate estimates of the observed noisy marginals, (2) estimate answers to new queries (e.g., marginals) based on a maximum entropy assumption, and (3) generate synthetic data that approximately preserves those marginals.  

The library is designed with the following core principles in mind:

* **Consistency:** This library produces consistent estimates of measured and unmeasured queries, (i.e., answers that could arise from some global data distribution).
* **Utility**: The estimation algorithms make the best use of the noisy measurements
by combining all sources of information in a principled manner (e.g., by maximizing the likelihood of the noisy observations).  The algorithms work well in practice.
* **Scalability**: The library scales effectively to high-dimensional datasets.
* **Flexibility**: The library supports generic loss functions defined over the low-dimensional marginals of a data distribution.  We provide APIs and utilities for constructing a loss function based on the broad class of noisy linear measurements of the marginals.  Flexibility beyond this is possible and easy to configure.  
* **Extensibility:** The library is designed to be built upon by future work.  We focus on the important but narrow problem of estimating a data distribution from noisy measurements, and provide a good solution for that.  This is only one component of a strong mechanism, and must be combined with principled methods for selecting *which* queries should be privately measured based on the data, privacy budget, downstream task, and other considerations.
* **Simplicity:** The library is simple to use, toy examples only require a few lines of code, and fairly powerful mechanisms can be developed on top of this library with surprisingly little code.  

Published articles: 

- McKenna, Ryan, Terrance Liu, "A simple recipe for private synthetic data generation."
[https://differentialprivacy.org/synth-data-1/](https://differentialprivacy.org/synth-data-1/)

- McKenna, Ryan, Daniel Sheldon, and Gerome Miklau. 2021. "Winning the NIST Contest: A scalable and general approach to differentially private synthetic data."  *Journal of Privacy and Confidentiality* 11 (3).  [![DOI:10.29012/jpc.778](https://zenodo.org/badge/DOI/10.29012/jpc.778.svg)](https://doi.org/10.29012/jpc.778) 

- McKenna, Ryan, Daniel Sheldon, and Gerome Miklau. "Graphical-model based estimation and inference for differential privacy." In Proceedings of the 36th International Conference on Machine Learning. 2019. https://arxiv.org/abs/1901.09136

- McKenna, Ryan, Siddhant Pradhan, Daniel Sheldon, and Gerome Miklau. "Relaxed Marginal Consistency for Differentially Private Query Answering".  In Proceedings of the 35th Conference on Neural Information Processing Systems, 2021. https://arxiv.org/pdf/2109.06153

- McKenna, Ryan, Brett Mullins, Daniel Sheldon, and Gerome Miklau. "AIM: An Adaptive and Iterative Mechanism for Differentially Private Synthetic Data".  In Proceedings of the VLDB Endowment, Vol 15, 2023. https://arxiv.org/abs/2201.12677