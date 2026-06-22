## MBI: Marginal-Based Estimation and Inference
**(with applications to differential privacy)**

<img src="pgm-logo.png" alt="drawing" width="123"/>

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5548533.svg)](https://doi.org/10.5281/zenodo.5548533)
[![Continuous integration](https://github.com/ryan112358/mbi/actions/workflows/main.yml/badge.svg)](https://github.com/ryan112358/mbi/actions/workflows/main.yml)
[![Documentation Status](https://readthedocs.org/projects/private-pgm/badge/?version=latest)](https://private-pgm.readthedocs.io/en/latest/)

![Metrics for ryan112358/mbi repository](https://raw.githubusercontent.com/ryan112358/ryan112358/main/metrics.mbi.svg)

Documentation can be found at
[https://private-pgm.readthedocs.io/en/latest/](https://private-pgm.readthedocs.io/en/latest/)!

Consider joining the [Google Differential Privacy community](https://join.slack.com/t/dp-open-source/shared_invite/zt-35hw483tz-nS5YOtGjxCHk3Ek7WiXvlg) in Slack.

### Quick Start

```python
from mbi import Domain, estimation, marginal_loss, callbacks

# Define a domain
domain = Domain(["age", "sex", "income"], [10, 2, 5])

# Provide noisy marginal measurements
measurements = [
    marginal_loss.LinearMeasurement(noisy_marginal_1, ("age", "sex")),
    marginal_loss.LinearMeasurement(noisy_marginal_2, ("sex", "income")),
]

# Estimate a graphical model
model = estimation.MirrorDescent().estimate(domain, measurements, iters=1000)

# Use the model
synthetic_data = model.synthetic_data(rows=1000)
age_sex_marginal = model.project(("age", "sex")).datavector()
```

### Estimators

All estimators share the same `Estimator` API:

```python
# Mirror Descent (recommended)
model = estimation.MirrorDescent().estimate(domain, measurements)

# Dual Averaging
model = estimation.DualAveraging().estimate(domain, measurements)

# Interior Gradient
model = estimation.InteriorGradient().estimate(domain, measurements)
```

Extension estimators provide alternative representations:

```python
from mbi.extensions.mixture_of_products import MixtureOfProductsEstimator
from mbi.extensions.reweighted_dataset import ReweightedDatasetEstimator

# Mixture of Products (scalable, no graphical model)
model = MixtureOfProductsEstimator(num_components=100).estimate(domain, measurements)

# Reweighted Dataset (produces a weighted dataset)
model = ReweightedDatasetEstimator(seed_data=data).estimate(domain, measurements)
```

### Callbacks

Monitor optimization progress with callbacks:

```python
cb = callbacks.default(measurements)
model = estimation.MirrorDescent().estimate(
    domain, measurements, iters=1000, callback_fn=cb,
)
```
