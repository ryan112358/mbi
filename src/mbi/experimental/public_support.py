
import jax
import numpy as np
from scipy.special import logsumexp

from .. import estimation, marginal_loss
from ..clique_vector import CliqueVector
from ..dataset import Dataset
from ..domain import Domain
from ..factor import Factor
from ..marginal_loss import LinearMeasurement

"""Experimental implementation of data synthesis using public data support.

This module provides an experimental function, `public_support`, which aims to
re-implement and generalize the technique presented in PMW^{Pub}
(https://arxiv.org/pdf/2102.08598.pdf). The core idea is to re-weight a public
dataset to match private marginal measurements, effectively generating synthetic
data that respects privacy constraints while leveraging public information.

Notable aspects and differences include:
- Adherence to the common interface for estimators within this repository.
- Support for unbounded differential privacy, including automatic total estimation.
- Flexibility to handle arbitrary measurements via `MarginalLossFn`.

Note: This implementation is experimental and not heavily optimized following
refactoring. Contributions for improvement are welcome.
"""


def entropic_mirror_descent(loss_and_grad, x0, total, iters=250):
    """Performs optimization using entropic mirror descent to find optimal weights."""
    logP = np.log(x0 + np.nextafter(0, 1)) + np.log(total) - np.log(x0.sum())
    P = np.exp(logP)
    P = x0 * total / x0.sum()
    loss, dL = loss_and_grad(P)
    alpha = 1.0
    begun = False

    for _ in range(iters):
        logQ = logP - alpha * dL
        logQ += np.log(total) - logsumexp(logQ)
        Q = np.exp(logQ)
        # Q = P * np.exp(-alpha*dL)
        # Q *= total / Q.sum()
        new_loss, new_dL = loss_and_grad(Q)

        if loss - new_loss >= 0.5 * alpha * dL.dot(P - Q):
            # print(alpha, loss)
            logP = logQ
            loss, dL = new_loss, new_dL
            # increase step size if we haven't already decreased it at least once
            if not begun:
                alpha *= 2
        else:
            alpha *= 0.5
            begun = True

    return np.exp(logP)

def _to_clique_vector(data, cliques):
    """Converts a Dataset object into a CliqueVector representation of its marginals."""
    arrays = {}
    for cl in cliques:
        dom = data.domain.project(cl)
        vals = data.project(cl).datavector(flatten=False)
        arrays[cl] = Factor(dom, vals)
    return CliqueVector(dom, cliques, arrays)


def public_support(
    domain: Domain,
    loss_fn: marginal_loss.MarginalLossFn | list[LinearMeasurement],
    *,
    public_data: Dataset,
    known_total=None
) -> Dataset:

    loss_fn, known_total, _ = estimation._initialize(domain, loss_fn, known_total, None)
    loss_and_grad_mu = jax.value_and_grad(loss_fn)

    cliques = loss_fn.cliques  # type: ignore

    def loss_and_grad(weights):
        """Calculates the loss and gradient with respect to the public data weights."""
        est = Dataset(public_data.data, public_data.domain, weights)
        mu = _to_clique_vector(est, cliques)
        loss, dL = loss_and_grad_mu(mu)
        dweights = np.zeros(weights.size)
        for cl in dL.cliques:
            idx = est.project(cl).data
            dweights += np.array(dL[cl].values[tuple(idx.T)])
        return loss, dweights

    weights = np.ones(public_data.records)
    weights = entropic_mirror_descent(loss_and_grad, weights, known_total)
    return Dataset(public_data.data, public_data.domain, weights)
