from mbi import Dataset, Factor, CliqueVector, marginal_loss, estimation, Domain, LinearMeasurement
from scipy.optimize import minimize
from collections import defaultdict
from jax import vjp
import jax.nn
from scipy.special import softmax
from functools import reduce
from scipy.sparse.linalg import lsmr
import jax.numpy as jnp
import optax
import numpy as np

""" This file is experimental.

It is a close approximation to the method described in RAP (https://arxiv.org/abs/2103.06641)
and an even closer approximation to RAP^{softmax} (https://arxiv.org/abs/2106.07153). This 
implementation is not very optimized.  If you would like to improve it, pull requests 
are welcome.

Notable differences:
- Code now shares the same interface as Private-PGM (see FactoredInference)
- Named model "MixtureOfProducts", as that is one interpretation for the relaxed tabular format
(at least when softmax is used).
- Added support for unbounded-DP, with automatic estimate of total.
"""


def adam(loss_and_grad, x0, iters=250):
    # TODO: Rewrite using optax
    a = 1.0
    b1, b2 = 0.9, 0.999
    eps = 10e-8

    x = x0
    m = jnp.zeros_like(x)
    v = jnp.zeros_like(x)
    for t in range(1, iters + 1):
        l, g = loss_and_grad(x)
        # print(l)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g ** 2
        mhat = m / (1 - b1 ** t)
        vhat = v / (1 - b2 ** t)
        x = x - a * mhat / (jnp.sqrt(vhat) + eps)
    return x


def synthetic_col(counts, total):
    counts *= total / counts.sum()
    frac, integ = np.modf(counts)
    integ = integ.astype(int)
    extra = total - integ.sum()
    if extra > 0:
        idx = np.random.choice(counts.size, extra, False, frac / frac.sum())
        integ[idx] += 1
    vals = np.repeat(np.arange(counts.size), integ)
    np.random.shuffle(vals)
    return vals


class MixtureOfProducts:
    def __init__(self, products, domain, total):
        self.products = products
        self.domain = domain
        self.total = total
        self.num_components = next(iter(products.values())).shape[0]

    def project(self, cols):
        products = {col: self.products[col] for col in cols}
        domain = self.domain.project(cols)
        return MixtureOfProducts(products, domain, self.total)

    def datavector(self, flatten=True):
        d = len(self.domain)
        letters = "bcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"[:d]
        formula = ",".join(["a%s" % l for l in letters]) + "->" + "".join(letters)
        components = [self.products[col] for col in self.domain]
        ans = jnp.einsum(formula, *components) * self.total / self.num_components
        return ans.flatten() if flatten else ans

    def synthetic_data(self, rows=None):
        total = rows or int(self.total)
        subtotal = total // self.num_components + 1

        data_list = []
        for i in range(self.num_components):
            comp_data = {}
            for col in self.products:
                counts = self.products[col][i]
                comp_data[col] = synthetic_col(counts, subtotal)

            # Convert comp_data (dict of arrays) to a structured representation or just a list of rows
            # Here we know the length is subtotal
            # We want a 2D array with columns in domain order
            current_block = np.stack([comp_data[col] for col in self.domain.attrs], axis=1)
            data_list.append(current_block)

        full_data = np.concatenate(data_list, axis=0)

        # Shuffle
        np.random.shuffle(full_data)

        # Truncate to total
        full_data = full_data[:total]

        return Dataset(full_data, self.domain)


def mixture_of_products(
    domain: Domain,
    loss_fn: marginal_loss.MarginalLossFn | list[LinearMeasurement],
    *,
    known_total: int | None = None,
    mixture_components: int = 100,
    iters: int = 2500,
    alpha: float = 0.1
) -> MixtureOfProducts:

    loss_fn, known_total, _ = estimation._initialize(domain, loss_fn, known_total, None)

    one_hot_features = sum(domain.shape)
    params = np.random.normal(
        loc=0, scale=0.25, size=(mixture_components, one_hot_features)
    )

    cliques = loss_fn.cliques  # type: ignore

    letters = "bcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

    def get_products(params):
        products = {}
        idx = 0
        for col in domain:
            n = domain[col]
            products[col] = jax.nn.softmax(params[:, idx : idx + n], axis=1)
            idx += n
        return products

    def marginals_from_params(params):
        products = get_products(params)
        arrays = {}
        for cl in cliques:
            let = letters[: len(cl)]
            formula = ",".join(["a%s" % l for l in let]) + "->" + "".join(let)
            components = [products[col] for col in cl]
            ans = jnp.einsum(formula, *components) * known_total / mixture_components
            arrays[cl] = Factor(domain.project(cl), ans)
        return CliqueVector(domain, cliques, arrays)

    def params_loss(params: jax.Array) -> float:
        mu = marginals_from_params(params)
        return loss_fn(mu)

    params_loss_and_grad = jax.value_and_grad(params_loss)

    params = adam(params_loss_and_grad, params, iters=iters)
    products = get_products(params)
    return MixtureOfProducts(products, domain, known_total)