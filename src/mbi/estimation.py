"""Algorithms for estimating graphical models from marginal-based loss functions.

This module provides a flexible set of optimization algorithms, each sharing the
the same API.  The supported algorithms are:
1. Mirror Descent [our recommended algorithm]
2. L-BFGS (using back-belief propagation)
3. Regularized Dual Averaging
4. Interior Gradient
5. Universal accelerated mirror descent

Each algorithm can be given an initial set of potentials, or can automatically
intialize the potentials to zero for you.  Any CliqueVector of potentials that
support the cliques of the marginal-based loss function can be used here.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, NamedTuple, Protocol

import jax
import jax.numpy as jnp
import numpy as np
import optax

from . import marginal_loss, marginal_oracles
from .approximate_oracles import StatefulMarginalOracle
from .clique_vector import CliqueVector
from .domain import Domain
from .factor import Factor, Projectable
from .marginal_loss import LinearMeasurement
from .markov_random_field import MarkovRandomField


class Estimator(Protocol):
    """
    Defines the callable signature for marginal-based estimators.

    An estimator estimates a discrete distribution, or more generally
    a `Projectable' object from a loss function defined over it's 
    low-dimensional marginals.

    Examples of conforming functions from `mbi.estimation`:
    - `mirror_descent`
    - `lbfgs`
    - `dual_averaging`
    - `interior_gradient`
    - `universal_accelerated_method`
    - ... and more from other modules
    """

    def __call__(
        self,
        domain: Domain,
        loss_fn: marginal_loss.MarginalLossFn | list[LinearMeasurement],
        *,
        known_total: float | None = None,
        **kwargs: Any
    ) -> Projectable:
        """
        Estimate a Projectable from noisy marginal measurements.

        Args:
            domain: The Domain object specifying the attributes and their
                cardinalities over which the model is defined.
            loss_fn: Either a MarginalLossFn object or a list of
                LinearMeasurement objects. This defines the objective function
                to be minimized.
            known_total: An optional float for the known or estimated total
                number of records. If not specified, the estimator will attempt
                to learn this automatically.
            **kwargs: Additional optional keyword arguments specific to the
                estimation algorithm.

        Returns:
            A Projectable object that is maximally consistent with the
            noisy measurements taken in some sense.
        """
        ...


def minimum_variance_unbiased_total(measurements: list[LinearMeasurement]) -> float:
    """Estimates the total count from measurements with identity queries."""
    # find the minimum variance estimate of the total given the measurements
    estimates, variances = [], []
    for M in measurements:
        y = M.noisy_measurement
        try:
            # TODO: generalize to support any linear measurement that supports total query
            if M.query == Factor.datavector:  # query = Identity
                estimates.append(y.sum())
                variances.append(M.stddev**2 * y.size)
        except Exception:
            continue
    estimates, variances = np.array(estimates), np.array(variances)
    if len(estimates) == 0:
        return 1

    variance = 1.0 / np.sum(1.0 / variances)
    estimate = variance * np.sum(estimates / variances)
    return max(1, estimate)


def _initialize(domain, loss_fn, known_total, potentials):
    """Initializes loss function, total records, and potentials for estimation algorithms."""
    if isinstance(loss_fn, list):
        if known_total is None:
            known_total = minimum_variance_unbiased_total(loss_fn)
        loss_fn = marginal_loss.from_linear_measurements(loss_fn, domain=domain)
    elif known_total is None:
        raise ValueError("Must set known_total if giving a custom MarginalLossFn")

    if potentials is None:
        potentials = CliqueVector.zeros(domain, loss_fn.cliques)

    if not all(potentials.supports(cl) for cl in loss_fn.cliques):
        potentials = potentials.expand(loss_fn.cliques)

    return loss_fn, known_total, potentials


def _get_stateful_oracle(
    marginal_oracle: marginal_oracles.MarginalOracle | StatefulMarginalOracle,
    stateful: bool,
) -> StatefulMarginalOracle:
    if stateful:
        return marginal_oracle

    def wrapper(theta, total, state):
        return marginal_oracle(theta, total), state

    return wrapper


def mirror_descent(
    domain: Domain,
    loss_fn: marginal_loss.MarginalLossFn | list[LinearMeasurement],
    *,
    known_total: float | None = None,
    potentials: CliqueVector | None = None,
    marginal_oracle: (
        marginal_oracles.MarginalOracle | StatefulMarginalOracle
    ) = marginal_oracles.message_passing_fast,
    iters: int = 1000,
    stateful: bool = False,
    stepsize: float | None = None,
    callback_fn: Callable[[CliqueVector], None] = lambda _: None,
    mesh: jax.sharding.Mesh | None = None,
):
    """Optimization using the Mirror Descent algorithm.

    This is a first-order proximal optimization algorithm for solving
    a (possibly nonsmooth) convex optimization problem over the marginal polytope.
    This is an  implementation of Algorithm 1 from the paper
    ["Graphical-model based estimation and inference for differential privacy"]
    (https://arxiv.org/pdf/1901.09136).  If stepsize is not provided, this algorithm
    uses a line search to automatically choose appropriate step sizes that satisfy
    the Armijo condition.

    Args:
        domain: The domain over which the model should be defined.
        loss_fn: A MarginalLossFn or a list of Linear Measurements.
        known_total: The known or estimated number of records in the data.
        potentials: The initial potentials.  Must be defind over a set of cliques
            that supports the cliques in the loss_fn.
        marginal_oracle: The function to use to compute marginals from potentials.
        iters: The maximum number of optimization iterations.
        stepsize: The step size for the optimization.  If not provided, this algorithm
            will use a line search to automatically choose appropriate step sizes.
        callback_fn: A function to call at each iteration with the iteration number.
        mesh: Determines how the marginal oracle and loss calculation
                will be sharded across devices.

    Returns:
        A MarkovRandomField object with the estimated potentials and marginals.
    """
    if stepsize is None and stateful:
        raise ValueError(
            "Stepsize should be manually tuned when using a stateful oracle."
        )

    loss_fn, known_total, potentials = _initialize(
        domain, loss_fn, known_total, potentials
    )
    marginal_oracle = functools.partial(marginal_oracle, mesh=mesh)
    marginal_oracle = _get_stateful_oracle(marginal_oracle, stateful)

    @jax.jit
    def update(theta, alpha, state=None):
        mu, state = marginal_oracle(theta, known_total, state)
        loss, dL = jax.value_and_grad(loss_fn)(mu)

        theta2 = theta - alpha * dL
        if stepsize is not None:
            return theta2, loss, alpha, mu, state

        mu2, _ = marginal_oracle(theta2, known_total, state)
        loss2 = loss_fn(mu2)

        sufficient_decrease = loss - loss2 >= 0.5 * alpha * dL.dot(mu - mu2)
        alpha = jax.lax.select(sufficient_decrease, 1.01 * alpha, 0.5 * alpha)
        theta = jax.lax.cond(sufficient_decrease, lambda: theta2, lambda: theta)
        loss = jax.lax.select(sufficient_decrease, loss2, loss)

        return theta, loss, alpha, mu, state

    # A reasonable initial learning rate seems to be 2.0 L / known_total,
    # where L is the Lipschitz constant.  Starting from a value too high
    # can be fine in some cases, but lead to incorrect behavior in others.
    # We don't currently take L as an argument, but for the most common case,
    # where our loss function is || mu - y ||_2^2, we have L = 1.
    alpha = 2.0 / known_total if stepsize is None else stepsize
    mu, state = marginal_oracle(potentials, known_total, state=None)
    for t in range(iters):
        potentials, loss, alpha, mu, state = update(potentials, alpha, state)
        callback_fn(mu)

    marginals, _ = marginal_oracle(potentials, known_total, state)
    return MarkovRandomField(
        potentials=potentials, marginals=marginals, total=known_total
    )


def _optimize(loss_and_grad_fn, params, iters=250, callback_fn=lambda _: None):
    """Runs an optimization loop using Optax L-BFGS."""
    def loss_fn(theta):
        return loss_and_grad_fn(theta)[0]

    @jax.jit
    def update(params, opt_state):
        loss, grad = loss_and_grad_fn(params)

        updates, opt_state = optimizer.update(
            grad, opt_state, params, value=loss, grad=grad, value_fn=loss_fn
        )

        return optax.apply_updates(params, updates), opt_state, loss

    optimizer = optax.lbfgs(
        memory_size=1,
        linesearch=optax.scale_by_zoom_linesearch(128, max_learning_rate=1),
    )
    state = optimizer.init(params)
    prev_loss = float("inf")
    for t in range(iters):
        params, state, loss = update(params, state)
        callback_fn(params)
        # if loss == prev_loss: break
        prev_loss = loss
    return params


def lbfgs(
    domain: Domain,
    loss_fn: marginal_loss.MarginalLossFn | list[LinearMeasurement],
    *,
    known_total: float | None = None,
    potentials: CliqueVector | None = None,
    marginal_oracle: marginal_oracles.MarginalOracle = marginal_oracles.message_passing_stable,
    iters: int = 1000,
    callback_fn: Callable[[CliqueVector], None] = lambda _: None,
    mesh: jax.sharding.Mesh | None = None,
):
    """Gradient-based optimization on the potentials (theta) via L-BFGS.

    This optimizer works by calculating the gradients with respect to the
    potentials by back-propagting through the marginal inference oracle.

    This is a standard approach for fitting the parameters of a graphical model
    without noise (i.e., when you know the exact marginals).  In this case,
    the loss function with respect to theta is convex, and therefore this approach
    enjoys convergence guarantees.  With generic marginal loss functions that arise
    for instance ith noisy marginals, the loss function is typically convex with
    respect to mu, but not with respect to theta.  Therefore, this optimizer is not
    guaranteed to converge to the global optimum in all cases.  In practice, it
    tends to work well in these settings despite non-convexities.  This approach
    appeared in the paper ["Learning Graphical Model Parameters with Approximate
    Marginal Inference"](https://arxiv.org/abs/1301.3193).

    Args:
      domain: The domain over which the model should be defined.
      loss_fn: A MarginalLossFn or a list of Linear Measurements.
      known_total: The known or estimated number of records in the data.
        If loss_fn is provided as a list of LinearMeasurements, this argument
        is optional.  Otherwise, it is required.
      potentials: The initial potentials.  Must be defined over a set of cliques
        that supports the cliques in the loss_fn.
      marginal_oracle: The function to use to compute marginals from potentials.
      iters: The maximum number of optimization iterations.
      callback_fn: ...
      mesh: Determines how the marginal oracle and loss calculation
                will be sharded across devices.
    """
    loss_fn, known_total, potentials = _initialize(
        domain, loss_fn, known_total, potentials
    )
    marginal_oracle = functools.partial(marginal_oracle, mesh=mesh)

    def theta_loss(theta):
        return loss_fn(marginal_oracle(theta, known_total))

    theta_loss_and_grad = jax.value_and_grad(theta_loss)

    def theta_callback_fn(theta):
        callback_fn(marginal_oracle(theta, known_total))

    potentials = _optimize(
        theta_loss_and_grad, potentials, iters=iters, callback_fn=theta_callback_fn
    )
    return MarkovRandomField(
        potentials=potentials,
        marginals=marginal_oracle(potentials, known_total),
        total=known_total,
    )


def mle_from_marginals(
    marginals: CliqueVector,
    known_total: float,
    iters: int = 250,
    marginal_oracle: marginal_oracles.MarginalOracle = marginal_oracles.message_passing_stable,
    callback_fn=lambda *_: None,
    mesh: jax.sharding.Mesh | None = None,
) -> MarkovRandomField:
    """Compute the MLE Graphical Model from the marginals.

    Args:
        marginals: The marginal probabilities.
        known_total: The known or estimated number of records in the data.

    Returns:
        A MarkovRandomField object with the final potentials and marginals.
    """

    def loss_and_grad_fn(theta):
        mu = marginal_oracle(theta, known_total, mesh)
        return -marginals.dot(mu.log()), mu - marginals

    potentials = CliqueVector.zeros(marginals.domain, marginals.cliques)
    potentials = _optimize(loss_and_grad_fn, potentials, iters=iters)
    return MarkovRandomField(
        potentials=potentials,
        marginals=marginal_oracle(potentials, known_total),
        total=known_total,
    )


def dual_averaging(
    domain: Domain,
    loss_fn: marginal_loss.MarginalLossFn | list[LinearMeasurement],
    *,
    known_total: float | None = None,
    potentials: CliqueVector | None = None,
    marginal_oracle: marginal_oracles.MarginalOracle = marginal_oracles.message_passing_stable,
    iters: int = 1000,
    callback_fn: Callable[[CliqueVector], None] = lambda _: None,
    mesh: jax.sharding.Mesh | None = None,
) -> MarkovRandomField:
    """Optimization using the Regularized Dual Averaging (RDA) algorithm.

    RDA is an accelerated proximal algorithm for solving a smooth convex optimization
    problem over the marginal polytope.  This algorithm requires knowledge of
    the Lipschitz constant of the gradient of the loss function.

    Args:
        domain: The domain over which the model should be defined.
        loss_fn: A MarginalLossFn or a list of Linear Measurements.
        lipschitz: The Lipschitz constant of the gradient of the loss function.
        known_total: The known or estimated number of records in the data.
        potentials: The initial potentials.  Must be defind over a set of cliques
            that supports the cliques in the loss_fn.
        marginal_oracle: The function to use to compute marginals from potentials.
        iters: The maximum number of optimization iterations.
        callback_fn: A function to call with intermediate solution at each iteration.
        mesh: Determines how the marginal oracle and loss calculation
                will be sharded across devices.

    Returns:
        A MarkovRandomField object with the final potentials and marginals.
    """
    loss_fn, known_total, potentials = _initialize(
        domain, loss_fn, known_total, potentials
    )
    if loss_fn.lipschitz is None:
        raise ValueError(
            "Dual Averaging requires a loss function with Lipschitz gradients."
        )

    D = np.sqrt(domain.size() * np.log(domain.size()))  # upper bound on entropy
    Q = 0  # upper bound on variance of stochastic gradients
    gamma = Q / D

    L = loss_fn.lipschitz / known_total

    @jax.jit
    def update(w, v, gbar, c, beta, t):
        u = (1 - c) * w + c * v
        g = jax.grad(loss_fn)(u) / known_total
        gbar = (1 - c) * gbar + c * g
        theta = -t * (t + 1) / (4 * L + beta) * gbar
        v = marginal_oracle(theta, known_total, mesh)
        w = (1 - c) * w + c * v
        return w, v, gbar

    w = v = marginal_oracle(potentials, known_total, mesh)
    gbar = CliqueVector.zeros(domain, loss_fn.cliques)
    for t in range(1, iters + 1):
        c = 2.0 / (t + 1)
        beta = gamma * (t + 1) ** 1.5 / 2
        w, v, gbar = update(w, v, gbar, c, beta, t)
        callback_fn(w)

    return mle_from_marginals(w, known_total)


def interior_gradient(
    domain: Domain,
    loss_fn: marginal_loss.MarginalLossFn | list[LinearMeasurement],
    *,
    known_total: float | None = None,
    potentials: CliqueVector | None = None,
    marginal_oracle: marginal_oracles.MarginalOracle = marginal_oracles.message_passing_stable,
    iters: int = 1000,
    callback_fn: Callable[[CliqueVector], None] = lambda _: None,
    mesh: jax.sharding.Mesh | None = None,
):
    """Optimization using the Interior Point Gradient Descent algorithm.

    Interior Gradient is an accelerated proximal algorithm for solving a smooth
    convex optimization problem over the marginal polytope.  This algorithm
    requires knowledge of the Lipschitz constant of the gradient of the loss function.
    This algorithm is based on the paper titled
    ["Interior Gradient and Proximal Methods for Convex and Conic Optimization"](https://epubs.siam.org/doi/abs/10.1137/S1052623403427823?journalCode=sjope8).

    Args:
        domain: The domain over which the model should be defined.
        loss_fn: A MarginalLossFn or a list of Linear Measurements.
        lipschitz: The Lipschitz constant of the gradient of the loss function.
        known_total: The known or estimated number of records in the data.
        potentials: The initial potentials.  Must be defind over a set of cliques
            that supports the cliques in the loss_fn.
        marginal_oracle: The function to use to compute marginals from potentials.
        iters: The maximum number of optimization iterations.
        callback_fn: A function to call at each iteration with the iteration number.
        mesh: Determines how the marginal oracle and loss calculation
                will be sharded across devices.

    Returns:
        A MarkovRandomField object with the optimized potentials and marginals.
    """
    loss_fn, known_total, potentials = _initialize(
        domain, loss_fn, known_total, potentials
    )
    if loss_fn.lipschitz is None:
        raise ValueError(
            "Interior Gradient requires a loss function with Lipschitz gradients."
        )

    # Algorithm parameters
    c = 1
    sigma = 1
    l = sigma / loss_fn.lipschitz

    @jax.jit
    def update(theta, c, x, y, z):
        a = (((c * l) ** 2 + 4 * c * l) ** 0.5 - l * c) / 2
        y = (1 - a) * x + a * z
        c = c * (1 - a)
        g = jax.grad(loss_fn)(y)
        theta = theta - a / c / known_total * g
        z = marginal_oracle(theta, known_total, mesh)
        x = (1 - a) * x + a * z
        return theta, c, x, y, z

    # If we remove jit from marginal oracle, then we'll need to wrap this in
    # a jitted "init" function.
    x = y = z = marginal_oracle(potentials, known_total, mesh)
    theta = potentials
    for t in range(1, iters + 1):
        theta, c, x, y, z = update(theta, c, x, y, z)
        callback_fn(x)

    return mle_from_marginals(x, known_total)


class _AcceleratedStepSearchState(NamedTuple):
    """State of the step search.

    Attributes:
        x: parameters defining the optimization algorithm (see Roulet and
        d'Aspremont Algorithm 2).
        z: same as x, see ref.
        u: dual variable corresponding to z.
        prev_stepsize: reciprocal of the estimate of the Lipshitz-continuity
        parameter of the gradient of the objective at the previous iteration of
        the algorithm.
        stepsize: reciprocal of the estimate of the Lipshitz-continuity parameter
        of the gradient of the objective at the current iteration of the
        algorithm.
        prev_theta: numerical value decreasing along iterates at the previous
        iteration of the algorithm, see ref.
        accept: whether the step is accepted or not.
        iter_search: iteration count of the search.

    References:
        Nesterov, [Universal Gradient Methods for Convex Optimization
        Problems](https://optimization-online.org/wp-content/uploads/2013/04/3833.pdf)

        Roulet and d'Aspremont, [Sharpness, Restart and
        Acceleration](https://arxiv.org/pdf/1702.03828)
    """

    x: CliqueVector
    z: CliqueVector
    u: CliqueVector
    prev_stepsize: jnp.ndarray | float
    stepsize: jnp.ndarray | float
    prev_theta: jnp.ndarray | float
    accept: jnp.ndarray | bool
    iter_search: jnp.ndarray | int


def _universal_accelerated_method_step_init(
    fun: Callable[[CliqueVector], jnp.ndarray],
    dual_init_params,
    dual_proj: Callable[..., Any],
    max_iter_search: int = 30,
    target_acc: float = 0.0,
    stepsize: float = 1.0,
    norm: int = 2,
    linesearch=True,
) -> tuple[
    _AcceleratedStepSearchState,
    Callable[[_AcceleratedStepSearchState], bool],
    Callable[[_AcceleratedStepSearchState], _AcceleratedStepSearchState],
]:
    """Accelerated first order method adapted to any smoothness.

    Minimizes fun(x) over a constraint set M.

    The algorithm requires an oracle "dual_proj(g)" that computes
    argmin_y <g, y> + h(y)
    s.t. y in M
    where h is a distance generating function.

    This method is inspired from ref 1 and the algorithm is described in
    essentially described in Algorithm 2 of ref 2. One difference is that we
    keep track of the dual variable returned by the dual_proj to avoid mapping
    back and forth between the primal and dual spaces.

    This function provides the initial state and the continuation and body
    functions for the step the method (which searches for a valid stepsize each
    time).

    Args:
        fun: objective to minimize.
        dual_init_params: initial parameters in dual space.
        dual_proj: projection onto some constraint set according to a bregman
        divergence.
        max_iter_search: maximal number of iterations to run the search.
        target_acc: target accuracy of the method. If `fun` is non-smooth, this
        needs to be set > 0. Convergence beyond that target accuracy is not
        guaranteed. If the function is smooth, set `target_acc=0`.
        stepsize: initial estimate of the stepsize.
        norm: type of norm measuring the smoothness of `fun`.
        linesearch: if true, uses linesearch to determine acceptance of step,
        otherwise use constant stepsize given by `stepsize`.

    Returns:
        (init_carry, cond_fun, body_fun) where
        init_carry: initial state of the step search.
        cond_fun: continuation criterion when searching for next step.
        body_fun: step when searching step.

    References:
        1 Nesterov, [Universal Gradient Methods for Convex Optimization
        Problems](https://optimization-online.org/wp-content/uploads/2013/04/3833.pdf)

        2 Roulet and d'Aspremont, [Sharpness, Restart and
        Acceleration](https://arxiv.org/pdf/1702.03828)
    """

    def cond_fun(carry: _AcceleratedStepSearchState) -> bool | jnp.ndarray:
        """Continuation criterion when searching for next step."""
        return jnp.logical_not(
            jnp.logical_or(carry.accept, carry.iter_search >= max_iter_search),
        )

    def body_fun(
        carry: _AcceleratedStepSearchState,
    ) -> _AcceleratedStepSearchState:
        """Step when searching step."""
        # Computes new theta
        prev_theta, prev_smooth_estim = carry.prev_theta, 1 / carry.prev_stepsize
        smooth_estim, stepsize = 1 / carry.stepsize, carry.stepsize
        aux = 1 + 4 * smooth_estim / (prev_theta**2 * prev_smooth_estim)
        new_theta = 2 / (1 + jnp.sqrt(aux))
        # We hardcode the first iteration to be prev_theta=-1
        theta = jnp.where(carry.prev_theta < 0.0, 1.0, new_theta)

        # Computes sequences of params
        y = (1 - theta) * carry.x + theta * carry.z
        value_y, grad_y = jax.value_and_grad(fun)(y)
        u = carry.u - stepsize / theta * grad_y
        z = dual_proj(u)
        x = (1 - theta) * carry.x + theta * z

        # Check condition
        if linesearch:
            new_value = fun(x)
            if norm == 1:
                sq_norm_diff = (
                    optax.tree.norm(optax.tree.sub(x, y), ord=1, squared=True)
                )
            elif norm == 2:
                sq_norm_diff = optax.tree.norm(
                    optax.tree_utils.tree_sub(x, y), ord=2, squared=True
                )
            else:
                raise ValueError(f"norm={norm} not supported")
            taylor_approx = (
                value_y + grad_y.dot(x - y) + 0.5 * smooth_estim * sq_norm_diff
            )
            accept = new_value <= (taylor_approx + 0.5 * target_acc * theta)
            new_stepsize = 1.1 * stepsize
        else:
            accept = True
            new_stepsize = stepsize

        candidate = _AcceleratedStepSearchState(
            x=x,
            z=z,
            u=u,
            prev_stepsize=stepsize,
            stepsize=new_stepsize,
            prev_theta=theta,
            accept=accept,
            iter_search=jnp.asarray(0),
        )
        base = carry._replace(
            stepsize=0.5 * carry.stepsize, iter_search=carry.iter_search + 1
        )
        return jax.tree.map(lambda x, y: jnp.where(accept, x, y), candidate, base)

    x = z = dual_proj(dual_init_params)
    u = dual_init_params
    init_carry = _AcceleratedStepSearchState(
        x=x,
        z=z,
        u=u,
        prev_stepsize=stepsize,
        stepsize=stepsize,
        prev_theta=jnp.asarray(-1.0),
        accept=jnp.asarray(False),
        iter_search=jnp.asarray(0),
    )
    return init_carry, cond_fun, body_fun


def universal_accelerated_method(
    domain: Domain,
    loss_fn: marginal_loss.MarginalLossFn | list[LinearMeasurement],
    *,
    known_total: float | None = None,
    potentials: CliqueVector | None = None,
    marginal_oracle: marginal_oracles.MarginalOracle = marginal_oracles.message_passing_stable,
    iters: int = 1000,
    callback_fn: Callable[[CliqueVector], None] = lambda _: None,
    mesh: jax.sharding.Mesh | None = None,
):
    """Optimization using the Universal Accelerated MD algorithm."""
    loss_fn, known_total, potentials = _initialize(
        domain, loss_fn, known_total, potentials
    )
    marginal_oracle = functools.partial(marginal_oracle, mesh=mesh)

    carry, cond_fun, body_fun = _universal_accelerated_method_step_init(
        fun=loss_fn,
        dual_init_params=potentials,
        dual_proj=lambda x: marginal_oracle(x, known_total),
        max_iter_search=30,
        target_acc=0.0,
        stepsize=1.0 / known_total,
        norm=2,
        linesearch=True,
    )
    for _ in range(iters):
        # jax.lax.while_loop traces the body function, so no need to jit it.
        carry = jax.lax.while_loop(cond_fun, body_fun, carry)
        carry = carry._replace(accept=jnp.asarray(False))
        callback_fn(carry.x)
    sol = carry.x
    return mle_from_marginals(sol, known_total)
