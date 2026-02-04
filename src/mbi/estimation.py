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

import attr
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
        **kwargs: Any,
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


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=["theta", "alpha", "mu", "loss", "oracle_state"],
    meta_fields=["stepsize_cfg"],
)
@attr.dataclass(frozen=True)
class _MirrorDescentState:
    theta: CliqueVector
    alpha: float
    mu: CliqueVector
    loss: float
    oracle_state: Any
    stepsize_cfg: float  # < 0 means None (use line search)

    @functools.partial(jax.jit, static_argnums=(1, 2))
    def update_fn(self, marginal_oracle, loss_fn):
        theta, alpha, state = self.theta, self.alpha, self.oracle_state
        # marginal_oracle should be (theta, state) -> (mu, state)
        mu, state = marginal_oracle(theta, state)
        loss, dL = jax.value_and_grad(loss_fn)(mu)

        theta2 = theta - alpha * dL

        def fixed_step():
            return theta2, loss, alpha, mu, state

        def line_search():
             mu2, _ = marginal_oracle(theta2, state)
             loss2 = loss_fn(mu2)
             sufficient_decrease = loss - loss2 >= 0.5 * alpha * dL.dot(mu - mu2)

             new_alpha = jax.lax.select(sufficient_decrease, 1.01 * alpha, 0.5 * alpha)
             new_theta = jax.lax.cond(sufficient_decrease, lambda: theta2, lambda: theta)
             new_loss = jax.lax.select(sufficient_decrease, loss2, loss)
             # Note: we return mu from the beginning of step
             return new_theta, new_loss, new_alpha, mu, state

        if self.stepsize_cfg >= 0.0:
            new_theta, new_loss, new_alpha, new_mu, new_state = fixed_step()
        else:
            new_theta, new_loss, new_alpha, new_mu, new_state = line_search()

        return attr.evolve(
            self,
            theta=new_theta,
            alpha=new_alpha,
            mu=new_mu,
            loss=new_loss,
            oracle_state=new_state
        )


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
        callback_fn: A function to call at each iteration with the current marginals.
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

    bound_oracle = lambda theta, state: marginal_oracle(theta, known_total, state)

    alpha = 2.0 / known_total if stepsize is None else stepsize
    mu, state = bound_oracle(potentials, None)

    state_obj = _MirrorDescentState(
        theta=potentials,
        alpha=alpha,
        mu=mu,
        loss=0.0,
        oracle_state=state,
        stepsize_cfg=-1.0 if stepsize is None else stepsize
    )

    for t in range(iters):
        state_obj = state_obj.update_fn(bound_oracle, loss_fn)
        callback_fn(state_obj.mu)

    marginals, _ = bound_oracle(state_obj.theta, state_obj.oracle_state)
    return MarkovRandomField(
        potentials=state_obj.theta, marginals=marginals, total=known_total
    )


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=["params", "opt_state", "loss"],
    meta_fields=[],
)
@attr.dataclass(frozen=True)
class _LBFGSState:
    params: CliqueVector
    opt_state: Any
    loss: float

    @functools.partial(jax.jit, static_argnums=(1, 2))
    def update_fn(self, marginal_oracle, loss_fn):
        # loss_fn here is loss_and_grad_fn(theta) -> (loss, grad)
        optimizer = optax.lbfgs(
            memory_size=1,
            linesearch=optax.scale_by_zoom_linesearch(128, max_learning_rate=1),
        )

        def value_fn(theta):
            return loss_fn(theta)[0]

        params, opt_state = self.params, self.opt_state
        loss, grad = loss_fn(params)

        updates, new_opt_state = optimizer.update(
            grad, opt_state, params, value=loss, grad=grad, value_fn=value_fn
        )

        new_params = optax.apply_updates(params, updates)

        return attr.evolve(
            self,
            params=new_params,
            opt_state=new_opt_state,
            loss=loss
        )


def _optimize(loss_and_grad_fn, params, iters=250, callback_fn=lambda _: None):
    """Runs an optimization loop using Optax L-BFGS."""
    optimizer = optax.lbfgs(
        memory_size=1,
        linesearch=optax.scale_by_zoom_linesearch(128, max_learning_rate=1),
    )
    opt_state = optimizer.init(params)

    state_obj = _LBFGSState(
        params=params,
        opt_state=opt_state,
        loss=float("inf")
    )

    for t in range(iters):
        state_obj = state_obj.update_fn(None, loss_and_grad_fn)
        callback_fn(state_obj.params)

    return state_obj.params


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
      callback_fn: A function to call at each iteration with the current marginals.
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


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=["w", "v", "gbar", "t"],
    meta_fields=["gamma", "L", "known_total"],
)
@attr.dataclass(frozen=True)
class _DualAveragingState:
    w: CliqueVector
    v: CliqueVector
    gbar: CliqueVector
    t: int
    gamma: float
    L: float
    known_total: float

    @functools.partial(jax.jit, static_argnums=(1, 2))
    def update_fn(self, marginal_oracle, loss_fn):
        w, v, gbar, t = self.w, self.v, self.gbar, self.t
        gamma, L, known_total = self.gamma, self.L, self.known_total

        c = 2.0 / (t + 1)
        beta = gamma * (t + 1) ** 1.5 / 2

        u = (1 - c) * w + c * v
        g = jax.grad(loss_fn)(u) / known_total
        gbar = (1 - c) * gbar + c * g
        theta = -t * (t + 1) / (4 * L + beta) * gbar

        v = marginal_oracle(theta)
        w = (1 - c) * w + c * v

        return attr.evolve(self, w=w, v=v, gbar=gbar, t=t+1)


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

    bound_oracle = lambda theta: marginal_oracle(theta, known_total, mesh)

    w = v = bound_oracle(potentials)
    gbar = CliqueVector.zeros(domain, loss_fn.cliques)

    state = _DualAveragingState(
        w=w, v=v, gbar=gbar, t=1, gamma=gamma, L=L, known_total=known_total
    )

    for t in range(iters):
        state = state.update_fn(bound_oracle, loss_fn)
        callback_fn(state.w)

    return mle_from_marginals(state.w, known_total)


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=["theta", "c", "x", "y", "z"],
    meta_fields=["l", "known_total"],
)
@attr.dataclass(frozen=True)
class _InteriorGradientState:
    theta: CliqueVector
    c: float
    x: CliqueVector
    y: CliqueVector
    z: CliqueVector
    l: float
    known_total: float

    @functools.partial(jax.jit, static_argnums=(1, 2))
    def update_fn(self, marginal_oracle, loss_fn):
        theta, c, x, y, z = self.theta, self.c, self.x, self.y, self.z
        l, known_total = self.l, self.known_total

        a = (((c * l) ** 2 + 4 * c * l) ** 0.5 - l * c) / 2
        y = (1 - a) * x + a * z
        c = c * (1 - a)
        g = jax.grad(loss_fn)(y)
        theta = theta - a / c / known_total * g
        z = marginal_oracle(theta)
        x = (1 - a) * x + a * z

        return attr.evolve(self, theta=theta, c=c, x=x, y=y, z=z)


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
    c = 1.0
    sigma = 1.0
    l = sigma / loss_fn.lipschitz

    bound_oracle = lambda theta: marginal_oracle(theta, known_total, mesh)

    x = y = z = bound_oracle(potentials)
    theta = potentials

    state = _InteriorGradientState(
        theta=theta, c=c, x=x, y=y, z=z, l=l, known_total=known_total
    )

    for t in range(iters):
        state = state.update_fn(bound_oracle, loss_fn)
        callback_fn(state.x)

    return mle_from_marginals(state.x, known_total)


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=["x", "z", "u", "prev_stepsize", "stepsize", "prev_theta", "accept", "iter_search"],
    meta_fields=["target_acc", "norm", "linesearch", "max_iter_search"],
)
@attr.dataclass(frozen=True)
class _UniversalAcceleratedMethodState:
    """State of the step search."""
    x: CliqueVector
    z: CliqueVector
    u: CliqueVector
    prev_stepsize: jnp.ndarray | float
    stepsize: jnp.ndarray | float
    prev_theta: jnp.ndarray | float
    accept: jnp.ndarray | bool
    iter_search: jnp.ndarray | int
    # Configuration
    target_acc: float
    norm: int
    linesearch: bool
    max_iter_search: int

    @functools.partial(jax.jit, static_argnums=(1, 2))
    def update_fn(self, marginal_oracle, loss_fn):
        # marginal_oracle: (theta) -> mu (which corresponds to dual_proj)
        # loss_fn: fun

        # We need to construct the loop functions
        dual_proj = marginal_oracle
        fun = loss_fn

        # Capture config from self (static)
        max_iter_search = self.max_iter_search
        target_acc = self.target_acc
        norm = self.norm
        linesearch = self.linesearch

        def cond_fun(carry: _UniversalAcceleratedMethodState) -> bool | jnp.ndarray:
            """Continuation criterion when searching for next step."""
            return jnp.logical_not(
                jnp.logical_or(carry.accept, carry.iter_search >= max_iter_search),
            )

        def body_fun(
            carry: _UniversalAcceleratedMethodState,
        ) -> _UniversalAcceleratedMethodState:
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
                    sq_norm_diff = optax.tree.norm(
                        optax.tree.sub(x, y), ord=1, squared=True
                    )
                elif norm == 2:
                    sq_norm_diff = optax.tree.norm(
                        optax.tree_utils.tree_sub(x, y), ord=2, squared=True
                    )
                else:
                    # Not raising ValueError inside JIT
                    sq_norm_diff = 0.0 # Should not happen if config is correct

                taylor_approx = (
                    value_y + grad_y.dot(x - y) + 0.5 * smooth_estim * sq_norm_diff
                )
                accept = new_value <= (taylor_approx + 0.5 * target_acc * theta)
                new_stepsize = 1.1 * stepsize
            else:
                accept = True
                new_stepsize = stepsize

            candidate = attr.evolve(
                carry,
                x=x,
                z=z,
                u=u,
                prev_stepsize=stepsize,
                stepsize=new_stepsize,
                prev_theta=theta,
                accept=accept,
                iter_search=jnp.asarray(0),
            )
            base = attr.evolve(
                carry,
                stepsize=0.5 * carry.stepsize, iter_search=carry.iter_search + 1
            )
            return jax.tree.map(lambda x, y: jnp.where(accept, x, y), candidate, base)

        # Run loop
        carry = jax.lax.while_loop(cond_fun, body_fun, self)
        # Reset accept for next outer iteration
        return attr.evolve(carry, accept=jnp.asarray(False))


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

    bound_oracle = lambda x: marginal_oracle(x, known_total)

    # Initialize state
    stepsize = 1.0 / known_total
    dual_init_params = potentials

    # Init logic
    x = z = bound_oracle(dual_init_params)
    u = dual_init_params

    state = _UniversalAcceleratedMethodState(
        x=x,
        z=z,
        u=u,
        prev_stepsize=stepsize,
        stepsize=stepsize,
        prev_theta=jnp.asarray(-1.0),
        accept=jnp.asarray(False),
        iter_search=jnp.asarray(0),
        target_acc=0.0,
        norm=2,
        linesearch=True,
        max_iter_search=30
    )

    for _ in range(iters):
        state = state.update_fn(bound_oracle, loss_fn)
        callback_fn(state.x)

    sol = state.x
    return mle_from_marginals(sol, known_total)
