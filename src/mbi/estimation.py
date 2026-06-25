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

import concurrent.futures
import functools
import math

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, NamedTuple

import attr
import jax
import jax.numpy as jnp
import numpy as np
import optax

from . import marginal_loss, marginal_oracles
from ._api import Model, Projectable  # noqa: F401  # pylint: disable=unused-import
from .clique_vector import CliqueVector
from .domain import Domain
from .factor import Factor
from .marginal_loss import LinearMeasurement, MarginalLossFn
from .markov_random_field import MarkovRandomField

# Shared thread pool for background JIT compilation.
_COMPILE_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=2)
CALLBACK_EVERY = 50


class Estimator(ABC):
    """An object that estimates a Model from a marginal-based loss function.

    Subclasses implement ``_init``, ``_step``, and ``_finalize``.  The ABC
    provides a default ``estimate`` loop and a shared asynchronous
    ``precompile`` that works for any estimator with a jitted ``_step``.

    **State convention:** the first element of the state tuple returned by
    ``_init`` / ``_step`` is the current solution (passed to ``callback_fn``).
    Override ``_callback_value`` only if the callback needs a transformation.

    Examples of subclasses:
        * ``MirrorDescent``
        * ``DualAveraging``
        * ``InteriorGradient``
        * ``extensions.MixtureOfProductsEstimator``
        * ``extensions.ReweightedDatasetEstimator``
    """

    marginal_oracle: marginal_oracles.MarginalOracle | None

    # ------------------------------------------------------------------
    # Abstract interface — subclasses must implement
    # ------------------------------------------------------------------

    @abstractmethod
    def _init(
        self,
        domain: Domain,
        loss_fn: MarginalLossFn,
        known_total: float,
        **kwargs: Any,
    ) -> Any:
        """Initialize the optimization state."""

    @abstractmethod
    def _step(
        self,
        state: Any,
        loss_fn: MarginalLossFn,
        known_total: float,
    ) -> Any:
        """Perform one optimization step (or one scan block)."""

    @abstractmethod
    def _finalize(self, state: Any, known_total: float) -> Model:
        """Convert the final optimization state into a Model."""

    # ------------------------------------------------------------------
    # Overridable hooks
    # ------------------------------------------------------------------

    def _callback_value(self, state: Any, known_total: float) -> Any:  # pylint: disable=unused-argument
        """Extract the value to pass to ``callback_fn``.

        Default: ``state[0]`` (first element of the state tuple).
        """
        return state[0]

    def _oracle(self, cliques, domain):
        """Return the marginal oracle, falling back to ``default_oracle``."""
        return self.marginal_oracle or marginal_oracles.default_oracle(
            cliques, domain
        )

    # ------------------------------------------------------------------
    # Default implementations
    # ------------------------------------------------------------------

    def estimate(
        self,
        domain: Domain,
        loss_fn: MarginalLossFn | list[LinearMeasurement],
        *,
        known_total: float | None = None,
        iters: int = 1000,
        callback_fn: Callable | None = None,
        **kwargs: Any,
    ) -> Model:
        """Estimate a Model from noisy marginal measurements."""
        if isinstance(loss_fn, list):
            if known_total is None:
                known_total = minimum_variance_unbiased_total(loss_fn)
            loss_fn = marginal_loss.from_linear_measurements(loss_fn, domain)
        if known_total is None:
            known_total = 1.0

        # Nothing to optimize when there are no cliques.
        if not loss_fn.cliques:
            potentials = CliqueVector.zeros(domain, ())
            oracle = self._oracle((), domain)
            return MarkovRandomField(
                potentials=potentials,
                marginals=oracle(potentials, known_total),
                total=known_total,
            )

        state = self._init(domain, loss_fn, known_total, **kwargs)
        for _ in range(math.ceil(iters / CALLBACK_EVERY)):
            state = self._multi_step(state, loss_fn, known_total)
            if callback_fn is not None:
                callback_fn(self._callback_value(state, known_total))
        return self._finalize(state, known_total)

    @jax.jit(static_argnames=["self"])
    def _multi_step(self, state, loss_fn, known_total):
        """Run ``CALLBACK_EVERY`` optimization steps as a fused scan."""

        def step(s, _):
            return self._step(s, loss_fn, known_total), None

        return jax.lax.scan(step, state, None, length=CALLBACK_EVERY)[0]

    def precompile(
        self,
        domain: Domain,
        measurements: list[LinearMeasurement] | None = None,
        *,
        extra_cliques: list[tuple[str, ...]] | None = None,
    ) -> concurrent.futures.Future:
        """Warm up the JIT cache for ``estimate`` asynchronously.

        Returns a ``Future`` that completes when compilation finishes.
        Callers may ignore the return value (fire-and-forget) or call
        ``future.result()`` to block until compilation is done.
        """
        all_measurements = list(measurements or [])
        for cl in extra_cliques or []:
            shape = (domain.project(cl).size(),)
            abstract_values = jax.ShapeDtypeStruct(shape, jnp.float32)
            all_measurements.append(
                marginal_loss.LinearMeasurement(abstract_values, cl)
            )

        loss_fn = marginal_loss.from_linear_measurements(
            all_measurements, domain
        )
        abstract_state = jax.eval_shape(
            functools.partial(self._init, domain), loss_fn, 1.0
        )
        lowered = self._multi_step.lower(self, abstract_state, loss_fn, 1.0)
        return _COMPILE_POOL.submit(lowered.compile)


def minimum_variance_unbiased_total(
    measurements: list[LinearMeasurement],
) -> float:
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
        except Exception:  # pylint: disable=broad-exception-caught
            continue
    estimates, variances = np.array(estimates), np.array(variances)
    if len(estimates) == 0:
        return 1.0
    else:
        weights = 1.0 / variances
        return max(1, float(np.average(estimates, weights=weights)))


def _initialize(domain, loss_fn, known_total, potentials):
    """Normalize inputs for estimation functions.

    Converts a list of ``LinearMeasurement`` objects to a ``MarginalLossFn``,
    estimates the total if not given, and creates zero potentials if needed.
    """
    if isinstance(loss_fn, list):
        if known_total is None:
            known_total = minimum_variance_unbiased_total(loss_fn)
        loss_fn = marginal_loss.from_linear_measurements(loss_fn, domain)

    if known_total is None:
        known_total = 1.0

    if potentials is not None:
        potentials = potentials.expand(loss_fn.cliques)
    else:
        potentials = CliqueVector.zeros(domain, loss_fn.cliques)

    return loss_fn, known_total, potentials


# ---------------------------------------------------------------------------
# State NamedTuples (module-level; they are JAX pytrees)
# ---------------------------------------------------------------------------


class MirrorDescentState(NamedTuple):
    """State for Algorithm 1 of https://arxiv.org/pdf/1901.09136."""

    mu: CliqueVector
    potentials: CliqueVector
    alpha: jax.Array | float
    loss: jax.Array | float


class DualAveragingState(NamedTuple):
    """State for Regularized Dual Averaging (https://proceedings.neurips.cc/paper_files/paper/2009/file/7cce53cf90577442771720a370c3c723-Paper.pdf)."""

    w: CliqueVector
    v: CliqueVector
    gbar: CliqueVector
    loss: jax.Array | float
    lipschitz: jax.Array | float
    gamma: jax.Array | float
    t: jax.Array | int


class InteriorGradientState(NamedTuple):
    """State for Interior Gradient (https://doi.org/10.1137/S1052623403427823)."""

    x: CliqueVector
    potentials: CliqueVector
    c: jax.Array | float
    y: CliqueVector
    z: CliqueVector
    loss: jax.Array | float
    inv_lipschitz: jax.Array | float


class LBFGSState(NamedTuple):
    """State for L-BFGS optimization on potentials."""

    potentials: CliqueVector
    opt_state: Any


class _AcceleratedStepSearchState(NamedTuple):
    """State for Universal Accelerated Method step search.

    References:
        Nesterov, `Universal Gradient Methods for Convex Optimization
        Problems <https://optimization-online.org/wp-content/uploads/2013/04/3833.pdf>`_

        Roulet and d'Aspremont, `Sharpness, Restart and
        Acceleration <https://arxiv.org/pdf/1702.03828>`_
    """

    x: CliqueVector
    z: CliqueVector
    u: CliqueVector
    prev_stepsize: jax.Array | float
    stepsize: jax.Array | float
    prev_theta: jax.Array | float
    accept: jax.Array | bool
    iter_search: jax.Array | int


# ---------------------------------------------------------------------------
# Class-based estimators
# ---------------------------------------------------------------------------


@attr.dataclass(frozen=True)
class MirrorDescent(Estimator):
    """Mirror descent estimator for graphical models.

    This is a first-order proximal optimization algorithm for solving
    a (possibly nonsmooth) convex optimization problem over the marginal polytope.
    This is an implementation of Algorithm 1 from the paper
    `"Graphical-model based estimation and inference for differential privacy"
    <https://arxiv.org/pdf/1901.09136>`_.

    Attributes:
        stepsize: Fixed step size, or ``None`` (default) to use Armijo line
            search.
        marginal_oracle: The function to compute marginals from potentials.
            If ``None`` (default), uses ``default_oracle()`` to auto-select.
        mesh: JAX sharding mesh.
    """

    stepsize: float | None = None
    marginal_oracle: marginal_oracles.MarginalOracle | None = None
    mesh: jax.sharding.Mesh | None = None

    def _init(
        self,
        domain: Domain,
        loss_fn: marginal_loss.MarginalLossFn,
        known_total: float,
        *,
        potentials: CliqueVector | None = None,
    ) -> MirrorDescentState:
        """Initialize the optimization state."""
        if potentials is None:
            potentials = CliqueVector.zeros(domain, loss_fn.cliques)
        else:
            potentials = potentials.expand(loss_fn.cliques)
        marginal_oracle = self._oracle(loss_fn.cliques, domain)
        # Theory suggests the initial learning rate should be inversely
        # proportional to L. We also divide by scaling factor to account for
        # the fact that gradients are scaled up by a factor of known_total.
        # See Eq 75. of https://www.cs.uic.edu/~zhangx/teaching/bregman.pdf.
        L = loss_fn.lipschitz or 1.0
        alpha = (
            2.0 / (L * known_total) if self.stepsize is None else self.stepsize
        )
        mu = marginal_oracle(potentials, known_total)
        initial_loss = loss_fn(mu)
        return MirrorDescentState(mu, potentials, alpha, initial_loss)

    def _step(
        self,
        state: MirrorDescentState,
        loss_fn: marginal_loss.MarginalLossFn,
        known_total: jax.Array | float,
    ) -> MirrorDescentState:
        """Perform a single mirror descent step."""
        marginal_oracle = self._oracle(loss_fn.cliques, state.potentials.domain)
        mu = marginal_oracle(state.potentials, known_total)
        loss, dL = jax.value_and_grad(loss_fn)(mu)
        theta2 = state.potentials - state.alpha * dL

        if self.stepsize is not None:
            # Fixed step size — no line search.
            return MirrorDescentState(mu, theta2, state.alpha, loss)

        # Armijo line search.
        mu2 = marginal_oracle(theta2, known_total)
        loss2 = loss_fn(mu2)

        sufficient_decrease = loss - loss2 >= 0.5 * state.alpha * dL.dot(
            mu - mu2
        )
        alpha = jax.lax.select(
            sufficient_decrease, 1.01 * state.alpha, 0.5 * state.alpha
        )
        potentials = jax.lax.cond(
            sufficient_decrease, lambda: theta2, lambda: state.potentials
        )
        accepted_loss = jax.lax.select(sufficient_decrease, loss2, loss)
        return MirrorDescentState(mu, potentials, alpha, accepted_loss)

    def _finalize(
        self,
        state: MirrorDescentState,
        known_total: float,
    ) -> MarkovRandomField:
        marginal_oracle = self._oracle(
            state.potentials.cliques, state.potentials.domain
        )
        marginals = marginal_oracle(state.potentials, known_total)
        return MarkovRandomField(
            potentials=state.potentials,
            marginals=marginals,
            total=known_total,
        )


@attr.dataclass(frozen=True)
class DualAveraging(Estimator):
    """Regularized Dual Averaging estimator for graphical models.

    RDA is an accelerated proximal algorithm for solving a smooth convex
    optimization problem over the marginal polytope.  This algorithm requires
    knowledge of the Lipschitz constant of the gradient of the loss function.

    Attributes:
        marginal_oracle: The function to compute marginals from potentials.
            If ``None`` (default), uses ``default_oracle()`` to auto-select.
        mesh: JAX sharding mesh.
    """

    marginal_oracle: marginal_oracles.MarginalOracle | None = None
    mesh: jax.sharding.Mesh | None = None

    def _init(
        self,
        domain: Domain,
        loss_fn: marginal_loss.MarginalLossFn,
        known_total: float,
        *,
        potentials: CliqueVector | None = None,
    ) -> DualAveragingState:
        """Initialize the optimization state."""
        if potentials is None:
            potentials = CliqueVector.zeros(domain, loss_fn.cliques)
        else:
            potentials = potentials.expand(loss_fn.cliques)
        marginal_oracle = self._oracle(loss_fn.cliques, domain)

        D = np.sqrt(
            domain.size() * np.log(domain.size())
        )  # upper bound on entropy
        Q = 0  # upper bound on variance of stochastic gradients
        gamma = Q / D
        L = (loss_fn.lipschitz or 1.0) / known_total

        w = v = marginal_oracle(potentials, known_total)
        gbar = CliqueVector.zeros(domain, loss_fn.cliques)
        initial_loss = loss_fn(w)
        return DualAveragingState(w, v, gbar, initial_loss, L, gamma, 1)

    def _step(
        self,
        state: DualAveragingState,
        loss_fn: marginal_loss.MarginalLossFn,
        known_total: jax.Array | float,
    ) -> DualAveragingState:
        """Perform a single dual averaging step."""
        marginal_oracle = self._oracle(loss_fn.cliques, state.w.domain)
        t = state.t
        c = 2.0 / (t + 1)
        beta = state.gamma * (t + 1) ** 1.5 / 2
        u = (1 - c) * state.w + c * state.v
        loss, g = jax.value_and_grad(loss_fn)(u)
        g = g / known_total
        gbar = (1 - c) * state.gbar + c * g
        theta = -t * (t + 1) / (4 * state.lipschitz + beta) * gbar
        v = marginal_oracle(theta, known_total)
        w = (1 - c) * state.w + c * v
        return DualAveragingState(
            w, v, gbar, loss, state.lipschitz, state.gamma, t + 1
        )

    def _finalize(
        self,
        state: DualAveragingState,
        known_total: float,
    ) -> MarkovRandomField:
        loss = marginal_loss.mle_loss_fn(state.w)
        return LBFGS(
            marginal_oracle=self.marginal_oracle,
        ).estimate(state.w.domain, loss, known_total=known_total)


@attr.dataclass(frozen=True)
class InteriorGradient(Estimator):
    """Interior Gradient estimator for graphical models.

    Interior Gradient is an accelerated proximal algorithm for solving a smooth
    convex optimization problem over the marginal polytope.  This algorithm
    requires knowledge of the Lipschitz constant of the gradient of the loss
    function.  Based on the paper
    `"Interior Gradient and Proximal Methods for Convex and Conic Optimization"
    <https://epubs.siam.org/doi/abs/10.1137/S1052623403427823>`_.

    Attributes:
        marginal_oracle: The function to compute marginals from potentials.
            If ``None`` (default), uses ``default_oracle()`` to auto-select.
        mesh: JAX sharding mesh.
    """

    marginal_oracle: marginal_oracles.MarginalOracle | None = None
    mesh: jax.sharding.Mesh | None = None

    def _init(
        self,
        domain: Domain,
        loss_fn: marginal_loss.MarginalLossFn,
        known_total: float,
        *,
        potentials: CliqueVector | None = None,
    ) -> InteriorGradientState:
        """Initialize the optimization state."""
        if potentials is None:
            potentials = CliqueVector.zeros(domain, loss_fn.cliques)
        else:
            potentials = potentials.expand(loss_fn.cliques)
        marginal_oracle = self._oracle(loss_fn.cliques, domain)

        inv_lipschitz = 1.0 / (loss_fn.lipschitz or 1.0)
        x = y = z = marginal_oracle(potentials, known_total)
        initial_loss = loss_fn(x)
        return InteriorGradientState(
            x, potentials, 1.0, y, z, initial_loss, inv_lipschitz
        )

    def _step(
        self,
        state: InteriorGradientState,
        loss_fn: marginal_loss.MarginalLossFn,
        known_total: jax.Array | float,
    ) -> InteriorGradientState:
        """Perform a single interior gradient step."""
        marginal_oracle = self._oracle(loss_fn.cliques, state.potentials.domain)
        l = state.inv_lipschitz
        a = (((state.c * l) ** 2 + 4 * state.c * l) ** 0.5 - l * state.c) / 2
        y = (1 - a) * state.x + a * state.z
        c = state.c * (1 - a)
        loss, g = jax.value_and_grad(loss_fn)(y)
        potentials = state.potentials - a / c / known_total * g
        z = marginal_oracle(potentials, known_total)
        x = (1 - a) * state.x + a * z
        return InteriorGradientState(
            x, potentials, c, y, z, loss, state.inv_lipschitz
        )

    def _finalize(
        self,
        state: InteriorGradientState,
        known_total: float,
    ) -> MarkovRandomField:
        loss = marginal_loss.mle_loss_fn(state.x)
        return LBFGS(
            marginal_oracle=self.marginal_oracle,
        ).estimate(state.x.domain, loss, known_total=known_total)


@attr.dataclass(frozen=True)
class LBFGS(Estimator):
    """L-BFGS estimator for graphical models.

    Optimizes the potentials (theta) directly via L-BFGS, back-propagating
    through the marginal inference oracle.  The loss is convex w.r.t. marginals
    but typically non-convex w.r.t. potentials; in practice, L-BFGS still
    converges well.

    See `"Learning Graphical Model Parameters with Approximate Marginal
    Inference" <https://arxiv.org/abs/1301.3193>`_.

    Attributes:
        marginal_oracle: The function to compute marginals from potentials.
            If ``None`` (default), uses ``default_oracle()`` to auto-select.
    """

    marginal_oracle: marginal_oracles.MarginalOracle | None = None

    def _init(self, domain, loss_fn, known_total, *, potentials=None):
        if potentials is None:
            potentials = CliqueVector.zeros(domain, loss_fn.cliques)
        else:
            potentials = potentials.expand(loss_fn.cliques)
        optimizer = optax.lbfgs(
            memory_size=1,
            linesearch=optax.scale_by_zoom_linesearch(128, max_learning_rate=1),
        )
        opt_state = optimizer.init(potentials)
        return LBFGSState(potentials, opt_state)

    def _step(self, state, loss_fn, known_total):
        marginal_oracle = self._oracle(loss_fn.cliques, state.potentials.domain)
        optimizer = optax.lbfgs(
            memory_size=1,
            linesearch=optax.scale_by_zoom_linesearch(128, max_learning_rate=1),
        )

        def theta_loss(theta):
            return loss_fn(marginal_oracle(theta, known_total))

        loss, grad = jax.value_and_grad(theta_loss)(state.potentials)
        updates, opt_state = optimizer.update(
            grad,
            state.opt_state,
            state.potentials,
            value=loss,
            grad=grad,
            value_fn=theta_loss,
        )
        potentials = optax.apply_updates(state.potentials, updates)
        return LBFGSState(potentials, opt_state)

    def _callback_value(self, state, known_total):
        marginal_oracle = self._oracle(
            state.potentials.cliques, state.potentials.domain
        )
        return marginal_oracle(state.potentials, known_total)

    def _finalize(self, state, known_total):
        marginal_oracle = self._oracle(
            state.potentials.cliques, state.potentials.domain
        )
        marginals = marginal_oracle(state.potentials, known_total)
        return MarkovRandomField(
            potentials=state.potentials,
            marginals=marginals,
            total=known_total,
        )


@attr.dataclass(frozen=True)
class UniversalAcceleratedMethod(Estimator):
    """Universal Accelerated Mirror Descent estimator.

    An accelerated first-order method that adapts to any smoothness level.
    Each optimization step performs an internal line-search via
    ``jax.lax.while_loop``.

    **Numerical stability:** Internally operates on the *unit* simplex
    (total=1) to keep potentials and gradients O(1).  The user-facing loss
    ``loss_fn`` is evaluated on the N-scaled marginals via the wrapper
    ``f_scaled(p) = loss_fn(N * p)``.  This prevents softmax saturation
    that otherwise causes the linesearch to degenerate for large ``N``.

    See `Nesterov (2015) <https://optimization-online.org/wp-content/uploads/2013/04/3833.pdf>`_
    and `Roulet & d'Aspremont (2017) <https://arxiv.org/pdf/1702.03828>`_.

    Attributes:
        marginal_oracle: The function to compute marginals from potentials.
            If ``None`` (default), uses ``default_oracle()`` to auto-select.
        max_iter_search: Max inner line-search iterations per step.
        target_acc: Target accuracy (set > 0 for non-smooth objectives).
        norm: Norm measuring smoothness (1 or 2).
        linesearch: Whether to use adaptive line-search.
    """

    marginal_oracle: marginal_oracles.MarginalOracle | None = None
    max_iter_search: int = 30
    target_acc: float = 0.0
    norm: int = 2
    linesearch: bool = True

    def _init(self, domain, loss_fn, known_total, *, potentials=None):
        if potentials is None:
            potentials = CliqueVector.zeros(domain, loss_fn.cliques)
        else:
            potentials = potentials.expand(loss_fn.cliques)
        marginal_oracle = self._oracle(loss_fn.cliques, domain)
        # Project onto unit simplex (total=1) for numerical stability.
        x = z = marginal_oracle(potentials, 1.0)
        # f_scaled has Hessian ~ N²·H_f; with H_f ~ 1/N we get L ~ N,
        # so initial stepsize ~ 1/N.
        stepsize = 1.0 / known_total
        return _AcceleratedStepSearchState(
            x=x,
            z=z,
            u=potentials,
            prev_stepsize=stepsize,
            stepsize=stepsize,
            prev_theta=jnp.asarray(-1.0),
            accept=jnp.asarray(False),
            iter_search=jnp.asarray(0),
        )

    def _step(self, state, loss_fn, known_total):
        marginal_oracle = self._oracle(loss_fn.cliques, state.x.domain)

        # Project onto unit simplex (total=1).
        def dual_proj(u):
            return marginal_oracle(u, 1.0)

        max_iter_search = self.max_iter_search
        target_acc = self.target_acc
        norm = self.norm
        use_linesearch = self.linesearch

        # Wrap loss to operate on unit simplex: f_scaled(p) = loss_fn(N*p).
        def scaled_loss(p):
            return loss_fn(p * known_total)

        def cond_fun(carry):
            return jnp.logical_not(
                jnp.logical_or(
                    carry.accept, carry.iter_search >= max_iter_search
                ),
            )

        def body_fun(carry):
            prev_theta = carry.prev_theta
            prev_smooth_estim = 1 / carry.prev_stepsize
            smooth_estim, stepsize = 1 / carry.stepsize, carry.stepsize
            aux = 1 + 4 * smooth_estim / (prev_theta**2 * prev_smooth_estim)
            new_theta = 2 / (1 + jnp.sqrt(aux))
            theta = jnp.where(prev_theta < 0.0, 1.0, new_theta)

            y = (1 - theta) * carry.x + theta * carry.z
            value_y, grad_y = jax.value_and_grad(scaled_loss)(y)
            u = carry.u - stepsize / theta * grad_y
            z = dual_proj(u)
            x = (1 - theta) * carry.x + theta * z

            if use_linesearch:
                new_value = scaled_loss(x)
                if norm == 1:
                    sq_norm_diff = optax.tree.norm(
                        optax.tree.sub(x, y), ord=1, squared=True
                    )
                elif norm == 2:
                    sq_norm_diff = optax.tree.norm(
                        optax.tree_utils.tree_sub(x, y),
                        ord=2,
                        squared=True,
                    )
                else:
                    raise ValueError(f"norm={norm} not supported")
                taylor_approx = (
                    value_y
                    + grad_y.dot(x - y)
                    + 0.5 * smooth_estim * sq_norm_diff
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
                stepsize=0.5 * carry.stepsize,
                iter_search=carry.iter_search + 1,
            )
            return jax.tree.map(
                lambda a, b: jnp.where(accept, a, b), candidate, base
            )

        carry = jax.lax.while_loop(cond_fun, body_fun, state)
        return carry._replace(accept=jnp.asarray(False))

    def _callback_value(self, state, known_total):
        # Scale back to N-simplex for user-facing callbacks.
        return state.x * known_total

    def _finalize(self, state, known_total):
        # Scale back to N-simplex and recover potentials via MLE.
        marginals = state.x * known_total
        loss = marginal_loss.mle_loss_fn(marginals)
        return LBFGS(
            marginal_oracle=self.marginal_oracle,
        ).estimate(marginals.domain, loss, known_total=known_total)
