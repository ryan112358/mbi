"""Approximate marginal oracles with convex counting numbers.

See the paper ["Relaxed Marginal Consistency for Differentially Private Query Answering"](https://arxiv.org/pdf/2109.06153) for more details.

This file implements one approximate marginal inference oracle: Convex-GBP
with fixed counting numbers of 1.0 for all regions.  We experimented with
others, but do not officially support them in this library.  If interested,
please see the following snapshot of this repository:

https://github.com/ryan112358/private-pgm/tree/approx-experiments-snapshot

Pull requests are welcome to add support for other approximate oracles.
"""

from __future__ import annotations

import functools
import itertools
from collections.abc import Sequence
from typing import Any, NamedTuple, Protocol

import dataclasses
import jax
import jax.numpy as jnp
import networkx as nx
from scipy.cluster.hierarchy import DisjointSet

from . import estimation
from . import marginal_loss
from .clique_utils import Clique
from .clique_vector import CliqueVector
from .domain import Domain
from .factor import Factor
from .marginal_loss import LinearMeasurement

# pylint: disable


class StatefulMarginalOracle(Protocol):
  """Defines the callable signature for stateful marginal oracle functions.

  A stateful marginal oracle computes (approximate) marginals from
  log-space potentials while also managing an internal state, often
  for optimization in iterative algorithms (e.g., preserving messages
  in message passing).
  """

  def __call__(
      self,
      potentials: CliqueVector,
      total: float = 1.0,
      state: Any = None,
      mesh: jax.sharding.Mesh | None = None,
  ) -> tuple[CliqueVector, Any]:
    """Computes marginals from log-space potentials and manages state.

    Args:
        potentials: A CliqueVector representing the log-space potentials
            of a graphical model.
        total: The normalization factor, typically the total number of
            records or a probability sum. Defaults to 1.0.
        state: An optional argument to pass state between calls.
            The oracle may use this state and return an updated version.
        mesh: Specifies how the computation will be sharded across devices.

    Returns:
        A tuple containing:
            - CliqueVector: The computed marginals.
            - Any: The updated state.
    """
    pass


def build_graph(domain: Domain, cliques: Sequence[Clique]) -> tuple[
    set[Clique],
    list[Clique],
    dict[tuple[Clique, Clique], Factor],
    list[tuple[Clique, Clique]],
    dict[Clique, list[Clique]],
    dict[Clique, list[Clique]],
]:
  """Builds the region graph for convex generalized belief propagation."""
  # Hard-code minimal=True, convex=True
  # Counting numbers = 1 for all regions
  # Alg 11.3 of Koller & Friedman
  regions = set(cliques)
  size = 0
  while len(regions) > size:
    size = len(regions)
    for r1, r2 in itertools.combinations(regions, 2):
      z = tuple(sorted(set(r1) & set(r2)))
      if len(z) > 0 and not z in regions:
        regions.update({z})

  G = nx.DiGraph()
  G.add_nodes_from(regions)
  for r1 in regions:
    for r2 in regions:
      if set(r2) < set(r1) and not any(
          set(r2) < set(r3) and set(r3) < set(r1) for r3 in regions
      ):
        G.add_edge(r1, r2)

  H = G.reverse()
  G1, H1 = nx.transitive_closure(G), nx.transitive_closure(H)

  children = {r: list(G.neighbors(r)) for r in regions}
  parents = {r: list(H.neighbors(r)) for r in regions}
  descendants = {r: list(G1.neighbors(r)) for r in regions}  # pyrefly: ignore[bad-argument-type]
  ancestors = {r: list(H1.neighbors(r)) for r in regions}  # pyrefly: ignore[bad-argument-type]
  forebears = {r: set([r] + ancestors[r]) for r in regions}
  downp = {r: set([r] + descendants[r]) for r in regions}

  min_edges = []
  for r in regions:
    ds = DisjointSet()
    for u in parents[r]:
      ds.add(u)
    for u, v in itertools.combinations(parents[r], 2):
      uv = set(ancestors[u]) & set(ancestors[v])
      if len(uv) > 0:
        ds.merge(u, v)
    canonical = set()
    for u in parents[r]:
      canonical.update({ds[u]})
    min_edges.extend([(u, r) for u in canonical])

  G = nx.DiGraph()
  G.add_nodes_from(regions)
  G.add_edges_from(min_edges)

  H = G.reverse()
  G1, H1 = nx.transitive_closure(G), nx.transitive_closure(H)

  children = {r: list(G.neighbors(r)) for r in regions}
  parents = {r: list(H.neighbors(r)) for r in regions}

  messages = {}
  message_order = []
  for ru in sorted(regions, key=len):
    for rd in children[ru]:
      message_order.append((ru, rd))
      messages[ru, rd] = Factor.zeros(domain.project(rd))
      messages[rd, ru] = Factor.zeros(
          domain.project(rd)
      )  # only for hazan et al

  return regions, list(cliques), messages, message_order, parents, children


_State = dict[tuple[Clique, Clique], Factor]


@functools.partial(jax.jit, static_argnames=["mesh", "iters"])
def convex_generalized_belief_propagation(
    potentials: CliqueVector,
    total: float = 1,
    state: _State | None = None,
    mesh: jax.sharding.Mesh | None = None,
    iters: int = 1,
    damping: float = 0.5,
) -> tuple[CliqueVector, _State]:
  """Convex generalized belief propagation for approximmate marginal inference.

      The algorithms implements the Algorithm 2 in our paper
      ["Relaxed Marginal Consistency for Differentially Private Query Answering"](https://arxiv.org/pdf/2109.06153), which itself is based on the paper titled
      ["Tightening Fractional Covering Upper Bounds on the Partition
  Function for High-Order Region Graphs"](https://arxiv.org/pdf/1210.4881).

  Args:
      potentials: A CliqueVector object containing the potentials of the graphical model.
      total: The total number of records in the dataset.
      state: The state of the message passing algorithm (i.e., the messages).  Useful when
          calling this within an iterative procedure for warm starting purposes.
      mesh: Specifies how the computation will be sharded across machines.
      iters: The number of iterations to run the algorithm.
      damping: The damping factor for the messages.

  Returns:
      A CliqueVector of pseudo-marginals for the cliques in the graphical model.
  """
  potentials = potentials.apply_sharding(mesh)
  domain, cliques = potentials.domain, potentials.cliques
  # We might need or want a sharding constraint on messages here
  regions, cliques, messages, message_order, parents, children = build_graph(
      domain, cliques
  )
  if state is not None:
    messages = state

  # Hardcode assumption that counting numbers are 1.0 for all regions.
  pot = potentials.expand(tuple(regions))

  cc = {}
  for r in regions:
    for p in parents[r]:
      cc[p, r] = 1 / (1 + len(parents[r]))

  for _ in range(iters):
    new = {}
    for r in regions:
      for p in parents[r]:
        new[p, r] = (
            (
                pot[p]
                + sum(messages[c, p] for c in children[p] if c != r)
                - sum(messages[p, p1] for p1 in parents[p])
            )
            .project(r, log=True)
            .normalize(log=True)
            .apply_sharding(mesh)
        )

    for r in regions:
      for p in parents[r]:
        new[r, p] = (
            (
                cc[p, r]
                * (
                    pot[r]
                    + sum(messages[c, r] for c in children[r])
                    + sum(messages[p1, r] for p1 in parents[r])
                )
                - messages[p, r]
            )
            .normalize(log=True)
            .apply_sharding(mesh)
        )

    # Damping is not described in paper, but is needed to get convergence for dense graphs
    rho = damping
    for p in regions:
      for r in children[p]:
        messages[p, r] = rho * messages[p, r] + (1.0 - rho) * new[p, r]
        messages[r, p] = rho * messages[r, p] + (1.0 - rho) * new[r, p]

  mu = {}
  for r in cliques:
    mu[r] = (
        (
            pot[r]
            + sum(messages[c, r] for c in children[r])
            - sum(messages[r, p] for p in parents[r])
        )
        .normalize(total, log=True)
        .exp()
        .apply_sharding(mesh)
    )

  return CliqueVector(domain, cliques, mu), messages


class ApproxMirrorDescentState(NamedTuple):
  """State for mirror descent with approximate marginal inference."""

  potentials: CliqueVector
  mu: CliqueVector
  messages: Any


@dataclasses.dataclass(frozen=True)
class ApproxMirrorDescent:
  """Mirror descent estimator using approximate marginal inference.

  Uses ``convex_generalized_belief_propagation`` as the marginal oracle and
  warm-starts messages between optimization iterations.  Unlike the exact
  mirror descent in ``estimation.py``, this does not produce a
  ``MarkovRandomField`` because approximate region graphs cannot generate
  synthetic data.

  This class holds optimizer configuration only.  All data (loss function,
  potentials, totals) is passed to methods as arguments.

  Example::

      estimator = ApproxMirrorDescent(stepsize=1.0)
      mu = estimator.estimate(domain, measurements)

  Attributes:
      stepsize: Fixed step size (required; no line search).
      oracle_iters: Belief propagation iterations per optimization step.
      damping: Damping factor for belief propagation messages.
      mesh: JAX sharding mesh.
  """

  stepsize: float
  oracle_iters: int = 1
  damping: float = 0.5
  mesh: jax.sharding.Mesh | None = None

  def _init(
      self,
      potentials: CliqueVector,
      total: float,
  ) -> ApproxMirrorDescentState:
    """Initialize the optimization state."""
    # Initialize messages so jit sees a consistent pytree structure.
    mu, messages = convex_generalized_belief_propagation(
        potentials,
        total,
        state=None,
        mesh=self.mesh,
        iters=self.oracle_iters,
        damping=self.damping,
    )
    return ApproxMirrorDescentState(potentials, mu, messages)

  @functools.partial(jax.jit, static_argnames=["self"])
  def _step(
      self,
      state: ApproxMirrorDescentState,
      total: jax.Array | float,
      loss_fn: marginal_loss.MarginalLossFn,
  ) -> ApproxMirrorDescentState:
    """Perform a single mirror descent step."""
    mu, messages = convex_generalized_belief_propagation(
        state.potentials,
        total,
        state=state.messages,
        mesh=self.mesh,
        iters=self.oracle_iters,
        damping=self.damping,
    )
    dL = jax.grad(loss_fn)(mu)
    potentials = state.potentials - self.stepsize * dL
    return ApproxMirrorDescentState(potentials, mu, messages)

  def estimate(
      self,
      domain: Domain,
      loss_fn: marginal_loss.MarginalLossFn | list[LinearMeasurement],
      *,
      known_total: float | None = None,
      potentials: CliqueVector | None = None,
      iters: int = 1000,
      callback_fn=lambda _: None,
  ) -> CliqueVector:
    """Run mirror descent and return pseudo-marginals.

    Args:
        domain: The domain over which the model is defined.
        loss_fn: A ``MarginalLossFn`` or a list of ``LinearMeasurement``.
        known_total: The known or estimated number of records.
        potentials: Initial potentials.
        iters: Number of optimization iterations.
        callback_fn: Called with pseudo-marginals at each iteration.

    Returns:
        Pseudo-marginals as a ``CliqueVector``.
    """
    loss_fn, known_total, potentials = estimation._initialize(
        domain, loss_fn, known_total, potentials
    )
    state = self._init(potentials, known_total)

    for _ in range(iters):
      state = self._step(state, known_total, loss_fn)
      callback_fn(state.mu)

    # Final oracle call with warm-started messages.
    mu, _ = convex_generalized_belief_propagation(
        state.potentials,
        known_total,
        state=state.messages,
        mesh=self.mesh,
        iters=self.oracle_iters,
        damping=self.damping,
    )
    return mu

  def precompile(
      self,
      domain: Domain,
      measurements: list[LinearMeasurement] | None = None,
      *,
      extra_cliques: list[tuple[str, ...]] | None = None,
  ) -> None:
    """Warm up the JIT cache for ``estimate``.

    Args:
        domain: The domain over which the model is defined.
        measurements: Optional list of ``LinearMeasurement`` objects.
        extra_cliques: Optional additional cliques to compile for.
    """
    all_measurements = list(measurements or [])
    for cl in extra_cliques or []:
      shape = (domain.project(cl).size(),)
      abstract_values = jax.ShapeDtypeStruct(shape, jnp.float32)
      # ``abstract_values`` is a ShapeDtypeStruct used only for JIT shape
      # inference via jax.eval_shape below; no concrete array is needed.
      all_measurements.append(LinearMeasurement(abstract_values, cl))  # pyrefly: ignore[bad-argument-type]
    all_measurements = jax.eval_shape(lambda x: x, all_measurements)

    loss_fn = marginal_loss.from_linear_measurements(all_measurements, domain)
    potentials = CliqueVector.abstract(domain, loss_fn.cliques)

    # Use eval_shape to get the abstract state pytree without
    # running any computation.
    abstract_state = jax.eval_shape(self._init, potentials, 1.0)

    # lower().compile() populates the jit cache without executing.
    self._step.lower(self, abstract_state, 1.0, loss_fn).compile()


def mirror_descent(
    domain: Domain,
    loss_fn,
    *,
    known_total: float | None = None,
    potentials: CliqueVector | None = None,
    stepsize: float,
    iters: int = 1000,
    oracle_iters: int = 1,
    damping: float = 0.5,
    callback_fn=lambda _: None,
    mesh=None,
) -> CliqueVector:
  """Mirror descent with approximate marginal inference.

  Convenience wrapper around :class:`ApproxMirrorDescent`.

  Args:
      domain: The domain over which the model should be defined.
      loss_fn: A ``MarginalLossFn`` or a list of ``LinearMeasurement``.
      known_total: The known or estimated number of records.
      potentials: Initial potentials.
      stepsize: Fixed step size (required; no line search).
      iters: Number of optimization iterations.
      oracle_iters: Belief propagation iterations per optimization step.
      damping: Damping factor for belief propagation messages.
      callback_fn: Called with pseudo-marginals at each iteration.
      mesh: JAX sharding mesh.

  Returns:
      Pseudo-marginals as a ``CliqueVector``.
  """
  estimator = ApproxMirrorDescent(
      stepsize=stepsize,
      oracle_iters=oracle_iters,
      damping=damping,
      mesh=mesh,
  )
  return estimator.estimate(
      domain,
      loss_fn,
      known_total=known_total,
      potentials=potentials,
      iters=iters,
      callback_fn=callback_fn,
  )
