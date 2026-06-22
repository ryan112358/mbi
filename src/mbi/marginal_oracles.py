"""Marginal oracles for computing marginals from graphical model potentials.

This module provides configurable marginal oracles that compute marginals
from the potentials of a discrete graphical model.  The core abstraction
is ``MessagePassingOracle``, which composes three independent choices:

- **Semiring**: The algebraic operations (log-sum-product for marginals,
  max-sum for MAP, or sum-product for probability-space potentials).
- **MessageSchedule**: How messages are routed on the junction tree
  (HUGIN, SHAFER_SHENOY, or IMPLICIT).
- **Contraction function**: How the inner sum-product is computed
  (only used by the IMPLICIT schedule).

Pre-built oracles are available as module-level names for convenience:
``message_passing_fast``, ``message_passing_stable``, and
``message_passing_shafer_shenoy``.
"""

import collections
import concurrent.futures
import dataclasses
import enum
import functools
import itertools
import math
import string
from collections.abc import Callable
from typing import Protocol

import jax
import jax.numpy as jnp
import networkx as nx

from . import junction_tree
from .clique_utils import clique_mapping
from .clique_vector import CliqueVector
from .domain import Domain
from .einsum import custom_einsum
from .factor import Factor

_EINSUM_LETTERS = list(string.ascii_lowercase) + list(string.ascii_uppercase)


@dataclasses.dataclass(frozen=True)
class Semiring:
    """Defines the algebraic operations for message passing on a junction tree.

    A semiring specifies how factors are combined and how variables are
    marginalized during inference.  The default LOG_SUM_PRODUCT semiring
    works with log-space potentials (combine=add, reduce=logsumexp),
    which is standard for marginal inference.  Other semirings enable
    different inference tasks:

    - MAX_SUM: MAP inference (combine=add, reduce=max)
    - SUM_PRODUCT: Probability-space inference (combine=multiply, reduce=sum)

    Attributes:
        combine_fn: Combines two Factor values element-wise.
            Log-space: jnp.add.  Probability-space: jnp.multiply.
        reduce_fn: Marginalizes a Factor over given attributes.
            Log-space: jax.scipy.special.logsumexp.  Probability-space: jnp.sum.
        log_space: Whether potentials are in log space.
        name: Human-readable name for display/debugging.
    """
    combine_fn: Callable = jnp.add
    reduce_fn: Callable = jax.scipy.special.logsumexp
    log_space: bool = True
    name: str = "log-sum-product"

    def combine(self, a: Factor, b: Factor) -> Factor:
        """Combine two factors using the semiring's combine operation."""
        return a._binaryop(self.combine_fn, b)

    def reduce(self, f: Factor, attrs: ...) -> Factor:
        """Marginalize a factor over the given attributes."""
        return f._aggregate(self.reduce_fn, attrs)

    def __repr__(self) -> str:
        return f"Semiring({self.name!r})"


LOG_SUM_PRODUCT = Semiring(
    combine_fn=jnp.add,
    reduce_fn=jax.scipy.special.logsumexp,
    log_space=True,
    name="log-sum-product",
)

MAX_SUM = Semiring(
    combine_fn=jnp.add,
    reduce_fn=jnp.max,
    log_space=True,
    name="max-sum",
)

SUM_PRODUCT = Semiring(
    combine_fn=jnp.multiply,
    reduce_fn=jnp.sum,
    log_space=False,
    name="sum-product",
)


class MarginalOracle(Protocol):
    """Callable signature for stateless marginal oracle functions.

    A marginal oracle consumes potentials of a graphical model and returns
    its marginals over the same set of cliques.  Different oracles may have
    different time/space complexities and numerical stabilities.

    The recommended way to obtain a marginal oracle is via
    ``MessagePassingOracle``, which composes a message schedule, a
    contraction function, and a semiring::

        oracle = MessagePassingOracle(
            schedule=MessageSchedule.IMPLICIT,
            contraction=einsum_semiring,
            semiring=LOG_SUM_PRODUCT,
        )

    Pre-built instances are also available as module-level names:

    - ``message_passing_fast``: IMPLICIT schedule + einsum_stabilized
    - ``message_passing_stable``: HUGIN schedule
    - ``message_passing_shafer_shenoy``: SHAFER_SHENOY schedule
    - ``brute_force_marginals``: Materializes the full joint distribution
    - ``einsum_marginals``: Direct einsum (not recommended for large models)
    """

    def __call__(
        self,
        potentials: CliqueVector,
        total: float = 1.0,
    ) -> CliqueVector:
        """Compute marginals from potentials.

        Args:
            potentials: Potentials of a graphical model.
            total: Normalization constant. Defaults to 1.0.

        Returns:
            Marginals as a CliqueVector.
        """


def sum_product(
    factors: list[Factor], dom: Domain, einsum_fn: Callable = jnp.einsum
) -> Factor:
    """Compute the sum-of-products of a list of Factors using einsum.

    Args:
        factors: A list of Factors.
        dom: The target domain of the output factor.

    Returns:
        sum_{S - D} prod_i F_i,
        where
            * F_i = factors[i]
            * D = dom
            * S = union of domains of F_i
    """

    attrs = sorted(set.union(*[set(f.domain) for f in factors]).union(set(dom)))
    mapping = dict(zip(attrs, _EINSUM_LETTERS))

    def convert(d):
        return "".join(mapping[a] for a in d.attributes)

    formula = ",".join(convert(f.domain) for f in factors) + "->" + convert(dom)
    values = einsum_fn(
        formula,
        *[f.values for f in factors],
        optimize="auto",
        precision=jax.lax.Precision.HIGHEST,
    )
    return Factor(dom, values)


def einsum_stabilized(
    log_factors: list[Factor], dom: Domain, semiring: Semiring = LOG_SUM_PRODUCT,
    einsum_fn: Callable = jnp.einsum,
) -> Factor:
    """Compute the generalized sum-product via the exp-normalize trick.

    Subtracts per-factor maxima for numerical stability, exponentiates,
    runs a standard multiply-sum einsum, then maps back to log space.
    Fast and memory-efficient (no super-factor materialization), but
    can produce NaN when factor magnitudes span a very wide range.

    Only valid for log-space semirings (``semiring.log_space`` must be True).

    Args:
        log_factors: Factors in log space.
        dom: Target domain for the output.
        semiring: Must be a log-space semiring. The reduce_fn is used
            after mapping back to log space.
        einsum_fn: Einsum implementation to use (default: ``jnp.einsum``).

    Returns:
        A Factor over *dom* containing the result in log space.
    """
    maxes = [f.max(f.domain.marginalize(dom).attributes) for f in log_factors]
    stable_factors = [(f - m).exp() for f, m in zip(log_factors, maxes)]
    return sum_product(stable_factors, dom, einsum_fn).log() + sum(maxes)


def einsum_materialized(
    log_factors: list[Factor], dom: Domain, semiring: Semiring = LOG_SUM_PRODUCT,
) -> Factor:
    """Compute the generalized sum-product by materializing the combined factor.

    Combines all factors into a single super-factor, then reduces over
    the non-target dimensions.  Numerically stable because no exp/log
    round-trip is needed, but may require O(product of all domain sizes)
    memory for the intermediate super-factor.

    Works with any semiring.

    Args:
        log_factors: Factors (in whatever space the semiring expects).
        dom: Target domain for the output.
        semiring: The semiring whose combine/reduce ops are used.

    Returns:
        A Factor over *dom*.
    """
    combined = functools.reduce(semiring.combine, log_factors)
    elim_attrs = combined.domain.marginalize(dom).attributes
    return semiring.reduce(combined, elim_attrs).transpose(dom.attributes)


def einsum_semiring(
    factors: list[Factor], dom: Domain, semiring: Semiring = LOG_SUM_PRODUCT,
) -> Factor:
    """Compute the generalized sum-product using ``custom_dot_general``.

    Delegates to ``custom_einsum`` with the semiring's combine and reduce
    functions, allowing the XLA compiler to fuse the combine and reduce
    steps.  This avoids both the exp/log round-trip of ``einsum_stabilized``
    and the super-factor materialization of ``einsum_materialized``.

    Works with any semiring.  On GPU, XLA often fuses the operations
    resulting in both good stability and good performance.

    Args:
        factors: Factors (in whatever space the semiring expects).
        dom: Target domain for the output.
        semiring: The semiring whose combine/reduce ops are used.

    Returns:
        A Factor over *dom*.
    """
    def _custom_einsum(formula, *arrays, **kwargs):
        kwargs.pop('optimize', None)
        kwargs.pop('precision', None)
        return custom_einsum(
            formula, *arrays,
            combine_fn=semiring.combine_fn,
            reduce_fn=semiring.reduce_fn,
        )

    return sum_product(factors, dom, einsum_fn=_custom_einsum)


# Backward-compatible aliases for contraction functions.
logspace_sum_product_fast = functools.partial(einsum_stabilized, semiring=LOG_SUM_PRODUCT)
logspace_sum_product_stable_v1 = functools.partial(einsum_materialized, semiring=LOG_SUM_PRODUCT)


class MessageSchedule(enum.Enum):
    """Message-passing schedule for junction tree inference.

    Attributes:
        HUGIN: Expand potentials to super-cliques, use belief subtraction
            for message computation (the classic HUGIN algorithm).
            Unstable when potentials contain ``-inf``.
        SHAFER_SHENOY: Expand potentials to super-cliques, collect messages
            from all neighbors except the target (no belief subtraction).
            More stable than HUGIN for ``-inf`` potentials.
        IMPLICIT: Keep potentials in their original factored form and compute
            messages via a configurable contraction function (einsum-based).
            Most memory-efficient — never materializes super-clique tables.
    """
    HUGIN = "hugin"
    SHAFER_SHENOY = "shafer_shenoy"
    IMPLICIT = "implicit"


# Type alias for contraction functions used by the IMPLICIT schedule.
ContractionFn = Callable[[list[Factor], Domain, Semiring], Factor]




@dataclasses.dataclass(frozen=True)
class MessagePassingOracle:
    """Configurable marginal oracle based on junction tree message passing.

    Composes three independent design choices:

    - **schedule**: How messages are routed on the junction tree
      (HUGIN, SHAFER_SHENOY, or IMPLICIT).
    - **contraction**: How the inner sum-product is computed.  Only used
      when ``schedule=IMPLICIT``.  For HUGIN and SHAFER_SHENOY, the
      semiring's reduce operation is applied directly to expanded
      super-clique beliefs.
    - **semiring**: The algebraic operations (log-sum-product, max-sum,
      or sum-product in probability space).

    Instances of this class satisfy the ``MarginalOracle`` protocol and
    can be passed anywhere a marginal oracle is expected.

    Examples:
        >>> oracle = MessagePassingOracle()  # defaults: IMPLICIT + einsum_stabilized + LOG_SUM_PRODUCT
        >>> hugin = MessagePassingOracle(schedule=MessageSchedule.HUGIN)
        >>> stable = MessagePassingOracle(contraction=einsum_semiring)
        >>> map_oracle = MessagePassingOracle(semiring=MAX_SUM)
    """
    schedule: MessageSchedule = MessageSchedule.IMPLICIT
    contraction: ContractionFn = einsum_stabilized
    semiring: Semiring = LOG_SUM_PRODUCT

    @jax.jit(static_argnames=["self", "jtree"])
    def __call__(
        self,
        potentials: CliqueVector,
        total: float = 1.0,
        jtree: nx.Graph | None = None,
    ) -> CliqueVector:
        """Compute marginals from potentials.

        Args:
            potentials: Potentials of a graphical model (log-space or
                probability-space, depending on the semiring).
            total: Normalization constant for the output marginals.
            jtree: Optional pre-computed junction tree.  If ``None``,
                one is constructed automatically from the potentials.

        Returns:
            Marginals as a CliqueVector over the same cliques.
        """
        return self.infer(potentials, total, jtree)[0]

    def infer(
        self,
        potentials: CliqueVector,
        total: float = 1.0,
        jtree: nx.Graph | None = None,
    ) -> tuple[CliqueVector, dict]:
        """Run inference and return both marginals and messages.

        This method has the same semantics as ``__call__`` but returns
        a ``(marginals, messages)`` tuple containing the intermediate
        messages in addition to the marginals.

        Args:
            potentials: Potentials of a graphical model.
            total: Normalization constant for the output marginals.
            jtree: Optional pre-computed junction tree.

        Returns:
            A tuple of (marginals, messages) where marginals is a
            CliqueVector and messages is a dict mapping
            ``(sender, receiver)`` clique pairs to message Factors.
        """
        if self.schedule == MessageSchedule.IMPLICIT:
            return self._implicit(potentials, total, jtree)
        elif self.schedule == MessageSchedule.HUGIN:
            return self._hugin(potentials, total, jtree)
        elif self.schedule == MessageSchedule.SHAFER_SHENOY:
            return self._shafer_shenoy(potentials, total, jtree)
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")

    def _normalize_beliefs(self, beliefs: CliqueVector, total: float) -> CliqueVector:
        """Convert beliefs to normalized marginals using the semiring."""
        if self.semiring.log_space:
            return beliefs.normalize(total, log=True).exp()
        else:
            return beliefs.normalize(total, log=False)

    def _hugin(self, potentials, total, jtree):
        """HUGIN message passing with belief subtraction."""
        if len(potentials.cliques) == 0:
            return CliqueVector(potentials.domain, [], {}), {}

        domain, cliques = potentials.domain, potentials.cliques

        if jtree is None:
            jtree = junction_tree.make_junction_tree(domain, cliques)[0]
        message_order = junction_tree.message_passing_order(jtree)
        maximal_cliques = junction_tree.maximal_cliques(jtree)

        mapping = clique_mapping(maximal_cliques, cliques, domain=domain)
        beliefs = potentials.expand(maximal_cliques)

        messages = {}
        for i, j in message_order:
            sep = beliefs[i].domain.invert(tuple(set(i) & set(j)))
            if (j, i) in messages:
                tau = beliefs[i] - messages[(j, i)] if self.semiring.log_space else beliefs[i] / messages[(j, i)]
            else:
                tau = beliefs[i]
            messages[(i, j)] = self.semiring.reduce(tau, sep)
            beliefs[j] = self.semiring.combine(beliefs[j], messages[(i, j)])

        marginals = (
            self._normalize_beliefs(beliefs, total)
            .contract(cliques)
        )
        return marginals, messages

    def _shafer_shenoy(self, potentials, total, jtree):
        """Shafer-Shenoy message passing with neighbor collection."""
        if len(potentials.cliques) == 0:
            return CliqueVector(potentials.domain, [], {}), {}

        domain, cliques = potentials.domain, potentials.cliques

        if jtree is None:
            jtree = junction_tree.make_junction_tree(domain, cliques)[0]
        message_order = junction_tree.message_passing_order(jtree)
        maximal_cliques = junction_tree.maximal_cliques(jtree)

        initial_beliefs = potentials.expand(maximal_cliques)

        messages = {}
        neighbors = {cl: list(jtree.neighbors(cl)) for cl in maximal_cliques}

        for i, j in message_order:
            tau = initial_beliefs[i]
            for k in neighbors[i]:
                if k == j:
                    continue
                tau = self.semiring.combine(tau, messages[(k, i)])

            sep = tau.domain.invert(tuple(set(i) & set(j)))
            messages[(i, j)] = self.semiring.reduce(tau, sep)

        beliefs = {}
        for cl in maximal_cliques:
            b = initial_beliefs[cl]
            for k in neighbors[cl]:
                b = self.semiring.combine(b, messages[(k, cl)])
            beliefs[cl] = b

        beliefs = CliqueVector(potentials.domain, maximal_cliques, beliefs)
        marginals = (
            self._normalize_beliefs(beliefs, total)
            .contract(cliques)
        )
        return marginals, messages

    def _implicit(self, potentials, total, jtree):
        """Implicit-factor message passing using a contraction function."""
        if len(potentials.cliques) == 0:
            return CliqueVector(potentials.domain, [], {}), {}

        domain, cliques = potentials.active_domain, potentials.cliques

        if jtree is None:
            jtree = junction_tree.make_junction_tree(domain, cliques)[0]
        message_order = junction_tree.message_passing_order(jtree)
        maximal_cliques = junction_tree.maximal_cliques(jtree)

        mapping = clique_mapping(maximal_cliques, cliques, domain=domain)
        inverse_mapping = collections.defaultdict(list)
        incoming_messages = collections.defaultdict(list)
        potential_mapping = collections.defaultdict(list)

        for cl in cliques:
            potential_mapping[mapping[cl]].append(potentials[cl])
            inverse_mapping[mapping[cl]].append(cl)

        for i, msg in enumerate(message_order):
            for j in range(i):
                msg2 = message_order[j]
                if msg[0] == msg2[1] and msg[1] != msg2[0]:
                    incoming_messages[msg].append(msg2)

        messages = {}
        for i, j in message_order:
            shared = domain.project(tuple(set(i) & set(j)))
            input_potentials = potential_mapping[i]
            input_messages = [messages[key] for key in incoming_messages[(i, j)]]
            inputs = input_potentials + input_messages

            for attr in shared.attributes:
                if not any(attr in input.domain.attributes for input in inputs):
                    inputs.append(Factor.zeros(domain.project([attr])))

            messages[(i, j)] = self.contraction(
                inputs, shared, self.semiring,
            )

        beliefs = {}
        for cl in maximal_cliques:
            input_potentials = potential_mapping[cl]
            input_messages = [val for key, val in messages.items() if key[1] == cl]
            inputs = input_potentials + input_messages
            for cl2 in inverse_mapping[cl]:
                belief = self.contraction(
                    inputs, domain.project(cl2), self.semiring,
                )
                if self.semiring.log_space:
                    beliefs[cl2] = belief.normalize(total, log=True).exp()
                else:
                    beliefs[cl2] = belief.normalize(total, log=False)

        return CliqueVector(potentials.domain, cliques, beliefs), messages

    def __repr__(self) -> str:
        parts = [f"schedule={self.schedule.value}"]
        if self.schedule == MessageSchedule.IMPLICIT:
            name = getattr(self.contraction, '__name__', repr(self.contraction))
            parts.append(f"contraction={name}")
        parts.append(f"semiring={self.semiring.name}")
        return f"MessagePassingOracle({', '.join(parts)})"


# ---- Backward-compatible oracle aliases ----
# These module-level instances replace the old functions.
# They satisfy the MarginalOracle protocol and can be used as drop-in replacements.

message_passing_fast = MessagePassingOracle(
    schedule=MessageSchedule.IMPLICIT,
    contraction=einsum_stabilized,
    semiring=LOG_SUM_PRODUCT,
)

message_passing_stable = MessagePassingOracle(
    schedule=MessageSchedule.HUGIN,
    semiring=LOG_SUM_PRODUCT,
)

message_passing_shafer_shenoy = MessagePassingOracle(
    schedule=MessageSchedule.SHAFER_SHENOY,
    semiring=LOG_SUM_PRODUCT,
)


def brute_force_marginals(
    potentials: CliqueVector,
    total: float = 1,
) -> CliqueVector:
    """Compute marginals from (log-space) potentials by materializing the full joint distribution."""
    if len(potentials.cliques) == 0:
        return CliqueVector(potentials.domain, [], {})

    P = (
        sum(potentials.arrays.values())
        .normalize(total, log=True)
        .exp()
    )
    marginals = {cl: P.project(cl) for cl in potentials.cliques}
    return CliqueVector(
        potentials.domain, potentials.cliques, marginals
    )


def einsum_marginals(
    potentials: CliqueVector,
    total: float = 1,
    einsum_fn: Callable = jnp.einsum,
) -> CliqueVector:
    """Compute marginals from (log-space) potentials by using einsum.

    This is a "brute-force" approach and is not recommended in practice.
    """
    # not strictly necessary, but consistent
    if len(potentials.cliques) == 0:
        return CliqueVector(potentials.domain, [], {})

    inputs = list(potentials.arrays.values())
    return CliqueVector(
        potentials.domain,
        potentials.cliques,
        {
            cl: (
                logspace_sum_product_fast(
                    inputs, potentials[cl].domain, einsum_fn=einsum_fn
                )
                .normalize(total, log=True)
                .exp()
            )
            for cl in potentials.cliques
        },
    )





Clique = tuple[str, ...]


def variable_elimination(
    potentials: CliqueVector,
    clique: Clique,
    total: float = 1,
    evidence: dict[str, int] | None = None,
) -> Factor:
    """Compute an out-of-model/unsupported marginal from the potentials.

    Args:
        potentials: The (log-space) potentials of a Graphical Model.
        clique: The subset of attributes whose marginal you want.
        total: The normalization factor.
        evidence: A dictionary mapping attribute names to observed values.

    Returns:
        The marginal defined over the domain of the input clique, where
        each entry is non-negative and sums to the input total.
    """
    clique = tuple(clique)
    evidence = evidence or {}
    if set(clique) & set(evidence.keys()):
        raise ValueError("Evidence attributes cannot be in the query clique.")

    k = len(potentials.cliques)
    psi = dict(zip(range(k), potentials.arrays.values()))

    if evidence:
        for i in list(psi.keys()):
            psi[i] = psi[i].slice(evidence)

    evidence_attr = "_mbi_evidence"
    has_vector_evidence = any(evidence_attr in psi[i].domain for i in psi)

    if has_vector_evidence:
        ev_size = next(
            psi[i].domain[evidence_attr]
            for i in psi
            if evidence_attr in psi[i].domain
        )
        extra = Domain([evidence_attr], [ev_size])
        domain = potentials.active_domain.marginalize(evidence.keys()).merge(
            extra
        )
        if evidence_attr not in clique:
            clique = (evidence_attr,) + clique
    else:
        domain = potentials.active_domain.marginalize(evidence.keys())

    cliques = [psi[i].domain.attributes for i in psi] + [clique]
    elim = domain.invert(clique)
    elim_order, _ = junction_tree.greedy_order(domain, cliques, elim=elim)

    for z in elim_order:
        psi2 = [psi.pop(i) for i in list(psi.keys()) if z in psi[i].domain]
        psi[k] = sum(psi2).logsumexp([z])
        k += 1
    # this expand covers the case when clique is not in the active domain
    if has_vector_evidence:
        vars_in_model = [v for v in clique if v != evidence_attr]
        base_dom = potentials.domain.project(vars_in_model)
        ev_size = domain[evidence_attr]
        newdom = base_dom.merge(Domain([evidence_attr], [ev_size])).project(
            clique
        )
    else:
        newdom = potentials.domain.project(clique)

    zero = Factor(Domain([], []), jnp.asarray(0.0))
    unnormalized = (
        sum(psi.values(), start=zero).expand(newdom)
    )

    if has_vector_evidence:
        sum_attrs = [
            a for a in unnormalized.domain.attributes if a != evidence_attr
        ]
        log_z = unnormalized.logsumexp(sum_attrs)
        normalized = unnormalized + jnp.log(total) - log_z
        return normalized.exp().project(clique)

    return (
        unnormalized.normalize(total, log=True)
        .exp()
        .project(clique)
    )


def bulk_variable_elimination(
    potentials: CliqueVector,
    marginal_queries: list[tuple[str, ...]],
    total: float = 1.0,
) -> CliqueVector:
    """Compute the marginals of the graphical model with the given potentials.

    Unlike other marginal oracles, which only compute marginals for cliques
    in the potentials vector, this function can compute arbitrary marginals
    from an arbitrary model. Both runtime and compilation time can be expensive
    when there are a large number of marginal queries. This function compiles
    and runs variable_elimination for one query at a time, using parallelism
    and asyncronous computation do do the compilation in the background, while
    running variable_eliminatoin sequentially one query at a time.

    Args:
      potentials: The (log-space) potentials of a Graphical Model.
      marginal_queries: A list of cliques to obtain marginals for.
      total: The normalization factor.

    Returns:
      A CliqueVector with the marginals computed over the specified cliques.
    """
    jitted = jax.jit(variable_elimination, static_argnums=(1,))

    # Async + parallel precompilation.
    def _precompile(query):
        return query, jitted.lower(potentials, query, total).compile()

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(_precompile, cl) for cl in marginal_queries]

        results = {}
        for future in concurrent.futures.as_completed(futures):
            query, compiled_fn = future.result()
            results[query] = compiled_fn(potentials, total)

        return CliqueVector(potentials.domain, marginal_queries, results)


def calculate_many_marginals(
    potentials: CliqueVector,
    marginal_queries: list[Clique],
    total: float = 1.0,
    belief_propagation_oracle: MarginalOracle = message_passing_stable,
) -> CliqueVector:
    """Calculate marginals for all projections using belief propagation.

    Implements Algorithm from section 10.3 in Koller and Friedman.
    This method may be faster than calling variable_elimination many times.
    Note: this implementation is experimental, and further work may be needed
    to optimize it. Contributions are welcome.

    Args:
        potentials: Potentials of a graphical model.
        marginal_queries: a list of cliques whose marginals are desired.

    Returns:
        A CliqueVector, where each defined over the list of input marginal_queries.
    """

    domain = potentials.domain
    jtree = junction_tree.make_junction_tree(
        potentials.domain, potentials.cliques
    )[0]
    max_cliques = junction_tree.maximal_cliques(jtree)
    neighbors = {i: tuple(jtree.neighbors(i)) for i in max_cliques}

    # TODO: let's see if we can get rid of this similar to message_passing_fast
    potentials = potentials.expand(max_cliques)

    # TODO: allow these to take in an optional junction tree
    marginals = belief_propagation_oracle(potentials, total)

    # first calculate P(Cj | Ci) for all neighbors Ci, Cj
    conditional = {}
    for Ci in max_cliques:
        for Cj in neighbors[Ci]:
            Cj: tuple[
                str, ...
            ]  # networkx does not seem to have the right type annotation.
            Sij = tuple(set(Cj) & set(Ci))
            Z = marginals.project(Cj)
            Z_sep = Z.project(Sij)
            denom = Z_sep.expand(Z.domain)
            safe_div = jnp.where(
                denom.values != 0, Z.values / denom.values, 0.0
            )
            conditional[(Cj, Ci)] = Factor(Z.domain, safe_div)

    # now iterate through pairs of cliques in order of distance
    # not sure why this API changed and why we need to do this hack.
    nx.set_edge_attributes(jtree, values=1.0, name="weight")  # type: ignore
    pred, dist = nx.floyd_warshall_predecessor_and_distance(
        jtree
    )  # , weight=None)

    def order_fn(x):
        return dist[x[0]][x[1]]

    results = {}
    for Ci, Cj in sorted(itertools.combinations(max_cliques, 2), key=order_fn):
        if dist[Ci][Cj] == math.inf:
            continue
        Cl = pred[Ci][Cj]
        Y = conditional[(Cj, Cl)]
        if Cl == Ci:
            X = marginals[Ci]
            results[(Ci, Cj)] = results[(Cj, Ci)] = X * Y
        else:
            X = results[(Ci, Cl)]
            S = set(Cl) - set(Ci) - set(Cj)
            results[(Ci, Cj)] = results[(Cj, Ci)] = (X * Y).sum(S)

    results = {
        domain.canonical(key[0] + key[1]): val for key, val in results.items()
    }

    answers = {}
    for cl in marginal_queries:
        for attr in results:
            if set(cl) <= set(attr):
                answers[cl] = results[attr].project(cl)
                break
        if cl not in answers:
            # just use variable elimination
            answers[cl] = variable_elimination(potentials, cl, total)

    return CliqueVector(domain, marginal_queries, answers)


def kron_query(
    potentials: CliqueVector,
    query_factors: dict[str, jax.Array],
    total: float = 1,
    suffix: str = "_answer",
) -> Factor:
    new_factors = {}
    extra_domain = {}
    extra_cliques = []
    target_clique = []

    for key in query_factors:
        key2 = key + suffix
        values = query_factors[key]
        dom = Domain([key2, key], values.shape)
        new_factors[(key2, key)] = Factor(dom, values).log()
        extra_domain[key2] = values.shape[0]
        extra_cliques.append((key2, key))
        target_clique.append(key2)

    domain = potentials.domain.merge(Domain.fromdict(extra_domain))
    cliques = potentials.cliques + tuple(extra_cliques)
    inputs = CliqueVector(domain, cliques, new_factors)
    return variable_elimination(inputs, tuple(target_clique), total)
