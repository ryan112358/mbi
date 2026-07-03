"""Marginal oracles for computing marginals from graphical model potentials.

This module provides marginal oracles that compute marginals from the
log-space potentials of a discrete graphical model.  Three junction tree
schedules are available:

- ``message_passing_implicit``: Implicit-factor message passing using a
  configurable contraction function (default: ``einsum_fused``).
- ``message_passing_hugin``: Classic HUGIN algorithm with belief subtraction.
- ``message_passing_shafer_shenoy``: Shafer-Shenoy algorithm with neighbor collection.

Use ``default_oracle()`` to automatically select the best oracle for your
hardware and clique structure::

    oracle = marginal_oracles.default_oracle(cliques=cliques, domain=domain)
    marginals = oracle(potentials)
"""

import collections
import concurrent.futures
import functools
import itertools
import math
import string
import warnings
from collections.abc import Callable, Sequence
from typing import Protocol

import jax
import jax.numpy as jnp
import networkx as nx

from . import junction_tree
from .clique_utils import clique_mapping
from .clique_vector import CliqueVector
from .constraint import Constraint
from .domain import Domain
from .einsum import custom_einsum
from .factor import Factor

_EINSUM_LETTERS = list(string.ascii_lowercase) + list(string.ascii_uppercase)


class MarginalOracle(Protocol):
    """Callable signature for stateless marginal oracle functions.

    A marginal oracle consumes log-space potentials of a graphical model
    and returns its marginals over the same set of cliques.  Different
    oracles may have different time/space complexities and numerical
    stabilities.

    Available oracles:

    - ``message_passing_implicit``: Implicit-factor message passing (configurable).
    - ``message_passing_hugin``: Classic HUGIN algorithm with belief subtraction.
    - ``message_passing_shafer_shenoy``: Shafer-Shenoy algorithm with neighbor collection.
    - ``brute_force_marginals``: Materializes the full joint distribution.
    - ``einsum_marginals``: Direct einsum (not recommended for large models).
    """

    def __call__(
        self,
        potentials: CliqueVector,
        total: float = 1.0,
        *,
        constraints: Sequence[Constraint] = (),
    ) -> CliqueVector:
        """Compute marginals from potentials.

        Args:
            potentials: Potentials of a graphical model.
            total: Normalization constant. Defaults to 1.0.
            constraints: Structural constraints. Oracles that support
                constraints fold them into the potentials or handle
                them natively; unsupported oracles raise ValueError.

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


def einsum_semistable(
    log_factors: list[Factor],
    dom: Domain,
    einsum_fn: Callable = jnp.einsum,
) -> Factor:
    """Compute the log-space sum-product via the exp-normalize trick.

    Subtracts per-factor maxima for numerical stability, exponentiates,
    runs a standard multiply-sum einsum, then maps back to log space.
    Fast and memory-efficient (no super-factor materialization), but
    can produce NaN when factor magnitudes span a very wide range.

    Args:
        log_factors: Factors in log space.
        dom: Target domain for the output.
        einsum_fn: Einsum implementation to use (default: ``jnp.einsum``).

    Returns:
        A Factor over *dom* containing the result in log space.
    """
    maxes = [f.max(f.domain.marginalize(dom).attributes) for f in log_factors]
    stable_factors = [(f - m).exp() for f, m in zip(log_factors, maxes)]
    return sum_product(stable_factors, dom, einsum_fn).log() + sum(maxes)


def einsum_materialized(
    log_factors: list[Factor],
    dom: Domain,
) -> Factor:
    """Compute the log-space sum-product by materializing the combined factor.

    Combines all log-factors into a single super-factor (via addition in
    log-space), then reduces over the non-target dimensions with logsumexp.
    Numerically stable because no exp/log round-trip is needed, but may
    require O(product of all domain sizes) memory for the intermediate
    super-factor.

    Args:
        log_factors: Factors in log space.
        dom: Target domain for the output.

    Returns:
        A Factor over *dom* in log space.
    """
    combined = functools.reduce(lambda a, b: a + b, log_factors)
    elim_attrs = combined.domain.marginalize(dom).attributes
    return combined.logsumexp(elim_attrs).transpose(dom.attributes)


def einsum_fused(
    log_factors: list[Factor],
    dom: Domain,
) -> Factor:
    """Compute the log-space sum-product using ``custom_dot_general``.

    Delegates to ``custom_einsum`` with add/logsumexp operations,
    allowing the XLA compiler to fuse the combine and reduce steps.
    This avoids both the exp/log round-trip of ``einsum_semistable``
    and the super-factor materialization of ``einsum_materialized``.

    On GPU, XLA often fuses the operations resulting in both good
    stability and good performance.

    Args:
        log_factors: Factors in log space.
        dom: Target domain for the output.

    Returns:
        A Factor over *dom* in log space.
    """

    def _custom_einsum(formula, *arrays, **kwargs):
        kwargs.pop("optimize", None)
        kwargs.pop("precision", None)
        return custom_einsum(
            formula,
            *arrays,
            combine_fn=jnp.add,
            reduce_fn=jax.scipy.special.logsumexp,
        )

    return sum_product(log_factors, dom, einsum_fn=_custom_einsum)


def _fold_constraints(
    potentials: CliqueVector,
    constraints: Sequence[Constraint],
) -> tuple[CliqueVector, tuple[tuple[str, ...], ...]]:
    """Fold constraints into potentials as -inf/0 log-space factors.

    Returns:
        A tuple of (expanded_potentials, input_cliques) where input_cliques
        are the original cliques before constraint factors were added.
    """
    input_cliques = potentials.cliques
    if not constraints:
        return potentials, input_cliques
    domain = potentials.domain
    cliques = list(input_cliques)
    arrays = {cl: potentials[cl] for cl in cliques}
    for c in constraints:
        cl = domain.canonical(c.clique)
        factor = c.potential
        if cl in arrays:
            arrays[cl] = arrays[cl] + factor
        else:
            cliques.append(cl)
            arrays[cl] = factor
    return CliqueVector(domain, cliques, arrays), input_cliques


@jax.jit(static_argnames=["jtree", "return_messages"])
def message_passing_hugin(
    potentials: CliqueVector,
    total: float = 1.0,
    jtree: nx.Graph | None = None,
    *,
    constraints: Sequence[Constraint] = (),
    return_messages: bool = False,
) -> CliqueVector | tuple[CliqueVector, dict]:
    """HUGIN message passing with belief subtraction.

    Expands potentials to super-cliques and uses belief subtraction for
    message computation.  Unstable when potentials contain ``-inf``.

    Args:
        potentials: Potentials of a graphical model.
        total: Normalization constant.
        jtree: Pre-computed junction tree (optional).
        constraints: Structural constraints folded into potentials as
            ``-inf``/``0`` factors before inference.
        return_messages: If True, return ``(marginals, messages)`` where
            *messages* is a dict mapping ``(clique_i, clique_j)`` to the
            log-space message Factor sent from *i* to *j*.
    """
    if constraints:
        warnings.warn(
            "HUGIN uses belief subtraction which is unstable with -inf"
            " potentials introduced by constraints. Consider using"
            " message_passing_shafer_shenoy instead.",
            stacklevel=2,
        )
    potentials, input_cliques = _fold_constraints(potentials, constraints)
    if len(potentials.cliques) == 0:
        result = CliqueVector(potentials.domain, [], {})
        return (result, {}) if return_messages else result

    domain, cliques = potentials.domain, potentials.cliques

    if jtree is None:
        jtree = junction_tree.make_junction_tree(domain, cliques)[0]
    message_order = junction_tree.message_passing_order(jtree)
    maximal_cliques = junction_tree.maximal_cliques(jtree)

    clique_mapping(maximal_cliques, cliques, domain=domain)
    beliefs = potentials.expand(maximal_cliques)

    messages = {}
    for i, j in message_order:
        sep = beliefs[i].domain.invert(tuple(set(i) & set(j)))
        if (j, i) in messages:
            tau = beliefs[i] - messages[(j, i)]
        else:
            tau = beliefs[i]
        messages[(i, j)] = tau.logsumexp(sep)
        beliefs[j] = beliefs[j] + messages[(i, j)]

    marginals = beliefs.normalize(total, log=True).exp().contract(input_cliques)
    return (marginals, messages) if return_messages else marginals


@jax.jit(static_argnames=["jtree", "return_messages"])
def message_passing_shafer_shenoy(
    potentials: CliqueVector,
    total: float = 1.0,
    jtree: nx.Graph | None = None,
    *,
    constraints: Sequence[Constraint] = (),
    return_messages: bool = False,
) -> CliqueVector | tuple[CliqueVector, dict]:
    """Shafer-Shenoy message passing with neighbor collection.

    Expands potentials to super-cliques and collects messages from all
    neighbors except the target.  More stable than HUGIN for ``-inf``
    potentials.

    Args:
        potentials: Potentials of a graphical model.
        total: Normalization constant.
        jtree: Pre-computed junction tree (optional).
        constraints: Structural constraints folded into potentials as
            ``-inf``/``0`` factors before inference.
        return_messages: If True, return ``(marginals, messages)`` where
            *messages* is a dict mapping ``(clique_i, clique_j)`` to the
            log-space message Factor sent from *i* to *j*.
    """
    potentials, input_cliques = _fold_constraints(potentials, constraints)
    if len(potentials.cliques) == 0:
        result = CliqueVector(potentials.domain, [], {})
        return (result, {}) if return_messages else result

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
            tau = tau + messages[(k, i)]

        sep = tau.domain.invert(tuple(set(i) & set(j)))
        messages[(i, j)] = tau.logsumexp(sep)

    beliefs = {}
    for cl in maximal_cliques:
        b = initial_beliefs[cl]
        for k in neighbors[cl]:
            b = b + messages[(k, cl)]
        beliefs[cl] = b

    beliefs = CliqueVector(potentials.domain, maximal_cliques, beliefs)
    marginals = beliefs.normalize(total, log=True).exp().contract(input_cliques)
    return (marginals, messages) if return_messages else marginals


@jax.jit(static_argnames=["jtree", "contraction", "return_messages"])
def message_passing_implicit(
    potentials: CliqueVector,
    total: float = 1.0,
    jtree: nx.Graph | None = None,
    *,
    constraints: Sequence[Constraint] = (),
    contraction: Callable = einsum_materialized,
    return_messages: bool = False,
) -> CliqueVector | tuple[CliqueVector, dict]:
    """Implicit-factor message passing using a contraction function.

    Keeps potentials in their original factored form and computes messages
    via the given contraction function.  Most memory-efficient — never
    materializes super-clique tables.

    The default contraction (``einsum_materialized``) is the fastest on GPU.
    For better ``-inf`` tolerance, use ``einsum_fused``.

    Args:
        potentials: Potentials of a graphical model.
        total: Normalization constant.
        jtree: Pre-computed junction tree (optional).
        constraints: Structural constraints folded into potentials as
            ``-inf``/``0`` factors before inference.
        contraction: Contraction function for log-space sum-product.
        return_messages: If True, return ``(marginals, messages)`` where
            *messages* is a dict mapping ``(clique_i, clique_j)`` to the
            log-space message Factor sent from *i* to *j*.
    """
    if constraints and contraction is einsum_semistable:
        raise ValueError(
            "einsum_semistable is not compatible with constraints"
            " (-inf potentials). Use einsum_fused or the default"
            " einsum_materialized instead."
        )
    potentials, input_cliques = _fold_constraints(potentials, constraints)
    if len(potentials.cliques) == 0:
        result = CliqueVector(potentials.domain, [], {})
        return (result, {}) if return_messages else result

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

        messages[(i, j)] = contraction(inputs, shared)

    beliefs = {}
    for cl in maximal_cliques:
        input_potentials = potential_mapping[cl]
        input_messages = [val for key, val in messages.items() if key[1] == cl]
        inputs = input_potentials + input_messages
        for cl2 in inverse_mapping[cl]:
            if cl2 not in input_cliques:
                continue
            belief = contraction(inputs, domain.project(cl2))
            beliefs[cl2] = belief.normalize(total, log=True).exp()

    marginals = CliqueVector(potentials.domain, input_cliques, beliefs)
    return (marginals, messages) if return_messages else marginals


# Backward-compatible aliases.
message_passing_fast = functools.partial(
    message_passing_implicit, contraction=einsum_semistable
)
message_passing_stable = message_passing_hugin


def default_oracle(
    cliques: tuple[tuple[str, ...], ...] | None = None,
    domain: Domain | None = None,
    backend: str | None = None,
    has_constraints: bool = True,
) -> MarginalOracle:
    """Select the best oracle for the given setting.

    Chooses based on hardware backend, clique structure, and whether the
    potentials may contain ``-inf`` entries:

    - **CPU**: ``message_passing_shafer_shenoy`` (has_constraints=True) or
      ``message_passing_hugin`` (has_constraints=False). Implicit is not used on
      CPU because the XLA compiler is less effective there.
    - **GPU/TPU, large cliques (>= 1M)**: ``message_passing_implicit``
      regardless of ``has_constraints`` (avoids materializing super-cliques).
    - **GPU/TPU, has_constraints=True** (default):
      ``message_passing_shafer_shenoy`` (numerically robust).
    - **GPU/TPU, has_constraints=False**: ``message_passing_hugin`` (faster when
      ``-inf`` is absent).

    Args:
        cliques: Cliques of the graphical model. Used to estimate max clique
            size when ``domain`` is also provided.
        domain: Domain of the graphical model. Used with ``cliques`` to
            estimate max clique size.
        backend: JAX backend string ('cpu', 'gpu', 'tpu'). If None, uses
            ``jax.default_backend()``.
        has_constraints: Whether potentials may contain ``-inf`` entries (e.g. from
            deterministic constraints or structural zeros). When True
            (default), Shafer-Shenoy is preferred for numerical robustness.
            When False, Hugin may be used for better performance.

    Returns:
        A callable satisfying the ``MarginalOracle`` protocol.

    Example::

        oracle = default_oracle(cliques=model.cliques, domain=model.domain)
        marginals = oracle(potentials)
    """
    if backend is None:
        backend = jax.default_backend()

    # CPU: SS or Hugin always (XLA compiler less effective on CPU).
    if backend == "cpu":
        if has_constraints:
            return message_passing_shafer_shenoy
        return message_passing_hugin

    # GPU/TPU with large cliques: implicit regardless of has_constraints.
    if cliques is not None and domain is not None:
        jtree = junction_tree.make_junction_tree(domain, cliques)[0]
        max_cliques = junction_tree.maximal_cliques(jtree)
        max_size = max(domain.project(cl).size() for cl in max_cliques)
        if max_size >= 1_000_000:
            return message_passing_implicit

    # GPU/TPU with small cliques.
    if has_constraints:
        return message_passing_shafer_shenoy
    return message_passing_hugin


def brute_force_marginals(
    potentials: CliqueVector,
    total: float = 1,
    *,
    constraints: Sequence[Constraint] = (),
) -> CliqueVector:
    """Compute marginals from (log-space) potentials by materializing the full joint distribution.

    Args:
        potentials: Potentials of a graphical model.
        total: Normalization constant.
        constraints: Structural constraints folded into potentials as
            ``-inf``/``0`` factors before inference.
    """
    potentials, input_cliques = _fold_constraints(potentials, constraints)
    if len(potentials.cliques) == 0:
        return CliqueVector(potentials.domain, [], {})

    P = sum(potentials.arrays.values()).normalize(total, log=True).exp()
    marginals = {cl: P.project(cl) for cl in input_cliques}
    return CliqueVector(potentials.domain, input_cliques, marginals)


def einsum_marginals(
    potentials: CliqueVector,
    total: float = 1,
    einsum_fn: Callable = jnp.einsum,
    *,
    constraints: Sequence[Constraint] = (),
) -> CliqueVector:
    """Compute marginals from (log-space) potentials by using einsum.

    This is a "brute-force" approach and is not recommended in practice.

    Args:
        potentials: Potentials of a graphical model.
        total: Normalization constant.
        einsum_fn: Einsum function to use.
        constraints: Structural constraints. Raises ValueError if non-empty.
    """
    if constraints:
        raise ValueError(
            "einsum_marginals does not support constraints. Use"
            " message_passing_shafer_shenoy or"
            " extensions.message_passing.shafer_shenoy."
        )
    # not strictly necessary, but consistent
    if len(potentials.cliques) == 0:
        return CliqueVector(potentials.domain, [], {})

    inputs = list(potentials.arrays.values())
    return CliqueVector(
        potentials.domain,
        potentials.cliques,
        {
            cl: (
                einsum_semistable(
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
    *,
    constraints: Sequence[Constraint] = (),
) -> Factor:
    """Compute an out-of-model/unsupported marginal from the potentials.

    Args:
        potentials: The (log-space) potentials of a Graphical Model.
        clique: The subset of attributes whose marginal you want.
        total: The normalization factor.
        evidence: Mapping from attribute names to observed scalar int values.
        constraints: Structural constraints folded into potentials as
            ``-inf``/``0`` factors before inference.

    Returns:
        The marginal defined over the domain of the input clique, where
        each entry is non-negative and sums to the input total.
    """
    potentials, _ = _fold_constraints(potentials, constraints)
    clique = tuple(clique)
    evidence = evidence or {}
    if set(clique) & set(evidence.keys()):
        raise ValueError("Evidence attributes cannot be in the query clique.")

    k = len(potentials.cliques)
    psi = dict(zip(range(k), potentials.arrays.values()))

    if evidence:
        for i in list(psi.keys()):
            psi[i] = psi[i].slice(evidence)

    domain = potentials.active_domain.marginalize(evidence.keys())
    cliques = [psi[i].domain.attributes for i in psi] + [clique]
    elim = domain.invert(clique)
    elim_order, _ = junction_tree.greedy_order(domain, cliques, elim=elim)

    for z in elim_order:
        psi2 = [psi.pop(i) for i in list(psi.keys()) if z in psi[i].domain]
        psi[k] = sum(psi2).logsumexp([z])
        k += 1

    newdom = potentials.domain.project(clique)
    zero = Factor(Domain([], []), jnp.asarray(0.0))
    unnormalized = sum(psi.values(), start=zero).expand(newdom)
    return unnormalized.normalize(total, log=True).exp().project(clique)


def bulk_variable_elimination(
    potentials: CliqueVector,
    marginal_queries: list[tuple[str, ...]],
    total: float = 1.0,
    *,
    constraints: Sequence[Constraint] = (),
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
      constraints: Structural constraints folded into potentials as
          ``-inf``/``0`` factors before inference.

    Returns:
      A CliqueVector with the marginals computed over the specified cliques.
    """
    potentials, _ = _fold_constraints(potentials, constraints)
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
    *,
    constraints: Sequence[Constraint] = (),
) -> CliqueVector:
    """Calculate marginals for all projections using belief propagation.

    Implements Algorithm from section 10.3 in Koller and Friedman.
    This method may be faster than calling variable_elimination many times.
    Note: this implementation is experimental, and further work may be needed
    to optimize it. Contributions are welcome.

    Args:
        potentials: Potentials of a graphical model.
        marginal_queries: a list of cliques whose marginals are desired.
        constraints: Structural constraints folded into potentials as
            ``-inf``/``0`` factors before inference.

    Returns:
        A CliqueVector, where each defined over the list of input marginal_queries.
    """
    potentials, _ = _fold_constraints(potentials, constraints)

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
    *,
    constraints: Sequence[Constraint] = (),
) -> Factor:
    """Compute a Kronecker-product query.

    Args:
        potentials: Potentials of a graphical model.
        query_factors: Mapping from attribute names to query matrices.
        total: Normalization constant.
        suffix: Suffix for the answer attributes.
        constraints: Structural constraints. Raises ValueError if non-empty.
    """
    if constraints:
        raise ValueError(
            "kron_query does not support constraints. Fold them into"
            " potentials before calling kron_query."
        )
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
