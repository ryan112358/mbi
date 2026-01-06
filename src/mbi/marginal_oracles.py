"""Functions for computing marginals from log-space potentials.

The functions in this library should all produce numerically identical
outputs on well-behaved inputs, but may have different stability characteristics
on poorly-behaved inputs, and different rutnime/memory performance characteristics.

We recommend using message_passing_stable with accelerated estimation algorithms like
Interior Gradient, but using message_passing_fast with mirror descent.
"""

import collections
import concurrent.futures
import functools
import itertools
import string
from typing import Protocol

import jax
import jax.numpy as jnp
import networkx as nx

from . import junction_tree
from .clique_utils import clique_mapping
from .clique_vector import CliqueVector
from .domain import Domain
from .factor import Factor

_EINSUM_LETTERS = list(string.ascii_lowercase) + list(string.ascii_uppercase)


class MarginalOracle(Protocol):
    """
    Defines the callable signature for stateless marginal oracle functions.

    A marginal oracle consumes log-space potentials (CliqueVector) of
    a graphical model and returns its marginals (CliqueVector).  The
    returned marginals will be defined over the same domain and set of cliques
    as the potentials.

    Different marginal oracles should usually produce identical results,
    but they may have different time/space complexities and numerical
    stabilities. Examples of conforming functions from `mbi.marginal_oracles`:

    - `message_passing_stable`: Computes marginals using message passing,
      operating in log-space for numerical stability.
    - `message_passing_fast`: A faster and more memory efficient message passing
      algorithm that uses einsum, but it is not as stable as message_passing_stable.
    - `brute_force_marginals`: Computes marginals by materializing the full
      joint distribution.
    - `einsum_marginals`: Computes marginals using einsum, generally not
      recommended for large models.
    """
    def __call__(
        self,
        potentials: CliqueVector,
        total: float = 1.0,
        mesh: jax.sharding.Mesh | None = None
    ) -> CliqueVector:
        """
        Computes marginals from log-space potentials.

        Args:
            potentials: A CliqueVector representing the log-space potentials
                of a graphical model.
            total: The normalization factor, typically the total number of
                records or a probability sum. Defaults to 1.0.
            mesh: An optional mesh which determines how the computation will be
                sharded across multiple machines.

        Returns:
            A CliqueVector of the computed marginals.
        """
        ...


def sum_product(factors: list[Factor], dom: Domain, einsum_fn=jnp.einsum) -> Factor:
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
        precision=jax.lax.Precision.HIGHEST
    )
    return Factor(dom, values)


def logspace_sum_product_fast(log_factors: list[Factor], dom: Domain, einsum_fn=jnp.einsum) -> Factor:
    """Numerically stable algorithm for computing sum product in log space.

    This seems to be the most stable algorithm for doing this computation that doesn't
    require materializing sum(log_factors). Materializing sum(log_factors) will
    in general give better numerical stability, but it comes at the cost of
    increased memory usage. This can be potentially mitigated by using
    scan_einsum with an appropriately chosen `sequential` kwarg from mbi.einsum.

    https://github.com/jax-ml/jax/issues/24915

    https://stackoverflow.com/questions/23630277/numerically-stable-way-to-multiply-log-probability-matrices-in-numpy

    Args:
        log_factors: a list of log-space factors.
        dom: The desired domain of the output factor.

    Returns:
        log sum_{S - D} prod_i exp(F_i),
        where
            * F_i = log_factors[i],
            * D is the input domain,
            * S is the union of the domains of F_i
    """
    maxes = [f.max(f.domain.marginalize(dom).attributes) for f in log_factors]
    stable_factors = [(f - m).exp() for f, m in zip(log_factors, maxes)]
    return sum_product(stable_factors, dom, einsum_fn).log() + sum(maxes)


def logspace_sum_product_stable_v1(log_factors: list[Factor], dom: Domain, einsum_fn=jnp.einsum) -> Factor:
    """More stable implementation of logspace_sum_product.

    This ipmlementation may (or may not) materialize a Factor over the domain
    of all elements of log_factors.  Without JIT, it will materialize this "super-factor".
    Under JIT, there may be some instances where the compiler can figure out
    that it does not need to materialize this intermediate to compute the final output.
    """
    del einsum_fn  # unused
    summed = sum(log_factors)  # Might help to put a sharding constraint here
    return summed.logsumexp(summed.domain.marginalize(dom).attributes).transpose(dom.attributes)


def brute_force_marginals(
    potentials: CliqueVector, total: float = 1, mesh: jax.sharding.Mesh | None = None
) -> CliqueVector:
    """Compute marginals from (log-space) potentials by materializing the full joint distribution."""
    P = sum(potentials.arrays.values()).normalize(total, log=True).exp().apply_sharding(mesh)
    marginals = {cl: P.project(cl) for cl in potentials.cliques}
    return CliqueVector(potentials.domain, potentials.cliques, marginals).apply_sharding(mesh)


def einsum_marginals(
    potentials: CliqueVector, total: float = 1, mesh: jax.sharding.Mesh | None = None, einsum_fn=jnp.einsum
) -> CliqueVector:
    """Compute marginals from (log-space) potentials by using einsum.

    This is a "brute-force" approach and is not recommended in practice.
    """
    inputs = list(potentials.arrays.values())
    return CliqueVector(
        potentials.domain,
        potentials.cliques,
        {
            cl: logspace_sum_product_fast(inputs, potentials[cl].domain, einsum_fn)
            .normalize(total, log=True)
            .exp()
            .apply_sharding(mesh)
            for cl in potentials.cliques
        },
    )


@functools.partial(jax.jit, static_argnums=[2])
def message_passing_stable(
    potentials: CliqueVector,
    total: float = 1,
    mesh: jax.sharding.Mesh | None = None,
    jtree: nx.Graph | None = None
) -> CliqueVector:
    """Compute marginals from (log-space) potentials using the message passing algorithm.

    This implementation operates completely in logspace, until the last step where it
    exponentiates the log-beliefs to get marginals.  It is very stable numerically,
    but in general could materialize factors defined over "super-cliques", which
    are the nodes in the junction tree implied by the cliques in potentials.
    Thus, it may require more memory than "message_passing_fast" below.

    Args:
        potentials: The (log-space) potentials of a graphical model.
        total: The normalization factor.
        mesh: The mesh over which the computation should be sharded.
        jtree: An optional junction tree that defines the message passing order.

    Returns:
        The marginals of the graphical model, defined over the same set of cliques
        as the input potentials.  Each marginal is non-negative and sums to "total".
    """
    potentials = potentials.apply_sharding(mesh)
    domain, cliques = potentials.domain, potentials.cliques

    if jtree is None:
      jtree = junction_tree.make_junction_tree(domain, cliques)[0]
    message_order = junction_tree.message_passing_order(jtree)
    maximal_cliques = junction_tree.maximal_cliques(jtree)

    mapping = clique_mapping(maximal_cliques, cliques)
    beliefs = potentials.expand(maximal_cliques).apply_sharding(mesh)

    messages = {}
    for i, j in message_order:
        sep = beliefs[i].domain.invert(tuple(set(i) & set(j)))
        if (j, i) in messages:
            tau = beliefs[i] - messages[(j, i)]
        else:
            tau = beliefs[i]
        messages[(i, j)] = tau.logsumexp(sep)
        beliefs[j] = beliefs[j] + messages[(i, j)]

    return beliefs.normalize(total, log=True).exp().contract(cliques).apply_sharding(mesh)

@functools.partial(jax.jit, static_argnums=[2, 3, 4, 5])
def message_passing_fast(
    potentials: CliqueVector,
    total: float = 1,
    mesh: jax.sharding.Mesh | None = None,
    einsum_fn=jnp.einsum,
    jtree: nx.Graph | None = None,
    logspace_sum_product_fn=logspace_sum_product_fast
) -> CliqueVector:
    """Compute marginals from (log-space) potentials using the message passing algorithm.

    This implementation leverages the "einsum" primitve to compute clique marginals
    without materializing marginals over the super cliques first (nodes in the
    junction tree).  It can be much faster and more memory efficient than
    message_passing_stable, but there are some cases where this
    implementation is not as stable.

    See the stackoverflow thread for the key difficulty here.
    https://stackoverflow.com/questions/23630277/numerically-stable-way-to-multiply-log-probability-matrices-in-numpy

    Args:
        potentials: The (log-space) potentials of a graphical model.
        total: The normalization factor.
        mesh: The mesh over which the computation should be sharded.
        einsum_fn: A function with the same API and semantics as jnp.einsum.
        jtree: An optional junction tree that defines the message passing order.

    Returns:
        The marginals of the graphical model, defined over the same set of cliques
        as the input potentials.  Each marginal is non-negative and sums to "total".
    """
    potentials = potentials.apply_sharding(mesh)
    domain, cliques = potentials.active_domain, potentials.cliques

    jtree = junction_tree.make_junction_tree(domain, cliques)[0]
    message_order = junction_tree.message_passing_order(jtree)
    # TODO: upstream this logic to message_passing_order function
    message_order = [(i, j) for i, j in message_order if len(set(i) & set(j)) > 0]
    maximal_cliques = junction_tree.maximal_cliques(jtree)

    mapping = clique_mapping(maximal_cliques, cliques)
    inverse_mapping = collections.defaultdict(list)
    incoming_messages = collections.defaultdict(list)
    potential_mapping = collections.defaultdict(list)

    for cl in cliques:
        potential_mapping[mapping[cl]].append(potentials[cl])
        inverse_mapping[mapping[cl]].append(cl)

    for i in range(len(message_order)):
        msg = message_order[i]
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

        messages[(i, j)] = logspace_sum_product_fn(inputs, shared, einsum_fn=einsum_fn).apply_sharding(mesh)

    beliefs = {}
    for cl in maximal_cliques:
        input_potentials = potential_mapping[cl]
        input_messages = [messages[key] for key in messages if key[1] == cl]
        inputs = input_potentials + input_messages
        for cl2 in inverse_mapping[cl]:
            beliefs[cl2] = (
                logspace_sum_product_fn(inputs, domain.project(cl2), einsum_fn=einsum_fn)
                .normalize(total, log=True)
                .exp()
                .apply_sharding(mesh)
            )

    return CliqueVector(potentials.domain, cliques, beliefs)

Clique = tuple[str, ...]

def variable_elimination(
    potentials: CliqueVector,
    clique: Clique,
    total: float = 1,
    mesh: jax.sharding.Mesh | None = None,
    evidence: dict[str, int] | None = None,
) -> Factor:
    """Compute an out-of-model/unsupported marginal from the potentials.

    Args:
        potentials: The (log-space) potentials of a Graphical Model.
        clique: The subset of attributes whose marginal you want.
        total: The normalization factor.
        mesh: The mesh over which the computation should be sharded.
        evidence: A dictionary mapping attribute names to observed values.

    Returns:
        The marginal defined over the domain of the input clique, where
        each entry is non-negative and sums to the input total.
    """
    clique = tuple(clique)
    evidence = evidence or {}
    target_clique = tuple(a for a in clique if a not in evidence)

    k = len(potentials.cliques)
    psi = dict(zip(range(k), potentials.arrays.values()))

    if evidence:
        for i in list(psi.keys()):
            psi[i] = psi[i].slice(evidence)

    cliques = [psi[i].domain.attributes for i in psi] + [target_clique]
    domain = potentials.active_domain.marginalize(evidence.keys())
    elim = domain.invert(target_clique)
    elim_order, _ = junction_tree.greedy_order(domain, cliques, elim=elim)

    for z in elim_order:
        psi2 = [psi.pop(i) for i in list(psi.keys()) if z in psi[i].domain]
        psi[k] = sum(psi2).logsumexp([z]).apply_sharding(mesh)
        k += 1
    # this expand covers the case when clique is not in the active domain
    newdom = potentials.domain.project(target_clique)
    zero = Factor(Domain([], []), 0)
    return (
        sum(psi.values(), start=zero)
        .expand(newdom)
        .apply_sharding(mesh)
        .normalize(total, log=True)
        .exp()
        .project(target_clique)
        .apply_sharding(mesh)
    )


def bulk_variable_elimination(
    potentials: CliqueVector,
    marginal_queries: list[tuple[str, ...]],
    total: float = 1.0,
    mesh: jax.sharding.Mesh | None = None,
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
      mesh: The mesh over which the computation should be sharded.

    Returns:
      A CliqueVector with the marginals computed over the specified cliques.
    """
    jitted = jax.jit(variable_elimination, static_argnums=(1, 3))

    # Async + parallel precompilation.
    def _precompile(query):
      return query, jitted.lower(potentials, query, total, mesh).compile()

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
    mesh: jax.sharding.Mesh | None = None,
) -> CliqueVector:
    """ Calculates marginals for all the projections in the list using

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
    jtree = junction_tree.make_junction_tree(potentials.domain, potentials.cliques)[0]
    max_cliques = junction_tree.maximal_cliques(jtree)
    neighbors = { i : tuple(jtree.neighbors(i)) for i in max_cliques }

    # TODO: let's see if we can get rid of this similar to message_passing_fast
    potentials = potentials.expand(max_cliques)

    # TODO: allow these to take in an optional junction tree
    marginals = belief_propagation_oracle(potentials, total, mesh)

    # first calculate P(Cj | Ci) for all neighbors Ci, Cj
    conditional = {}
    for Ci in max_cliques:
        for Cj in neighbors[Ci]:
            Cj: tuple[str, ...]  # networkx does not seem to have the right type annotation.
            Sij = tuple(set(Cj) & set(Ci))
            Z = marginals.project(Cj)
            conditional[(Cj,Ci)] = Z / Z.project(Sij)

    # now iterate through pairs of cliques in order of distance
    # not sure why this API changed and why we need to do this hack.
    nx.set_edge_attributes(jtree, values=1.0, name="weight")  # type: ignore
    pred, dist = nx.floyd_warshall_predecessor_and_distance(jtree) #, weight=None)

    def order_fn(x):
        return dist[x[0]][x[1]]

    results = {}
    for Ci, Cj in sorted(itertools.combinations(max_cliques, 2), key=order_fn):
        Cl = pred[Ci][Cj]
        Y = conditional[(Cj,Cl)]
        if Cl == Ci:
            X = marginals[Ci]
            results[(Ci, Cj)] = results[(Cj, Ci)] = X*Y
        else:
            X = results[(Ci, Cl)]
            S = set(Cl) - set(Ci) - set(Cj)
            results[(Ci, Cj)] = results[(Cj, Ci)] = (X*Y).sum(S)

    results = { domain.canonical(key[0]+key[1]) : results[key] for key in results }

    answers = { }
    for cl in marginal_queries:
        for attr in results:
            if set(cl) <= set(attr):
                answers[cl] = results[attr].project(cl)
                break
        if cl not in answers:
            # just use variable elimination
            answers[cl] = variable_elimination(potentials, cl, total, mesh)

    return CliqueVector(domain, marginal_queries, answers)

def kron_query(
    potentials: CliqueVector,
    query_factors: dict[str, jax.Array],
    total: float = 1,
    mesh: jax.sharding.Mesh | None = None,
    suffix: str = "_answer"
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
    cliques = potentials.cliques + extra_cliques
    inputs = CliqueVector(domain, cliques, new_factors)
    return variable_elimination(inputs, tuple(target_clique), total, mesh)
