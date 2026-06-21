"""Efficient handling of deterministic variable constraints.

Provides utilities for exploiting deterministic relationships between variables
(e.g., A' = f(A) where A' is a coarsening of A) during message passing in
graphical models. Instead of materializing full |A| x |A'| constraint factors,
messages are routed through the constraint in O(|A|) time using coarsen/refine
operations.
"""

from __future__ import annotations

import functools

import attr
import jax
import jax.numpy as jnp
import networkx as nx
import numpy as np

from .. import junction_tree
from ..clique_vector import CliqueVector
from ..domain import Domain
from ..factor import Factor


@attr.dataclass(frozen=True, hash=False, eq=False)
class DeterministicConstraint:
    """A deterministic relationship: coarse = f(fine).

    Attributes:
        fine: The finer-grained variable name.
        coarse: The coarser variable name.
        mapping: Array of shape (|fine|,) mapping fine values to coarse values.
    """

    fine: str
    coarse: str
    mapping: np.ndarray

    def __hash__(self):
        return hash((self.fine, self.coarse, self.mapping.tobytes()))

    def __eq__(self, other):
        if not isinstance(other, DeterministicConstraint):
            return NotImplemented
        return (
            self.fine == other.fine
            and self.coarse == other.coarse
            and np.array_equal(self.mapping, other.mapping)
        )

    @property
    def clique(self) -> tuple[str, ...]:
        return tuple(sorted((self.fine, self.coarse)))

    @property
    def n_fine(self) -> int:
        return len(self.mapping)

    @property
    def n_coarse(self) -> int:
        return int(self.mapping.max()) + 1


def _replace_variable(factor, old, new, new_size):
    """Replace a variable name and size in a Factor's domain."""
    attrs = list(factor.domain.attributes)
    shape = list(factor.domain.shape)
    idx = attrs.index(old)
    attrs[idx] = new
    shape[idx] = new_size
    return Factor(Domain(attrs, shape), factor.values)


def _segment_logsumexp(values, mapping, n_segments, axis):
    """Logsumexp grouped by segment IDs along the given axis."""
    values = jnp.moveaxis(values, axis, 0)
    max_vals = jnp.full((n_segments,) + values.shape[1:], -jnp.inf)
    max_vals = max_vals.at[mapping].max(values)
    shifted = jnp.exp(values - max_vals[mapping])
    summed = jnp.zeros_like(max_vals)
    summed = summed.at[mapping].add(shifted)
    result = jnp.log(summed) + max_vals
    return jnp.moveaxis(result, 0, axis)


def coarsen(factor, constraint):
    """Log-space fine-to-coarse aggregation via segment-logsumexp."""
    axis = factor.domain.axes((constraint.fine,))[0]
    values = _segment_logsumexp(
        factor.values, constraint.mapping, constraint.n_coarse, axis
    )
    return _replace_variable(
        Factor(factor.domain, values),
        constraint.fine,
        constraint.coarse,
        constraint.n_coarse,
    )


def refine(factor, constraint):
    """Log-space coarse-to-fine scatter via indexing."""
    axis = factor.domain.axes((constraint.coarse,))[0]
    values = jnp.take(factor.values, constraint.mapping, axis=axis)
    return _replace_variable(
        Factor(factor.domain, values),
        constraint.coarse,
        constraint.fine,
        constraint.n_fine,
    )


def project_to_coarse(factor, constraint):
    """Probability-space fine-to-coarse aggregation via segment-sum."""
    axis = factor.domain.axes((constraint.fine,))[0]
    values = jnp.moveaxis(factor.values, axis, 0)
    result = jnp.zeros(
        (constraint.n_coarse,) + values.shape[1:], dtype=values.dtype
    )
    result = result.at[constraint.mapping].add(values)
    values = jnp.moveaxis(result, 0, axis)
    return _replace_variable(
        Factor(factor.domain, values),
        constraint.fine,
        constraint.coarse,
        constraint.n_coarse,
    )


def _constraint_slice(factor, constraint):
    """Remove the coarse variable by selecting valid constraint entries.

    For each value of the fine variable a, selects coarse = mapping[a].
    """
    fine_axis = factor.domain.axes((constraint.fine,))[0]
    coarse_axis = factor.domain.axes((constraint.coarse,))[0]

    idx_shape = [1] * len(factor.domain.shape)
    idx_shape[fine_axis] = constraint.n_fine
    indices = jnp.array(constraint.mapping).reshape(idx_shape)

    values = jnp.take_along_axis(factor.values, indices, axis=coarse_axis)
    values = jnp.squeeze(values, axis=coarse_axis)

    attrs = [
        a for i, a in enumerate(factor.domain.attributes) if i != coarse_axis
    ]
    shape = [s for i, s in enumerate(factor.domain.shape) if i != coarse_axis]
    return Factor(Domain(attrs, shape), values)


def _constraint_expand(factor, constraint):
    """Re-introduce the coarse variable using the constraint mapping.

    Inverse of _constraint_slice. For each fine value a, places the
    factor's values at coarse = mapping[a] and zeros elsewhere.
    """
    fine_axis = factor.domain.axes((constraint.fine,))[0]
    coarse_axis = fine_axis + 1

    indicator = jnp.zeros(
        (constraint.n_fine, constraint.n_coarse), dtype=factor.values.dtype
    )
    indicator = indicator.at[
        jnp.arange(constraint.n_fine), constraint.mapping
    ].set(1.0)

    ind_shape = [1] * (len(factor.domain.shape) + 1)
    ind_shape[fine_axis] = constraint.n_fine
    ind_shape[coarse_axis] = constraint.n_coarse

    values = jnp.expand_dims(factor.values, axis=coarse_axis)
    result_values = values * indicator.reshape(ind_shape)

    attrs = list(factor.domain.attributes)
    shape = list(factor.domain.shape)
    attrs.insert(coarse_axis, constraint.coarse)
    shape.insert(coarse_axis, constraint.n_coarse)
    return Factor(Domain(attrs, shape), result_values)


def _add_optional(a, b):
    """Add two Factors where either may be None."""
    if a is None:
        return b
    if b is None:
        return a
    return a + b


def _partition_by_variable(items, variable):
    """Partition Factors into those containing variable and those not."""
    has_var, no_var = None, None
    for f in items:
        if variable in f.domain:
            has_var = _add_optional(has_var, f)
        else:
            no_var = _add_optional(no_var, f)
    return has_var, no_var


def _constraint_message(constraint, inputs, target_has_fine):
    """Compute the outgoing message from a pure constraint clique.

    Routes messages through the constraint in O(|fine|) time.
    """
    # Pre-process: slice any input that has both fine and coarse.
    clean = []
    for f in inputs:
        if constraint.fine in f.domain and constraint.coarse in f.domain:
            clean.append(_constraint_slice(f, constraint))
        else:
            clean.append(f)
    fine_sum, coarse_sum = _partition_by_variable(clean, constraint.fine)

    if target_has_fine:
        result = _add_optional(
            fine_sum,
            refine(coarse_sum, constraint) if coarse_sum else None,
        )
    else:
        result = _add_optional(
            coarsen(fine_sum, constraint) if fine_sum else None,
            coarse_sum,
        )
    return result


def _embedded_message(tau, sep_vars, constraints_list):
    """Outgoing message from a clique with an embedded constraint."""
    for c in constraints_list:
        if c.fine not in tau.domain or c.coarse not in tau.domain:
            continue
        tau = _constraint_slice(tau, c)
        if c.coarse in sep_vars and c.fine not in sep_vars:
            # Only coarse in separator — must coarsen to reach it.
            keep = tuple((sep_vars - {c.coarse}) | {c.fine})
            extra = set(tau.domain.attributes) - set(keep)
            if extra:
                tau = tau.logsumexp(tau.domain.invert(keep))
            return coarsen(tau, c)
        # Fine in separator (or both) — keep per-element info.
        sep_without_coarse = tuple(sep_vars - {c.coarse})
        return tau.logsumexp(tau.domain.invert(sep_without_coarse))
    return tau.logsumexp(tau.domain.invert(tuple(sep_vars)))


def _classify_cliques(max_cliques, constraints):
    """Classify maximal cliques by their relationship to constraints.

    Returns:
        pure_constraint: dict mapping mc -> constraint for standalone
            {fine, coarse} nodes.
        embedded: dict mapping mc -> [constraints] for cliques where the
            constraint is absorbed into a larger maximal clique.
    """
    pure_constraint = {}
    embedded = {}
    for mc in max_cliques:
        mc_set = set(mc)
        cs = [c for c in constraints if c.fine in mc_set and c.coarse in mc_set]
        if not cs:
            continue
        if len(cs) == 1 and mc_set == {cs[0].fine, cs[0].coarse}:
            pure_constraint[mc] = cs[0]
        else:
            embedded[mc] = cs
    return pure_constraint, embedded


def _derive_pure_marginal(
    clique,
    pure_constraint,
    constraint_init,
    neighbors,
    messages,
    total,
    domain,
):
    """Derive a marginal for a clique subsumed by a pure constraint."""
    for con_mc, c in pure_constraint.items():
        if not set(clique) <= set(con_mc):
            continue

        inputs = [
            messages[(k, con_mc)]
            for k in neighbors[con_mc]
            if (k, con_mc) in messages
        ]
        inputs.extend(constraint_init.get(con_mc, []))
        fine_sum, coarse_sum = _partition_by_variable(inputs, c.fine)

        # Reduce any inputs containing both variables to fine-only.
        if fine_sum is not None and c.coarse in fine_sum.domain:
            fine_sum = _constraint_slice(fine_sum, c)

        belief = _add_optional(
            fine_sum, refine(coarse_sum, c) if coarse_sum else None
        )
        if belief is None:
            belief = Factor.zeros(domain.project((c.fine,)))

        belief = belief.normalize(total, log=True).exp()
        if c.fine in clique and c.coarse in clique:
            return _constraint_expand(belief, c).project(clique)
        if c.fine in clique:
            return belief.project(clique)
        return project_to_coarse(belief, c).project(clique)

    raise ValueError(f'Clique {clique} not supported by any constraint.')


def _try_expand_belief(clique, result_cv, constraints):
    """Re-expand a belief by re-introducing sliced constraint variables."""
    cl_set = set(clique)
    for belief_cl in result_cv.cliques:
        missing = cl_set - set(belief_cl)
        if not missing or not set(belief_cl) & cl_set:
            continue
        expansions = []
        for var in missing:
            c = next(
                (
                    c
                    for c in constraints
                    if c.coarse == var and c.fine in belief_cl
                ),
                None,
            )
            if c is None:
                break
            expansions.append(c)
        else:
            if cl_set - missing <= set(belief_cl):
                factor = result_cv[belief_cl]
                for c in expansions:
                    factor = _constraint_expand(factor, c)
                return factor.project(clique)
    return None


@functools.partial(jax.jit, static_argnums=[2, 3, 4])
def message_passing_with_constraints(
    potentials: CliqueVector,
    total: float = 1,
    mesh: jax.sharding.Mesh | None = None,
    jtree: nx.Graph | None = None,
    constraints: tuple[DeterministicConstraint, ...] = (),
) -> CliqueVector:
    """Compute marginals using Shafer-Shenoy with constraint-aware shortcuts.

    Extends message_passing_shafer_shenoy to handle deterministic constraints
    without materializing |fine| x |coarse| factors.

    Args:
        potentials: The (log-space) potentials of a graphical model.
        total: The normalization factor.
        mesh: The mesh over which the computation should be sharded.
        jtree: An optional junction tree that defines the message passing
            order.
        constraints: Deterministic constraints to handle efficiently.

    Returns:
        The marginals of the graphical model.
    """
    if len(potentials.cliques) == 0:
        return CliqueVector(potentials.domain, [], {})

    potentials = potentials.apply_sharding(mesh)
    domain, cliques = potentials.domain, potentials.cliques

    # Build the junction tree, including constraint edges.
    extra = [
        domain.canonical(c.clique)
        for c in constraints
        if domain.canonical(c.clique) not in cliques
    ]
    if jtree is None:
        jtree = junction_tree.make_junction_tree(
            domain, cliques + tuple(extra)
        )[0]
    message_order = junction_tree.message_passing_order(jtree)
    max_cliques = junction_tree.maximal_cliques(jtree)
    pure_constraint, embedded = _classify_cliques(max_cliques, constraints)
    neighbors = {cl: list(jtree.neighbors(cl)) for cl in max_cliques}

    # Partition user potentials by pure-constraint membership.
    constraint_init = {}
    non_constraint_cliques = []
    for cl in cliques:
        con_mc = next(
            (mc for mc in pure_constraint if set(cl) <= set(mc)), None
        )
        if con_mc:
            constraint_init.setdefault(con_mc, []).append(potentials[cl])
        else:
            non_constraint_cliques.append(cl)

    # Initialize beliefs for non-pure-constraint maximal cliques.
    non_pure = [mc for mc in max_cliques if mc not in pure_constraint]
    if non_constraint_cliques:
        nc = CliqueVector(
            domain,
            non_constraint_cliques,
            {cl: potentials[cl] for cl in non_constraint_cliques},
        )
        beliefs0 = nc.expand(non_pure).apply_sharding(mesh)
    else:
        beliefs0 = CliqueVector(
            domain,
            non_pure,
            {mc: Factor.zeros(domain.project(mc)) for mc in non_pure},
        )

    # Forward-backward message passing.
    messages = {}
    for i, j in message_order:
        sep_vars = set(i) & set(j)

        if i in pure_constraint:
            c = pure_constraint[i]
            inputs = [
                messages[(k, i)]
                for k in neighbors[i]
                if k != j and (k, i) in messages
            ]
            inputs.extend(constraint_init.get(i, []))
            msg = _constraint_message(c, inputs, c.fine in sep_vars)
            if msg is None:
                sizes = {c.fine: c.n_fine, c.coarse: c.n_coarse}
                sep = tuple(sep_vars)
                msg = Factor.zeros(Domain(list(sep), [sizes[s] for s in sep]))
            messages[(i, j)] = msg
        else:
            tau = beliefs0[i]
            for k in neighbors[i]:
                if k != j and (k, i) in messages:
                    tau = tau + messages[(k, i)]
            if i in embedded:
                messages[(i, j)] = _embedded_message(tau, sep_vars, embedded[i])
            else:
                messages[(i, j)] = tau.logsumexp(
                    tau.domain.invert(tuple(sep_vars))
                )

    # Collect final beliefs, applying constraint slicing.
    belief_map = {}
    belief_cliques = []
    for cl in max_cliques:
        if cl in pure_constraint:
            continue
        b = beliefs0[cl]
        for k in neighbors[cl]:
            if (k, cl) in messages:
                b = b + messages[(k, cl)]
        if cl in embedded:
            for c in embedded[cl]:
                b = _constraint_slice(b, c)
        key = tuple(b.domain.attributes)
        belief_map[key] = b
        belief_cliques.append(key)

    result_cv = CliqueVector(domain, belief_cliques, belief_map)
    result_cv = result_cv.normalize(total, log=True).exp()

    # Project back to user cliques.
    result = {}
    for cl in cliques:
        if result_cv.supports(cl):
            result[cl] = result_cv.project(cl)
        elif any(set(cl) <= set(mc) for mc in pure_constraint):
            # Clique subsumed by a pure constraint — derive from
            # constraint messages and absorbed potentials.
            result[cl] = _derive_pure_marginal(
                cl,
                pure_constraint,
                constraint_init,
                neighbors,
                messages,
                total,
                domain,
            )
        else:
            # Try re-expanding beliefs where constraint slicing removed
            # a coarse variable the user needs.
            expanded = _try_expand_belief(cl, result_cv, constraints)
            if expanded is not None:
                result[cl] = expanded
            else:
                result[cl] = _derive_pure_marginal(
                    cl,
                    pure_constraint,
                    constraint_init,
                    neighbors,
                    messages,
                    total,
                    domain,
                )

    return CliqueVector(domain, cliques, result).apply_sharding(mesh)
