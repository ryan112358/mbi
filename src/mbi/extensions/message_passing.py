"""Extension message-passing oracles with efficient constraint handling.

Alternative implementations of the Shafer-Shenoy and implicit message-passing
oracles from :mod:`mbi.marginal_oracles`.  Key differences:

1. **More efficient constraint handling.**  For deterministic (mapping)
   constraints, messages are routed through the constraint in O(|fine|)
   time via ``coarsen``/``refine`` operations, avoiding the full
   O(|fine| × |coarse|) potential materialization used by the core oracles.
2. **AI-written, less well-tested.**  These implementations were developed
   with AI assistance and have narrower test coverage than the core oracles.
   They are kept separate until they mature enough to replace the originals.

Usage::

    from mbi.extensions.message_passing import shafer_shenoy, implicit
"""

from __future__ import annotations

import collections
from collections.abc import Callable, Sequence


import jax
import jax.numpy as jnp
import networkx as nx

from .. import junction_tree, marginal_oracles
from ..clique_utils import clique_mapping
from ..clique_vector import CliqueVector
from ..constraint import Constraint
from ..domain import Domain
from ..factor import Factor


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
  fine, coarse = constraint.domain.attributes
  n_coarse = constraint.domain.shape[1]
  axis = factor.domain.axes((fine,))[0]
  values = _segment_logsumexp(factor.values, constraint.mapping, n_coarse, axis)
  return _replace_variable(
      Factor(factor.domain, values), fine, coarse, n_coarse
  )


def refine(factor, constraint):
  """Log-space coarse-to-fine scatter via indexing."""
  fine, coarse = constraint.domain.attributes
  n_fine = constraint.domain.shape[0]
  axis = factor.domain.axes((coarse,))[0]
  values = jnp.take(factor.values, constraint.mapping, axis=axis)
  return _replace_variable(Factor(factor.domain, values), coarse, fine, n_fine)


def project_to_coarse(factor, constraint):
  """Probability-space fine-to-coarse aggregation via segment-sum."""
  fine, coarse = constraint.domain.attributes
  n_coarse = constraint.domain.shape[1]
  axis = factor.domain.axes((fine,))[0]
  values = jnp.moveaxis(factor.values, axis, 0)
  result = jnp.zeros((n_coarse,) + values.shape[1:], dtype=values.dtype)
  result = result.at[constraint.mapping].add(values)
  values = jnp.moveaxis(result, 0, axis)
  return _replace_variable(
      Factor(factor.domain, values), fine, coarse, n_coarse
  )


def _constraint_slice(factor, constraint):
  """Remove the coarse variable by selecting valid constraint entries.

  For each value of the fine variable a, selects coarse = mapping[a].
  """
  fine, coarse = constraint.domain.attributes
  n_fine = constraint.domain.shape[0]
  fine_axis = factor.domain.axes((fine,))[0]
  coarse_axis = factor.domain.axes((coarse,))[0]

  idx_shape = [1] * len(factor.domain.shape)
  idx_shape[fine_axis] = n_fine
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
  fine, coarse = constraint.domain.attributes
  n_fine, n_coarse = constraint.domain.shape
  fine_axis = factor.domain.axes((fine,))[0]
  coarse_axis = fine_axis + 1

  indicator = jnp.zeros((n_fine, n_coarse), dtype=factor.values.dtype)
  indicator = indicator.at[jnp.arange(n_fine), constraint.mapping].set(1.0)

  ind_shape = [1] * (len(factor.domain.shape) + 1)
  ind_shape[fine_axis] = n_fine
  ind_shape[coarse_axis] = n_coarse

  values = jnp.expand_dims(factor.values, axis=coarse_axis)
  result_values = values * indicator.reshape(ind_shape)

  attrs = list(factor.domain.attributes)
  shape = list(factor.domain.shape)
  attrs.insert(coarse_axis, coarse)
  shape.insert(coarse_axis, n_coarse)
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
  fine, coarse = constraint.domain.attributes
  # Pre-process: slice any input that has both fine and coarse.
  clean = []
  for f in inputs:
    if fine in f.domain and coarse in f.domain:
      clean.append(_constraint_slice(f, constraint))
    else:
      clean.append(f)
  fine_sum, coarse_sum = _partition_by_variable(clean, fine)

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
    fine, coarse = c.domain.attributes
    if fine not in tau.domain or coarse not in tau.domain:
      continue
    tau = _constraint_slice(tau, c)
    if coarse in sep_vars and fine not in sep_vars:
      # Only coarse in separator — must coarsen to reach it.
      keep = tuple((sep_vars - {coarse}) | {fine})
      extra = set(tau.domain.attributes) - set(keep)
      if extra:
        tau = tau.logsumexp(tau.domain.invert(keep))
      return coarsen(tau, c)
    # Fine in separator (or both) — keep per-element info.
    sep_without_coarse = tuple(sep_vars - {coarse})
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
    cs = [
        c
        for c in constraints
        if c.domain.attributes[0] in mc_set and c.domain.attributes[1] in mc_set
    ]
    if not cs:
      continue
    if len(cs) == 1 and mc_set == set(cs[0].domain.attributes):
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
    fine, coarse = c.domain.attributes

    inputs = [
        messages[(k, con_mc)]
        for k in neighbors[con_mc]
        if (k, con_mc) in messages
    ]
    inputs.extend(constraint_init.get(con_mc, []))
    fine_sum, coarse_sum = _partition_by_variable(inputs, fine)

    # Reduce any inputs containing both variables to fine-only.
    if fine_sum is not None and coarse in fine_sum.domain:
      fine_sum = _constraint_slice(fine_sum, c)

    belief = _add_optional(
        fine_sum, refine(coarse_sum, c) if coarse_sum else None
    )
    if belief is None:
      belief = Factor.zeros(domain.project((fine,)))

    belief = belief.normalize(total, log=True).exp()
    if fine in clique and coarse in clique:
      return _constraint_expand(belief, c).project(clique)
    if fine in clique:
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
              if c.domain.attributes[1] == var
              and c.domain.attributes[0] in belief_cl
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


def _fold_non_deterministic(potentials, constraints):
  """Fold non-deterministic constraints as materialized potentials."""
  non_det = [c for c in constraints if not c.is_deterministic]
  if not non_det:
    return potentials
  domain = potentials.domain
  cliques = list(potentials.cliques)
  arrays = {cl: potentials[cl] for cl in cliques}
  for c in non_det:
    cl = domain.canonical(c.clique)
    factor = c.potential
    if cl in arrays:
      arrays[cl] = arrays[cl] + factor
    else:
      cliques.append(cl)
      arrays[cl] = factor
  return CliqueVector(domain, cliques, arrays)


@jax.jit(static_argnames=['jtree'])
def shafer_shenoy(
    potentials: CliqueVector,
    total: float = 1,
    jtree: nx.Graph | None = None,
    constraints: Sequence[Constraint] = (),
) -> CliqueVector:
  """Compute marginals using Shafer-Shenoy with constraint-aware shortcuts.

  Deterministic (mapping) constraints are routed through efficient O(|fine|)
  coarsen/refine operations. General (valid/invalid) constraints are folded
  into potentials as materialized ``-inf``/``0`` factors.

  Args:
      potentials: The (log-space) potentials of a graphical model.
      total: The normalization factor.
      jtree: An optional junction tree that defines the message passing
          order.
      constraints: Structural constraints to handle.

  Returns:
      The marginals of the graphical model.
  """
  if len(potentials.cliques) == 0:
    return CliqueVector(potentials.domain, [], {})

  # Fold non-deterministic constraints as materialized potentials.
  potentials = _fold_non_deterministic(potentials, constraints)
  det_constraints = tuple(c for c in constraints if c.is_deterministic)

  domain, cliques = potentials.domain, potentials.cliques

  # Build the junction tree, including constraint edges.
  extra = [
      domain.canonical(c.clique)
      for c in det_constraints
      if domain.canonical(c.clique) not in cliques
  ]
  if jtree is None:
    jtree = junction_tree.make_junction_tree(
        domain, tuple(cliques) + tuple(extra)
    )[0]
  message_order = junction_tree.message_passing_order(jtree)
  max_cliques = junction_tree.maximal_cliques(jtree)
  pure_constraint, embedded = _classify_cliques(max_cliques, det_constraints)
  neighbors = {cl: list(jtree.neighbors(cl)) for cl in max_cliques}

  # Partition user potentials by pure-constraint membership.
  constraint_init = {}
  non_constraint_cliques = []
  for cl in cliques:
    con_mc = next((mc for mc in pure_constraint if set(cl) <= set(mc)), None)
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
    beliefs0 = nc.expand(non_pure)
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
      fine, coarse = c.domain.attributes
      msg = _constraint_message(c, inputs, fine in sep_vars)
      if msg is None:
        sizes = dict(zip(c.domain.attributes, c.domain.shape))
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
        messages[(i, j)] = tau.logsumexp(tau.domain.invert(tuple(sep_vars)))

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
      expanded = _try_expand_belief(cl, result_cv, det_constraints)
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

  return CliqueVector(domain, cliques, result)


# ---------------------------------------------------------------------------
# Hybrid implicit + constraint-aware message passing
# ---------------------------------------------------------------------------


@jax.jit(static_argnames=['jtree', 'contraction'])
def implicit(
    potentials: CliqueVector,
    total: float = 1,
    jtree: nx.Graph | None = None,
    *,
    constraints: Sequence[Constraint] = (),
    contraction: Callable = marginal_oracles.einsum_materialized,
) -> CliqueVector:
  """Hybrid message passing: implicit for standard cliques, SS for constraints.

  Uses the memory-efficient implicit contraction for cliques that do not
  involve deterministic constraints, and falls back to Shafer-Shenoy style
  constraint routing for pure and embedded constraint cliques. General
  (valid/invalid) constraints are folded in as materialized potentials.

  Args:
      potentials: The (log-space) potentials of a graphical model.
      total: The normalization factor.
      jtree: An optional junction tree.
      constraints: Structural constraints to handle.
      contraction: Contraction function for log-space sum-product.

  Returns:
      The marginals of the graphical model.
  """

  if len(potentials.cliques) == 0:
    return CliqueVector(potentials.domain, [], {})

  # Fold non-deterministic constraints as materialized potentials.
  potentials = _fold_non_deterministic(potentials, constraints)
  det_constraints = tuple(c for c in constraints if c.is_deterministic)

  domain, cliques = potentials.domain, potentials.cliques

  # Build junction tree with constraint edges.
  extra = [
      domain.canonical(c.clique)
      for c in det_constraints
      if domain.canonical(c.clique) not in cliques
  ]
  if jtree is None:
    jtree = junction_tree.make_junction_tree(domain, list(cliques) + extra)[0]
  message_order = junction_tree.message_passing_order(jtree)
  max_cliques = junction_tree.maximal_cliques(jtree)
  pure_constraint, embedded = _classify_cliques(max_cliques, det_constraints)
  neighbors = {cl: list(jtree.neighbors(cl)) for cl in max_cliques}

  # Map user cliques → maximal cliques.
  mapping = clique_mapping(max_cliques, cliques, domain=domain)
  inverse_mapping = collections.defaultdict(list)
  potential_mapping = collections.defaultdict(list)

  for cl in cliques:
    mc = mapping[cl]
    potential_mapping[mc].append(potentials[cl])
    inverse_mapping[mc].append(cl)

  # Separate potentials absorbed by pure constraint cliques.
  constraint_init = {}
  for mc in pure_constraint:
    constraint_init[mc] = potential_mapping.pop(mc, [])

  non_pure = [mc for mc in max_cliques if mc not in pure_constraint]

  # Pre-build incoming message graph for implicit contraction.
  incoming_messages = collections.defaultdict(list)
  for i, msg in enumerate(message_order):
    for j in range(i):
      msg2 = message_order[j]
      if msg[0] == msg2[1] and msg[1] != msg2[0]:
        incoming_messages[msg].append(msg2)

  # ---- Forward-backward message passing ----
  messages = {}
  for i, j in message_order:
    sep_vars = set(i) & set(j)

    if i in pure_constraint:
      # Constraint routing — O(|fine|).
      c = pure_constraint[i]
      inputs = [
          messages[(k, i)]
          for k in neighbors[i]
          if k != j and (k, i) in messages
      ]
      inputs.extend(constraint_init.get(i, []))
      fine, coarse = c.domain.attributes
      msg = _constraint_message(c, inputs, fine in sep_vars)
      if msg is None:
        sizes = dict(zip(c.domain.attributes, c.domain.shape))
        sep = tuple(sep_vars)
        msg = Factor.zeros(Domain(list(sep), [sizes[s] for s in sep]))
      messages[(i, j)] = msg

    elif i in embedded:
      # Normalize-to-fine + contraction (no super-clique expansion).
      shared = domain.project(tuple(sep_vars))
      input_potentials = potential_mapping.get(i, [])
      input_messages = [messages[key] for key in incoming_messages[(i, j)]]
      inputs = input_potentials + input_messages

      for c in embedded[i]:
        inputs = _normalize_to_fine(inputs, c)

      target = shared
      post_ops = []
      for c in embedded[i]:
        target, post_op = _target_for_constraint(target, c)
        post_ops.append((c, post_op))

      for attr in target.attributes:
        if not any(attr in inp.domain.attributes for inp in inputs):
          inputs.append(Factor.zeros(domain.project([attr])))

      msg = contraction(inputs, target)
      for c, post_op in post_ops:
        msg = _postprocess_log(msg, c, post_op, shared)
      messages[(i, j)] = msg

    else:
      # Implicit contraction — no super-clique expansion.
      shared = domain.project(tuple(sep_vars))
      input_potentials = potential_mapping.get(i, [])
      input_messages = [messages[key] for key in incoming_messages[(i, j)]]
      inputs = input_potentials + input_messages
      for attr in shared.attributes:
        if not any(attr in inp.domain.attributes for inp in inputs):
          inputs.append(Factor.zeros(domain.project([attr])))
      messages[(i, j)] = contraction(inputs, shared)

  # ---- Compute beliefs ----
  result = {}
  for mc in non_pure:
    if mc in pure_constraint:
      continue

    input_potentials = potential_mapping.get(mc, [])
    input_messages = [val for key, val in messages.items() if key[1] == mc]
    inputs = input_potentials + input_messages

    if mc in embedded:
      # Normalize to fine, contract, then post-process.
      for c in embedded[mc]:
        inputs = _normalize_to_fine(inputs, c)

    for cl in inverse_mapping.get(mc, []):
      target = domain.project(cl)

      if mc in embedded:
        adj_target = target
        post_ops = []
        for c in embedded[mc]:
          adj_target, post_op = _target_for_constraint(adj_target, c)
          post_ops.append((c, post_op))

        adj_inputs = list(inputs)
        for attr in adj_target.attributes:
          if not any(attr in inp.domain.attributes for inp in adj_inputs):
            adj_inputs.append(Factor.zeros(domain.project([attr])))

        belief = contraction(adj_inputs, adj_target)
        # Normalize/exp to probability space, then post-process.
        belief = belief.normalize(total, log=True).exp()
        for c, post_op in post_ops:
          belief = _postprocess_prob(belief, c, post_op, target)
        result[cl] = belief
      else:
        adj_inputs = list(inputs)
        for attr in target.attributes:
          if not any(attr in inp.domain.attributes for inp in adj_inputs):
            adj_inputs.append(Factor.zeros(domain.project([attr])))
        belief = contraction(adj_inputs, target)
        result[cl] = belief.normalize(total, log=True).exp()

  # Pure constraint cliques: derive marginals.
  for cl in cliques:
    if cl in result:
      continue
    con_mc = next((mc for mc in pure_constraint if set(cl) <= set(mc)), None)
    if con_mc is not None:
      result[cl] = _derive_pure_marginal(
          cl,
          pure_constraint,
          constraint_init,
          neighbors,
          messages,
          total,
          domain,
      )

  return CliqueVector(potentials.domain, cliques, result)


# ---------------------------------------------------------------------------
# Helpers for embedded constraint handling via implicit contraction
# ---------------------------------------------------------------------------


def _normalize_to_fine(inputs, constraint):
  """Normalize all inputs to fine-variable space."""
  fine, coarse = constraint.domain.attributes
  cleaned = []
  for f in inputs:
    has_fine = fine in f.domain
    has_coarse = coarse in f.domain
    if has_fine and has_coarse:
      cleaned.append(_constraint_slice(f, constraint))
    elif has_coarse and not has_fine:
      cleaned.append(refine(f, constraint))
    else:
      cleaned.append(f)
  return cleaned


def _target_for_constraint(target_domain, constraint):
  """Adjust target domain and return (domain, post_op) for post-processing."""
  fine, coarse = constraint.domain.attributes
  n_fine = constraint.domain.shape[0]
  has_fine = fine in target_domain
  has_coarse = coarse in target_domain

  if not has_fine and not has_coarse:
    return target_domain, 'none'
  if has_fine and not has_coarse:
    return target_domain, 'fine_only'
  if has_coarse and not has_fine:
    attrs = list(target_domain.attributes)
    shape = list(target_domain.shape)
    idx = attrs.index(coarse)
    attrs[idx] = fine
    shape[idx] = n_fine
    return Domain(attrs, shape), 'coarsen'
  # Both — drop coarse from target, will expand later.
  attrs = [a for a in target_domain.attributes if a != coarse]
  shape = [
      s
      for a, s in zip(target_domain.attributes, target_domain.shape)
      if a != coarse
  ]
  return Domain(attrs, shape), 'expand'


def _postprocess_prob(result, constraint, post_op, original_target):
  """Post-process a probability-space result."""
  if post_op in {'none', 'fine_only'}:
    return result
  if post_op == 'coarsen':
    return project_to_coarse(result, constraint)
  if post_op == 'expand':
    expanded = _constraint_expand(result, constraint)
    return expanded.project(tuple(original_target.attributes))
  raise ValueError(f'Unknown post_op: {post_op}')


def _postprocess_log(result, constraint, post_op, original_target):
  """Post-process a log-space result (for messages)."""
  if post_op in {'none', 'fine_only'}:
    return result
  if post_op == 'coarsen':
    return coarsen(result, constraint)
  if post_op == 'expand':
    return result
  raise ValueError(f'Unknown post_op: {post_op}')


def default_oracle(
    cliques: tuple[tuple[str, ...], ...] | None = None,
    domain: Domain | None = None,
    backend: str | None = None,
) -> marginal_oracles.MarginalOracle:
  """Select the best constraint-aware oracle for the given setting.

  When constraints are present, this oracle uses the extensions
  implementations which route messages through deterministic constraints
  efficiently.  When no constraints are passed at call time, behavior is
  identical to the core ``marginal_oracles.default_oracle``.

  The selection heuristic mirrors the core oracle:

  - **CPU**: ``shafer_shenoy`` (XLA compiler less effective on CPU).
  - **GPU/TPU, large cliques (>= 1M)**: ``implicit``.
  - **GPU/TPU, small cliques**: ``shafer_shenoy``.
  """
  if backend is None:
    backend = jax.default_backend()

  # CPU: always SS (XLA compiler less effective on CPU).
  if backend == 'cpu':
    return shafer_shenoy

  # GPU/TPU with large cliques: implicit.
  if cliques is not None and domain is not None:
    jtree = junction_tree.make_junction_tree(domain, cliques)[0]
    max_cliques = junction_tree.maximal_cliques(jtree)
    max_size = max(domain.project(cl).size() for cl in max_cliques)
    if max_size >= 1_000_000:
      return implicit

  # GPU/TPU with small cliques.
  return shafer_shenoy
