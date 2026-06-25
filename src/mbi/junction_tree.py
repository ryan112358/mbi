"""Utilities for constructing and working with junction trees.

This module provides functions for building junction trees from a given domain
and set of cliques. Junction trees are fundamental structures in graphical model
inference, enabling efficient message passing algorithms. Functions include
finding maximal cliques, determining message passing orders, graph triangulation,
computing greedy elimination orders, and estimating model size.
"""

import itertools
from collections import OrderedDict
from collections.abc import Collection, Sequence
from typing import TypeAlias

import networkx as nx
import numpy as np

from .domain import Domain

Clique: TypeAlias = tuple[str, ...]


def maximal_cliques(junction_tree: nx.Graph) -> list[Clique]:
    """Return the list of maximal cliques in the model."""
    return list(nx.dfs_preorder_nodes(junction_tree))


def message_passing_order(
    junction_tree: nx.Graph,
) -> list[tuple[Clique, Clique]]:
    """Return a valid message passing order."""
    edges = set()
    messages = list(junction_tree.edges()) + [
        (b, a) for a, b in junction_tree.edges()
    ]
    for m1 in messages:
        for m2 in messages:
            if m1[1] == m2[0] and m1[0] != m2[1]:
                edges.add((m1, m2))
    graph = nx.DiGraph()
    graph.add_nodes_from(messages)
    graph.add_edges_from(edges)
    return list(nx.topological_sort(graph))


def _make_graph(domain: Domain, cliques: Collection[Clique]) -> nx.Graph:
    """Create a graph from the domain and cliques."""
    graph = nx.Graph()
    graph.add_nodes_from(domain.attributes)
    for cl in cliques:
        graph.add_edges_from(itertools.combinations(cl, 2))
    return graph


def _triangulated(graph: nx.Graph, order: list[str]) -> nx.Graph:
    """Triangulate the graph using the given elimination order."""
    edges = set()
    graph2 = nx.Graph(graph)
    for node in order:
        tmp = set(itertools.combinations(graph2.neighbors(node), 2))
        edges |= tmp
        graph2.add_edges_from(tmp)
        graph2.remove_node(node)
    tri = nx.Graph(graph)
    tri.add_edges_from(edges)
    return tri


def greedy_order(
    domain: Domain,
    cliques: Sequence[Clique],
    stochastic: bool = False,
    elim: list[str] | None = None,
) -> tuple[list[str], int]:
    """Compute a greedy elimination order."""
    order = []
    unmarked = elim if elim is not None else list(domain.attributes)
    cliques = set(cliques)
    total_cost = 0
    for _ in range(len(unmarked)):
        cost = OrderedDict()
        for a in unmarked:
            neighbors = [cl for cl in cliques if a in cl]
            variables = tuple(set.union(set(), *[set(n) for n in neighbors]))
            newdom = domain.project(variables)
            cost[a] = newdom.size()

        if stochastic:
            choices = list(unmarked)
            costs = np.array([cost[a] for a in choices], dtype=float)
            probas = np.max(costs) - costs + 1
            probas /= probas.sum()
            i = np.random.choice(probas.size, p=probas)
            a = choices[i]
        else:
            a = min(cost, key=lambda a: cost[a])

        order.append(a)
        unmarked.remove(a)
        neighbors = [cl for cl in cliques if a in cl]
        variables = tuple(set.union(set(), *[set(n) for n in neighbors]) - {a})
        cliques -= set(neighbors)
        cliques.add(variables)
        total_cost += cost[a]

    return order, total_cost


def make_junction_tree(
    domain: Domain,
    cliques: Collection[Clique],
    elimination_order: list[str] | int | None = None,
) -> tuple[nx.Graph, list[str]]:
    """Create a junction tree."""
    cliques = [tuple(cl) for cl in cliques]
    graph = _make_graph(domain, cliques)

    if (
        elimination_order is None
        and not nx.is_empty(graph)
        and nx.is_tree(graph)
    ):
        elimination_order = list(nx.dfs_postorder_nodes(graph))
    elif elimination_order is None:
        elimination_order = greedy_order(domain, cliques, stochastic=False)[0]
    elif isinstance(elimination_order, int):
        orders = [greedy_order(domain, cliques, stochastic=False)] + [
            greedy_order(domain, cliques, stochastic=True)
            for _ in range(elimination_order)
        ]
        elimination_order = min(orders, key=lambda x: x[1])[0]

    tri = _triangulated(graph, elimination_order)
    cliques = sorted([domain.canonical(c) for c in nx.find_cliques(tri)])
    complete = nx.Graph()
    complete.add_nodes_from(cliques)
    for c1, c2 in itertools.combinations(cliques, 2):
        wgt = len(set(c1) & set(c2))
        if wgt > 0:
            complete.add_edge(c1, c2, weight=-wgt)
    spanning = nx.minimum_spanning_tree(complete)
    return spanning, elimination_order


def hypothetical_model_size(domain: Domain, cliques: Sequence[Clique]) -> float:
    """Size of the full junction tree parameters, measured in megabytes."""
    jtree, _ = make_junction_tree(domain, cliques)
    max_cliques = maximal_cliques(jtree)
    cells = sum(domain.size(cl) for cl in max_cliques)
    size_mb = cells * 8 / 2**20
    return size_mb


def _fmt_size(n: int) -> str:
    """Format a number of cells as a human-readable string."""
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    if n >= 1e3:
        return f"{n / 1e3:.2f}K"
    return str(n)


def _fmt_bytes(nbytes: int) -> str:
    """Format a byte count as a human-readable string."""
    if nbytes >= 2**30:
        return f"{nbytes / 2**30:.2f} GiB"
    if nbytes >= 2**20:
        return f"{nbytes / 2**20:.2f} MiB"
    if nbytes >= 2**10:
        return f"{nbytes / 2**10:.2f} KiB"
    return f"{nbytes} B"


def model_summary(
    domain: Domain,
    cliques: Sequence[Clique],
    jtree: nx.Graph | None = None,
    marginal_oracle: "MarginalOracle | None" = None,
    bytes_per_cell: int = 4,
) -> str:
    """Return a human-readable summary of the model structure.

    Surfaces key diagnostic information for debugging and capacity planning:
    clique statistics, junction tree node and message sizes, treewidth,
    memory estimates, and (optionally) XLA compilation cost analysis.

    Args:
        domain: The full data domain.
        cliques: The measurement cliques (before maximal subsumption).
        jtree: An optional pre-built junction tree (``nx.Graph``).
            If ``None``, one is constructed via ``make_junction_tree``.
        marginal_oracle: A marginal oracle function. If ``None``, uses
            ``marginal_oracles.default_oracle(cliques, domain)``.
        bytes_per_cell: Bytes per table cell (4 for float32, 8 for float64).

    Returns:
        A multi-line string with the summary.
    """
    from .clique_utils import maximal_subset  # avoid circular import

    input_cliques = [tuple(cl) for cl in cliques]
    maximal = maximal_subset(input_cliques)

    if jtree is None:
        jtree, elim_order = make_junction_tree(domain, input_cliques)
    else:
        elim_order = None
    jtree_nodes = maximal_cliques(jtree)

    # --- Clique stats ---
    clique_sizes = [domain.size(cl) for cl in input_cliques]
    maximal_sizes = [domain.size(cl) for cl in maximal]

    # --- Junction tree node stats ---
    node_sizes = [domain.size(cl) for cl in jtree_nodes]

    # --- Message stats (separators = intersections of adjacent nodes) ---
    messages = []
    for u, v in jtree.edges():
        sep = tuple(set(u) & set(v))
        messages.append((sep, domain.size(sep) if sep else 1))
    msg_sizes = [s for _, s in messages]

    # --- Treewidth ---
    treewidth = max(len(cl) for cl in jtree_nodes) - 1

    # --- Memory ---
    total_node_cells = sum(node_sizes)
    total_msg_cells = sum(msg_sizes) * 2  # messages go both directions
    total_cells = total_node_cells + total_msg_cells
    mem_bytes = total_cells * bytes_per_cell

    lines = [
        "=== Model Summary ===",
        "",
        (
            f"Domain: {len(domain)} attributes, "
            f"{_fmt_size(domain.size())} total cells"
        ),
        "",
        "Cliques:",
        f"  Input cliques:   {len(input_cliques)}",
        f"  Maximal cliques: {len(maximal)}",
        (
            f"  Largest clique:  {max(clique_sizes)} cells "
            f"({max(input_cliques, key=lambda c: domain.size(c))})"
        ),
        f"  Total clique cells: {_fmt_size(sum(clique_sizes))}",
        "",
        "Junction Tree:",
        f"  Treewidth:       {treewidth}",
        f"  Nodes:           {len(jtree_nodes)}",
        (
            f"  Largest node:    {_fmt_size(max(node_sizes))} cells "
            f"({max(jtree_nodes, key=lambda c: domain.size(c))})"
        ),
        f"  Total node cells: {_fmt_size(total_node_cells)}",
        "",
    ]

    if msg_sizes:
        largest_msg_idx = max(
            range(len(messages)), key=lambda i: messages[i][1]
        )
        largest_msg_sep, largest_msg_size = messages[largest_msg_idx]
        lines += [
            "Messages:",
            f"  Edges:           {len(messages)}",
            (
                f"  Largest message: {_fmt_size(largest_msg_size)} cells "
                f"({largest_msg_sep})"
            ),
            (
                f"  Total message cells: {_fmt_size(sum(msg_sizes))} "
                f"(x2 = {_fmt_size(total_msg_cells)})"
            ),
            "",
        ]

    dtype_name = {4: "float32", 8: "float64"}.get(
        bytes_per_cell, f"{bytes_per_cell}B"
    )
    lines += [
        f"Memory ({dtype_name}):",
        f"  Nodes:    {_fmt_bytes(total_node_cells * bytes_per_cell)}",
        f"  Messages: {_fmt_bytes(total_msg_cells * bytes_per_cell)}",
        f"  Total:    {_fmt_bytes(mem_bytes)}",
    ]

    # --- XLA compilation analysis ---
    try:
        import jax  # pylint: disable=import-outside-toplevel

        from .clique_vector import CliqueVector  # avoid circular import
        from . import marginal_oracles as mo  # avoid circular import

        if marginal_oracle is None:
            marginal_oracle = mo.default_oracle(
                cliques=tuple(input_cliques), domain=domain
            )

        potentials = CliqueVector.zeros(domain, input_cliques)
        compiled = (
            jax.jit(lambda p: marginal_oracle(p, 1.0))
            .lower(potentials)
            .compile()
        )

        lines += ["", f"XLA Compilation ({marginal_oracle.__name__}):"]

        cost = compiled.cost_analysis()
        if cost:
            flops = cost.get("flops", 0)
            transcendentals = cost.get("transcendentals", 0)
            bytes_accessed = cost.get("bytes accessed", 0)
            lines += [
                f"  FLOPs:           {_fmt_size(flops)}",
                f"  Transcendentals: {_fmt_size(transcendentals)}",
                f"  Bytes accessed:  {_fmt_bytes(int(bytes_accessed))}",
            ]

        try:
            mem = compiled.memory_analysis()
            lines += [
                f"  Arg size:        {_fmt_bytes(mem.argument_size_in_bytes)}",
                f"  Output size:     {_fmt_bytes(mem.output_size_in_bytes)}",
                f"  Temp size:       {_fmt_bytes(mem.temp_size_in_bytes)}",
            ]
        except Exception:  # pylint: disable=broad-exception-caught
            pass

    except Exception as e:  # pylint: disable=broad-exception-caught
        lines += ["", f"XLA Compilation: unavailable ({e})"]

    return "\n".join(lines)
