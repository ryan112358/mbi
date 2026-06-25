"""Human-readable model diagnostics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

import jax
import networkx as nx

from . import junction_tree, marginal_oracles
from .clique_vector import CliqueVector
from .domain import Domain

if TYPE_CHECKING:
    from .marginal_oracles import MarginalOracle

Clique = tuple[str, ...]


def _fmt_size(n: int | float) -> str:
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


def summarize(
    domain: Domain,
    cliques: Sequence[Clique],
    jtree: nx.Graph | None = None,
    marginal_oracle: MarginalOracle | None = None,
    compile: bool = False,
    bytes_per_cell: int | None = None,
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
        marginal_oracle: A marginal oracle function. If ``None`` and
            ``compile=True``, uses
            ``marginal_oracles.default_oracle(cliques, domain)``.
        compile: If True, compile the marginal oracle and surface XLA
            cost/memory analysis. Defaults to False.
        bytes_per_cell: Bytes per table cell. If ``None`` (default),
            auto-detects from ``jax.config.jax_enable_x64``
            (8 for float64, 4 for float32).

    Returns:
        A multi-line string with the summary.
    """
    if bytes_per_cell is None:
        bytes_per_cell = 8 if jax.config.jax_enable_x64 else 4
    input_cliques = [tuple(cl) for cl in cliques]

    if jtree is None:
        jtree, _ = junction_tree.make_junction_tree(domain, input_cliques)
    jtree_nodes = junction_tree.maximal_cliques(jtree)

    clique_sizes = [domain.size(cl) for cl in input_cliques]
    node_sizes = [domain.size(cl) for cl in jtree_nodes]

    messages = []
    for u, v in jtree.edges():
        u, v = cast(tuple[str, ...], u), cast(tuple[str, ...], v)
        sep = tuple(set(u) & set(v))
        messages.append((sep, domain.size(sep) if sep else 1))
    msg_sizes = [s for _, s in messages]

    treewidth = max(len(cl) for cl in jtree_nodes) - 1
    total_node_cells = sum(node_sizes)
    total_msg_cells = sum(msg_sizes) * 2  # messages go both directions
    mem_bytes = (total_node_cells + total_msg_cells) * bytes_per_cell

    lines = [
        "=== Model Summary ===",
        "",
        (
            f"Domain: {len(domain)} attributes, "
            f"{_fmt_size(domain.size())} total cells"
        ),
        "",
        "Cliques:",
        f"  Input:    {len(input_cliques)}",
        (
            f"  Largest:  {_fmt_size(max(clique_sizes))} cells "
            f"({max(input_cliques, key=lambda c: domain.size(c))})"
        ),
        f"  Total:    {_fmt_size(sum(clique_sizes))} cells",
        "",
        "Junction Tree:",
        f"  Treewidth: {treewidth}",
        f"  Nodes:     {len(jtree_nodes)}",
        (
            f"  Largest:   {_fmt_size(max(node_sizes))} cells "
            f"({max(jtree_nodes, key=lambda c: domain.size(c))})"
        ),
        f"  Total:     {_fmt_size(total_node_cells)} cells",
        "",
    ]

    if msg_sizes:
        largest_msg_idx = max(
            range(len(messages)), key=lambda i: messages[i][1]
        )
        largest_msg_sep, largest_msg_size = messages[largest_msg_idx]
        lines += [
            "Messages:",
            f"  Edges:   {len(messages)}",
            (
                f"  Largest: {_fmt_size(largest_msg_size)} cells "
                f"({largest_msg_sep})"
            ),
            f"  Total:   {_fmt_size(total_msg_cells)} cells (x2 directions)",
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

    if not compile:
        return "\n".join(lines)

    try:
        if marginal_oracle is None:
            marginal_oracle = marginal_oracles.default_oracle(
                cliques=tuple(input_cliques), domain=domain
            )

        potentials = CliqueVector.abstract(domain, input_cliques)
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
