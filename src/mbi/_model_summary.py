"""Human-readable model diagnostics."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

import jax
import networkx as nx

from . import junction_tree, marginal_oracles
from .clique_utils import Clique
from .clique_vector import CliqueVector
from .domain import Domain

if TYPE_CHECKING:
  from .marginal_oracles import MarginalOracle


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


@dataclasses.dataclass(frozen=True)
class ModelSummary:
  """Structured summary of a model's clique and junction tree structure."""

  num_attributes: int
  domain_size: int
  num_cliques: int
  largest_clique: Clique
  largest_clique_size: int
  total_clique_cells: int
  treewidth: int
  num_jtree_nodes: int
  largest_jtree_node: Clique
  largest_jtree_node_size: int
  total_jtree_cells: int
  num_messages: int
  largest_message: Clique
  largest_message_size: int
  total_message_cells: int
  bytes_per_cell: int
  xla_flops: int | None = None
  xla_transcendentals: int | None = None
  xla_bytes_accessed: int | None = None
  xla_arg_bytes: int | None = None
  xla_output_bytes: int | None = None
  xla_temp_bytes: int | None = None

  @property
  def memory_bytes(self) -> int:
    return (
        self.total_jtree_cells + self.total_message_cells
    ) * self.bytes_per_cell

  @property
  def short(self) -> str:
    """One-line summary: potentials, junction tree, and messages size."""
    pot_bytes = self.total_clique_cells * self.bytes_per_cell
    msg_bytes = self.total_message_cells * self.bytes_per_cell
    return (
        f"potentials={_fmt_bytes(pot_bytes)},"
        f" jtree={_fmt_bytes(self.total_jtree_cells * self.bytes_per_cell)},"
        f" messages={_fmt_bytes(msg_bytes)}"
    )

  def __str__(self) -> str:
    dtype_name = {4: "float32", 8: "float64"}.get(
        self.bytes_per_cell, f"{self.bytes_per_cell}B"
    )
    lines = [
        "=== Model Summary ===",
        "",
        (
            f"Domain: {self.num_attributes} attributes, "
            f"{_fmt_size(self.domain_size)} total cells"
        ),
        "",
        "Cliques:",
        f"  Input:    {self.num_cliques}",
        (
            f"  Largest:  {_fmt_size(self.largest_clique_size)} cells "
            f"({self.largest_clique})"
        ),
        f"  Total:    {_fmt_size(self.total_clique_cells)} cells",
        "",
        "Junction Tree:",
        f"  Treewidth: {self.treewidth}",
        f"  Nodes:     {self.num_jtree_nodes}",
        (
            f"  Largest:   {_fmt_size(self.largest_jtree_node_size)} cells "
            f"({self.largest_jtree_node})"
        ),
        f"  Total:     {_fmt_size(self.total_jtree_cells)} cells",
        "",
    ]

    if self.num_messages > 0:
      lines += [
          "Messages:",
          f"  Edges:   {self.num_messages}",
          (
              f"  Largest: {_fmt_size(self.largest_message_size)} cells "
              f"({self.largest_message})"
          ),
          (
              "  Total:   "
              f"{_fmt_size(self.total_message_cells)} cells "
              "(x2 directions)"
          ),
          "",
      ]

    lines += [
        f"Memory ({dtype_name}):",
        (
            "  Nodes:   "
            f" {_fmt_bytes(self.total_jtree_cells * self.bytes_per_cell)}"
        ),
        (
            "  Messages:"
            f" {_fmt_bytes(self.total_message_cells * self.bytes_per_cell)}"
        ),
        f"  Total:    {_fmt_bytes(self.memory_bytes)}",
    ]

    if self.xla_flops is not None:
      lines += [
          "",
          "XLA Compilation:",
          f"  FLOPs:           {_fmt_size(self.xla_flops)}",
          f"  Transcendentals: {_fmt_size(self.xla_transcendentals or 0)}",
          f"  Bytes accessed:  {_fmt_bytes(self.xla_bytes_accessed or 0)}",
      ]
      if self.xla_arg_bytes is not None:
        lines += [
            f"  Arg size:        {_fmt_bytes(self.xla_arg_bytes)}",
            f"  Output size:     {_fmt_bytes(self.xla_output_bytes or 0)}",
            f"  Temp size:       {_fmt_bytes(self.xla_temp_bytes or 0)}",
        ]

    return "\n".join(lines)


def summarize(
    domain: Domain,
    cliques: Sequence[Clique],
    jtree: nx.Graph | None = None,
    marginal_oracle: MarginalOracle | None = None,
    compile: bool = False,
    bytes_per_cell: int | None = None,
) -> ModelSummary:
  """Return a structured summary of the model.

  Args:
      domain: The full data domain.
      cliques: The measurement cliques (before maximal subsumption).
      jtree: An optional pre-built junction tree.
          If ``None``, one is constructed via ``make_junction_tree``.
      marginal_oracle: A marginal oracle function.  If ``None`` and
          ``compile=True``, uses ``default_oracle``.
      compile: If True, compile the marginal oracle and surface XLA
          cost/memory analysis. Defaults to False.
      bytes_per_cell: Bytes per table cell. If ``None`` (default),
          auto-detects from ``jax.config.jax_enable_x64``.

  Returns:
      A ``ModelSummary`` dataclass.
  """
  if bytes_per_cell is None:
    bytes_per_cell = 8 if jax.config.jax_enable_x64 else 4
  input_cliques = [tuple(cl) for cl in cliques]

  if jtree is None:
    jtree, _ = junction_tree.make_junction_tree(domain, input_cliques)
  jtree_nodes = junction_tree.maximal_cliques(jtree)

  clique_sizes = {cl: domain.size(cl) for cl in input_cliques}
  node_sizes = {cl: domain.size(cl) for cl in jtree_nodes}

  messages: list[tuple[Clique, int]] = []
  for u, v in jtree.edges():
    u, v = cast(tuple[str, ...], u), cast(tuple[str, ...], v)
    sep = tuple(set(u) & set(v))
    messages.append((sep, domain.size(sep) if sep else 1))

  largest_clique = max(input_cliques, key=lambda c: clique_sizes[c])
  largest_node = max(jtree_nodes, key=lambda c: node_sizes[c])

  if messages:
    largest_msg_idx = max(range(len(messages)), key=lambda i: messages[i][1])
    largest_msg, largest_msg_size = messages[largest_msg_idx]
  else:
    largest_msg, largest_msg_size = (), 0

  total_msg_cells = sum(s for _, s in messages) * 2

  xla_kwargs: dict = {}
  if compile:
    try:
      if marginal_oracle is None:
        marginal_oracle = marginal_oracles.default_oracle(
            cliques=tuple(input_cliques), domain=domain
        )
      potentials = CliqueVector.abstract(domain, input_cliques)
      compiled = (
          jax.jit(lambda p: marginal_oracle(p, 1.0)).lower(potentials).compile()
      )
      cost = compiled.cost_analysis()
      if cost:
        xla_kwargs["xla_flops"] = cost.get("flops", 0)
        xla_kwargs["xla_transcendentals"] = cost.get("transcendentals", 0)
        xla_kwargs["xla_bytes_accessed"] = int(cost.get("bytes accessed", 0))
      try:
        mem = compiled.memory_analysis()
        xla_kwargs["xla_arg_bytes"] = mem.argument_size_in_bytes
        xla_kwargs["xla_output_bytes"] = mem.output_size_in_bytes
        xla_kwargs["xla_temp_bytes"] = mem.temp_size_in_bytes
      except Exception:  # pylint: disable=broad-exception-caught
        pass
    except Exception:  # pylint: disable=broad-exception-caught
      pass

  return ModelSummary(
      num_attributes=len(domain),
      domain_size=domain.size(),
      num_cliques=len(input_cliques),
      largest_clique=largest_clique,
      largest_clique_size=clique_sizes[largest_clique],
      total_clique_cells=sum(clique_sizes.values()),
      treewidth=max(len(cl) for cl in jtree_nodes) - 1,
      num_jtree_nodes=len(jtree_nodes),
      largest_jtree_node=largest_node,
      largest_jtree_node_size=node_sizes[largest_node],
      total_jtree_cells=sum(node_sizes.values()),
      num_messages=len(messages),
      largest_message=largest_msg,
      largest_message_size=largest_msg_size,
      total_message_cells=total_msg_cells,
      bytes_per_cell=bytes_per_cell,
      **xla_kwargs,
  )
