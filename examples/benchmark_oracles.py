"""Benchmark marginal oracles on census-like graphical models.

Usage:
    .venv/bin/python examples/benchmark_oracles.py
"""

import functools
import time

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from mbi.clique_vector import CliqueVector
from mbi.domain import Domain
from mbi.marginal_oracles import (
    message_passing_implicit,
    einsum_fused,
    einsum_materialized,
    einsum_semistable,
    message_passing_hugin,
    message_passing_shafer_shenoy,
)

ORACLES = [
    ("message_passing_implicit+materialized", message_passing_implicit),
    (
        "message_passing_implicit+fused",
        functools.partial(message_passing_implicit, contraction=einsum_fused),
    ),
    (
        "message_passing_implicit+semistable",
        functools.partial(
            message_passing_implicit, contraction=einsum_semistable
        ),
    ),
    ("message_passing_hugin", message_passing_hugin),
    ("message_passing_shafer_shenoy", message_passing_shafer_shenoy),
]


def make_census_graph(attr_sizes, clique_specs):
    """Build a domain and cliques from attribute sizes and index specs."""
    attrs = {f"a{i}": s for i, s in enumerate(attr_sizes)}
    domain = Domain.fromdict(attrs)
    names = list(attrs)
    cliques = list(
        dict.fromkeys(tuple(names[i] for i in spec) for spec in clique_specs)
    )
    return domain, cliques


def benchmark_oracle(oracle, potentials, num_iters):
    """Return (warmup_ms, median_ms) or (None, None) on failure."""
    try:
        t0 = time.perf_counter()
        jax.block_until_ready(oracle(potentials))
        warmup = (time.perf_counter() - t0) * 1000

        times = []
        for _ in range(num_iters):
            t0 = time.perf_counter()
            jax.block_until_ready(oracle(potentials))
            times.append((time.perf_counter() - t0) * 1000)
        return warmup, np.median(times)
    except Exception as e:
        print(f"    FAILED: {e}")
        return None, None


def run_scale(label, domain, cliques, num_iters=5):
    """Run all oracle variants on a given scale."""
    potentials = CliqueVector.random(domain, cliques)
    max_size = max(domain.size(cl) for cl in cliques)

    print(f"\n{'='*70}")
    print(f"  {label}")
    print(
        f"  {len(cliques)} cliques, {len(domain)} attrs,"
        f" max_clique_size={max_size:,.0f}"
    )
    print(f"{'='*70}")
    print(f"  {'Oracle':<35} {'Warmup':>10} {'Median':>10}")
    print(f"  {'-'*35} {'-'*10} {'-'*10}")

    for name, oracle in ORACLES:
        warmup, median = benchmark_oracle(oracle, potentials, num_iters)
        if warmup is not None:
            print(f"  {name:<35} {warmup:>8.1f}ms {median:>8.1f}ms")


def main():
    print(f"\nBackend: {jax.default_backend()}")
    print(f"Devices: {jax.devices()}")

    # ~1e5 max clique size: 10 attrs, chain + 3-way + skip edges
    sizes = [10, 15, 20, 12, 25, 8, 15, 10, 20, 12]
    specs = (
        [(i, i + 1) for i in range(9)]
        + [(i, i + 1, i + 2) for i in range(0, 8, 3)]
        + [(0, 5), (3, 7)]
    )
    run_scale(
        "~1e5 max_clique_size", *make_census_graph(sizes, specs), num_iters=10
    )

    # ~1e6 max clique size: 16 attrs with a 100*100*100 3-clique
    sizes = [100, 100, 100, 50, 50, 30, 20, 20, 15, 10, 10, 8, 8, 6, 6, 5]
    specs = [(i, i + 1) for i in range(15)] + [
        (0, 1, 2),
        (3, 4, 5),
        (0, 3),
        (2, 5),
    ]
    run_scale("~1e6 max_clique_size", *make_census_graph(sizes, specs))

    # Wide graph: 50 attrs, many small cliques
    np.random.seed(42)
    sizes = list(np.random.choice([5, 8, 10, 15, 20], size=50))
    specs = (
        [(i, i + 1) for i in range(49)]
        + [(i, i + 1, i + 2) for i in range(0, 48, 5)]
        + [(0, 12), (12, 25), (25, 37)]
    )
    run_scale("Wide (50 attrs)", *make_census_graph(sizes, specs), num_iters=10)


if __name__ == "__main__":
    main()
