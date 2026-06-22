"""Performance regression benchmark for marginal oracle refactoring.

Compares timing of marginal oracles on census-like graphical models
at multiple scales. Run on both master and refactor branch to verify
no performance regressions.

Usage:
    .venv/bin/python benchmark_oracles.py
"""

import time
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from mbi.domain import Domain
from mbi.clique_vector import CliqueVector
from mbi import marginal_oracles


def make_census_like_graph(num_attrs, attr_sizes, clique_specs):
    """Build a domain and cliques from specifications."""
    attrs = {f"a{i}": attr_sizes[i] for i in range(num_attrs)}
    domain = Domain.fromdict(attrs)
    names = list(attrs.keys())
    seen = set()
    cliques = []
    for spec in clique_specs:
        cl = tuple(names[i] for i in spec)
        if cl not in seen:
            seen.add(cl)
            cliques.append(cl)
    return domain, cliques


def benchmark_oracle(oracle_fn, potentials, num_iters, name):
    """Benchmark an oracle, returning (warmup_ms, median_ms)."""
    try:
        # Warmup / compile
        t0 = time.perf_counter()
        result = jax.block_until_ready(oracle_fn(potentials))
        warmup = (time.perf_counter() - t0) * 1000

        # Timed iterations
        times = []
        for _ in range(num_iters):
            t0 = time.perf_counter()
            result = jax.block_until_ready(oracle_fn(potentials))
            times.append((time.perf_counter() - t0) * 1000)

        median = np.median(times)
        return warmup, median
    except Exception as e:
        print(f"    {name}: FAILED - {e}")
        return None, None


def run_scale(label, domain, cliques, num_iters=5):
    """Run all oracle variants on a given scale."""
    potentials = CliqueVector.random(domain, cliques)

    # Compute max clique size for reference
    max_cl_size = max(domain.size(cl) for cl in cliques)

    print(f"\n{'='*70}")
    print(f"  {label}")
    print(
        f"  {len(cliques)} cliques, {len(domain)} attrs,"
        f" max_clique_size={max_cl_size:,.0f}"
    )
    print(f"  {num_iters} timed iterations")
    print(f"{'='*70}")
    print(f"  {'Oracle':<35} {'Warmup':>10} {'Median':>10}")
    print(f"  {'-'*35} {'-'*10} {'-'*10}")

    oracles = [
        (
            "message_passing_fast",
            lambda p: marginal_oracles.message_passing_fast(p, total=1.0),
        ),
        (
            "message_passing_stable (HUGIN)",
            lambda p: marginal_oracles.message_passing_stable(p, total=1.0),
        ),
        (
            "message_passing_shafer_shenoy",
            lambda p: marginal_oracles.message_passing_shafer_shenoy(
                p, total=1.0
            ),
        ),
    ]

    # Check if new API is available (refactor branch)
    if hasattr(marginal_oracles, "MessagePassingOracle"):
        from mbi.marginal_oracles import (
            MessagePassingOracle,
            MessageSchedule,
            einsum_stabilized,
            einsum_materialized,
            einsum_semiring,
        )

        oracles.extend([
            (
                "MPO(IMPLICIT, stabilized)",
                MessagePassingOracle(
                    schedule=MessageSchedule.IMPLICIT,
                    contraction=einsum_stabilized,
                ),
            ),
            (
                "MPO(IMPLICIT, materialized)",
                MessagePassingOracle(
                    schedule=MessageSchedule.IMPLICIT,
                    contraction=einsum_materialized,
                ),
            ),
            (
                "MPO(IMPLICIT, semiring)",
                MessagePassingOracle(
                    schedule=MessageSchedule.IMPLICIT,
                    contraction=einsum_semiring,
                ),
            ),
            (
                "MPO(HUGIN)",
                MessagePassingOracle(
                    schedule=MessageSchedule.HUGIN,
                ),
            ),
            (
                "MPO(SHAFER_SHENOY)",
                MessagePassingOracle(
                    schedule=MessageSchedule.SHAFER_SHENOY,
                ),
            ),
        ])

    results = {}
    for name, oracle_fn in oracles:
        warmup, median = benchmark_oracle(
            oracle_fn, potentials, num_iters, name
        )
        if warmup is not None:
            print(f"  {name:<35} {warmup:>8.1f}ms {median:>8.1f}ms")
            results[name] = (warmup, median)

    return results


def main():
    print(f"\nBackend: {jax.default_backend()}")
    print(f"Devices: {jax.devices()}")

    # === Scale 1: max_clique_size ~ 1e5 ===
    sizes_1e5 = [
        10,
        15,
        8,
        20,
        12,
        6,
        25,
        10,
        8,
        15,
        20,
        10,
        6,
        8,
        12,
        15,
        10,
        8,
        20,
        12,
    ]
    specs_1e5 = (
        [(i, i + 1) for i in range(19)]
        + [(i, i + 1, i + 2) for i in range(0, 18, 3)]
        + [(0, 5), (5, 10), (10, 15)]
    )
    domain_1e5, cliques_1e5 = make_census_like_graph(20, sizes_1e5, specs_1e5)
    run_scale(
        "Census-like ~1e5 max_clique_size",
        domain_1e5,
        cliques_1e5,
        num_iters=10,
    )

    # === Scale 2: max_clique_size ~ 1e6 ===
    sizes_1e6 = [100, 100, 100, 50, 50, 30, 20, 20, 15, 10, 10, 8, 8, 6, 6, 5]
    specs_1e6 = (
        [(i, i + 1) for i in range(15)]
        + [(0, 1, 2)]  # 100*100*100 = 1e6
        + [(3, 4, 5)]  # 50*50*30 = 75k
        + [(0, 3), (2, 5)]
    )
    domain_1e6, cliques_1e6 = make_census_like_graph(16, sizes_1e6, specs_1e6)
    run_scale(
        "Census-like ~1e6 max_clique_size", domain_1e6, cliques_1e6, num_iters=5
    )

    # === Scale 3: Wide graph (many small cliques, like real census) ===
    np.random.seed(42)
    n_attrs = 50
    sizes_wide = list(np.random.choice([5, 8, 10, 15, 20], size=n_attrs))
    specs_wide = (
        [(i, i + 1) for i in range(n_attrs - 1)]  # chain
        + [(i, i + 1, i + 2) for i in range(0, n_attrs - 2, 5)]  # 3-way every 5
        + [
            (0, n_attrs // 4),
            (n_attrs // 4, n_attrs // 2),
            (n_attrs // 2, 3 * n_attrs // 4),
        ]  # skip links
    )
    domain_wide, cliques_wide = make_census_like_graph(
        n_attrs, sizes_wide, specs_wide
    )
    run_scale(
        "Wide census-like (50 attrs, many cliques)",
        domain_wide,
        cliques_wide,
        num_iters=10,
    )


if __name__ == "__main__":
    main()
