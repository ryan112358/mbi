"""Comprehensive oracle benchmark across devices, topologies, and stability.

Tests all oracle variants across:
  - Graph topologies: chain, chain+3way, dense, wide-sparse
  - Scales: ~1e3, ~1e5, ~1e7 max clique size
  - Potential properties: random, scaled (magnitude sensitivity), -inf entries
  - Metrics: runtime, correctness (TV distance), NaN/OOM detection

Usage:
    .venv/bin/python examples/evaluate_oracles.py
"""

import functools
import time
from collections import defaultdict

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from mbi.clique_vector import CliqueVector
from mbi.domain import Domain
from mbi.marginal_oracles import (
    brute_force_marginals,
    einsum_fused,
    einsum_materialized,
    einsum_semistable,
    message_passing_hugin,
    message_passing_implicit,
    message_passing_shafer_shenoy,
)

ORACLES = {
    "implicit+fused": message_passing_implicit,
    "implicit+semistable": functools.partial(
        message_passing_implicit, contraction=einsum_semistable
    ),
    "implicit+materialized": functools.partial(
        message_passing_implicit, contraction=einsum_materialized
    ),
    "hugin": message_passing_hugin,
    "shafer_shenoy": message_passing_shafer_shenoy,
}

# ---------------------------------------------------------------------------
# Graph construction helpers
# ---------------------------------------------------------------------------


def make_graph(attr_sizes, clique_specs):
    attrs = {f"a{i}": s for i, s in enumerate(attr_sizes)}
    domain = Domain.fromdict(attrs)
    names = list(attrs)
    cliques = list(
        dict.fromkeys(tuple(names[i] for i in spec) for spec in clique_specs)
    )
    return domain, cliques


def chain_graph(sizes):
    n = len(sizes)
    specs = [(i, i + 1) for i in range(n - 1)]
    return make_graph(sizes, specs)


def chain_3way_graph(sizes):
    n = len(sizes)
    specs = [(i, i + 1) for i in range(n - 1)] + [
        (i, i + 1, i + 2) for i in range(0, n - 2, 3)
    ]
    return make_graph(sizes, specs)


def dense_graph(sizes):
    n = len(sizes)
    specs = (
        [(i, i + 1) for i in range(n - 1)]
        + [(i, i + 1, i + 2) for i in range(0, n - 2, 2)]
        + [(0, n // 3), (n // 3, 2 * n // 3), (0, 2 * n // 3)]
    )
    return make_graph(sizes, specs)


def wide_sparse_graph(n_attrs, rng):
    sizes = list(rng.choice([5, 8, 10, 15], size=n_attrs))
    specs = [(i, i + 1) for i in range(n_attrs - 1)]
    return make_graph(sizes, specs)


# ---------------------------------------------------------------------------
# Benchmark scenarios
# ---------------------------------------------------------------------------

TOPOLOGIES = {
    "chain_small": lambda: chain_graph([10, 15, 20, 12, 8]),
    "chain_medium": lambda: chain_graph(
        [10, 15, 20, 12, 25, 8, 15, 10, 20, 12]
    ),
    "chain3_small": lambda: chain_3way_graph([10, 15, 20, 12, 8]),
    "chain3_medium": lambda: chain_3way_graph(
        [10, 15, 20, 12, 25, 8, 15, 10, 20, 12]
    ),
    "dense_small": lambda: dense_graph([10, 15, 20, 12, 8, 25]),
    "dense_medium": lambda: dense_graph(
        [10, 15, 20, 12, 25, 8, 15, 10, 20, 12]
    ),
    "dense_large": lambda: dense_graph(
        [50, 50, 30, 20, 20, 100, 100, 100, 15, 10]
    ),
    "wide_sparse": lambda: wide_sparse_graph(40, np.random.RandomState(42)),
}


# ---------------------------------------------------------------------------
# Potential generators
# ---------------------------------------------------------------------------


def random_potentials(domain, cliques, scale=1.0, seed=0):
    rng = np.random.RandomState(seed)
    arrays = {}
    for cl in cliques:
        shape = tuple(domain.size(a) for a in cl)
        arrays[cl] = jnp.array(scale * rng.randn(*shape).astype("f"))
    from mbi.factor import Factor

    factors = {cl: Factor(domain.project(cl), arrays[cl]) for cl in cliques}
    return CliqueVector(domain, cliques, factors)


def potentials_with_neginf(domain, cliques, frac=0.1, seed=0):
    """Random potentials with `frac` of entries set to -inf."""
    rng = np.random.RandomState(seed)
    arrays = {}
    for cl in cliques:
        shape = tuple(domain.size(a) for a in cl)
        vals = rng.randn(*shape).astype("f")
        mask = rng.rand(*shape) < frac
        vals[mask] = -np.inf
        arrays[cl] = jnp.array(vals)
    from mbi.factor import Factor

    factors = {cl: Factor(domain.project(cl), arrays[cl]) for cl in cliques}
    return CliqueVector(domain, cliques, factors)


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------


def time_oracle(oracle, potentials, n_iters=5):
    """Return (warmup_ms, median_ms) or (None, None) on failure."""
    try:
        t0 = time.perf_counter()
        result = oracle(potentials)
        jax.block_until_ready(result)
        warmup = (time.perf_counter() - t0) * 1000

        # Check for NaN
        for cl in result.cliques:
            if jnp.any(jnp.isnan(result[cl].values)):
                return warmup, None  # NaN

        times = []
        for _ in range(n_iters):
            t0 = time.perf_counter()
            jax.block_until_ready(oracle(potentials))
            times.append((time.perf_counter() - t0) * 1000)
        return warmup, np.median(times)
    except Exception as e:
        print(f"      ERROR: {e}")
        return None, None


def tv_distance(mu1, mu2):
    """Max total variation distance across cliques."""
    max_tv = 0.0
    for cl in mu1.cliques:
        v1 = np.array(mu1[cl].datavector(), dtype=np.float64)
        v2 = np.array(mu2[cl].datavector(), dtype=np.float64)
        tv = 0.5 * np.abs(v1 - v2).sum()
        max_tv = max(max_tv, tv)
    return max_tv


def measure_stability(oracle, potentials, reference):
    """Return TV distance vs reference, or 'NaN'/'ERROR'."""
    try:
        result = oracle(potentials)
        jax.block_until_ready(result)
        for cl in result.cliques:
            if jnp.any(jnp.isnan(result[cl].values)):
                return "NaN"
        return tv_distance(result, reference)
    except Exception:
        return "ERROR"


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------


def run_speed_benchmark():
    """Benchmark runtime across topologies."""
    print("\n" + "=" * 80)
    print("  SPEED BENCHMARK")
    print("=" * 80)

    for topo_name, topo_fn in TOPOLOGIES.items():
        domain, cliques = topo_fn()
        potentials = random_potentials(domain, cliques)
        max_size = max(domain.size(cl) for cl in cliques)

        print(
            f"\n--- {topo_name} ({len(cliques)} cliques,"
            f" max_size={max_size:,.0f}) ---"
        )
        print(f"  {'Oracle':<28} {'Warmup':>10} {'Median':>10}")
        print(f"  {'-'*28} {'-'*10} {'-'*10}")

        for name, oracle in ORACLES.items():
            warmup, median = time_oracle(oracle, potentials)
            if warmup is None:
                print(f"  {name:<28} {'OOM':>10} {'':>10}")
            elif median is None:
                print(f"  {name:<28} {warmup:>8.1f}ms {'NaN':>10}")
            else:
                print(f"  {name:<28} {warmup:>8.1f}ms {median:>8.1f}ms")


def run_stability_benchmark():
    """Test numerical stability across potential magnitudes."""
    print("\n" + "=" * 80)
    print("  STABILITY BENCHMARK (TV distance vs shafer_shenoy@float64)")
    print("=" * 80)

    domain, cliques = chain_3way_graph([10, 15, 20, 12, 8])
    scales = [1, 5, 10, 20, 50, 100]

    header = f"  {'Oracle':<28}" + "".join(
        f" {'s=' + str(s):>8}" for s in scales
    )
    print(f"\n{header}")
    print(f"  {'-'*28}" + " --------" * len(scales))

    for name, oracle in ORACLES.items():
        row = f"  {name:<28}"
        for scale in scales:
            potentials = random_potentials(domain, cliques, scale=scale)
            ref = message_passing_shafer_shenoy(potentials)
            result = measure_stability(oracle, potentials, ref)
            if isinstance(result, str):
                row += f" {result:>8}"
            else:
                row += f" {result:>8.1e}"
        print(row)


def run_neginf_benchmark():
    """Test handling of -inf potentials."""
    print("\n" + "=" * 80)
    print("  -INF POTENTIALS BENCHMARK")
    print("=" * 80)

    domain, cliques = chain_3way_graph([10, 15, 20, 12, 8])
    fracs = [0.0, 0.05, 0.1, 0.2, 0.5]

    header = f"  {'Oracle':<28}" + "".join(
        f" {'f=' + str(f):>8}" for f in fracs
    )
    print(f"\n{header}")
    print(f"  {'-'*28}" + " --------" * len(fracs))

    for name, oracle in ORACLES.items():
        row = f"  {name:<28}"
        for frac in fracs:
            potentials = potentials_with_neginf(domain, cliques, frac=frac)
            # Use shafer_shenoy as reference (handles -inf correctly)
            ref = message_passing_shafer_shenoy(potentials)
            result = measure_stability(oracle, potentials, ref)
            if isinstance(result, str):
                row += f" {result:>8}"
            else:
                row += f" {result:>8.1e}"
        print(row)


def run_compilation_benchmark():
    """Measure JIT warmup (compilation) time."""
    print("\n" + "=" * 80)
    print("  COMPILATION TIME BENCHMARK")
    print("=" * 80)

    scenarios = [
        ("chain_small", chain_graph([10, 15, 20, 12, 8])),
        ("chain_medium", chain_graph([10, 15, 20, 12, 25, 8, 15, 10, 20, 12])),
        ("wide_sparse", wide_sparse_graph(40, np.random.RandomState(42))),
    ]

    for scenario_name, (domain, cliques) in scenarios:
        potentials = random_potentials(domain, cliques)
        print(f"\n--- {scenario_name} ---")
        print(f"  {'Oracle':<28} {'Compile (ms)':>12}")
        print(f"  {'-'*28} {'-'*12}")

        for name, oracle in ORACLES.items():
            # Clear JAX cache to force recompilation
            jax.clear_caches()
            warmup, _ = time_oracle(oracle, potentials, n_iters=1)
            if warmup is None:
                print(f"  {name:<28} {'ERROR':>12}")
            else:
                print(f"  {name:<28} {warmup:>10.1f}ms")


def main():
    backend = jax.default_backend()
    print(f"\nBackend: {backend}")
    print(f"Devices: {jax.devices()}")
    print(f"Float64 enabled: {jax.config.jax_enable_x64}")

    run_speed_benchmark()
    run_stability_benchmark()
    run_neginf_benchmark()
    run_compilation_benchmark()


if __name__ == "__main__":
    main()
