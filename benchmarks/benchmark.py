"""Benchmark: Compare relaxed projection implementations.

Compares three implementations:
  1. NEW: mbi.extensions.mixture_of_products (this PR)
  2. PR48: KaiChen9909's relaxed_projection_estimation (PR #48)
  3. EXPERIMENTAL: mbi.experimental.mixture_of_products (existing)

Metrics:
  - Final L2 loss (lower is better)
  - Wall-clock time (seconds)
  - Max marginal error (L-inf norm between fitted and true marginals)
  - Non-negativity violations (count of negative marginal entries)
"""

import time
import traceback

import jax
import jax.numpy as jnp
import numpy as np
import optax

from mbi import Domain, marginal_loss
from mbi.clique_vector import CliqueVector
from mbi.factor import Factor
from mbi.marginal_loss import LinearMeasurement

# Implementation 1: New extensions
from mbi.extensions.mixture_of_products import (
    MixtureOfProducts,
    mixture_of_products,
)

# Implementation 3: Existing experimental
from mbi.experimental.mixture_of_products import (
    mixture_of_products as experimental_mop,
)

# Implementation 2: PR #48
try:
    from mbi.relaxed_projection import (
        relaxed_projection_estimation,
        ProjectableData,
    )
    HAS_PR48 = True
except Exception as e:
    print(f"Warning: Could not import PR #48 code: {e}")
    HAS_PR48 = False


CONFIGS = {
    "small_tree": {
        "domain": Domain(["a", "b", "c", "d"], [2, 3, 4, 5]),
        "cliques": [("a", "b"), ("b", "c"), ("c", "d")],
    },
    "small_dense": {
        "domain": Domain(["a", "b", "c", "d"], [2, 3, 4, 5]),
        "cliques": [
            ("a", "b"), ("a", "c"), ("a", "d"),
            ("b", "c"), ("b", "d"), ("c", "d"),
        ],
    },
    "medium_tree": {
        "domain": Domain(["a", "b", "c", "d", "e", "f"], [3, 4, 5, 3, 4, 5]),
        "cliques": [
            ("a", "b"), ("b", "c"), ("c", "d"),
            ("d", "e"), ("e", "f"),
        ],
    },
    "medium_dense": {
        "domain": Domain(["a", "b", "c", "d", "e", "f"], [3, 4, 5, 3, 4, 5]),
        "cliques": [
            ("a", "b"), ("b", "c"), ("c", "d"),
            ("d", "e"), ("e", "f"), ("a", "c"),
            ("b", "d"), ("c", "e"), ("d", "f"),
        ],
    },
    "large_sparse": {
        "domain": Domain(
            [f"x{i}" for i in range(10)],
            [3, 4, 3, 4, 3, 4, 3, 4, 3, 4],
        ),
        "cliques": [(f"x{i}", f"x{i+1}") for i in range(9)],
    },
}

ITERS = 1000
NUM_COMPONENTS = 50
LEARNING_RATE = 0.1
SEED = 42


def make_measurements(domain, cliques, noise_stddev=0.01):
    np.random.seed(SEED)
    P = Factor.random(domain)
    P = P / P.sum()
    measurements = []
    for cl in cliques:
        true_marginal = P.project(cl).datavector()
        noise = np.random.normal(0, noise_stddev, true_marginal.shape)
        y = true_marginal + noise
        measurements.append(LinearMeasurement(y, cl, stddev=noise_stddev))
    return measurements, P


def compute_loss(model, measurements, domain):
    loss_fn_obj = marginal_loss.from_linear_measurements(measurements)
    cliques = loss_fn_obj.cliques
    try:
        arrays = {}
        for cl in cliques:
            marg = model.project(cl)
            if isinstance(marg, Factor):
                arrays[cl] = marg
            elif hasattr(marg, "datavector"):
                vals = marg.datavector(flatten=False)
                arrays[cl] = Factor(domain.project(cl), vals)
            else:
                arrays[cl] = Factor(domain.project(cl), marg.values)
        mu = CliqueVector(domain, cliques, arrays)
        return float(loss_fn_obj(mu))
    except Exception as e:
        print(f"    compute_loss error: {e}")
        return float("inf")


def compute_max_error(model, measurements):
    max_err = 0.0
    neg_count = 0
    for M in measurements:
        marg = model.project(M.clique)
        if isinstance(marg, Factor):
            actual = np.array(marg.datavector())
        elif hasattr(marg, "datavector"):
            actual = np.array(marg.datavector(flatten=True))
        else:
            continue
        expected = np.array(M.noisy_measurement)
        if actual.sum() > 0 and expected.sum() > 0:
            a_prob = actual / actual.sum()
            e_prob = expected / expected.sum()
            max_err = max(max_err, float(np.max(np.abs(a_prob - e_prob))))
        neg_count += int(np.sum(np.array(actual) < -1e-10))
    return max_err, neg_count


def run_config(name, domain, cliques):
    print(f"\n{'='*70}")
    print(f"Config: {name}")
    print(f"  Domain: {len(domain)} attrs, total size = {domain.size()}")
    print(f"  Cliques: {len(cliques)} cliques")
    print(f"{'='*70}")

    measurements, P = make_measurements(domain, cliques)
    results = []

    # --- Method 1: NEW MixtureOfProducts ---
    print(f"  [1/3] NEW (extensions)...", end="", flush=True)
    t0 = time.time()
    try:
        model = mixture_of_products(
            domain, measurements,
            num_components=NUM_COMPONENTS, iters=ITERS,
            learning_rate=LEARNING_RATE, seed=SEED,
        )
        t = time.time() - t0
        loss = compute_loss(model, measurements, domain)
        max_err, neg = compute_max_error(model, measurements)
        print(f" {t:.1f}s, loss={loss:.6f}")
        results.append(("NEW (extensions)", loss, t, max_err, neg, None))
    except Exception as e:
        t = time.time() - t0
        print(f" FAILED ({t:.1f}s): {e}")
        results.append(("NEW (extensions)", None, t, None, None, str(e)))

    # --- Method 2: PR #48 ---
    print(f"  [2/3] PR #48...", end="", flush=True)
    if HAS_PR48:
        t0 = time.time()
        try:
            model48 = relaxed_projection_estimation(
                domain, measurements, iters=ITERS,
                optimizer=optax.adam(learning_rate=LEARNING_RATE),
                known_total=1000, seed=SEED,
            )
            t = time.time() - t0
            loss = compute_loss(model48, measurements, domain)
            max_err, neg = compute_max_error(model48, measurements)
            print(f" {t:.1f}s, loss={loss:.6f}")
            results.append(("PR #48", loss, t, max_err, neg, None))
        except Exception as e:
            t = time.time() - t0
            print(f" FAILED ({t:.1f}s): {e}")
            traceback.print_exc()
            results.append(("PR #48", None, t, None, None, str(e)))
    else:
        print(" SKIPPED (import failed)")
        results.append(("PR #48", None, None, None, None, "import failed"))

    # --- Method 3: EXPERIMENTAL ---
    print(f"  [3/3] EXPERIMENTAL...", end="", flush=True)
    t0 = time.time()
    try:
        model_exp = experimental_mop(
            domain, measurements,
            mixture_components=NUM_COMPONENTS, iters=ITERS,
            alpha=LEARNING_RATE,
        )
        t = time.time() - t0
        loss = compute_loss(model_exp, measurements, domain)
        max_err, neg = compute_max_error(model_exp, measurements)
        print(f" {t:.1f}s, loss={loss:.6f}")
        results.append(("EXPERIMENTAL", loss, t, max_err, neg, None))
    except Exception as e:
        t = time.time() - t0
        print(f" FAILED ({t:.1f}s): {e}")
        results.append(("EXPERIMENTAL", None, t, None, None, str(e)))

    return results


def main():
    print("=" * 70)
    print("MixtureOfProducts Benchmark")
    print(f"  Iters: {ITERS}, Components: {NUM_COMPONENTS}, LR: {LEARNING_RATE}")
    print("=" * 70)

    all_results = {}
    for name, cfg in CONFIGS.items():
        all_results[name] = run_config(name, cfg["domain"], cfg["cliques"])

    # Summary table
    print(f"\n\n{'='*100}")
    print("SUMMARY")
    print(f"{'='*100}")
    print(
        f"{'Config':<20} {'Method':<22} {'Loss':>12} {'Time(s)':>10} "
        f"{'MaxErr':>10} {'Neg':>5} {'Error'}"
    )
    print("-" * 100)
    for name, results in all_results.items():
        for method, loss, t, max_err, neg, err in results:
            ls = f"{loss:.6f}" if loss is not None else "FAILED"
            ts = f"{t:.1f}" if t is not None else "N/A"
            es = f"{max_err:.6f}" if max_err is not None else "N/A"
            ns = str(neg) if neg is not None else "N/A"
            er = err or ""
            print(f"{name:<20} {method:<22} {ls:>12} {ts:>10} {es:>10} {ns:>5} {er}")
        print()


if __name__ == "__main__":
    main()
