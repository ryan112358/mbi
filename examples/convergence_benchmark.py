"""Convergence benchmark for MBI estimators on the Adult dataset.

This script serves two purposes:

1. **Regression test**: Runs each estimator for a fixed number of iterations
   with a fixed seed, and asserts that the train loss drops below a known
   threshold.  If a future code change breaks convergence, the assertion will
   fire.

2. **Convergence demo**: When run with ``--plot``, produces a matplotlib figure
   showing train-loss convergence curves for every estimator.

The 2-way cliques are selected greedily by mutual information, subject to a
junction-tree max-node-size constraint of 5 000 cells.  This keeps each run
fast (a few seconds on CPU) while exercising the full estimation pipeline.

Usage
-----
::

    # Quick regression check (default 5 000 iters, ~80 s total on CPU):
    python convergence_benchmark.py

    # With convergence plot:
    python convergence_benchmark.py --plot

    # Longer run for tighter thresholds:
    python convergence_benchmark.py --iters=25000 --plot

    # Save plot to file:
    python convergence_benchmark.py --iters=25000 --plot --save=convergence.pdf
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import jax
import numpy as np

import mbi
from mbi import estimation, marginal_loss

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Adult dataset paths (relative to this file).
_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data"
)
_DOMAIN_PATH = os.path.join(_DATA_DIR, "adult-domain.json")
_CSV_PATH = os.path.join(_DATA_DIR, "adult.csv")

# Fixed random seed for reproducibility.
SEED = 42

# Noise level (sigma) for synthetic measurements.
SIGMA = 10.0

# 2-way cliques selected by greedy mutual-information ranking with a
# junction-tree max-node-size constraint of 5 000 cells.  These were computed
# from the bundled adult dataset (N=48 842, 14 attributes).
# JTree: 10 nodes, max=4 200, total=22 303.
TWO_WAY_CLIQUES: list[tuple[str, ...]] = [
    ("marital-status", "relationship"),
    ("workclass", "occupation"),
    ("relationship", "sex"),
    ("age", "marital-status"),
    ("education-num", "occupation"),
    ("age", "relationship"),
    ("occupation", "hours-per-week"),
    ("relationship", "income>50K"),
    ("age", "education-num"),
    ("marital-status", "sex"),
    ("marital-status", "income>50K"),
    ("race", "native-country"),
    ("capital-gain", "income>50K"),
    ("education-num", "native-country"),
    ("fnlwgt", "native-country"),
    ("relationship", "capital-gain"),
    ("workclass", "education-num"),
    ("capital-loss", "income>50K"),
    ("sex", "income>50K"),
    ("relationship", "capital-loss"),
    ("occupation", "race"),
    ("education-num", "race"),
    ("sex", "capital-gain"),
    ("sex", "capital-loss"),
]

# Per-estimator convergence thresholds.  Each value is the maximum allowable
# train loss after ``DEFAULT_ITERS`` iterations.  These were calibrated from
# 5 000-iteration runs on the adult dataset with σ=10 and seed=42.
# A 1.5× headroom factor is applied to absorb platform-level floating-point
# variation while still catching real regressions (e.g. broken step sizes).
#
# Calibrated values: IG=4109, DA=8440, UAM=4081, MD=4897.
DEFAULT_ITERS = 5_000
THRESHOLDS: dict[str, float] = {
    "IG": 6_500.0,
    "DA": 13_000.0,
    "UAM": 6_500.0,
    "MD": 7_500.0,
}

# How often to record loss during optimization.
CALLBACK_EVERY_ITERS = 100

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_problem(sigma: float = SIGMA, seed: int = SEED):
    """Load the adult dataset and construct noisy measurements.

    Returns
    -------
    domain : mbi.Domain
    measurements : list[mbi.LinearMeasurement]
    loss_fn : mbi.MarginalLossFn
    N : float
    """
    dataset = mbi.Dataset.load(_CSV_PATH, _DOMAIN_PATH)
    domain = dataset.domain
    N = float(dataset.records)

    attrs = list(domain.attrs)
    one_way = [(a,) for a in attrs]
    cliques = one_way + list(TWO_WAY_CLIQUES)

    # Compute true marginals and add Gaussian noise.
    rng = np.random.default_rng(seed)
    measurements = []
    for cl in cliques:
        x = dataset.project(cl).datavector().astype(np.float64)
        noisy = x + rng.normal(0, sigma, size=x.size)
        measurements.append(mbi.LinearMeasurement(noisy, cl, sigma))

    loss_fn = marginal_loss.from_linear_measurements(measurements, domain)
    return domain, measurements, loss_fn, N


def run_estimator(
    name: str,
    domain: mbi.Domain,
    loss_fn,
    N: float,
    iters: int,
) -> tuple[list[float], list[float]]:
    """Run a single estimator and return (iterations, train_losses)."""
    from mbi.estimation import CALLBACK_EVERY

    iterations: list[float] = []
    train_losses: list[float] = []
    step_counter = [0]

    def callback(marginals):
        step_counter[0] += CALLBACK_EVERY
        step = step_counter[0]
        if step % CALLBACK_EVERY_ITERS == 0:
            loss_val = float(loss_fn(marginals))
            iterations.append(step)
            train_losses.append(loss_val)

    if name == "IG":
        est = estimation.InteriorGradient()
    elif name == "DA":
        est = estimation.DualAveraging()
    elif name == "UAM":
        est = estimation.UniversalAcceleratedMethod(linesearch=True)
    elif name == "MD":
        est = estimation.MirrorDescent()
    else:
        raise ValueError(f"Unknown estimator: {name}")

    est.estimate(
        domain, loss_fn, known_total=N, iters=iters, callback_fn=callback
    )
    return iterations, train_losses


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ESTIMATORS = ["IG", "DA", "UAM", "MD"]


def main():
    parser = argparse.ArgumentParser(
        description="MBI estimator convergence benchmark on Adult."
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=DEFAULT_ITERS,
        help="Optimizer iterations per estimator (default: %(default)s).",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=SIGMA,
        help="Noise standard deviation (default: %(default)s).",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show a matplotlib convergence plot.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="",
        help="Save the plot to this path (e.g. convergence.pdf).",
    )
    parser.add_argument(
        "--no-assert",
        action="store_true",
        help="Skip convergence assertions (useful for exploratory runs).",
    )
    args = parser.parse_args()

    jax.config.update("jax_enable_x64", True)

    print(f"Loading Adult dataset from {_CSV_PATH}")
    domain, _, loss_fn, N = load_problem(sigma=args.sigma)
    print(
        f"  N={N:.0f}, {len(TWO_WAY_CLIQUES)} 2-way + "
        f"{len(domain.attrs)} 1-way cliques, "
        f"σ={args.sigma}"
    )

    results: dict[str, tuple[list[float], list[float]]] = {}
    for name in ESTIMATORS:
        print(f"\nRunning {name} for {args.iters} iterations...")
        t0 = time.time()
        iters_list, losses = run_estimator(
            name,
            domain,
            loss_fn,
            N,
            iters=args.iters,
        )
        elapsed = time.time() - t0
        final = losses[-1] if losses else float("inf")
        print(f"  {name}: final loss = {final:.2f}  ({elapsed:.1f}s)")
        results[name] = (iters_list, losses)

    # --- Assertions --------------------------------------------------------
    if not args.no_assert:
        print("\n--- Convergence assertions ---")
        all_pass = True
        for name in ESTIMATORS:
            _, losses = results[name]
            final = losses[-1] if losses else float("inf")
            threshold = THRESHOLDS[name]
            ok = final <= threshold
            status = "PASS" if ok else "FAIL"
            print(f"  {name}: {final:.2f} <= {threshold:.2f}? {status}")
            if not ok:
                all_pass = False
        if not all_pass:
            print("\nFAILED: One or more estimators did not converge.")
            sys.exit(1)
        print("\nAll estimators converged within thresholds.")

    # --- Plotting ----------------------------------------------------------
    if args.plot or args.save:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        for name in ESTIMATORS:
            iters_list, losses = results[name]
            ax.plot(iters_list, losses, label=name, linewidth=1.5)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Train Loss")
        ax.set_title(
            f"Estimator Convergence — Adult (N={N:.0f}, σ={args.sigma})"
        )
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        if args.save:
            fig.savefig(args.save, dpi=150)
            print(f"Plot saved to {args.save}")
        if args.plot:
            plt.show()


if __name__ == "__main__":
    main()
