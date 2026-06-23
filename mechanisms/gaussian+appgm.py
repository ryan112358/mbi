from mbi import approximate_oracles, LinearMeasurement, callbacks
import numpy as np
import argparse
from mbi import Dataset
import itertools
import os
import pandas as pd
from scipy import sparse

"""
This is an implementation of the Gaussian Mechanism + APPGM

APPGM, short for Approx-Private-PGM is described in the paper:
"Relaxed Marginal Consistency for Differentially Private Query Answering"

This is a mechanism for answering a workload of marginal queries under 
(epsilon, delta)-DP.  This file depends on the autodp library which can be installed with:
The former can be installed with 

$ pip install autodp
"""


def default_params():
    """
    Return default parameters to run this program

    :returns: a dictionary of default parameter settings for each command line argument
    """
    params = {}
    params["dataset"] = "../data/adult.csv"
    params["domain"] = "../data/adult-domain.json"
    params["epsilon"] = 1.0
    params["delta"] = 1e-9
    params["degree"] = 2
    params["num_marginals"] = None
    params["max_cells"] = 10000
    params["noise"] = "gaussian"
    params["pgm_iters"] = 5000
    params["restarts"] = 1
    params["seed"] = 0
    params["save"] = None

    return params


if __name__ == "__main__":
    description = ""
    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(
        description=description, formatter_class=formatter
    )
    parser.add_argument("--dataset", type=str, help="path to dataset file")
    parser.add_argument("--domain", type=str, help="path to domain file")
    parser.add_argument("--epsilon", type=float, help="privacy  parameter")
    parser.add_argument("--delta", type=float, help="privacy parameter")

    parser.add_argument(
        "--degree", type=int, help="degree of marginals in workload"
    )
    parser.add_argument(
        "--num_marginals", type=int, help="number of marginals in workload"
    )
    parser.add_argument(
        "--max_cells",
        type=int,
        help="maximum number of cells for marginals in workload",
    )

    parser.add_argument(
        "--pgm_iters", type=int, help="number of optimization iterations"
    )
    parser.add_argument("--restarts", type=int, help="number of HDMM restarts")
    parser.add_argument("--seed", type=int, help="random seed")
    parser.add_argument("--save", type=str, help="path to save results")

    parser.set_defaults(**default_params())
    args = parser.parse_args()

    prng = np.random.RandomState(args.seed)
    data = Dataset.load(args.dataset, args.domain)

    workload = list(itertools.combinations(data.domain, args.degree))
    workload = [cl for cl in workload if data.domain.size(cl) <= args.max_cells]
    if args.num_marginals is not None and args.num_marginals < len(workload):
        workload = [
            workload[i]
            for i in prng.choice(
                len(workload), args.num_marginals, replace=False
            )
        ]

    try:
        from autodp import privacy_calibrator

        sigma = privacy_calibrator.gaussian_mech(args.epsilon, args.delta)[
            "sigma"
        ] * np.sqrt(len(workload))
    except:
        print("AutoDP not installed or configured correctly, using sigma=50")
        sigma = 50
    measurements = []
    for cl in workload:
        Q = sparse.eye(data.domain.size(cl))
        x = data.project(cl).datavector()
        y = x + np.random.normal(loc=0, scale=sigma, size=x.size)
        measurements.append(LinearMeasurement(y, cl, sigma))

    callback_fn = callbacks.default(measurements, data)
    stepsize = 5e-4

    model = approximate_oracles.mirror_descent(
        data.domain,
        measurements,
        iters=args.pgm_iters,
        callback_fn=callback_fn,
        stepsize=stepsize,
    )

    errors = []
    for proj in workload:
        X = data.project(proj).datavector()
        Y = model.project(proj).datavector()
        e = 0.5 * np.linalg.norm(X / X.sum() - Y / Y.sum(), 1)
        errors.append(e)
    print("Average Error: ", np.mean(errors))
