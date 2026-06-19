from mbi.experimental import LocalInference
import numpy as np
import argparse
from hdmm import workload, templates, error
from mbi import Dataset
import itertools
from autodp import privacy_calibrator
import os
import pandas as pd
import pickle

"""
This is an implementation of HDMM+APPGM as described in the paper

"Relaxed Marginal Consistency for Differentially Private Query Answering"

This is a mechanism for answering a workload of marginal queries under epsilon or 
(epsilon, delta)-DP.  This file depends on the autodp library and the hdmm library.  
The former can be installed with 

$ pip install autodp

The latter can be installed by following the instructions on the official HDMM repository

https://github.com/dpcomp-org/hdmm

Note that HDMM optimizes for a measure of overall error.  It achieves this by adding adding 
correlated non-uniform noise to different queries.  If you are interested in evaluating max error, 
HDMM will be outperformed by a simpler mechanism that just answers every query with the same amount
of noise, at least for the special case of marginal query workloads.  

Note that this file can take some time to run.  The strategy selection step takes a good amount of time, but is 
saved so subsequent calls will take less time than the first call.
"""


def convert_matrix(domain, cliques):
    weights = {}
    for proj in cliques:
        tpl = tuple([domain.attrs.index(i) for i in proj])
        weights[tpl] = 1.0
    return workload.Marginals.fromtuples(domain.shape, weights)


def convert_back(domain, Q):
    cliques = []
    weights = []
    for Qi in Q.matrices:
        wgt = Qi.weight
        key = tuple([domain.attrs[i] for i in Qi.base.tuple()])
        cliques.append(key)
        weights.append(wgt)
    return cliques, weights


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
    params["parameterization"] = "OPT+"
    params["pgm_iters"] = 2500
    params["restarts"] = 1
    params["seed"] = 0
    params["save"] = None

    return params


def optm(queries, approx=False):
    W = convert_matrix(data.domain, queries)
    if os.path.exists(strategy_path):
        print("loading strategy from file")
        A = pickle.load(open(strategy_path, "rb"))
    else:
        print("optimizing strategy, could take a while")
        best_obj = np.inf
        for _ in range(args.restarts):
            if args.parameterization == "OPTM":
                temp = templates.Marginals(
                    data.domain.shape,
                    approx=args.noise == "gaussian",
                    seed=args.seed,
                )
            else:
                temp = templates.MarginalUnionKron(
                    data.domain.shape,
                    len(queries),
                    approx=args.noise == "gaussian",
                )
            obj = temp.optimize(W)
            if obj < best_obj:
                best_obj = obj
                A = temp.strategy()
    return convert_back(data.domain, A)


def opt_plus(queries, approx=False):
    # Just return the workload itself, appropriately weighted
    weights = np.ones(len(queries))
    return queries, weights


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
        "--parameterization",
        choices=["OPTM", "OPT+"],
        help="Strategy parameterization to optimize over",
    )
    parser.add_argument(
        "--noise",
        choices=["laplace", "gaussian"],
        help="noise distribution to use",
    )

    parser.add_argument(
        "--workload", type=int, help="number of marginals in workload"
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

    print("%d Dimensional Domain" % len(data.domain))
    if len(data.domain) >= 13 and args.parameterization == "OPTM":
        print(
            "Time complexity of strategy optimization using OPT_M is O(4^d),"
            " could be slow for this domain"
        )

    queries = list(itertools.combinations(data.domain, args.degree))
    queries = [cl for cl in queries if data.domain.size(cl) <= args.max_cells]
    if args.num_marginals is not None and args.num_marginals < len(queries):
        queries = [
            queries[i]
            for i in prng.choice(
                len(queries), args.num_marginals, replace=False
            )
        ]

    key = (
        args.degree,
        args.seed,
        data.domain.shape,
        args.noise == "gaussian",
        args.max_cells,
        args.parameterization == "OPTM",
    )
    print(hash(key))
    strategy_path = "hdmm-%d.pkl" % hash(key)

    W = convert_matrix(data.domain, queries)
    if os.path.exists(strategy_path):
        print("loading strategy from file")
        strategy = pickle.load(open(strategy_path, "rb"))
    else:
        if args.parameterization == "OPTM":
            strategy = optm(queries, approx=args.noise == "gaussian")
        else:
            strategy = opt_plus(queries, approx=args.noise == "gaussian")
        pickle.dump(strategy, open(strategy_path, "wb"))

    cliques, weights = strategy

    prng = np.random
    if args.noise == "laplace":
        var = 2.0 / args.epsilon**2
        sensitivity = np.linalg.norm(weights, 1)
        add_noise = lambda x: x + sensitivity * prng.laplace(
            loc=0, scale=1.0 / args.epsilon, size=x.size
        )
    else:
        sigma = privacy_calibrator.gaussian_mech(args.epsilon, args.delta)[
            "sigma"
        ]
        var = sigma**2
        sensitivity = np.linalg.norm(weights, 2)
        add_noise = lambda x: x + sensitivity * prng.normal(
            loc=0, scale=sigma, size=x.size
        )

    measurements = []
    for proj, wgt in zip(cliques, weights):
        Q = wgt * workload.Identity(data.domain.size(proj))
        x = data.project(proj).datavector()
        y = add_noise(Q @ x)
        measurements.append((Q, y, 1.0, proj))

    engine = LocalInference(data.domain, iters=args.pgm_iters, log=True)
    model = engine.estimate(measurements)

    errors = []
    for proj in queries:
        X = data.project(proj).datavector()
        Y = model.project(proj).datavector()
        e = 0.5 * np.linalg.norm(X / X.sum() - Y / Y.sum(), 1)
        errors.append(e)
    print("Average Error: ", np.mean(errors))
