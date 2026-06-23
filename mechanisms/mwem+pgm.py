import numpy as np
import itertools
from mbi import Dataset, LinearMeasurement, estimation, callbacks, junction_tree
from scipy.special import softmax
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from cdp2adp import cdp_rho
import argparse


"""
This file contains an implementation of MWEM+PGM that is designed specifically for marginal query workloads.
Unlike mwem.py, which selects a single query in each round, this implementation selects an entire marginal 
in each step.  It leverages parallel composition to answer many more queries using the same privacy budget.

This enhancement of MWEM was described in the original paper in section 3.3 (https://arxiv.org/pdf/1012.4763.pdf).

There are two additional improvements not described in the original Private-PGM paper:
- In each round we only consider candidate cliques to select if they result in sufficiently small model sizes
- At the end of the mechanism, we generate synthetic data (rather than query answers)
"""


def worst_approximated(
    workload_answers, est, workload, eps, penalty=True, bounded=False
):
    """Select a (noisy) worst-approximated marginal for measurement.

    :param workload_answers: a dictionary of true answers to the workload
        keys are cliques
        values are numpy arrays, corresponding to the counts in the marginal
    :param est: a GraphicalModel object that approximates the data distribution
    :param: workload: The list of candidates to consider in the exponential mechanism
    :param eps: the privacy budget to use for this step.
    """
    errors = np.array([])
    for cl in workload:
        bias = est.domain.size(cl) if penalty else 0
        x = workload_answers[cl]
        xest = est.project(cl).datavector()
        errors = np.append(errors, np.abs(x - xest).sum() - bias)
    sensitivity = 2.0 if bounded else 1.0
    prob = softmax(0.5 * eps / sensitivity * (errors - errors.max()))
    key = np.random.choice(len(errors), p=prob)
    return workload[key]


def hypothetical_model_size(domain, cliques):
    jtree, _ = junction_tree.make_junction_tree(domain, cliques)
    maximal_cliques = junction_tree.maximal_cliques(jtree)
    cells = sum(domain.size(cl) for cl in maximal_cliques)
    size_mb = cells * 8 / 2**20
    return size_mb


def mwem_pgm(
    data,
    epsilon,
    delta=0.0,
    workload=None,
    rounds=None,
    maxsize_mb=25,
    pgm_iters=1000,
    noise="gaussian",
    bounded=False,
    alpha=0.9,
):
    """
    Implementation of MWEM + PGM

    :param data: an mbi.Dataset object
    :param epsilon: privacy budget
    :param delta: privacy parameter (ignored)
    :param workload: A list of cliques (attribute tuples) to include in the workload (default: all pairs of attributes)
    :param rounds: The number of rounds of MWEM to run (default: number of attributes)
    :param maxsize_mb: [New] a limit on the size of the model (in megabytes), used to filter out candidate cliques from selection.
        Used to avoid MWEM+PGM failure modes (intractable model sizes).
        Set to np.inf if you would like to run MWEM as originally described without this modification
        (Note it may exceed resource limits if run for too many rounds)

    Implementation Notes:
    - During each round of MWEM, one clique will be selected for measurement, but only if measuring the clique does
        not increase size of the graphical model too much
    """
    if workload is None:
        workload = list(itertools.combinations(data.domain, 2))
    if rounds is None:
        rounds = len(data.domain)

    if noise == "laplace":
        eps_per_round = epsilon / rounds
        sigma = 1.0 / (alpha * eps_per_round)
        exp_eps = (1 - alpha) * eps_per_round
        marginal_sensitivity = 2 if bounded else 1.0
    else:
        rho = cdp_rho(epsilon, delta)
        rho_per_round = rho / rounds
        sigma = np.sqrt(0.5 / (alpha * rho_per_round))
        exp_eps = np.sqrt(8 * (1 - alpha) * rho_per_round)
        marginal_sensitivity = np.sqrt(2) if bounded else 1.0

    domain = data.domain
    total = data.records if bounded else None

    workload_answers = {cl: data.project(cl).datavector() for cl in workload}

    measurements = []
    est = estimation.mirror_descent(domain, measurements, known_total=total)
    # import IPython; IPython.embed()
    cliques = []
    for i in range(1, rounds + 1):
        # [New] Only consider candidates that keep the model sufficiently small
        candidates = [
            cl
            for cl in workload
            if hypothetical_model_size(domain, cliques + [cl])
            <= maxsize_mb * i / rounds
        ]
        ax = worst_approximated(workload_answers, est, candidates, exp_eps)
        model_size = hypothetical_model_size(domain, cliques)
        print("Round", i, "Selected", ax, "Model Size (MB)", model_size)
        n = domain.size(ax)
        x = data.project(ax).datavector()
        if noise == "laplace":
            y = x + np.random.laplace(
                loc=0, scale=marginal_sensitivity * sigma, size=n
            )
        else:
            y = x + np.random.normal(
                loc=0, scale=marginal_sensitivity * sigma, size=n
            )
        measurements.append(LinearMeasurement(y, ax))
        est = estimation.mirror_descent(
            domain,
            measurements,
            known_total=total,
            potentials=est.potentials,
            callback_fn=callbacks.default(measurements, data),
        )
        cliques.append(ax)

    print("Generating Data...")
    return est.synthetic_data()


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
    params["rounds"] = None
    params["noise"] = "gaussian"
    params["max_model_size"] = 25
    params["pgm_iters"] = 1000
    params["degree"] = 2
    params["num_marginals"] = None
    params["max_cells"] = 10000

    return params


if __name__ == "__main__":

    description = ""
    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(
        description=description, formatter_class=formatter
    )
    parser.add_argument("--dataset", help="dataset to use")
    parser.add_argument("--domain", help="domain to use")
    parser.add_argument("--epsilon", type=float, help="privacy parameter")
    parser.add_argument("--delta", type=float, help="privacy parameter")
    parser.add_argument(
        "--rounds", type=int, help="number of rounds of MWEM to run"
    )
    parser.add_argument(
        "--noise",
        choices=["laplace", "gaussian"],
        help="noise distribution to use",
    )
    parser.add_argument(
        "--max_model_size",
        type=float,
        help="maximum size (in megabytes) of model",
    )

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

    parser.add_argument("--pgm_iters", type=int, help="number of iterations")
    parser.add_argument("--save", type=str, help="path to save synthetic data")

    parser.set_defaults(**default_params())
    args = parser.parse_args()

    data = Dataset.load(args.dataset, args.domain)

    workload = list(itertools.combinations(data.domain, args.degree))
    workload = [cl for cl in workload if data.domain.size(cl) <= args.max_cells]
    if args.num_marginals is not None and args.num_marginals < len(workload):
        workload = [
            workload[i]
            for i in np.random.choice(
                len(workload), args.num_marginals, replace=False
            )
        ]

    synth = mwem_pgm(
        data,
        args.epsilon,
        args.delta,
        workload=workload,
        rounds=args.rounds,
        maxsize_mb=args.max_model_size,
        pgm_iters=args.pgm_iters,
    )

    if args.save is not None:
        synth.df.to_csv(args.save, index=False)

    errors = []
    for proj in workload:
        X = data.project(proj).datavector()
        Y = synth.project(proj).datavector()
        e = 0.5 * np.linalg.norm(X / X.sum() - Y / Y.sum(), 1)
        errors.append(e)
    print("Average Error: ", np.mean(errors))
