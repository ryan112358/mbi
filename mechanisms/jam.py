"""
Implementation of JAM (Joint Adaptive Measurements) for DP Synthetic Data.
Combines private and public measurements through iterative selection.

Note that this method uses bounded DP as the neighborhood relation and should
be compared to other bounded DP methods.

For full technical details see: 
Fuentes, M., Mullins, B.C., McKenna, R., Miklau, G. &amp; Sheldon, D.. (2024). 
Joint Selection: Adaptively Incorporating Public Information for Private Synthetic Data. 
Proceedings of The 27th International Conference on Artificial Intelligence and Statistics, 
in Proceedings of Machine Learning Research 238:2404-2412 Available from https://proceedings.mlr.press/v238/fuentes24a.html.
"""

import argparse
import itertools
import os

import numpy as np

from mbi import (
    Dataset,
    estimation,
    junction_tree,
    LinearMeasurement,
)
from mechanism import Mechanism


def powerset(iterable):
    """powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"""
    s = list(iterable)
    return itertools.chain.from_iterable(
        itertools.combinations(s, r) for r in range(1,
                                                    len(s) + 1))


def downward_closure(workloads):
    """Computes the downward closure of a set of workloads."""
    ans = set()
    for proj in workloads:
        ans.update(powerset(proj))
    return list(sorted(ans, key=len))


def all_k_way(attrs, k):
    """Generates all k-way marginals (cliques) from attributes."""
    return list(itertools.combinations(attrs, k))


def compile_workload(workload):
    """
    Compiles a workload into a dictionary of downward-closed cliques
    with aggregated scores.
    """
    weights = dict(workload)
    workload_cliques_keys = weights.keys()

    def score(clique):
        return sum(weights[workload_cl] * len(set(clique) & set(workload_cl))
                   for workload_cl in workload_cliques_keys)

    return {clique: score(clique) for clique in downward_closure(workload_cliques_keys)}


def adaptive_split(rho_total, rho_used, num_rounds, current_round, alpha):
    """Splits remaining privacy budget (rho) for selection and measurement."""
    rho_remaining = rho_total - rho_used
    rounds_left = num_rounds - current_round
    if rho_remaining <= 0 or rounds_left <= 0:
        return 0.0, 0.0
    rho_per_round = rho_remaining / rounds_left
    rho_select = rho_per_round * alpha
    rho_measure = rho_per_round * (1 - alpha)
    return rho_select, rho_measure


def exponential_mech_eps(rho_select):
    """Converts a rho budget for selection into an epsilon for Exponential Mechanism."""
    # Consistent with AIM's epsilon derivation (epsilon = sqrt(8 * rho))
    return np.sqrt(8 * rho_select)


def gaussian_sigma(rho_measure, sensitivity):
    """Calculates Gaussian noise stddev (sigma) from rho and sensitivity."""
    if rho_measure <= 0:
        return float('inf')
    return sensitivity / np.sqrt(2 * rho_measure)


def size_filter(domain, current_cliques, candidate_clique,
                max_model_size_limit):
    """Filters candidates that would exceed model size limit."""
    if candidate_clique in current_cliques:
        return True
    return junction_tree.hypothetical_model_size(
        domain, current_cliques + [candidate_clique]) <= max_model_size_limit


def expected_priv_error(sigma, num_cells):
    """Calculates expected L1 error from Gaussian noise."""
    return np.sqrt(2 / np.pi) * sigma * num_cells


# --- JAM Mechanism Class ---


class JAM(Mechanism):
    """
    Implementation of JAM (Joint Adaptive Measurements) for DP Synthetic Data.
    Combines private and public measurements through iterative selection.
    """

    def __init__(self, epsilon, delta, prng=None, alpha=0.2, degree=3,
                 optim_iters=1000, size_limit=80.0, rounds=30):
        # pylint: disable=too-many-arguments,too-many-locals,too-many-positional-arguments
        super().__init__(epsilon, delta, prng)
        self.alpha = alpha
        self.degree = degree
        self.optim_iters = optim_iters
        self.size_limit = size_limit
        self.rounds = rounds

    def run(self, data, pub_data, workload):
        # pylint: disable=arguments-differ,too-many-locals,too-many-statements
        """
        Runs the JAM mechanism to generate synthetic data.

        Args:
            data (Dataset): The private dataset.
            pub_data (Dataset): The public dataset. This is a mandatory argument
                                for JAM, unlike the base Mechanism class.
            workload (list): A list of cliques defining the initial workload.
                             Weights are uniformly set to 1.0.

        Returns:
            tuple: (model, synth) - The estimated graphical model and the
                                    generated synthetic dataset.
        """
        npriv = data.records
        npub = pub_data.records
        nfactor = npriv / npub

        domain = data.domain

        initial_workload_tuples = [(cl, 1.0) for cl in workload]
        candidates = compile_workload(initial_workload_tuples)

        all_candidates = list(candidates.keys())

        # Get true answers for private and (scaled) public data
        priv_answers = {
            marg: data.project(marg).datavector() for marg in all_candidates
        }
        pub_answers = {
            marg: pub_data.project(marg).datavector() * nfactor
            for marg in all_candidates
        }
        pub_errors = {
            marg: np.linalg.norm(priv_answers[marg] - pub_answers[marg], 1)
            for marg in all_candidates
        }

        model = None
        rho_used = 0

        marg_l2_sensitivity = np.sqrt(2.0)  # L2 sensitivity for counts
        score_sensitivity_exp_mech = 4.0  # L1 error sensitivity for scores

        # --- Iterative Selection Loop ---
        t = 0
        measurements = []
        while t < self.rounds:
            rho_select, rho_measure = adaptive_split(self.rho, rho_used,
                                                     self.rounds, t, self.alpha)
            selection_eps = exponential_mech_eps(rho_select)
            measurement_sigma = gaussian_sigma(rho_measure, marg_l2_sensitivity)

            # Filter candidates by model size
            current_cliques_in_model = list(set(m.clique for m in measurements))
            current_size_limit = (
                1 / self.rounds) * self.size_limit if t == 0 else (
                    rho_used / self.rho) * self.size_limit

            candidates_filtered = [
                marg for marg in all_candidates if size_filter(
                    domain, current_cliques_in_model, marg, current_size_limit)
            ]

            # Calculate scores for selection
            priv_scores = []
            pub_scores = []

            if not measurements:  # First round
                priv_scores = [
                    -1 *
                    expected_priv_error(measurement_sigma, domain.size(marg))
                    for marg in candidates_filtered
                ]
                pub_scores = [
                    -1 * pub_errors[marg] for marg in candidates_filtered
                ]
            else:
                model_error = {
                    marg:
                        np.linalg.norm(
                            priv_answers[marg] -
                            model.project(marg).datavector(), 1)
                    for marg in candidates_filtered
                }
                priv_scores = [
                    model_error[marg] -
                    expected_priv_error(measurement_sigma, domain.size(marg))
                    for marg in candidates_filtered
                ]
                pub_scores = [
                    model_error[marg] - pub_errors[marg]
                    for marg in candidates_filtered
                ]

            # Make selection using Exponential Mechanism
            score_dict = {
                (clique, 'PRIV'): score
                for clique, score in zip(candidates_filtered, priv_scores)
            }
            score_dict.update({
                (clique, 'PUB'): score
                for clique, score in zip(candidates_filtered, pub_scores)
            })

            selected_marg, mtype = self.exponential_mechanism(
                score_dict,
                selection_eps,
                sensitivity=score_sensitivity_exp_mech)
            rho_used += rho_select

            if mtype == 'PRIV':
                x = priv_answers[selected_marg]
                y = x + self.gaussian_noise(measurement_sigma, x.size)
                measurements.append(
                    LinearMeasurement(y, selected_marg, stddev=1.0))
                rho_used += rho_measure
            else:
                x = pub_answers[selected_marg]
                y = x
                measurements.append(
                    LinearMeasurement(y, selected_marg, stddev=1.0))

            # Update Model
            potentials = model.potentials if model else None

            print(
                f'Measuring Clique {selected_marg} ({mtype}) - Round {t+1}/{self.rounds} '
                f'Meas_Sigma={measurement_sigma:.3g} Sel_Eps={selection_eps:.3g} '
                f'Budget Used={rho_used:.3g}/{self.rho:.3g}',
                flush=True)

            model = estimation.mirror_descent(domain,
                                              measurements,
                                              iters=self.optim_iters,
                                              potentials=potentials,
                                              callback_fn=lambda *_: None)
            model_size = junction_tree.hypothetical_model_size(
                domain, model.cliques)
            print(f'Model Size: {model_size:.2f} MB', flush=True)

            t += 1
            if rho_used >= self.rho:
                print(
                    f"[{self.__class__.__name__}] Privacy budget exhausted. Terminating early."
                )
                break

        # --- Final Model Estimation and Synthetic Data Generation ---
        print("Generating Data...", flush=True)
        final_potentials = model.potentials if model else None
        final_model = estimation.mirror_descent(domain,
                                                measurements,
                                                iters=self.optim_iters,
                                                potentials=final_potentials)

        synth = final_model.synthetic_data(rows=npriv)

        return final_model, synth


# --- Example Usage ---
def default_params():
    """Returns a dictionary of default parameters for the JAM mechanism."""
    params = {}
    params["dataset"] = "adult_pub025_01"
    params["data_dir"] = os.path.join("..", "data")
    params["epsilon"] = 1.0
    params["delta"] = 1e-9
    params["alpha"] = 0.2
    params["degree"] = 3
    params["seed"] = 17
    params["optim_iters"] = 1000
    params["size_limit"] = 80.0
    params["rounds"] = 30
    return params


if __name__ == '__main__':
    DESCRIPTION = "Run JAM (Joint Adaptive Measurements) mechanism for DP Synthetic Data."
    FORMATTER = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=DESCRIPTION,
                                     formatter_class=FORMATTER)

    parser.add_argument("--dataset",
                        type=str,
                        help="dataset name (e.g., adult)")
    parser.add_argument("--data_dir",
                        type=str,
                        help="base directory for datasets")
    parser.add_argument("--epsilon",
                        type=float,
                        help="privacy parameter epsilon")
    parser.add_argument("--delta", type=float, help="privacy parameter delta")
    parser.add_argument("--alpha",
                        type=float,
                        help="selection budget proportion")
    parser.add_argument("--degree",
                        type=int,
                        help="degree of marginals in workload")
    parser.add_argument("--seed",
                        type=int,
                        help="PRNG seed")
    parser.add_argument("--optim_iters",
                        type=int,
                        help="number of optimization iterations")
    parser.add_argument("--size_limit",
                        type=float,
                        help="model size limit in MB (approx)")
    parser.add_argument("--rounds",
                        type=int,
                        help="number of iterative selection rounds")
    parser.add_argument("--save_synth",
                        type=str,
                        help="path to save synthetic data CSV")
    parser.add_argument("--save_errors",
                        type=str,
                        help="path to save errors CSV")

    parser.set_defaults(**default_params())
    args = parser.parse_args()

    print(f"Running JAM with args: {args}", flush=True)

    data_folder = os.path.join(args.data_dir, args.dataset)
    private_csv_path = os.path.join(data_folder, "priv.csv")
    public_csv_path = os.path.join(data_folder, "pub.csv")
    domain_json_path = os.path.join(data_folder, f"{args.dataset}-domain.json")

    private_data = Dataset.load(path=private_csv_path, domain=domain_json_path)
    public_data = Dataset.load(path=public_csv_path, domain=domain_json_path)

    mech = JAM(
        epsilon=args.epsilon,
        delta=args.delta,
        prng=np.random.default_rng(args.seed),
        alpha=args.alpha,
        degree=args.degree,
        optim_iters=args.optim_iters,
        size_limit=args.size_limit,
        rounds=args.rounds,
    )

    workload_cliques = all_k_way(private_data.domain.attrs, args.degree)

    # Pass public_data as a mandatory argument to run
    model, synth_data = mech.run(private_data, public_data, workload_cliques)

    # --- Compute Error and Save Results ---
    errors = []
    normalization_factor = 1 / private_data.records if private_data.records > 0 else 1.0

    for cl in workload_cliques:
        X = private_data.project(cl).datavector()
        Y = synth_data.project(cl).datavector()
        e = np.linalg.norm(X - Y, 1) * normalization_factor
        errors.append(e)

    err_avg = np.mean(errors)
    err_max = np.max(errors)

    if args.save_synth:
        synth_data.df.to_csv(args.save_synth, index=False)
        print(f"Synthetic data saved to {args.save_synth}")

    if args.save_errors:
        results_dir = os.path.dirname(args.save_errors)
        if results_dir:
            os.makedirs(results_dir, exist_ok=True)

        with open(args.save_errors, 'a', encoding='utf-8') as f:
            f.write(
                f'{args.size_limit},{args.seed},{args.epsilon},{err_avg},{err_max}\n'
            )
        print(f"Errors saved to {args.save_errors}")

    print(
        f"Results: size_limit={args.size_limit}, seed={args.seed}, epsilon={args.epsilon}, "
        f"avg_error={err_avg:.6f}, max_error={err_max:.6f}",
        flush=True)
