import os
import numpy as np
import pandas as pd
import torch
import tomli
import json
import itertools
import argparse
import jax.numpy as jnp
from mbi import Dataset, Domain, relaxed_projection_estimation, LinearMeasurement
from cdp2adp import cdp_rho


def exponential_mechanism(score, rho, sensitivity):
    max_score = np.max(score)
    scaled_score = [s - max_score for s in score]
    exp_score = [np.exp(np.sqrt(2*rho)/sensitivity * s) for s in scaled_score]
    sample_prob = [score/sum(exp_score) for score in exp_score]
    id = np.random.choice(np.arange(len(exp_score)), p = sample_prob)
    return id


def scale_vector(vec):
    temp_vec = np.clip(vec, 0, np.inf)
    return temp_vec/np.sum(temp_vec)


class RPGen():
    def __init__(
            self, 
            epsilon: float,
            delta: float,
            dataset: Dataset,  
            domain: Domain
        ):
        self.rho = 0 if delta == 0 else cdp_rho(epsilon, delta)
        self.dataset = dataset
        self.domain = domain
        self.save_loss = False


    def fit(self):
        """
        This is a simple example, fitting RP with all 
        one-way and two-way marginals without DP
        """
        print('Start fitting all')
        selected_marginals = []
        marginals = list(itertools.combinations(self.domain.attrs, 1)) + list(itertools.combinations(self.domain.attrs, 2))

        for i in range(len(marginals)):
            cl = marginals[i]
            marginal = self.dataset.project(cl).datavector()
            selected_marginals.append(
                LinearMeasurement(scale_vector(marginal), cl)
            )

        self.model = relaxed_projection_estimation(
            self.domain,
            selected_marginals
        )


    def fit_noisy_marginal(self):
        """
        This is a simple example, fitting RP with all 
        one-way and two-way marginals with DP
        """
        print('Start fitting all')
        selected_marginals = []
        marginals = list(itertools.combinations(self.domain.attrs, 1)) + list(itertools.combinations(self.domain.attrs, 2))
        rho = self.rho/len(marginals)

        for i in range(len(marginals)):
            cl = marginals[i]
            marginal = self.dataset.project(cl).datavector()
            marginal += np.random.normal(loc=0, scale=1/np.sqrt(2*rho), size=marginal.shape)

            selected_marginals.append(
                LinearMeasurement(scale_vector(marginal), cl)
            )

        self.model = relaxed_projection_estimation(
            self.domain,
            selected_marginals
        )
        

    def sample(self, num_samples):
        return self.model.synthetic_data(num_samples)
   

# --- Example Usage ---
def default_params():
    """Returns a dictionary of default parameters for the JAM mechanism."""
    params = {}
    params["dataset"] = "../data/adult.csv"
    params["domain"] = "../data/adult-domain.json"
    params["epsilon"] = 1.0
    params["delta"] = 1e-9
    
    return params


if __name__ == '__main__':
    description = ""
    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=description, formatter_class=formatter)
    parser.add_argument("--dataset", help="dataset to use")
    parser.add_argument("--domain", help="domain to use")
    parser.add_argument("--epsilon", type=float, help="privacy parameter")
    parser.add_argument("--delta", type=float, help="privacy parameter")
    parser.set_defaults(**default_params())
    args = parser.parse_args()

    data = Dataset.load(args.dataset, args.domain)

    generator = RPGen(
        epsilon = args.epsilon,
        delta = args.delta,
        dataset = data,
        domain = data.domain
    )
    generator.fit()
    synth = generator.sample(data.df.shape[0])

    workload = list(itertools.combinations(data.domain, 2))
    errors = []
    for proj in workload:
        X = data.project(proj).datavector()
        Y = synth.project(proj).datavector()
        e = 0.5 * np.linalg.norm(X / X.sum() - Y / Y.sum(), 1)
        errors.append(e)
    print("Average Error: ", np.mean(errors))