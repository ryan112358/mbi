import numpy as np
# from autodp import privacy_calibrator
from functools import partial
from cdp2adp import cdp_rho
from scipy.special import softmax


def pareto_efficient(costs):
    eff = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if eff[i]:
            eff[eff] = np.any(
                costs[eff] <= c, axis=1
            )  # Keep any point with a lower cost
    return np.nonzero(eff)[0]


def generalized_em_scores(q, ds, t):
    q = -q
    idx = pareto_efficient(np.vstack([q, ds]).T)
    r = q + t * ds
    r = r[:, None] - r[idx][None, :]
    z = ds[:, None] + ds[idx][None, :]
    s = (r / z).max(axis=1)
    return -s


class Mechanism:

    def __init__(self, epsilon, delta, bounded, prng=np.random):
        """
        Base class for a mechanism.
        :param epsilon: privacy parameter
        :param delta: privacy parameter
        :param bounded: privacy definition (bounded vs unbounded DP)
        :param prng: pseudo random number generator
        """
        self.epsilon = epsilon
        self.delta = delta
        self.rho = 0 if delta == 0 else cdp_rho(epsilon, delta)
        self.bounded = bounded
        self.prng = prng

    def run(self, dataset, workload):
        pass

    def generalized_exponential_mechanism(
        self, qualities, sensitivities, epsilon, t=None, base_measure=None
    ):
        if t is None:
            t = 2 * np.log(len(qualities) / 0.5) / epsilon
        if isinstance(qualities, dict):
            keys = list(qualities.keys())
            qualities = np.array([qualities[key] for key in keys])
            sensitivities = np.array([sensitivities[key] for key in keys])
            if base_measure is not None:
                base_measure = np.log([base_measure[key] for key in keys])
        else:
            keys = np.arange(qualities.size)
        scores = generalized_em_scores(qualities, sensitivities, t)
        key = self.exponential_mechanism(
            scores, epsilon, 1.0, base_measure=base_measure
        )
        return keys[key]

    def permute_and_flip(self, qualities, epsilon, sensitivity=1.0):
        """Sample a candidate from the permute-and-flip mechanism"""
        q = qualities - qualities.max()
        p = np.exp(0.5 * epsilon / sensitivity * q)
        for i in np.random.permutation(p.size):
            if np.random.rand() <= p[i]:
                return i

    def exponential_mechanism(
        self, qualities, epsilon, sensitivity=1.0, base_measure=None
    ):
        if isinstance(qualities, dict):
            # import pandas as pd
            # print(pd.Series(list(qualities.values()), list(qualities.keys())).sort_values().tail())
            keys = list(qualities.keys())
            qualities = np.array([qualities[key] for key in keys])
            if base_measure is not None:
                base_measure = np.log([base_measure[key] for key in keys])
        else:
            qualities = np.array(qualities)
            keys = np.arange(qualities.size)

        """ Sample a candidate from the permute-and-flip mechanism """
        q = qualities - qualities.max()
        if base_measure is None:
            p = softmax(0.5 * epsilon / sensitivity * q)
        else:
            p = softmax(0.5 * epsilon / sensitivity * q + base_measure)

        return keys[self.prng.choice(p.size, p=p)]

    def gaussian_noise_scale(self, l2_sensitivity, epsilon, delta):
        """Return the Gaussian noise necessary to attain (epsilon, delta)-DP"""
        raise ValueError('Temporarily Deprecated due to broken dependency')

    def laplace_noise_scale(self, l1_sensitivity, epsilon):
        """Return the Laplace noise necessary to attain epsilon-DP"""
        if self.bounded:
            l1_sensitivity *= 2.0
        return l1_sensitivity / epsilon

    def gaussian_noise(self, sigma, size):
        """Generate iid Gaussian noise  of a given scale and size"""
        return self.prng.normal(0, sigma, size)

    def laplace_noise(self, b, size):
        """Generate iid Laplace noise  of a given scale and size"""
        return self.prng.laplace(0, b, size)

    def best_noise_distribution(
        self, l1_sensitivity, l2_sensitivity, epsilon, delta
    ):
        """Adaptively determine if Laplace or Gaussian noise will be better, and
        return a function that samples from the appropriate distribution"""
        b = self.laplace_noise_scale(l1_sensitivity, epsilon)
        sigma = self.gaussian_noise_scale(l2_sensitivity, epsilon, delta)
        dist = (
            self.gaussian_noise
            if np.sqrt(2) * b > sigma
            else self.laplace_noise
        )
        if np.sqrt(2) * b < sigma:
            return partial(self.laplace_noise, b)
        return partial(self.gaussian_noise, sigma)
