import numpy as np
from mbi import estimation, Dataset, Domain, LinearMeasurement
from scipy import sparse
from scipy.cluster.hierarchy import DisjointSet
import networkx as nx
import itertools
from cdp2adp import cdp_rho
from scipy.special import logsumexp
import argparse
import pandas as pd
import os
import json

"""
This is a generalization of the winning mechanism from the 
2018 NIST Differential Privacy Synthetic Data Competition.

Unlike the original implementation, this one can work for any discrete dataset,
and does not rely on public provisional data for measurement selection.  
"""


def MST(data, epsilon, delta):
  rho = cdp_rho(epsilon, delta)
  sigma = np.sqrt(3 / (2 * rho))
  cliques = [(col,) for col in data.domain]
  log1 = measure(data, cliques, sigma)
  data, log1, undo_compress_fn = compress_domain(data, log1)
  cliques = select(data, rho / 3.0, log1)
  log2 = measure(data, cliques, sigma)
  est = estimation.mirror_descent(data.domain, log1+log2, iters=10000)
  synth = est.synthetic_data()
  return undo_compress_fn(synth)


def measure(data, cliques, sigma, weights=None):
  if weights is None:
    weights = np.ones(len(cliques))
  weights = np.array(weights) / np.linalg.norm(weights)
  measurements = []
  for proj, wgt in zip(cliques, weights):
    x = data.project(proj).datavector()
    y = x + np.random.normal(loc=0, scale=sigma / wgt, size=x.size)
    measurements.append(LinearMeasurement(y, proj, sigma / wgt))
  return measurements


def compress_domain(data, measurements):
  supports = {}
  new_measurements = []
  for M in measurements:
    col = M.clique[0]
    y = M.noisy_measurement
    sup = y >= 3 * M.stddev
    supports[col] = sup
    if supports[col].all():
      new_measurements.append(M)
    else:  # need to re-express measurement over the new domain
      y2 = np.append(y[sup], y[~sup].sum())
      I2 = np.ones(y2.size)
      I2[-1] = 1.0 / np.sqrt(y.size - y2.size + 1.0)
      y2[-1] /= np.sqrt(y.size - y2.size + 1.0)
      # temporary hack to get MST working again
      query = (lambda I2: lambda mu: mu.datavector() * I2)(I2)  
      new_measurements.append(LinearMeasurement(y2, M.clique, M.stddev, query=query))
  undo_compress_fn = lambda data: reverse_data(data, supports)
  return transform_data(data, supports), new_measurements, undo_compress_fn


def exponential_mechanism(q, eps, sensitivity, prng=np.random, monotonic=False):
  coef = 1.0 if monotonic else 0.5
  scores = coef * eps / sensitivity * q
  probas = np.exp(scores - logsumexp(scores))
  return prng.choice(q.size, p=probas)


def select(data, rho, measurement_log, cliques=[]):

  est = estimation.mirror_descent(data.domain, measurement_log, iters=2500)

  weights = {}
  candidates = list(itertools.combinations(data.domain.attrs, 2))
  for a, b in candidates:
    xhat = est.project([a, b]).datavector()
    x = data.project([a, b]).datavector()
    weights[a, b] = np.linalg.norm(x - xhat, 1)

  T = nx.Graph()
  T.add_nodes_from(data.domain.attrs)
  ds = DisjointSet(data.domain.attrs)

  for e in cliques:
    T.add_edge(*e)
    ds.merge(*e)

  r = len(list(nx.connected_components(T)))
  epsilon = np.sqrt(8 * rho / (r - 1))
  for i in range(r - 1):
    candidates = [e for e in candidates if not ds.connected(*e)]
    wgts = np.array([weights[e] for e in candidates])
    idx = exponential_mechanism(wgts, epsilon, sensitivity=1.0)
    e = candidates[idx]
    T.add_edge(*e)
    ds.merge(*e)

  return list(T.edges)


def transform_data(data, supports):
  df = pd.DataFrame(data.to_dict(), columns=data.domain.attrs)
  newdom = {}
  for col in data.domain:
    support = supports[col]
    size = support.sum()
    newdom[col] = int(size)
    if size < support.size:
      newdom[col] += 1
    mapping = {}
    idx = 0
    for i in range(support.size):
      mapping[i] = size
      if support[i]:
        mapping[i] = idx
        idx += 1
    assert idx == size
    df[col] = df[col].map(mapping)
  newdom = Domain.fromdict(newdom)
  return Dataset(df.values.astype(int), newdom)


def reverse_data(data, supports):
  df = pd.DataFrame(data.to_dict(), columns=data.domain.attrs)
  newdom = {}
  for col in data.domain:
    support = supports[col]
    mx = support.sum()
    newdom[col] = int(support.size)
    idx, extra = np.where(support)[0], np.where(~support)[0]
    mask = df[col] == mx
    if extra.size == 0:
      pass
    else:
      df.loc[mask, col] = np.random.choice(extra, mask.sum())
    df.loc[~mask, col] = idx[df.loc[~mask, col]]
  newdom = Domain.fromdict(newdom)
  return Dataset(df.values, newdom)


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

  return params


if __name__ == "__main__":

  description = ""
  formatter = argparse.ArgumentDefaultsHelpFormatter
  parser = argparse.ArgumentParser(description=description, formatter_class=formatter)
  parser.add_argument("--dataset", help="dataset to use")
  parser.add_argument("--domain", help="domain to use")
  parser.add_argument("--epsilon", type=float, help="privacy parameter")
  parser.add_argument("--delta", type=float, help="privacy parameter")

  parser.add_argument("--degree", type=int, help="degree of marginals in workload")
  parser.add_argument(
    "--num_marginals", type=int, help="number of marginals in workload"
  )
  parser.add_argument(
    "--max_cells",
    type=int,
    help="maximum number of cells for marginals in workload",
  )

  parser.add_argument("--save", type=str, help="path to save synthetic data")

  parser.set_defaults(**default_params())
  args = parser.parse_args()

  df = pd.read_csv(os.path.join('../data/', args.dataset))
  config = json.load(open(os.path.join('../data/', args.domain), 'r'))
  domain = Domain.fromdict(config)
  data = Dataset(df, domain)

  workload = list(itertools.combinations(data.domain, args.degree))
  workload = [cl for cl in workload if data.domain.size(cl) <= args.max_cells]
  if args.num_marginals is not None:
    prng = np.random.RandomState(None)
    workload = [
      workload[i]
      for i in prng.choice(len(workload), args.num_marginals, replace=False)
    ]

  synth = MST(data, args.epsilon, args.delta)

  if args.save is not None:
    df = pd.DataFrame(synth.to_dict(), columns=synth.domain.attrs)
    df.to_csv(args.save, index=False)

  errors = []
  for proj in workload:
    X = data.project(proj).datavector()
    Y = synth.project(proj).datavector()
    e = 0.5 * np.linalg.norm(X / X.sum() - Y / Y.sum(), 1)
    errors.append(e)
  print("Average Error: ", np.mean(errors))
