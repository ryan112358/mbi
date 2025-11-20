from mbi import (
  Dataset,
  Domain,
  marginal_loss,
  estimation
)
import numpy as np

# load adult dataset

data = Dataset.load("../data/adult.csv", "../data/adult-domain.json")
domain = data.domain
total = data.records

print(domain)

# spend half of privacy budget to measure all 1 way marginals
np.random.seed(0)

cliques = [
  ("age", "education-num"),
  ("marital-status", "race"),
  ("sex", "hours-per-week"),
  ("hours-per-week", "income>50K"),
  ("native-country", "marital-status", "occupation"),
]


epsilon = 1.0
epsilon_split = epsilon / (len(data.domain) + len(cliques))
sigma = 2.0 / epsilon_split

measurements = []
for col in data.domain:
  x = data.project(col).datavector()
  y = x + np.random.laplace(loc=0, scale=sigma, size=x.size)
  measurements.append(marginal_loss.LinearMeasurement(y, (col,)))

# spend half of privacy budget to measure some more 2 and 3 way marginals

for cl in cliques:
  x = data.project(cl).datavector()
  y = x + np.random.laplace(loc=0, scale=sigma, size=x.size)
  measurements.append(marginal_loss.LinearMeasurement(y, cl))

# now estimate the data distribution from the noisy measurements

estimated_total = estimation.minimum_variance_unbiased_total(measurements)
loss_fn = marginal_loss.from_linear_measurements(measurements)
model = estimation.mirror_descent(domain, loss_fn, known_total=estimated_total, iters=2500, stepsize=4e-5)

# now answer new queries

y1 = model.project(("sex", "income>50K")).datavector()
y2 = model.project(("race", "occupation")).datavector()

# and compute error:

x1 = data.project(("sex", "income>50K")).datavector()
print('Error on (sex, income>50K)', np.linalg.norm(x1 - y1, 1) / x1.sum())
