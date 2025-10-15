from mbi import Dataset, Domain
from mbi import marginal_loss, estimation, relaxed_projection
import numpy as np

# discrete domain with attributes A, B, C and corresponding size 4 x 5 x 6
domain = Domain(['A', 'B', 'C'], [2, 3, 4])

# synthetic dataset with 1000 rows
data = Dataset.synthetic(domain, 1000)

# project data onto subset of cols, and vectorize
ab = data.project(['A', 'B']).datavector()
bc = data.project(['B', 'C']).datavector()

# add noise to preserve differential privacy
epsilon = np.sqrt(2)
sigma = np.sqrt(2.0) / epsilon

np.random.seed(0)
yab = ab + np.random.laplace(loc=0, scale=sigma, size=ab.size)
ybc = bc + np.random.laplace(loc=0, scale=sigma, size=bc.size)

print(yab)
print(ybc)

measurements = [marginal_loss.LinearMeasurement(yab, ['A', 'B']), marginal_loss.LinearMeasurement(ybc, ['B', 'C'])]

loss_fn = marginal_loss.from_linear_measurements(measurements)

# estimate the data distribution
# model = estimation.mirror_descent(domain, loss_fn, known_total=1000)

# estimation via RP
model = relaxed_projection.relaxed_projection_estimation(domain, loss_fn, known_total=1000)

# recover consistent estimates of measurements
ab2 = model.project(['A', 'B']).datavector()
bc2 = model.project(['B', 'C']).datavector()

print(ab2)
print(bc2)
print('L1 loss of ab:', np.linalg.norm(ab2/np.sum(ab2) - yab/np.sum(yab), 1))
print('L1 loss of bc:', np.linalg.norm(bc2/np.sum(bc2) - ybc/np.sum(ybc), 1))

# estimate answer to unmeasured queries
ac2 = model.project(['A', 'C']).datavector()
#print(ac2)

# generate synthetic data
synth = model.synthetic_data()
print(synth.df.head())
