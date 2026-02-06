import sys
import os
import numpy as np
import itertools

# Add src to python path to import mbi
sys.path.insert(0, os.path.abspath('src'))

import mbi
from mbi import Dataset

def main():
    try:
        data = Dataset.load('data/adult.csv', 'data/adult-domain.json')
    except Exception as e:
        print(f"Failed to load data: {e}")
        return

    domain = data.domain

    clique = ('marital-status',
     'occupation',
     'relationship',
     'race',
     'sex',
     'native-country',
     'income>50K')

    joint = data.project(clique) + 0.0001
    joint_marginals = mbi.CliqueVector(domain, [clique], { clique: joint })

    model = mbi.MarkovRandomField(
        potentials = joint_marginals.log(),
        marginals=joint_marginals,
        total=joint.values.sum()
    )

    print("Generating synthetic data (method='round')...")
    np.random.seed(0)
    synth = model.synthetic_data(method='round')

    print("Evaluating all 2-way marginals within the clique...")
    errors = []

    # Generate all pairs of attributes in the clique
    pairs = list(itertools.combinations(clique, 2))

    for pair in pairs:
        model_marg = model.project(pair).datavector(flatten=False)
        synth_marg = synth.project(pair).datavector(flatten=False)

        # L1 Error normalized by N
        err = np.abs(model_marg - synth_marg).sum() / data.records
        errors.append(err)
        # print(f"Pair {pair}: {err:.6f}")

    avg_error = np.mean(errors)
    max_error = np.max(errors)
    min_error = np.min(errors)

    print(f"Average 2-way Error: {avg_error:.6f}")
    print(f"Max 2-way Error: {max_error:.6f}")
    print(f"Min 2-way Error: {min_error:.6f}")

    # Also verify the original problematic clique specifically
    cl = ('marital-status', 'relationship')
    if cl in pairs or cl[::-1] in pairs:
        # Just to be explicit
        m = model.project(cl).datavector(flatten=False)
        s = synth.project(cl).datavector(flatten=False)
        e = np.abs(m - s).sum() / data.records
        print(f"Original problematic pair {cl}: {e:.6f}")

if __name__ == "__main__":
    main()
