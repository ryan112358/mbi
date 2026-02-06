import sys
import os
import numpy as np

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

    cl = ('marital-status', 'relationship')

    print(f"Running reproduction on clique {cl}")

    for method in ['round', 'sample']:
        np.random.seed(0)
        synth = model.synthetic_data(method=method)
        model_ans = model.project(cl).datavector(flatten=False).astype(int)
        synth_ans = synth.project(cl).datavector(flatten=False).astype(int)
        error = np.abs(model_ans - synth_ans).sum() / data.records
        print(f"{method}: {error}")

if __name__ == "__main__":
    main()
