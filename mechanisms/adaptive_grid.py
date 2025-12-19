import numpy as np
import pandas as pd
import json
from mbi import FactoredInference, Factor, Dataset, Domain
from scipy import sparse
from scipy.special import logsumexp
import itertools
import networkx as nx
from scipy.cluster.hierarchy import DisjointSet
from mechanisms.cdp2adp import cdp_rho
import argparse


def powerset(iterable):
    """Returns powerset of set consisting of elements in ``iterable``.
    Args:
        iterable (iterable): Set that powerset will be taken over

    Returns:
        iterator: the powerset

    Example:
        >>> pset = powerset(['a','b'])
        >>> list(pset)
        [('a',), ('b',), ('a', 'b')]
    """
    s = list(iterable)
    return itertools.chain.from_iterable(
        itertools.combinations(s, r) for r in range(1, len(s) + 1)
    )


def downward_closure(cliques):
    """Returns the 'downward closure' of a list of cliques. The 'downward closure'
    is the union of powersets of each individual clique in a list. Elements within
    each clique are sorted, but the list of cliques is not.

    Args:
        cliques ([iterable]): List of cliques

    Returns:
        list: the downward closure of the set of variables in cliques

    Example:
        >>> downward_closure([[1,2],[2,3]])
        [(2,), (1,), (3,), (1, 2), (2, 3)]
    """
    ans = set()
    for proj in cliques:
        ans.update(powerset(proj))
    return list(sorted(ans, key=len))


def get_permutation_matrix(cl1, cl2, domain):
    # permutation matrix that maps datavector of cl1 factor to datavector of cl2 factor

    """Using the vector-of-counts representation of a database detailed in
    [Li 2012], we create a permutation matrix which maps the database with
    attributes in order cl1 to database with attributes in order cl2. Note that
    cl1 and cl2 contain the same elements, just in different order.

    Example of Concept:
        Let us define two example databases:

        Database A
           id a b c
            1 0 1 0
            2 0 0 0

        Database B
           id b c a
            1 1 0 0
            2 0 0 0

        We know that A = B since only the ordering of the attributes is changed.

        Let #(.) operation return the number of elements in a database
        which satisfy the condition within the parenthesis and let vec(.)
        be the vector-of-counts operation. Thus:

        vec(A) = [#(a=0,b=0,c=0),#(a=0,b=0,c=1),#(a=0,b=1,c=0),...,#(a=1,b=1,c=1)]
               = [1,0,1,0,0,0,0,0]
        vec(B) = [#(b=0,a=0,c=0),#(b=0,a=0,c=1),#(b=0,a=1,c=0),...,#(b=1,a=1,c=1)]
               = [1,0,0,0,1,0,0,0]

        Observe that vec(A) and vec(B) have the same values, but are just
        rearranged. Then, for any two equivalent databases A and B, the
        permutation matrix P is an 8x8 matrix such that:

        P @ vec(A) = vec(B).T

        For two identical database A and B.

    Args:
        cl1 (iterable): Input clique that permutation matrix maps from.
        cl2 (iterable): Target clique that permutation matrix maps to.
        domain (mbi.Domain): A mbi Domain object which holds the shape and names
            of each variable in the domain.

    Returns:
        scipy.sparse.csr_matrix: Sparse permutation matrix.

    Example:
        >>> domain = Domain(attrs=[1,2],shape=[2,2])
        >>> get_permutation_matrix([1,2],[2,1], domain).todense()
        matrix([[1., 0., 0., 0.],
                [0., 0., 1., 0.],
                [0., 1., 0., 0.],
                [0., 0., 0., 1.]])
    """
    assert set(cl1) == set(cl2)
    n = domain.size(cl1)
    fac = Factor(domain.project(cl1), np.arange(n))
    new = fac.transpose(cl2)
    data = np.ones(n)
    row_ind = fac.datavector()
    col_ind = new.datavector()
    return sparse.csr_matrix((data, (row_ind, col_ind)), shape=(n, n))


def get_aggregate(cl, matrices, domain):
    """Returns additional measurement matrices by taking the Kronecker
    product between Identity and previous measurements.

    Args:
        cl (iterable): A clique marginal.
        matrices (dict): A dictionary of measurement matrices where the key is
            the clique and the value is the matrix.
        domain (mbi.Domain): A mbi Domain object which holds the shape and names
            of each variable in the domain.

    Returns:
        scipy.sparse.csr_matrix: Sparse matrix containing additional
            measurements.
    """
    children = [r for r in matrices if set(r) < set(cl) and len(r) + 1 == len(cl)]
    ans = [sparse.csr_matrix((0, domain.size(cl)))]
    for c in children:
        coef = 1.0 / np.sqrt(len(children))
        a = tuple(set(cl) - set(c))
        cl2 = a + c
        Qc = matrices[c]
        P = get_permutation_matrix(cl, cl2, domain)
        T = np.ones(domain.size(a))
        Q = sparse.kron(T, Qc) @ P
        ans.append(coef * Q)
    return sparse.vstack(ans)


def get_identity(cl, post_plausibility, domain):
    """Determine which cells in the cl marginal *could* have a count above
    threshold based on previous measurements.

    Args:
        cl (iterable): A clique marginal
        post_plausibility (dict): Dictionary of previously taken measurements.
            The key is the clique and value is a Factor object.
        domain (mbi.Domain): A mbi Domain object which holds the shape and names
            of each variable in the domain

    Returns:
        scipy.sparse.csr_matrix: Sparse matrix object where cells identified as
        probably containing counts above threshold have value 1.
    """
    children = [
        r for r in post_plausibility if set(r) < set(cl) and len(r) + 1 == len(cl)
    ]
    plausibility = Factor.ones(domain.project(cl))
    for c in children:
        plausibility *= post_plausibility[c]

    row_ind = col_ind = np.nonzero(plausibility.datavector())[0]
    data = np.ones_like(row_ind)
    n = domain.size(cl)
    Q = sparse.csr_matrix((data, (row_ind, col_ind)), (n, n))
    return Q


def exponential_mechanism(q, eps, sensitivity, prng=np.random, monotonic=False):
    """Performs the exponential mechanism. Returned results satisfy eps-DP.

    Args:
        q (ndarray): Weights for each item.
        eps (float): Privacy parameter.
        sensitivity (float): Sensitivity of the query.
        prng (np.random): Pseudo-random number generator to be used.
        monotonic (boolean): True if the addition of an new element to the
            selection set cannot cause the value of the query to increase, False
            otherwise.

    Returns:
        A list containing indices of chosen items.
    """
    if eps == np.inf:
        eps = np.finfo(np.float64).max
    coef = 1.0 if monotonic else 0.5
    scores = coef * eps / sensitivity * (q - q.max())
    probas = np.exp(scores - logsumexp(scores))
    return prng.choice(q.size, p=probas)


def select(data, model, rho, targets=[]):
    """Selects additional measurements using Minimum Spanning Tree based method
    with the exponential mechanism being used to privately select candidate
    edges. Weights for each edge of the tree are based on the L1 norm between
    the marginal counts from the data and the marginal counts from the model.

    Args:
        data (mbi.Dataset): The sensitive dataset.
        model (mbi.GraphicalModel): The DP graphical model learned from the
            first round of measurements.
        rho (float): Remaining privacy budget, calculated using Gaussian DP.
        targets (list, optional): Target columns specified by the user. Default
            is ``[]``.

    Returns:
        List of additional measurements selected by the algorithm.
    """
    weights = {}
    candidates = list(itertools.combinations(data.domain.invert(targets), 2))
    for a, b in candidates:
        xhat = model.project([a, b] + targets).datavector()
        x = data.project([a, b] + targets).datavector()
        weights[a, b] = np.linalg.norm(x - xhat, 1)

    T = nx.Graph()
    T.add_nodes_from(data.domain.attrs)
    ds = DisjointSet()

    r = len(data.domain) - len(targets)
    epsilon = np.sqrt(8 * rho / (r - 1))
    for i in range(r - 1):
        candidates = [e for e in candidates if not ds.connected(*e)]
        wgts = np.array([weights[e] for e in candidates])
        idx = exponential_mechanism(wgts, epsilon, sensitivity=1.0)
        e = candidates[idx]
        T.add_edge(*e)
        ds.merge(*e)

    return [e + tuple(targets) for e in T.edges]


def adagrid(
    data, epsilon, delta, threshold, targets=[], split_strategy=None, **mbi_args
):
    """Implements the Adagrid mechanism used in Sprint 3 of NIST 2021
    Competition by Team Minutemen.

    Args:
        data (mbi.Dataset): The sensitive dataset.
        epsilon (float): Privacy parameter.
        delta (float): Delta parameter in approximate DP. Set to ``0`` if pure
            DP is required.
        threshold (float): Threshold for deciding which cells are
            likely to have non-zero counts .
        targets (iterable, optional): List of target columns. Default is
            ``[]``.
        iters (int): Number of iterations for Mirror Descent algorithm to run.
        split_strategy ([floats]): List of floats, each containing the
            fraction of the zCDP budget allocated to each step of the algorithm.
        mbi_args (kwargs): Args to pass to mbi.FactoredInference. Please refer
            to the comments within this class to determine which parameters to pass.

    Returns:
        mbi.Dataset: Dataset object holding synthetic dataset satisfying
            (epsilon, delta) DP
    """
    rho = cdp_rho(epsilon, delta)
    if not split_strategy:
        rho_step_1 = rho_step_2 = rho_step_3 = rho / 3
    else:
        assert len(split_strategy) == 3
        frac_1, frac_2, frac_3 = np.array(split_strategy) / sum(split_strategy)
        rho_step_1 = rho * frac_1
        rho_step_2 = rho * frac_2
        rho_step_3 = rho * frac_3

    domain = data.domain
    measurements = []
    post_plausibility = {}
    matrices = {}

    step1_outer = [(a,) + tuple(targets) for a in domain if a not in targets]
    step1_all = downward_closure(step1_outer)
    step1_sigma = np.sqrt(0.5 / rho_step_1) * np.sqrt(len(step1_all))

    # Step 1: Measure all 1-way marginals involving target(s)
    for k in range(1, len(targets) + 2):
        split = [cl for cl in step1_all if len(cl) == k]
        print()
        for cl in split:
            I = sparse.eye(domain.size(cl))
            Q1 = get_identity(
                cl, post_plausibility, domain
            )  # get fine-granularity measurements
            Q2 = get_aggregate(cl, matrices, domain) @ (
                I - Q1
            )  # get remaining aggregate measurements
            Q1 = Q1[Q1.getnnz(1) > 0]  # remove all-zero rows
            Q = sparse.vstack([Q1, Q2])
            Q.T = sparse.csr_matrix(Q.T)  # a trick to improve efficiency of Private-PGM
            # Q has sensitivity 1 by construction
            print(
                "Measuring %s, L2 sensitivity %.6f"
                % (cl, np.sqrt(Q.power(2).sum(axis=0).max()))
            )
            #########################################
            ### This code uses the sensitive data ###
            #########################################
            mu = data.project(cl).datavector()
            y = Q @ mu + np.random.normal(loc=0, scale=step1_sigma, size=Q.shape[0])
            #########################################
            est = Q1.T @ y[: Q1.shape[0]]

            post_plausibility[cl] = Factor(
                domain.project(cl), est >= step1_sigma * threshold
            )
            matrices[cl] = Q
            measurements.append((Q, y, 1.0, cl))

    engine = FactoredInference(domain, log=False, **mbi_args)
    engine.estimate(measurements)

    # Step 2: select more marginals using an MST-style approach
    step2_queries = select(data, engine.model, rho_step_2, targets)

    print()
    # step 3: measure those marginals
    step3_sigma = np.sqrt(len(step2_queries)) * np.sqrt(0.5 / rho_step_3)
    for cl in step2_queries:
        I = sparse.eye(domain.size(cl))
        Q1 = get_identity(
            cl, post_plausibility, domain
        )  # get fine-granularity measurements
        Q2 = get_aggregate(cl, matrices, domain) @ (
            I - Q1
        )  # get remaining aggregate measurements
        Q1 = Q1[Q1.getnnz(1) > 0]  # remove all-zero rows
        Q = sparse.vstack([Q1, Q2])
        Q.T = sparse.csr_matrix(Q.T)  # a trick to improve efficiency of Private-PGM
        # Q has sensitivity 1 by construction
        print(
            "Measuring %s, L2 sensitivity %.6f"
            % (cl, np.sqrt(Q.power(2).sum(axis=0).max()))
        )
        #########################################
        ### This code uses the sensitive data ###
        #########################################
        mu = data.project(cl).datavector()
        y = Q @ mu + np.random.normal(loc=0, scale=step3_sigma, size=Q.shape[0])
        #########################################

        measurements.append((Q, y, 1.0, cl))

    print()
    print("Post-processing with Private-PGM, will take some time...")
    model = engine.estimate(measurements)
    return model.synthetic_data()


def default_params():
    """
    Return default parameters to run this program

    :returns: a dictionary of default parameter settings for each command line argument
    """
    params = {}
    params["dataset"] = "datasets/adult.zip"
    params["domain"] = "datasets/adult-domain.json"
    params["epsilon"] = 1.0
    params["delta"] = 1e-10
    params["targets"] = []
    params["pgm_iters"] = 2500
    params["warm_start"] = True
    params["metric"] = "L2"
    params["threshold"] = 5.0
    params["split_strategy"] = [0.1, 0.1, 0.8]
    params["save"] = "out.csv"

    return params


if __name__ == "__main__":

    description = "A generalization of the Adaptive Grid Mechanism that won 2nd place in the 2020 NIST temporal map challenge"
    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=description, formatter_class=formatter)
    parser.add_argument("--dataset", help="dataset to use")
    parser.add_argument("--domain", help="domain to use")
    parser.add_argument("--epsilon", type=float, help="privacy parameter")
    parser.add_argument("--delta", type=float, help="privacy parameter")
    parser.add_argument(
        "--targets", type=str, nargs="+", help="target columns to preserve"
    )
    parser.add_argument("--pgm_iters", type=int, help="number of iterations")
    parser.add_argument("--warm_start", type=bool, help="warm start PGM")
    parser.add_argument(
        "--metric", choices=["L1", "L2"], help="loss function metric to use"
    )
    parser.add_argument("--threshold", type=float, help="adagrid treshold parameter")
    parser.add_argument(
        "--split_strategy", type=float, nargs="+", help="budget split for 3 steps"
    )
    parser.add_argument("--save", type=str, help="path to save synthetic data")

    parser.set_defaults(**default_params())
    args = parser.parse_args()

    df = pd.read_csv(args.dataset)
    domain = Domain.fromdict(json.load(open(args.domain, "r")))

    # Ensure columns are in correct order for numpy array
    df = df[list(domain.attrs)]

    data = Dataset(df.values, domain)
    mbi_args = {
        "iters": args.pgm_iters,
        "warm_start": args.warm_start,
        "metric": args.metric,
    }
    synth = adagrid(
        data,
        args.epsilon,
        args.delta,
        args.threshold,
        split_strategy=args.split_strategy,
        targets=args.targets,
        **mbi_args
    )

    # Convert back to pandas for saving
    df_synth = pd.DataFrame(synth.to_dict(), columns=synth.domain.attrs)
    df_synth.to_csv(args.save, index=False)
