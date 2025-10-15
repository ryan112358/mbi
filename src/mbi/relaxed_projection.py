"""
This algorithm mainly refers to RAP, RAP++. See their papers in the following link:
    - https://proceedings.mlr.press/v139/aydore21a/aydore21a.pdf
    - https://proceedings.neurips.cc/paper_files/paper/2022/file/7428310c0f97f1c6bb2ef1be99c1ec2a-Paper-Conference.pdf

Their code repo is: 
    - https://github.com/amazon-science/relaxed-adaptive-projection
"""


import numpy as np
import pandas as pd
import itertools

import jax
import jax.numpy as jnp
import optax
from dataclasses import dataclass, field

from typing import Callable, List, Optional
from . import marginal_loss
from .dataset import Dataset
from .domain import Domain
from .marginal_loss import LinearMeasurement
from .clique_vector import CliqueVector
from .factor import Factor


@jax.tree_util.register_dataclass
@dataclass()
class ProjectableData():
    """
    A class storing soft-hot data, supporting projection, vector 
    query, and synthesize data operation.
    
    Args:
        D: a jax.Array, representing a soft-hot form of the data.
        domain: a Domain class.
    """
    D_prime: jax.Array
    domain: Domain = field(metadata=dict(static=True))
    
    @property 
    def feature_indices_map(self):
        map_dict = {}
        s = 0
        for col in self.domain.attrs:
            e = s + self.domain.size(col)
            map_dict[col] = list(range(s, e))
            s = e
        return map_dict

    def datavector(self, count=False, flatten=True):
        """ 
            This will return a vector/matrix of PROBABILITY distribution 
            or frequency COUNT 

            Args:
                count: bool value indicating whether return a frequency, 
                        default to False
                flatten: bool value indicating whether return a flattened 
                        vector, default to True
        """
        splits = [self.D_prime[:, self.feature_indices_map[attr]] for attr in self.domain.attrs]

        letters = "bcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        assert(
            len(letters) >= len(self.domain)
        ), 'High dim domain'

        input_dims = ','.join(f'a{letters[i]}' for i in range(len(splits)))
        output_dims = ''.join(f'{letters[i]}' for i in range(len(splits)))
        einsum_str = f'{input_dims}->a{output_dims}'

        if not count:
            joint = jnp.einsum(einsum_str, *splits).mean(axis=0) # Probability
        else:
            joint = jnp.einsum(einsum_str, *splits).sum(axis=0) # Frequency count
        
        return joint if not flatten else joint.flatten()


    def project(self, cl: tuple):
        cols = list(itertools.chain.from_iterable(self.feature_indices_map[attr] for attr in cl))
        D_proj = self.D_prime[:, cols]
        domain_proj = self.domain.project(cl)
        return ProjectableData(D_prime = D_proj, domain = domain_proj)
    

    def synthetic_data(self, num_samples=1000, seed=0):
        n, _ = self.D_prime.shape
        key = jax.random.PRNGKey(seed)
        key, sk = jax.random.split(key)
        sampled_indices = jax.random.choice(key, n, shape=(num_samples,), replace=True)
        sampled_data = self.D_prime[sampled_indices]

        result = []
        for attr in self.domain.attrs:
            p = sampled_data[:, self.feature_indices_map[attr]]
            key, sk = jax.random.split(key)
            ids = jax.random.categorical(sk, jnp.log(jnp.clip(p, a_min=1e-30, a_max=1.0)), axis=-1)
            result.append(ids)

        sampled_df = pd.DataFrame(np.stack([np.array(r) for r in result], axis=1), columns = list(self.domain.attrs))
        sampled_data = Dataset(sampled_df, self.domain)
        return sampled_data


def relaxed_projection_estimation(
    domain: Domain,
    loss_fn: marginal_loss.MarginalLossFn | list[LinearMeasurement],
    iters: int = 1000,
    optimizer = optax.adam(learning_rate=0.01),
    D_start: Optional[jnp.ndarray] = None,
    known_total: Optional[int] = None,
    seed: int = 0,
    **kwargs
) -> ProjectableData:
    """
    API function for relaxed projection mechanism, return a ProjectableData 
    class, which can further be used to conduct projection and synthesize data.

    Args:
        domain: a Domain class.
        loss_fn: a list of LinearMeasurement, or a callable MarginalLossFn.
        iters: number of iterations in training.
        D_start: initial data on which we start the optimization. If 
                    None, the optimization will start from a randomly 
                    initialized data.
        known_total: number of records in optimizable data (``D_prime.shape[0]``), 
                        which can be regarded as the `batch size`.
        optimizer: an optax optimizer, default to adam with lr=0.01.
        seed: random seed, default to 0.
        kwargs: args that may be used for extra config:
            log: whether return optimization log (start loss and end loss), default 
                 to False (no log)
    
    NOTICE: Here we provide two implementations for loss function. One is based 
        on a parallel computation for multiple marginals, which will be used when 
        loss_fn is a list of LinearMeasurement. The other one is the same as PGM, 
        which receives a MarginalLossFn to calculate loss.
        The reason is that we find that parallel computation is more efficient 
        after running them on some toy examples. 
    """
    if known_total is not None and known_total < 100:
        raise ValueError('You are using a too small known_total, which may influence model capacity.')
    key = jax.random.PRNGKey(seed)

    if isinstance(loss_fn, List):
        # if given a list of LinearMeasurement, define the loss function by it
        if D_start is None:
            known_total = 1000 if known_total is None else known_total
            D_start = _initialize_synthetic_dataset(key, num_generated_points=known_total, data_dimension=np.sum(domain.shape))

        stat_dim = _obtain_dim(measurements = loss_fn)
        statistics = [MarginalStatistics(domain, dim) for dim in stat_dim]
        selected_query_index, measured_ans = _marginal_stat(loss_fn, stat_dim, statistics)

        exact_fn = [None]*len(stat_dim)
        for i in range(len(stat_dim)):
            exact_fn[i] = statistics[i].get_exact_statistics_fn()

        @jax.jit
        def progress_loss(D_prime, query_idx, target, _exact_fn=exact_fn):
            loss = 0.0
            for i in range(len(stat_dim)):
                stats = _exact_fn[i](D_prime, query_idx[i])
                loss += jnp.linalg.norm(target[i] - stats)**2
            return loss
        loss_fn = jax.jit(lambda D: progress_loss(D, selected_query_index, measured_ans))

        # -- initialize optimizer and update function --
        feats_cum = jnp.array([0] + list(domain.shape)).cumsum()
        feats_idx = [list(range(feats_cum[i], feats_cum[i+1])) for i in range(len(feats_cum)-1)]
        opt_init_fn = optimizer.init


        @jax.jit
        def update_fn(D, opt_state):
            value, grads = jax.value_and_grad(loss_fn)(D)
            updates, opt_state = optimizer.update(grads, opt_state, D)
            D = optax.apply_updates(D, updates)
            D = jnp.clip(_sparsemax_project(D, feats_idx), 0, 1)
            return D, opt_state, value


        # -- main optimization step: find a D' that minimized the loss function -- 
        progress_loss_start = loss_fn(D_start)
        D_prime = _optimize_D(
            D_start,
            opt_init_fn,
            loss_fn,
            update_fn,
            iters
        )
        progress_loss_final = loss_fn(D_prime)
        if kwargs.get('log', False):
            print(
                f'marginal fitting results: start loss = {progress_loss_start}, end loss = {progress_loss_final}'
            )
        return ProjectableData(D_prime = D_prime, domain = domain)
    
    else:
        # if given a MarginalLossFn, we transform D_start into CliqueVector and optimize it (same as PGM)
        if D_start is None:
            if known_total is not None:
                D_start = _initialize_synthetic_dataset(key, num_generated_points=known_total, data_dimension=np.sum(domain.shape))
            else:
                raise ValueError('Relaxed Projection reuqires an estimiated sample number if given MarginalLossFn')
        
        @jax.jit
        def loss_fn_wrap(D):
            ProjectableD = ProjectableData(D_prime = D, domain = domain)
            cliques = [tuple(cl) for cl in loss_fn.cliques]
            arrays = {
                cl: Factor(domain.project(cl), ProjectableD.project(cl).datavector(count = True, flatten=False)) 
                for cl in cliques
            }
            return loss_fn(CliqueVector(domain, cliques, arrays))

        feats_cum = jnp.array([0] + list(domain.shape)).cumsum()
        feats_idx = [list(range(feats_cum[i], feats_cum[i+1])) for i in range(len(feats_cum)-1)]
        opt_init_fn = optimizer.init

        @jax.jit
        def update_fn(D, opt_state):
            value, grads = jax.value_and_grad(loss_fn_wrap)(D)
            updates, opt_state = optimizer.update(grads, opt_state, D)
            D = optax.apply_updates(D, updates)
            D = jnp.clip(_sparsemax_project(D, feats_idx), 0, 1)
            return D, opt_state, value

        progress_loss_start = loss_fn_wrap(D_start)
        D_prime = _optimize_D(
            D_start,
            opt_init_fn,
            loss_fn_wrap,
            update_fn,
            iters
        )
        progress_loss_final = loss_fn_wrap(D_prime)
        if kwargs.get('log', False):
            print(
                f'marginal fitting results: start loss = {progress_loss_start}, end loss = {progress_loss_final}'
            )
        return ProjectableData(D_prime = D_prime, domain = domain)





def _obtain_dim(measurements):
    dims = [len(measure.clique) for measure in measurements]
    return list(set(dims))


def _scale_vector(vec):
    temp_vec = jnp.clip(vec, 0, jnp.inf)
    return temp_vec/jnp.sum(temp_vec)


def _marginal_stat(measurements, stat_dim, statistics):
    selected_query_index = [None]*len(stat_dim)
    measured_ans = [None]*len(stat_dim)
    for measure in measurements:
        cl = measure.clique
        marginal = measure.noisy_measurement
        if jnp.sum(marginal) > 1.1:
            # if marginal is a count vector, scale it to probability vector
            marginal = _scale_vector(marginal) 
        try:
            stat_id = stat_dim.index(len(cl))
            if selected_query_index[stat_id] is None:
                selected_query_index[stat_id] = [statistics[stat_id].workload.index(cl)]
                measured_ans[stat_id] = [jnp.hstack([jnp.array(marginal), jnp.zeros((statistics[stat_id].max_size - len(marginal),))])]
            else:
                selected_query_index[stat_id].append(statistics[stat_id].workload.index(cl))
                measured_ans[stat_id].append(
                    np.hstack([jnp.array(marginal), jnp.zeros((statistics[stat_id].max_size - len(marginal),))])
                )
        except:
            raise ValueError(f'Unsupported marginal dimension: {len(cl)}')

    for i in range(len(stat_dim)):
        if selected_query_index[i] is not None:
            selected_query_index[i] = jnp.array(selected_query_index[i])
            measured_ans[i] = jnp.array(measured_ans[i])

    return selected_query_index, measured_ans



def _initialize_synthetic_dataset(
    key: jnp.ndarray, num_generated_points, data_dimension
):
    shape = (num_generated_points, data_dimension)
    random_initial = jax.random.uniform(key=key, shape=shape)
    return random_initial



@jax.jit
def _sparsemax(input):
    """
    forward pass for _sparsemax to transform the input to a probability 
    vector, which sums to 1.
    """
    # sort logits
    z_sorted = jnp.sort(input, axis=1)[:, ::-1]

    # calculate k(z)
    z_cumsum = jnp.cumsum(z_sorted, axis=1)
    k = jnp.arange(1, input.shape[1] + 1)
    z_check = 1 + k * z_sorted > z_cumsum
    k_z = input.shape[1] - jnp.argmax(z_check[:, ::-1], axis=1)

    # calculate tau(logits)
    tau_sum = z_cumsum[jnp.arange(0, input.shape[0]), k_z - 1]
    tau_z = ((tau_sum - 1) / k_z).reshape(-1, 1)

    return jnp.maximum(0, input - tau_z)


@jax.jit
def _sparsemax_project(D, feats_idx):
    return jnp.hstack(
        [_sparsemax(D[:, q]) if len(q) > 1 else D[:, q] for q in feats_idx]
    )


def _optimize_D(
    D_init: jnp.ndarray,
    opt_init_fn: Callable,
    progress_loss_fn: Callable,
    update_fn: Callable,
    iters: int
) -> jnp.ndarray:
    opt_state = opt_init_fn(D_init)
    D_best = D_init
    best_loss = progress_loss_fn(D_best)

    def step(carry, _):
        D, opt_state, D_best, best_loss, prev_loss = carry
        D_new, opt_state_new, loss = update_fn(D, opt_state)

        # track best solution
        improved = loss < best_loss
        best_loss = jnp.where(improved, loss, best_loss)
        D_best = jnp.where(improved, D_new, D_best)
        return (D_new, opt_state_new, D_best, best_loss, loss), None

    init_carry = (D_init, opt_state, D_best, best_loss, best_loss)
    carry_final, _ = jax.lax.scan(step, init_carry, None, length=iters)
    return carry_final[2]

_optimize_D = jax.jit(_optimize_D, static_argnums=(1, 2, 3, 4))




class MarginalStatistics():
    """
    This class is used to manage all marginals and to allow us to perform 
    parallel queries on measurements, improving efficiency.
    """
    def __init__(self, domain, k, max_number_rows=5000):
        self.domain = domain
        self.K = min(k, len(self.domain.attrs))
        self.max_number_rows = max_number_rows

        self.workload = list(itertools.combinations(self.domain.attrs, self.K))
        self.max_size = max([self.domain.size(cl) for cl in self.workload])
        self.feature_indices_map = {}
        s = 0
        for col in self.domain.attrs:
            e = s + self.domain.size(col)
            self.feature_indices_map[col] = list(range(s, e))
            s = e

        queries = []
        for marginal in self.workload:
            positions = [] # columns
            for col in marginal:
                positions.append(self.feature_indices_map[col])
            indices = [] # unqiue values
            for tup in itertools.product(*positions):
                indices.append(tup)
            # K-way Index vector
            idx = []
            for k in range(self.K):
                idx.append(np.array([q[k] for q in indices]))
            idx = np.array(idx)
            queries.append(idx)
        queries = jnp.array(
            [
                jnp.hstack(
                    [jnp.array(query_idx), -jnp.ones((self.K, self.max_size - query_idx.shape[1]))]
                )
                for query_idx in queries
            ]
        ).astype(int)
        extra_zeroes = -jnp.ones((1, self.K, self.max_size), dtype=int)
        self.queries = jnp.vstack([queries, extra_zeroes])


    def compute_stat(self, D_zeroes, query_index):
        sub_marginal_queries = self.queries[query_index]
        D_temp = D_zeroes[:, sub_marginal_queries]
        answers = D_temp.prod(axis=1)
        stats = answers.sum(axis=0)
        return stats

    def get_num_queries(self):
        return self.queries.shape[0] - 1

    def get_workload_index(self, workload):
        try:
            idx = self.workload.index(workload)
        except:
            idx = -1
        return idx
    
    def get_exact_statistics_fn(self):
        compute_stats_vmap = jax.vmap(self.compute_stat, in_axes=(None, 0))

        @jax.jit
        def compute_statistics_fn(D, queries_idx: jnp.ndarray):
            queries_idx = queries_idx.reshape(-1).astype(int)

            D_zeroes = jnp.hstack(
                [D, jnp.zeros((D.shape[0], 1))]
            )  # add a column of zeros
            D_zeroes = jnp.asarray(D_zeroes)
            rows = D_zeroes.shape[0]

            num_D_split = max(2, int(rows / self.max_number_rows + 0.5))
            D_splits = jnp.array_split(D_zeroes, num_D_split)
            stats = jnp.array(
                [compute_stats_vmap(D_sub, queries_idx) for D_sub in D_splits]
            ).sum(axis=0)
            stats_normed = stats / D.shape[0]

            return stats_normed

        return compute_statistics_fn
