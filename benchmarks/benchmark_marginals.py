"""
Benchmark Results:
(run on Linux 6.8.0, x86_64, Python 3.12.12, JAX 0.8.2, CPU)

N          | Method                              | Time (s)
-----------------------------------------------------------------
1000       | NumPy Dataset                       | 1.0173
1000       | JIT CliqueVector                    | 0.1659
1000       | JIT JaxDataset.project              | 660.1992
1000       | No JIT                              | 194.3960
-----------------------------------------------------------------
10000      | NumPy Dataset                       | 1.7162
10000      | JIT CliqueVector                    | 0.1620
10000      | JIT JaxDataset.project              | Skipped (>10m)
10000      | No JIT                              | 80.6897
-----------------------------------------------------------------
100000     | NumPy Dataset                       | 5.4920
100000     | JIT CliqueVector                    | 1.6443
100000     | JIT JaxDataset.project              | Skipped (>10m)
100000     | No JIT                              | 77.6430
-----------------------------------------------------------------
1000000    | NumPy Dataset                       | 59.5640
1000000    | JIT CliqueVector                    | OOM
1000000    | JIT JaxDataset.project              | Skipped (>10m)
1000000    | No JIT                              | 152.4771
"""

import sys
import os
import time
import platform
import jax
import logging
import itertools
import numpy as np
import functools
import jax.numpy as jnp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import mbi
from mbi.dataset import JaxDataset

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

def get_system_info():
    info = []
    info.append(f"System: {platform.system()} {platform.release()}")
    info.append(f"Processor: {platform.processor()}")
    info.append(f"Machine: {platform.machine()}")
    info.append(f"Python: {sys.version.split()[0]}")
    info.append(f"JAX: {jax.__version__}")
    try:
        import psutil
        info.append(f"RAM: {psutil.virtual_memory().total / (1024**3):.2f} GB")
    except ImportError:
        info.append("RAM: (psutil not available)")

    try:
        devices = jax.devices()
        info.append(f"JAX Devices: {devices}")
    except Exception as e:
        info.append(f"JAX Devices: Error getting devices: {e}")

    return "\n".join(info)

# --- Variants implementation ---

# Variant 1: JIT on CliqueVector.from_projectable
@functools.partial(jax.jit, static_argnums=(1,))
def jit_cv_from_projectable(data, cliques):
    # cliques is expected to be a tuple of tuples for hashing in static_argnums
    cliques_list = [list(c) for c in cliques]
    return mbi.CliqueVector.from_projectable(data, cliques_list)

# Variant 2: JIT on JaxDataset.project
# We need to cache the jitted functions to avoid recompilation loop within a run
_project_jit_cache = {}
def get_project_jit(cols):
    if cols not in _project_jit_cache:
         # static_argnums=(1,) means 'cols' is static
         _project_jit_cache[cols] = jax.jit(JaxDataset.project, static_argnums=(1,))
    return _project_jit_cache[cols]

class JittedProjectWrapper:
    def __init__(self, data):
        self.data = data
        self.domain = data.domain

    def project(self, cols):
        # We assume cols is a tuple (hashable)
        return get_project_jit(cols)(self.data, cols)

# -----------------------------

def run_benchmark():
    domain_dict = { "metro": 3, "metarea": 40, "metaread": 40, "city": 113, "sizepl": 15, "urban": 2, "sea": 130, "gq": 5, "gqtype": 8, "gqtyped": 11, "gqfunds": 8, "farm": 2, "ownershp": 3, "ownershpd": 3, "rent": 65, "valueh": 14, "split": 2, "slrec": 2, "respondt": 3, "famsize": 17, "nchlt5": 6, "sex": 2, "age": 88, "agemonth": 13, "marst": 6, "marrno": 4, "agemarr": 12, "chborn": 10, "race": 6, "hispan": 5, "hispand": 6, "bpl": 80, "bpld": 83, "mbpl": 31, "mbpld": 34, "fbpl": 31, "fbpld": 32, "nativity": 6, "citizen": 6, "mtongue": 8, "mtongued": 10, "spanname": 2, "hisprule": 8, "school": 2, "higrade": 23, "higraded": 23, "educ": 13, "educd": 23, "empstat": 4, "empstatd": 12, "labforce": 3, "classwkr": 4, "classwkrd": 8, "occ": 112, "occ1950": 118, "ind": 109, "ind1950": 111, "wkswork2": 7, "hrswork2": 9, "uocc95": 20, "uclasswk": 8, "incwage": 37, "incnonwg": 3, "occscore": 43, "sei": 65, "presgl": 104, "erscor50": 101, "edscor50": 95, "npboss50": 101, "migrate5": 5, "migrate5d": 9, "migplac5": 45, "migcity5": 28, "samesea5": 6, "vetstat": 4, "vetstatd": 4, "vet1940": 4, "vetwwi": 2, "vetper": 4, "vetchild": 5, "ssenroll": 3 }

    domain = mbi.Domain.fromdict(domain_dict)

    cliques = list(itertools.combinations(domain.attrs, 2))
    cliques_tuple = tuple(tuple(c) for c in cliques) # For static arg
    logger.info(f"Number of cliques (pairs): {len(cliques)}")

    N_values = [1000, 10000, 100000, 1000000]

    logger.info("Starting Benchmark...")
    print(f"{'N':<10} | {'Method':<35} | {'Time (s)':<15}")
    print("-" * 65)

    for N in N_values:
        # Generate data
        data_np = mbi.Dataset.synthetic(domain, N=N)
        jax_dataset = JaxDataset(
            data={k: jnp.array(v) for k, v in data_np.to_dict().items()},
            domain=domain,
            weights=jnp.array(data_np.weights)
        )

        # Method 0: NumPy Dataset (Baseline)
        start_time = time.time()
        cv = mbi.CliqueVector.from_projectable(data_np, cliques)
        jax.tree.map(lambda x: x.block_until_ready(), cv)
        elapsed_np = time.time() - start_time
        print(f"{N:<10} | {'NumPy Dataset':<35} | {elapsed_np:.4f}")

        # Method 1: JIT CliqueVector.from_projectable
        try:
            # Warmup
            # Note: For N=1M, this might OOM.
            if N < 1000000:
                _ = jit_cv_from_projectable(jax_dataset, cliques_tuple)
                jax.tree.map(lambda x: x.block_until_ready(), _)

                # Measure
                start_time = time.time()
                cv = jit_cv_from_projectable(jax_dataset, cliques_tuple)
                jax.tree.map(lambda x: x.block_until_ready(), cv)
                elapsed_m1 = time.time() - start_time
                print(f"{N:<10} | {'JIT CliqueVector':<35} | {elapsed_m1:.4f}")
            else:
                # Based on experience, N=1M OOMs.
                print(f"{N:<10} | {'JIT CliqueVector':<35} | {'OOM':<15}")
        except Exception as e:
             print(f"{N:<10} | {'JIT CliqueVector':<35} | {'Failed/OOM':<15}")

        # Method 2: JIT JaxDataset.project
        if N > 1000:
            print(f"{N:<10} | {'JIT JaxDataset.project':<35} | {'Skipped (>10m)':<15}")
        else:
            wrapped_data = JittedProjectWrapper(jax_dataset)
            start_time = time.time()
            cv = mbi.CliqueVector.from_projectable(wrapped_data, cliques)
            jax.tree.map(lambda x: x.block_until_ready(), cv)
            elapsed_m2 = time.time() - start_time
            print(f"{N:<10} | {'JIT JaxDataset.project':<35} | {elapsed_m2:.4f}")

        # Method 3: No JIT
        start_time = time.time()
        cv = mbi.CliqueVector.from_projectable(jax_dataset, cliques)
        jax.tree.map(lambda x: x.block_until_ready(), cv)
        elapsed_m3 = time.time() - start_time
        print(f"{N:<10} | {'No JIT':<35} | {elapsed_m3:.4f}")

if __name__ == "__main__":
    print("System Information:")
    print(get_system_info())
    print("-" * 20)
    try:
        run_benchmark()
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
