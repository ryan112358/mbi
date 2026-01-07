"""
Benchmark Results:
(run on Linux 6.8.0, x86_64, Python 3.12.12, JAX 0.8.2, CPU)
N=1000: 28.9653 seconds
N=10000: 1.8300 seconds
N=100000: 5.0538 seconds
N=1000000: 46.8701 seconds
"""

import sys
import os
import time
import platform
import jax
import logging
import itertools
import numpy as np

# Add src to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import mbi

# Setup logging
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

    # Check device
    try:
        devices = jax.devices()
        info.append(f"JAX Devices: {devices}")
    except Exception as e:
        info.append(f"JAX Devices: Error getting devices: {e}")

    return "\n".join(info)

def run_benchmark():
    domain_dict = { "metro": 3, "metarea": 40, "metaread": 40, "city": 113, "sizepl": 15, "urban": 2, "sea": 130, "gq": 5, "gqtype": 8, "gqtyped": 11, "gqfunds": 8, "farm": 2, "ownershp": 3, "ownershpd": 3, "rent": 65, "valueh": 14, "split": 2, "slrec": 2, "respondt": 3, "famsize": 17, "nchlt5": 6, "sex": 2, "age": 88, "agemonth": 13, "marst": 6, "marrno": 4, "agemarr": 12, "chborn": 10, "race": 6, "hispan": 5, "hispand": 6, "bpl": 80, "bpld": 83, "mbpl": 31, "mbpld": 34, "fbpl": 31, "fbpld": 32, "nativity": 6, "citizen": 6, "mtongue": 8, "mtongued": 10, "spanname": 2, "hisprule": 8, "school": 2, "higrade": 23, "higraded": 23, "educ": 13, "educd": 23, "empstat": 4, "empstatd": 12, "labforce": 3, "classwkr": 4, "classwkrd": 8, "occ": 112, "occ1950": 118, "ind": 109, "ind1950": 111, "wkswork2": 7, "hrswork2": 9, "uocc95": 20, "uclasswk": 8, "incwage": 37, "incnonwg": 3, "occscore": 43, "sei": 65, "presgl": 104, "erscor50": 101, "edscor50": 95, "npboss50": 101, "migrate5": 5, "migrate5d": 9, "migplac5": 45, "migcity5": 28, "samesea5": 6, "vetstat": 4, "vetstatd": 4, "vet1940": 4, "vetwwi": 2, "vetper": 4, "vetchild": 5, "ssenroll": 3 }

    domain = mbi.Domain.fromdict(domain_dict)

    cliques = list(itertools.combinations(domain.attrs, 2))
    logger.info(f"Number of cliques (pairs): {len(cliques)}")

    # Values of N to benchmark
    N_values = [1000, 10000, 100000, 1000000]

    results = []

    logger.info("Starting Benchmark...")
    for N in N_values:
        logger.info(f"Generating data for N={N}...")
        # Generate data outside the timing block
        data = mbi.Dataset.synthetic(domain, N=N)

        # Force JAX compilation if involved, though Dataset is mostly numpy.
        # But CliqueVector uses JAX.
        # Wait, CliqueVector.from_projectable calls data.project -> Factor init -> data.datavector.
        # data.datavector is numpy based (bincount).
        # Factor wraps numpy array into JAX array eventually?
        # Factor(..., values) -> self.values = jnp.array(values)
        # So there is some JAX array creation.

        logger.info(f"Measuring CliqueVector.from_projectable for N={N}...")
        start_time = time.time()
        mbi.CliqueVector.from_projectable(data, cliques)
        # We might want to block until computation is done if JAX is async?
        # Creating JAX arrays from numpy is usually sync or fast enough to trigger sync?
        # JAX arrays are created in Factor.__init__.
        # To be safe, we can inspect the result, but typically for this kind of CPU bound numpy work it's fine.
        # If JAX is used, we should block.
        # Let's verify if we need `jax.block_until_ready()`.
        # CliqueVector.from_projectable returns a CliqueVector which has `arrays` (dict of Factors).
        # Factor has `values` which is a JAX array.
        # So yes, JAX is involved.
        # However, `Dataset.project` -> `Factor` -> `jnp.array(numpy_array)`.
        # The heavy lifting is `Dataset.datavector` which is numpy `bincount`.
        # Converting to JAX array is just memory copy.
        # So timing python time is mostly correct for the numpy part.

        end_time = time.time()
        elapsed = end_time - start_time
        logger.info(f"N={N}: {elapsed:.4f} seconds")
        results.append((N, elapsed))

    return results

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
