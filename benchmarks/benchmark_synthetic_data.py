"""
Benchmark Results (run on Linux 6.8.0, x86_64, Python 3.12.12, JAX 0.8.2, CPU):

JIT Compilation Time (N=1): 11.4774 seconds

Generation Times:
N=1000: 13.9146 seconds
N=10000: 2.0952 seconds
N=100000: 8.6045 seconds
N=1000000: 86.8311 seconds
"""

import sys
import os
import time
import platform
import jax
import logging

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

    cliques = [('spanname', 'famsize'), ('urban', 'vetchild'), ('urban', 'nativity'), ('urban', 'vetstat'), ('gqtyped', 'erscor50'), ('gqtyped', 'citizen'), ('ind1950', 'occ1950'), ('ind1950', 'marst'), ('ind1950', 'respondt'), ('vetstatd', 'nchlt5'), ('vetstatd', 'famsize'), ('migplac5', 'hrswork2'), ('migplac5', 'sizepl'), ('migplac5', 'ssenroll'), ('migplac5', 'edscor50'), ('hisprule', 'presgl'), ('hisprule', 'migrate5d'), ('hisprule', 'empstat'), ('classwkrd', 'agemonth'), ('incnonwg', 'empstatd'), ('sei', 'mbpl'), ('sei', 'gqtype'), ('sei', 'vet1940'), ('sex', 'nativity'), ('sex', 'gqfunds'), ('mtongue', 'agemarr'), ('sea', 'migrate5d'), ('city', 'empstat'), ('city', 'age'), ('classwkr', 'edscor50'), ('occ', 'mbpl'), ('occ1950', 'famsize'), ('higraded', 'school'), ('hispand', 'slrec'), ('hispand', 'uclasswk'), ('hispand', 'school'), ('hispand', 'wkswork2'), ('fbpl', 'agemarr'), ('fbpl', 'marrno'), ('fbpl', 'ssenroll'), ('hispan', 'npboss50'), ('bpl', 'npboss50'), ('empstat', 'valueh'), ('labforce', 'chborn'), ('labforce', 'ownershpd'), ('nchlt5', 'agemarr'), ('race', 'gqtype'), ('valueh', 'sizepl'), ('mbpld', 'presgl'), ('fbpld', 'bpld'), ('fbpld', 'occscore'), ('fbpld', 'metarea'), ('empstatd', 'agemarr'), ('metarea', 'rent'), ('educ', 'ind'), ('educ', 'farm'), ('npboss50', 'presgl'), ('npboss50', 'educd'), ('npboss50', 'vetper'), ('npboss50', 'incwage'), ('age', 'mtongued'), ('nativity', 'incwage'), ('split', 'respondt'), ('gq', 'agemarr'), ('uclasswk', 'metro'), ('uclasswk', 'gqtype'), ('incwage', 'bpld'), ('incwage', 'migrate5'), ('migcity5', 'metaread'), ('sizepl', 'agemonth'), ('school', 'samesea5'), ('school', 'chborn'), ('respondt', 'uocc95'), ('mbpl', 'farm'), ('mbpl', 'metaread'), ('mbpl', 'vetwwi'), ('metro', 'erscor50'), ('uocc95', 'higrade'), ('ownershp', 'metaread'), ('chborn', 'hrswork2')]

    domain = mbi.Domain.fromdict(domain_dict)

    logger.info("Initializing Potentials...")
    potentials = mbi.CliqueVector.zeros(domain, cliques)

    logger.info("Calculating Marginals...")
    marginals = mbi.marginal_oracles.message_passing_stable(potentials)

    model = mbi.MarkovRandomField(potentials=potentials, marginals=marginals, total=1)

    # Warmup / Compilation time (N=1)
    logger.info("Measuring JIT Compilation (N=1)...")
    start_time = time.time()
    model.synthetic_data(rows=1)
    end_time = time.time()
    jit_time = end_time - start_time
    logger.info(f"JIT Compilation Time: {jit_time:.4f} seconds")

    # Values of N to benchmark
    N_values = [1000, 10000, 100000, 1000000]

    results = []

    logger.info("Starting Benchmark...")
    for N in N_values:
        start_time = time.time()
        model.synthetic_data(rows=N)
        end_time = time.time()
        elapsed = end_time - start_time
        logger.info(f"N={N}: {elapsed:.4f} seconds")
        results.append((N, elapsed))

    return jit_time, results

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
