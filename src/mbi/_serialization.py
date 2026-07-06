"""Save and load JAX pytrees as ``.npz`` checkpoints.

Checkpoint files are tied to the library version that created them.
They are intended for resuming interrupted jobs, not long-term archival.
"""

from __future__ import annotations

import io
import os
import pickle
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


def save(obj: Any, file: str | os.PathLike | io.IOBase) -> None:
    """Save a JAX pytree to ``.npz`` format.

    The pytree leaves (arrays) are stored as named entries and the
    tree structure (including all static / auxiliary data) is stored
    as a pickled ``PyTreeDef``.

    Common pytrees include ``CliqueVector``, ``MarkovRandomField``,
    and ``list[LinearMeasurement]``, but any JAX-compatible pytree
    is accepted.

    Args:
        obj: An arbitrary JAX pytree.
        file: Path string or writable binary file-like object.
    """
    leaves, treedef = jax.tree.flatten(obj)
    arrays = {f"leaf_{i}": np.asarray(leaf) for i, leaf in enumerate(leaves)}
    arrays["_treedef"] = np.frombuffer(pickle.dumps(treedef), dtype=np.uint8)
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    data = buf.getvalue()
    if isinstance(file, (str, os.PathLike)):
        with open(file, "wb") as f:
            f.write(data)
    else:
        file.write(data)


def load(file: str | os.PathLike | io.IOBase) -> Any:
    """Load a JAX pytree from ``.npz`` format.

    Leaf values are returned as ``jax.Array`` so that JIT-compiled
    functions see consistent types.

    Args:
        file: Path string or readable binary file-like object.

    Returns:
        The reconstructed pytree.
    """
    if isinstance(file, (str, os.PathLike)):
        with open(file, "rb") as f:
            raw = f.read()
    else:
        raw = file.read()
    npz = np.load(io.BytesIO(raw), allow_pickle=True)
    treedef = pickle.loads(npz["_treedef"].tobytes())
    leaves = [jnp.asarray(npz[f"leaf_{i}"]) for i in range(treedef.num_leaves)]
    return jax.tree.unflatten(treedef, leaves)
