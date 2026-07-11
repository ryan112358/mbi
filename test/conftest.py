"""Global test configuration for mbi."""

import jax


def pytest_configure(config):
  """Disable JAX persistent compilation cache.

  MBI tests produce many small jitted programs (one per unique clique
  structure / domain combination), which makes the persistent cache
  expensive to maintain and slow to populate.
  """
  del config  # Unused.
  jax.config.update("jax_enable_compilation_cache", False)
