import concurrent.futures
import logging


class RobustFuture(concurrent.futures.Future):
  """A Future that intercepts and logs exceptions to prevent crashing downstream callers.

  If the wrapped task fails, this future resolves to ``None`` and logs the error,
  avoiding an exception when `.result()` is called.
  """

  def __init__(self, wrapped_future: concurrent.futures.Future):
    super().__init__()
    self._wrapped_future = wrapped_future

    def callback(f):
      try:
        self.set_result(f.result())
      except Exception as e:  # pylint: disable=broad-exception-caught
        logging.info(
            "Background precompilation failed (fallback to JIT at runtime): %s",
            e,
        )
        # Resolve to None without throwing an exception.
        self.set_result(None)

    self._wrapped_future.add_done_callback(callback)

  def cancel(self):
    """Cancel the wrapped future."""
    # We must cancel the inner future, and depending on success, cancel ourselves.
    if self._wrapped_future.cancel():
      return super().cancel()
    return False
