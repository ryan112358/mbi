"""Defines callback mechanisms for monitoring optimization processes.

This module provides a `Callback` class that can be used to track and log
various metrics during iterative algorithms, such as those used in estimating
marginals. It logs loss values and other relevant statistics.
"""

import dataclasses
from typing import Any

from jax.typing import ArrayLike
import jax

from . import estimation
from . import marginal_loss
from .clique_vector import CliqueVector
from .domain import Domain
from .factor import Projectable
from .marginal_loss import LinearMeasurement

_log_fn = print


def set_log_fn(fn):
  """Override the library-wide log function (default: print)."""
  global _log_fn
  _log_fn = fn


def log(*args, **kwargs):
  """Log a message using the configured log function."""
  _log_fn(*args, **kwargs)


def _pad(string: str, length: int):
  """Pads a string with spaces on both sides to a target length."""
  if len(string) > length:
    return string[:length]
  left_pad = (length - len(string)) // 2
  right_pad = length - len(string) - left_pad
  return " " * left_pad + string + " " * right_pad


@jax.jit
def _compute_row(
    loss_fns: tuple[marginal_loss.MarginalLossFn, ...], marginals: CliqueVector
) -> list[ArrayLike]:
  """Evaluates all loss functions on the marginals in one fused, jitted pass."""
  # Passing loss_fns as a traced argument keeps measurement data dynamic (not
  # baked into the compiled program) and lets XLA share projections across them.
  return [loss_fn(marginals) for loss_fn in loss_fns]


def _primal_feasibility_loss(
    marginals: CliqueVector, unused_data: Any
) -> ArrayLike:
  """Adapts primal_feasibility to the (marginals, data) loss_fn signature."""
  return marginal_loss.primal_feasibility(marginals)


@dataclasses.dataclass
class Callback:
  loss_fns: dict[str, marginal_loss.MarginalLossFn]
  _call: int = 0
  _logs: list = dataclasses.field(default_factory=list)

  def __call__(self, marginals: CliqueVector) -> None:
    step = self._call * estimation.CALLBACK_EVERY
    if self._call == 0:
      header = "|".join([_pad(x, 12) for x in ["step", *self.loss_fns.keys()]])
      log(header)
      log("=" * len(header))
    row = _compute_row(tuple(self.loss_fns.values()), marginals)
    self._logs.append([step] + row)
    padded_step = str(step) + " " * (9 - len(str(step)))
    log(padded_step, *[f"{v:.6f}"[:6] for v in row], sep="   |   ")
    self._call += 1

  @property
  def summary(self) -> dict[str, list]:
    return {
        "columns": ["step"] + list(self.loss_fns.keys()),
        "data": self._logs,
    }


def default(
    measurements: list[LinearMeasurement],
    domain: Domain,
    data: Projectable | None = None,
) -> Callback:
  """Creates a default Callback with standard loss functions."""
  loss_fns = {
      "L2 Loss": marginal_loss.from_linear_measurements(
          measurements, domain, norm="l2", normalize=True
      ),
      "L1 Loss": marginal_loss.from_linear_measurements(
          measurements,
          domain,
          norm="l1",
          normalize=True,
      ),
  }

  if data is not None:
    ground_truth = [
        LinearMeasurement(
            M.query(data.project(M.clique)),
            clique=M.clique,
            stddev=1,
            query=M.query,
        )
        for M in measurements
    ]
    loss_fns["L2 Error"] = marginal_loss.from_linear_measurements(
        ground_truth, domain, norm="l2", normalize=True
    )
    loss_fns["L1 Error"] = marginal_loss.from_linear_measurements(
        ground_truth, domain, norm="l1", normalize=True
    )

  loss_fns["Primal Feas"] = marginal_loss.MarginalLossFn(
      cliques=(), loss_fn=_primal_feasibility_loss
  )

  return Callback(loss_fns)
