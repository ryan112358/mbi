"""Defines callback mechanisms for monitoring optimization processes.

This module provides a `Callback` class that can be used to track and log
various metrics during iterative algorithms, such as those used in estimating
marginals. It logs loss values and other relevant statistics.
"""

import dataclasses
import jax

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


@dataclasses.dataclass
class Callback:
    loss_fns: dict[str, marginal_loss.MarginalLossFn]
    _step: int = 0
    _logs: list = dataclasses.field(default_factory=list)

    def __call__(self, marginals: CliqueVector) -> None:
        if self._step == 0:
            header = "|".join(
                [_pad(x, 12) for x in ["step", *self.loss_fns.keys()]]
            )
            log(header)
            log("=" * len(header))
        row = [self.loss_fns[key](marginals) for key in self.loss_fns]
        self._logs.append([self._step] + row)
        padded_step = str(self._step) + " " * (9 - len(str(self._step)))
        log(padded_step, *[f"{v:.6f}"[:6] for v in row], sep="   |   ")
        self._step += 1

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

    loss_fns = {key: jax.jit(val.__call__) for key, val in loss_fns.items()}
    loss_fns["Primal Feas"] = jax.jit(marginal_loss.primal_feasibility)

    return Callback(loss_fns)
