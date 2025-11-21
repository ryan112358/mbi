"""Defines callback mechanisms for monitoring optimization processes.

This module provides a `Callback` class that can be used to track and log
various metrics during iterative algorithms, such as those used in estimating
marginals. It logs loss values and other relevant statistics.
"""
import attr
import jax

from . import marginal_loss
from .clique_vector import CliqueVector
from .factor import Projectable
from .marginal_loss import LinearMeasurement


def _pad(string: str, length: int):
    """Pads a string with spaces on both sides to a target length."""
    if len(string) > length:
        return string[:length]
    left_pad = (length - len(string)) // 2
    right_pad = length - len(string) - left_pad
    return " " * left_pad + string + " " * right_pad


@attr.dataclass
class Callback:
    loss_fns: dict[str, marginal_loss.MarginalLossFn]
    frequency: int = 50
    # Internal state
    _step: int = 0
    _logs: list = attr.field(factory=list)

    def __call__(self, marginals: CliqueVector):
        if self._step == 0:
            header = "|".join([_pad(x, 12) for x in ["step", *self.loss_fns.keys()]])
            print(header)
            print("=" * len(header))
        if self._step % self.frequency == 0:
            row = [self.loss_fns[key](marginals) for key in self.loss_fns]
            self._logs.append([self._step] + row)
            padded_step = str(self._step) + " " * (9 - len(str(self._step)))
            print(padded_step, *[("%.6f" % v)[:6] for v in row], sep="   |   ")
        self._step += 1

    @property
    def summary(self):
        return {
            "columns": ["step"] + list(self.loss_fns.keys()),
            "data": self._logs,
        }


def default(
    measurements: list[LinearMeasurement],
    data: Projectable | None = None,
    frequency: int = 50,
) -> Callback:
    """Creates a default Callback with standard loss functions (L1/L2 Loss/Error, Primal Feas)."""
    loss_fns = {}
    # Measures distance between input marginals and noisy marginals.
    loss_fns["L2 Loss"] = marginal_loss.from_linear_measurements(
        measurements, norm="l2", normalize=True
    )
    loss_fns["L1 Loss"] = marginal_loss.from_linear_measurements(
        measurements,
        norm="l1",
        normalize=True,
    )

    if data is not None:
        # Measures distance between input marginals and true marginals.
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
            ground_truth, norm="l2", normalize=True
        )
        loss_fns["L1 Error"] = marginal_loss.from_linear_measurements(
            ground_truth, norm="l1", normalize=True
        )

    loss_fns = {key: jax.jit(loss_fns[key].__call__) for key in loss_fns}
    loss_fns["Primal Feas"] = jax.jit(marginal_loss.primal_feasibility)

    return Callback(loss_fns, frequency)
