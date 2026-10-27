import numpy as np

from . import (
    FitDiagnostics,
    build_analysis,
    fit_diagnostics,
    residual_bootstrap,
    standard_uncertainty,
    trajectory_arrays,
)
from ..schema import Analysis


def analyze(
    time_s: list[float] | np.ndarray,
    magnitude: list[float] | np.ndarray,
    *,
    bootstrap_samples: int = 200,
    seed: int = 0,
) -> tuple[Analysis, FitDiagnostics]:
    times, values = trajectory_arrays(time_s, magnitude, positive_position=True)
    elapsed = times - times[0]
    log_values = np.log(values)

    def fit_rate(log_observations: np.ndarray) -> tuple[float, np.ndarray]:
        slope, intercept = np.polyfit(elapsed, log_observations, 1)
        return float(slope), slope * elapsed + intercept

    rate_a, fitted_log = fit_rate(log_values)
    rate_b = _rate_from_crossing(elapsed, values)
    bootstrap_a = residual_bootstrap(
        log_values,
        fitted_log,
        lambda observations: fit_rate(observations)[0],
        samples=bootstrap_samples,
        seed=seed,
    )
    bootstrap_b = residual_bootstrap(
        log_values,
        fitted_log,
        lambda observations: _rate_from_crossing(elapsed, np.exp(observations)),
        samples=bootstrap_samples,
        seed=seed + 1,
    )
    return (
        build_analysis(
            "linear regression of log magnitude",
            rate_a,
            standard_uncertainty(bootstrap_a),
            "half-life or doubling-time crossing",
            rate_b,
            standard_uncertainty(bootstrap_b),
            "1/s",
        ),
        fit_diagnostics(log_values, fitted_log),
    )


def _rate_from_crossing(elapsed: np.ndarray, values: np.ndarray) -> float:
    initial = float(values[0])
    if values[-1] < initial:
        target = initial / 2
        candidates = np.flatnonzero(values <= target)
        if candidates.size == 0:
            raise ValueError("trajectory does not reach its half-life within the clip")
        index = int(candidates[0])
        direction = -1.0
    elif values[-1] > initial:
        target = initial * 2
        candidates = np.flatnonzero(values >= target)
        if candidates.size == 0:
            raise ValueError("trajectory does not reach its doubling time within the clip")
        index = int(candidates[0])
        direction = 1.0
    else:
        raise ValueError("magnitude must show measurable growth or decay")

    if index == 0:
        crossing_time = float(elapsed[0])
    else:
        x0, x1 = float(values[index - 1]), float(values[index])
        fraction = (target - x0) / (x1 - x0)
        crossing_time = float(
            elapsed[index - 1] + fraction * (elapsed[index] - elapsed[index - 1])
        )
    if crossing_time <= 0:
        raise ValueError("growth/decay crossing time must be positive")
    return direction * float(np.log(2) / crossing_time)
