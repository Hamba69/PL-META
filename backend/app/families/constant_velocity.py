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
    position_m: list[float] | np.ndarray,
    *,
    bootstrap_samples: int = 200,
    seed: int = 0,
) -> tuple[Analysis, FitDiagnostics]:
    times, positions = trajectory_arrays(time_s, position_m)
    elapsed = times - times[0]

    def fit_velocity(values: np.ndarray) -> tuple[float, np.ndarray]:
        slope, intercept = np.polyfit(elapsed, values, 1)
        return float(slope), slope * elapsed + intercept

    velocity_a, fitted = fit_velocity(positions)
    velocity_b = float((positions[-1] - positions[0]) / elapsed[-1])
    bootstrap_a = residual_bootstrap(
        positions,
        fitted,
        lambda values: fit_velocity(values)[0],
        samples=bootstrap_samples,
        seed=seed,
    )
    bootstrap_b = residual_bootstrap(
        positions,
        fitted,
        lambda values: float((values[-1] - values[0]) / elapsed[-1]),
        samples=bootstrap_samples,
        seed=seed + 1,
    )
    return (
        build_analysis(
            "linear regression slope",
            velocity_a,
            standard_uncertainty(bootstrap_a),
            "endpoint displacement over time",
            velocity_b,
            standard_uncertainty(bootstrap_b),
            "m/s",
        ),
        fit_diagnostics(positions, fitted),
    )
