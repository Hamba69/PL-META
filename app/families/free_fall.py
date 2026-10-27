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
    vertical_position_m: list[float] | np.ndarray,
    *,
    bootstrap_samples: int = 200,
    seed: int = 0,
) -> tuple[Analysis, FitDiagnostics]:
    """Estimate downward gravitational acceleration; vertical position is positive upward."""
    times, positions = trajectory_arrays(time_s, vertical_position_m)
    elapsed = times - times[0]

    def fit_acceleration(values: np.ndarray) -> tuple[float, np.ndarray]:
        coefficients = np.polyfit(elapsed, values, 2)
        fitted = np.polyval(coefficients, elapsed)
        return float(-2 * coefficients[0]), fitted

    acceleration_a, fitted = fit_acceleration(positions)
    duration = float(elapsed[-1])
    acceleration_b = float(2 * (positions[0] - positions[-1]) / duration**2)

    bootstrap_a = residual_bootstrap(
        positions,
        fitted,
        lambda values: fit_acceleration(values)[0],
        samples=bootstrap_samples,
        seed=seed,
    )
    bootstrap_b = residual_bootstrap(
        positions,
        fitted,
        lambda values: float(2 * (values[0] - values[-1]) / duration**2),
        samples=bootstrap_samples,
        seed=seed + 1,
    )
    return (
        build_analysis(
            "quadratic fit",
            acceleration_a,
            standard_uncertainty(bootstrap_a),
            "endpoint kinematic",
            acceleration_b,
            standard_uncertainty(bootstrap_b),
            "m/s^2",
        ),
        fit_diagnostics(positions, fitted),
    )
