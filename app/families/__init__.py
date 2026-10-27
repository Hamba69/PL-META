"""Deterministic, provider-independent experiment analyses."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FitDiagnostics:
    r_squared: float
    normalized_rmse: float


def trajectory_arrays(
    time_s: list[float] | np.ndarray,
    position: list[float] | np.ndarray,
    *,
    minimum_points: int = 3,
    positive_position: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(time_s, dtype=np.float64)
    values = np.asarray(position, dtype=np.float64)
    if times.ndim != 1 or values.ndim != 1 or times.size != values.size:
        raise ValueError("time and position must be one-dimensional arrays of equal length")
    if times.size < minimum_points:
        raise ValueError(f"at least {minimum_points} observations are required")
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(values)):
        raise ValueError("time and position values must be finite")
    if np.any(np.diff(times) <= 0):
        raise ValueError("time values must be strictly increasing")
    if positive_position and np.any(values <= 0):
        raise ValueError("position values must be strictly positive")
    return times, values


def fit_diagnostics(observed: np.ndarray, fitted: np.ndarray) -> FitDiagnostics:
    residuals = observed - fitted
    rmse = float(np.sqrt(np.mean(np.square(residuals))))
    value_range = float(np.ptp(observed))
    normalized_rmse = rmse / value_range if value_range > 0 else (0.0 if rmse == 0 else float("inf"))
    total = float(np.sum(np.square(observed - np.mean(observed))))
    r_squared = 1.0 - float(np.sum(np.square(residuals))) / total if total > 0 else 1.0
    return FitDiagnostics(r_squared=r_squared, normalized_rmse=normalized_rmse)


def residual_bootstrap(
    observed: np.ndarray,
    fitted: np.ndarray,
    estimator,
    *,
    samples: int,
    seed: int,
) -> np.ndarray:
    if samples < 2:
        raise ValueError("at least two bootstrap samples are required")
    residuals = observed - fitted
    residuals = residuals - np.mean(residuals)
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        synthetic = fitted + rng.choice(residuals, size=observed.size, replace=True)
        estimates[index] = estimator(synthetic)
    return estimates


def standard_uncertainty(samples: np.ndarray) -> float:
    if samples.size < 2 or not np.all(np.isfinite(samples)):
        raise ValueError("bootstrap did not produce enough finite estimates")
    return float(np.std(samples, ddof=1))


def build_analysis(
    name_a: str,
    value_a: float,
    uncertainty_a: float,
    name_b: str,
    value_b: float,
    uncertainty_b: float,
    unit: str,
):
    from ..schema import Analysis, AnalysisMethod

    agreement = _agreement_pct(value_a, value_b)
    return Analysis(
        method_a=AnalysisMethod(
            name=name_a,
            value=float(value_a),
            unit=unit,
            uncertainty=float(uncertainty_a),
        ),
        method_b=AnalysisMethod(
            name=name_b,
            value=float(value_b),
            unit=unit,
            uncertainty=float(uncertainty_b),
        ),
        agreement_pct=agreement,
        final_value=float((value_a + value_b) / 2),
        final_uncertainty=float(np.hypot(uncertainty_a, uncertainty_b) / 2),
        unit=unit,
    )


def _agreement_pct(value_a: float, value_b: float) -> float:
    denominator = (abs(value_a) + abs(value_b)) / 2
    if denominator == 0:
        return 0.0
    return float(abs(value_a - value_b) / denominator * 100)
