import numpy as np
from scipy.optimize import least_squares
from scipy.signal import find_peaks

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
    position: list[float] | np.ndarray,
    *,
    bootstrap_samples: int = 100,
    seed: int = 0,
) -> tuple[Analysis, FitDiagnostics]:
    times, values = trajectory_arrays(time_s, position, minimum_points=12)
    intervals = np.diff(times)
    median_interval = float(np.median(intervals))
    if np.max(np.abs(intervals - median_interval)) > median_interval * 0.02:
        raise ValueError("sinusoid fitting requires uniformly sampled time values")
    elapsed = times - times[0]
    period_a, fitted = _fit_period(elapsed, values)
    period_b = _peak_period(elapsed, values, period_a)

    bootstrap_a = residual_bootstrap(
        values,
        fitted,
        lambda observations: _fit_period(elapsed, observations)[0],
        samples=bootstrap_samples,
        seed=seed,
    )
    bootstrap_b = residual_bootstrap(
        values,
        fitted,
        lambda observations: _peak_period(elapsed, observations, period_a),
        samples=bootstrap_samples,
        seed=seed + 1,
    )
    return (
        build_analysis(
            "sinusoidal least-squares fit",
            period_a,
            standard_uncertainty(bootstrap_a),
            "mean peak-to-peak timing",
            period_b,
            standard_uncertainty(bootstrap_b),
            "s",
        ),
        fit_diagnostics(values, fitted),
    )


def pendulum_gravity(
    period_analysis: Analysis,
    length_m: float,
    *,
    length_uncertainty_m: float = 0.0,
) -> Analysis:
    if not np.isfinite(length_m) or length_m <= 0:
        raise ValueError("pendulum length must be finite and positive")
    if not np.isfinite(length_uncertainty_m) or length_uncertainty_m < 0:
        raise ValueError("length uncertainty must be finite and non-negative")
    if period_analysis.unit != "s":
        raise ValueError("pendulum gravity requires an oscillation period in seconds")

    def gravity_and_uncertainty(period: float, period_uncertainty: float) -> tuple[float, float]:
        if not np.isfinite(period) or period <= 0:
            raise ValueError("pendulum period must be finite and positive")
        gravity = float(4 * np.pi**2 * length_m / period**2)
        relative_length_uncertainty = length_uncertainty_m / length_m
        relative_period_uncertainty = 2 * period_uncertainty / period
        uncertainty = gravity * float(
            np.hypot(relative_length_uncertainty, relative_period_uncertainty)
        )
        return gravity, uncertainty

    gravity_a, uncertainty_a = gravity_and_uncertainty(
        period_analysis.method_a.value,
        period_analysis.method_a.uncertainty,
    )
    gravity_b, uncertainty_b = gravity_and_uncertainty(
        period_analysis.method_b.value,
        period_analysis.method_b.uncertainty,
    )
    return build_analysis(
        "pendulum relation using fitted period",
        gravity_a,
        uncertainty_a,
        "pendulum relation using peak-timed period",
        gravity_b,
        uncertainty_b,
        "m/s^2",
    )


def _fit_period(times: np.ndarray, values: np.ndarray) -> tuple[float, np.ndarray]:
    duration = float(times[-1])
    if duration <= 0:
        raise ValueError("oscillation duration must be positive")
    centered = values - np.mean(values)
    frequencies = np.fft.rfftfreq(values.size, d=duration / (values.size - 1))
    spectrum = np.abs(np.fft.rfft(centered))
    if spectrum.size < 2:
        raise ValueError("not enough observations to fit an oscillation")
    spectrum[0] = 0
    frequency_guess = float(frequencies[int(np.argmax(spectrum))])
    minimum_frequency = 1 / duration
    maximum_frequency = 0.5 / float(np.median(np.diff(times)))
    frequency_guess = float(np.clip(frequency_guess, minimum_frequency, maximum_frequency))
    omega_guess = 2 * np.pi * frequency_guess
    max_omega = 2 * np.pi * maximum_frequency

    def model(parameters: np.ndarray) -> np.ndarray:
        amplitude_cos, amplitude_sin, offset, omega = parameters
        return (
            amplitude_cos * np.cos(omega * times)
            + amplitude_sin * np.sin(omega * times)
            + offset
        )

    initial = np.array([np.ptp(values) / 2, 0.0, np.mean(values), omega_guess])
    lower = np.array([-np.inf, -np.inf, -np.inf, 2 * np.pi / duration])
    upper = np.array([np.inf, np.inf, np.inf, max_omega])
    result = least_squares(
        lambda parameters: model(parameters) - values,
        initial,
        bounds=(lower, upper),
    )
    if not result.success:
        raise ValueError("sinusoidal fit did not converge")
    period = float(2 * np.pi / result.x[3])
    return period, model(result.x)


def _peak_period(times: np.ndarray, values: np.ndarray, fitted_period: float) -> float:
    sample_interval = float(np.median(np.diff(times)))
    minimum_distance = max(1, int(0.5 * fitted_period / sample_interval))
    peaks, _ = find_peaks(values, distance=minimum_distance)
    if peaks.size < 3:
        raise ValueError("at least three visible peaks are required for independent period timing")
    intervals = np.diff(times[peaks])
    if np.any(intervals <= 0):
        raise ValueError("peak times must be strictly increasing")
    return float(np.mean(intervals))
