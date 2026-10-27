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

STANDARD_GRAVITY_M_S2 = 9.80665


def analyze(
    time_s: list[float] | np.ndarray,
    distance_along_ramp_m: list[float] | np.ndarray,
    incline_angle_rad: float,
    *,
    gravity_m_s2: float = STANDARD_GRAVITY_M_S2,
    bootstrap_samples: int = 200,
    seed: int = 0,
) -> tuple[Analysis, FitDiagnostics]:
    if not 0 < incline_angle_rad < np.pi / 2:
        raise ValueError("incline angle must be between zero and pi/2 radians")
    if not np.isfinite(gravity_m_s2) or gravity_m_s2 <= 0:
        raise ValueError("gravity must be finite and positive")
    times, positions = trajectory_arrays(time_s, distance_along_ramp_m)
    elapsed = times - times[0]
    distance = positions - positions[0]
    feature = 0.5 * np.square(elapsed)
    denominator = float(np.dot(feature, feature))
    if denominator <= 0:
        raise ValueError("trajectory duration must be positive")

    def fit_acceleration(values: np.ndarray) -> tuple[float, np.ndarray]:
        relative = values - values[0]
        acceleration = float(np.dot(feature, relative) / denominator)
        return acceleration, acceleration * feature + values[0]

    acceleration, fitted = fit_acceleration(positions)
    mu_a = _friction_from_acceleration(acceleration, incline_angle_rad, gravity_m_s2)
    mu_b = _friction_from_energy(
        elapsed,
        positions,
        incline_angle_rad,
        gravity_m_s2,
    )

    bootstrap_a = residual_bootstrap(
        positions,
        fitted,
        lambda values: _friction_from_acceleration(
            fit_acceleration(values)[0],
            incline_angle_rad,
            gravity_m_s2,
        ),
        samples=bootstrap_samples,
        seed=seed,
    )
    bootstrap_b = residual_bootstrap(
        positions,
        fitted,
        lambda values: _friction_from_energy(
            elapsed,
            values,
            incline_angle_rad,
            gravity_m_s2,
        ),
        samples=bootstrap_samples,
        seed=seed + 1,
    )
    return (
        build_analysis(
            "quadratic acceleration fit",
            mu_a,
            standard_uncertainty(bootstrap_a),
            "energy balance with endpoint velocity",
            mu_b,
            standard_uncertainty(bootstrap_b),
            "1",
        ),
        fit_diagnostics(distance, fitted - positions[0]),
    )


def _friction_from_acceleration(
    acceleration: float,
    incline_angle_rad: float,
    gravity_m_s2: float,
) -> float:
    return float(
        (gravity_m_s2 * np.sin(incline_angle_rad) - acceleration)
        / (gravity_m_s2 * np.cos(incline_angle_rad))
    )


def _friction_from_energy(
    elapsed: np.ndarray,
    positions: np.ndarray,
    incline_angle_rad: float,
    gravity_m_s2: float,
) -> float:
    distance = float(positions[-1] - positions[0])
    duration = float(elapsed[-1])
    if distance <= 0 or duration <= 0:
        raise ValueError("object must move a positive distance down the ramp")
    window_start = max(0, positions.size - 5)
    local_time = elapsed[window_start:] - duration
    local_fit = np.polyfit(local_time, positions[window_start:], 2)
    velocity = float(local_fit[1])
    height_lost = distance * np.sin(incline_angle_rad)
    kinetic_energy_per_mass = 0.5 * velocity**2
    work_per_mass = gravity_m_s2 * np.cos(incline_angle_rad) * distance
    if work_per_mass <= 0:
        raise ValueError("incline geometry does not permit friction estimation")
    return float((gravity_m_s2 * height_lost - kinetic_energy_per_mass) / work_per_mass)
