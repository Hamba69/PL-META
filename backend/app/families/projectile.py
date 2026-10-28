import numpy as np

from . import FitDiagnostics
from .constant_velocity import analyze as analyze_horizontal_motion
from .free_fall import analyze as analyze_vertical_motion
from ..schema import Analysis


def analyze(
    time_s: list[float] | np.ndarray,
    horizontal_position_m: list[float] | np.ndarray,
    vertical_position_m: list[float] | np.ndarray,
    *,
    bootstrap_samples: int = 200,
    seed: int = 0,
) -> tuple[Analysis, FitDiagnostics, Analysis, FitDiagnostics]:
    """Analyze vertical gravity and horizontal constant velocity independently."""
    gravity, vertical_diagnostics = analyze_vertical_motion(
        time_s,
        vertical_position_m,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    horizontal_velocity, horizontal_diagnostics = analyze_horizontal_motion(
        time_s,
        horizontal_position_m,
        bootstrap_samples=bootstrap_samples,
        seed=seed + 2,
    )
    return gravity, vertical_diagnostics, horizontal_velocity, horizontal_diagnostics
