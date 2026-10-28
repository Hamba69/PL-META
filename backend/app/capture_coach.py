from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .schema import CaptureQuality, ScaleReferenceType


@dataclass(frozen=True)
class CaptureCoachConfig:
    minimum_resolution: tuple[int, int] = (640, 480)
    minimum_sharpness_variance: float = 80.0
    maximum_frames: int = 15_000
    sharpness_sample_count: int = 30
    claimed_fps_tolerance_fraction: float = 0.05
    scale_reference_minimum_area_fraction: float = 0.005

    def __post_init__(self) -> None:
        if min(self.minimum_resolution) <= 0:
            raise ValueError("minimum resolution dimensions must be positive")
        if self.minimum_sharpness_variance < 0:
            raise ValueError("sharpness threshold must be non-negative")
        if self.maximum_frames <= 0 or self.sharpness_sample_count <= 0:
            raise ValueError("frame limits and sample count must be positive")
        if not 0 <= self.claimed_fps_tolerance_fraction < 1:
            raise ValueError("FPS tolerance must be in [0, 1)")
        if not 0 < self.scale_reference_minimum_area_fraction < 1:
            raise ValueError("scale reference area fraction must be in (0, 1)")


def inspect_video(
    video_path: str | Path,
    *,
    claimed_fps: float | None = None,
    scale_reference_type: ScaleReferenceType = ScaleReferenceType.A4_SHEET,
    config: CaptureCoachConfig = CaptureCoachConfig(),
) -> CaptureQuality:
    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"video file does not exist: {path}")
    if claimed_fps is not None and (not np.isfinite(claimed_fps) or claimed_fps <= 0):
        raise ValueError("claimed FPS must be finite and positive")

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"video could not be opened: {path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not np.isfinite(fps) or fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
            raise ValueError("video has invalid FPS, frame-count, or resolution metadata")
        sample_indices = np.unique(
            np.linspace(
                0,
                frame_count - 1,
                num=min(frame_count, config.sharpness_sample_count),
                dtype=np.int64,
            )
        )
        sharpness_scores: list[float] = []
        scale_reference_detected = False
        for frame_index in sample_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            read_success, frame = capture.read()
            if not read_success or frame is None:
                raise ValueError(f"could not decode sampled frame {int(frame_index)}")
            grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            laplacian = cv2.Laplacian(grayscale, cv2.CV_64F)
            sharpness_scores.append(float(laplacian.var()))
            scale_reference_detected = scale_reference_detected or detect_scale_reference(
                frame,
                scale_reference_type,
                config.scale_reference_minimum_area_fraction,
            )

        sharpness_score = float(np.median(sharpness_scores))
        reasons: list[str] = []
        if frame_count > config.maximum_frames:
            reasons.append(
                f"Clip exceeds the {config.maximum_frames}-frame tracking limit."
            )
        if sharpness_score < config.minimum_sharpness_variance:
            reasons.append("Footage is blurry; improve focus, light, or shutter speed.")
        if claimed_fps is not None and _fps_mismatch(
            fps, claimed_fps, config.claimed_fps_tolerance_fraction
        ):
            reasons.append("Container FPS does not match the claimed frame rate.")
        if width < config.minimum_resolution[0] or height < config.minimum_resolution[1]:
            reasons.append("Resolution is below the tracking minimum.")
        if not scale_reference_detected:
            reasons.append(
                "No scale reference matching the selected reference type was detected."
            )
        return CaptureQuality(
            sharpness_score=sharpness_score,
            measured_fps=fps,
            resolution=(width, height),
            scale_ref_detected=scale_reference_detected,
            passed=not reasons,
            reason_if_failed=" ".join(reasons) if reasons else None,
        )
    finally:
        capture.release()


def detect_scale_reference(
    frame: np.ndarray,
    reference_type: ScaleReferenceType,
    minimum_area_fraction: float = 0.005,
) -> bool:
    if not isinstance(reference_type, ScaleReferenceType):
        raise ValueError("scale reference type is not supported")
    if not 0 < minimum_area_fraction < 1:
        raise ValueError("minimum area fraction must be in (0, 1)")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("scale detection expects a three-channel BGR frame")
    height, width = frame.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("frame dimensions must be positive")
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(grayscale, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    frame_area = float(width * height)
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area / frame_area < minimum_area_fraction:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 0:
            continue
        polygon = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(polygon) != 4 or not cv2.isContourConvex(polygon):
            continue
        rectangle = cv2.minAreaRect(polygon)
        box_width, box_height = rectangle[1]
        if box_width <= 0 or box_height <= 0:
            continue
        ratio = max(box_width, box_height) / min(box_width, box_height)
        if _matches_reference_aspect_ratio(ratio, reference_type):
            return True
    return False


def _matches_reference_aspect_ratio(
    ratio: float,
    reference_type: ScaleReferenceType,
) -> bool:
    if reference_type is ScaleReferenceType.A4_SHEET:
        return 1.30 <= ratio <= 1.53
    if reference_type is ScaleReferenceType.BANKNOTE:
        return 1.20 <= ratio <= 1.80
    return 1.10 <= ratio <= 2.50


def _fps_mismatch(actual: float, claimed: float, tolerance_fraction: float) -> bool:
    return abs(actual - claimed) / claimed > tolerance_fraction
