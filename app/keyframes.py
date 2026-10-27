from pathlib import Path

import cv2
import numpy as np


def extract_keyframes(
    video_path: str | Path,
    output_directory: str | Path,
    *,
    count: int = 8,
) -> list[Path]:
    if count <= 0:
        raise ValueError("keyframe count must be positive")
    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"video file does not exist: {path}")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"video could not be opened: {path}")
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            raise ValueError("video has no readable frames")
        frame_indices = np.unique(
            np.linspace(0, frame_count - 1, num=min(count, frame_count), dtype=np.int64)
        )
        destination = Path(output_directory)
        destination.mkdir(parents=True, exist_ok=True)
        output_paths: list[Path] = []
        for ordinal, frame_index in enumerate(frame_indices):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            success, frame = capture.read()
            if not success or frame is None:
                raise ValueError(f"could not decode keyframe {int(frame_index)}")
            frame_path = destination / f"keyframe_{ordinal:03d}_{int(frame_index):06d}.jpg"
            if not cv2.imwrite(str(frame_path), frame):
                raise OSError(f"failed to write extracted keyframe: {frame_path}")
            output_paths.append(frame_path)
        return output_paths
    finally:
        capture.release()
