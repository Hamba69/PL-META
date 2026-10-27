"""The gated first end-to-end pipeline: one free-fall experiment family."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from uuid import uuid4

from .cache import ContentCache, sha256_file
from .capture_coach import inspect_video
from .certificate import evaluate
from .cost_tracker import CostTracker
from .keyframes import extract_keyframes
from .provider_clients import ProviderClients, ProviderError, TrackBox
from .sandbox_runner import SandboxExecutionError, analyze_free_fall_in_sandbox
from .schema import (
    CardInput,
    Cost,
    CostBreakdown,
    ExperimentFamily,
    PhysicsCard,
    ScaleReference,
    ScaleReferenceType,
    Tracking,
)

BACKEND_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = BACKEND_ROOT / "data"
COST_TRACKER = CostTracker(BACKEND_ROOT / "data" / "costs.sqlite3")
RESULT_CACHE = ContentCache(BACKEND_ROOT / "cache" / "results.sqlite3")
PIPELINE_VERSION = "free-fall-v1"


class CaptureRejected(ValueError):
    def __init__(self, quality):
        self.quality = quality
        super().__init__(quality.reason_if_failed or "Capture failed quality checks.")


def run_free_fall(
    video_path: str | Path,
    *,
    scale_reference_type: ScaleReferenceType,
    real_world_length_m: float,
    object_prompt: str,
    voice_path: str | Path | None = None,
) -> PhysicsCard:
    video = Path(video_path)
    if not video.is_file():
        raise FileNotFoundError(f"video does not exist: {video}")
    if not 0.01 <= real_world_length_m <= 10:
        raise ValueError("scale-reference length must be between 0.01 m and 10 m.")
    object_prompt = object_prompt.strip()
    if not object_prompt or len(object_prompt) > 80:
        raise ValueError("object prompt must contain 1 to 80 characters.")
    video_hash = sha256_file(video)
    voice_hash = sha256_file(voice_path) if voice_path is not None else None
    cache_parameters = {
        "scale_reference_type": scale_reference_type.value,
        "real_world_length_m": real_world_length_m,
        "object_prompt": object_prompt,
        "voice_note_hash": voice_hash,
        "pipeline": PIPELINE_VERSION,
    }
    cached = RESULT_CACHE.get(video_hash, "free_fall", PIPELINE_VERSION, cache_parameters)
    if cached is not None:
        return PhysicsCard.model_validate(cached.value)

    run_id = str(uuid4())
    quality = inspect_video(
        video,
        scale_reference_type=scale_reference_type,
    )
    if not quality.passed:
        raise CaptureRejected(quality)

    # Persist the source only after the local quality gate accepts it. The
    # digest-derived name makes retries idempotent and prevents path traversal.
    clips_dir = DATA_ROOT / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    retained_video = clips_dir / f"{video_hash}{video.suffix.lower()}"
    if not retained_video.exists():
        shutil.copyfile(video, retained_video)

    evidence_dir = DATA_ROOT / "evidence" / video_hash
    evidence_paths = extract_keyframes(retained_video, evidence_dir, count=8)
    cost_tracker = COST_TRACKER
    providers = ProviderClients(cost_tracker)
    transcript = ""
    if voice_path is not None:
        transcription = providers.transcribe_audio(voice_path, run_id)
        transcript = transcription.transcript

    target_boxes, target_cost = providers.segment_video(
        retained_video,
        object_prompt,
        run_id,
    )
    reference_prompt = {
        ScaleReferenceType.A4_SHEET: "sheet of paper",
        ScaleReferenceType.BANKNOTE: "banknote",
        ScaleReferenceType.CUSTOM: "scale reference",
    }[scale_reference_type]
    reference_boxes, reference_cost = providers.segment_video(
        retained_video,
        reference_prompt,
        run_id,
        call_kind="track",
    )
    scale_box = _first_box(reference_boxes)
    pixel_length = max(scale_box.width, scale_box.height)
    if pixel_length <= 0:
        raise ProviderError("SAM 3.1 returned an unusable scale-reference box.")
    metres_per_pixel = real_world_length_m / pixel_length

    target = _one_track(target_boxes)
    fps = quality.measured_fps
    times = [box.frame_index / fps for box in target]
    # Image coordinates increase downwards; the free-fall convention is
    # positive upwards. Absolute origin does not affect the fitted acceleration.
    vertical_positions = [-box.center_y * metres_per_pixel for box in target]
    try:
        analysis, diagnostics = analyze_free_fall_in_sandbox(
            times,
            vertical_positions,
        )
    except SandboxExecutionError as error:
        raise ProviderError(str(error)) from error
    certificate = evaluate(
        ExperimentFamily.FREE_FALL,
        analysis,
        diagnostics.normalized_rmse,
    )

    trajectory_path = DATA_ROOT / "runs" / run_id / "trajectory.json"
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    trajectory_path.write_text(
        json.dumps(
            [
                {
                    "frame": box.frame_index,
                    "time_s": time_s,
                    "x_px": box.center_x,
                    "y_px": box.center_y,
                    "x_m": box.center_x * metres_per_pixel,
                    "y_m": position_m,
                }
                for box, time_s, position_m in zip(target, times, vertical_positions)
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    report_prompt = _report_prompt(analysis, certificate.certificate, certificate.reason_codes)
    report = providers.spark_text(
        report_prompt,
        run_id,
        purpose="critique_or_report",
    )
    breakdown = COST_TRACKER.run_breakdown(run_id)
    cost_breakdown = CostBreakdown(
        transcribe=breakdown.get("transcribe", 0.0),
        classify=breakdown.get("classify", 0.0),
        track=breakdown.get("track", 0.0),
        analyze=0.0,
        critique_or_report=breakdown.get("critique_or_report", 0.0),
    )
    total_cost = COST_TRACKER.run_total(run_id)
    card = PhysicsCard(
        card_id=uuid4(),
        created_at=datetime.now(timezone.utc),
        experiment_family=ExperimentFamily.FREE_FALL,
        input=CardInput(
            video_file_hash=video_hash,
            voice_note_transcript=transcript,
            scale_reference=ScaleReference(
                type=scale_reference_type,
                real_world_length_m=real_world_length_m,
                pixel_length=pixel_length,
            ),
        ),
        capture_quality=quality,
        tracking=Tracking(
            model="sam-3.1",
            frames_processed=len(target),
            cost_usd=target_cost + reference_cost,
            trajectory_px_path=f"/api/runs/{run_id}/trajectory",
        ),
        analysis=analysis,
        certificate=certificate.certificate,
        cost=Cost(total_usd=total_cost, breakdown=cost_breakdown),
        report_markdown=report.text,
        evidence_frame_paths=[
            f"/api/evidence/{video_hash}/{path.name}" for path in evidence_paths
        ],
    )
    RESULT_CACHE.put(
        video_hash,
        "free_fall",
        PIPELINE_VERSION,
        cache_parameters,
        card.model_dump(mode="json", by_alias=True),
    )
    return card


def _first_box(boxes: list[TrackBox]) -> TrackBox:
    return min(boxes, key=lambda box: box.frame_index)


def _one_track(boxes: list[TrackBox]) -> list[TrackBox]:
    tracks: dict[int, list[TrackBox]] = {}
    for box in boxes:
        tracks.setdefault(box.object_index, []).append(box)
    if not tracks:
        raise ProviderError("SAM 3.1 returned no usable trajectory.")
    selected = max(tracks.values(), key=len)
    selected.sort(key=lambda box: box.frame_index)
    unique_frames: dict[int, TrackBox] = {}
    for box in selected:
        unique_frames[box.frame_index] = box
    result = list(unique_frames.values())
    if len(result) < 3:
        raise ProviderError("SAM 3.1 tracked fewer than three usable frames.")
    return result


def _report_prompt(analysis, certificate, reason_codes: list[str]) -> str:
    payload = {
        "experiment": "free_fall",
        "method_a": analysis.method_a.model_dump(),
        "method_b": analysis.method_b.model_dump(),
        "final_value": analysis.final_value,
        "final_uncertainty": analysis.final_uncertainty,
        "unit": analysis.unit,
        "certificate": certificate.model_dump(mode="json"),
        "reason_codes": reason_codes,
    }
    return (
        "Write a concise PhoneLab measurement report using only these supplied "
        "results. Never change, invent, or round the measured values. State "
        "whether the certificate passed. If it failed, explain the listed "
        "reason codes as specific re-film guidance without claiming the clip "
        "was fixed. Return Markdown only.\n\n"
        + json.dumps(payload, allow_nan=False)
    )
