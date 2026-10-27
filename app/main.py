from __future__ import annotations

import os
from pathlib import Path
import re
import tempfile

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from .pipeline import (
    BACKEND_ROOT,
    DATA_ROOT,
    CaptureRejected,
    run_free_fall,
)
from .provider_clients import ProviderError
from .schema import PhysicsCard, ScaleReferenceType

PROJECT_ROOT = BACKEND_ROOT.parent
load_dotenv(PROJECT_ROOT / ".env.local")
load_dotenv(PROJECT_ROOT / ".env")

MAX_VIDEO_UPLOAD_BYTES = 100 * 1024 * 1024
MAX_AUDIO_UPLOAD_BYTES = 32 * 1024 * 1024
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov"}

app = FastAPI(
    title="PhoneLab API",
    description="Infrastructure for trusted measurements from phone video.",
    version="0.2.0",
)


@app.get("/healthz", tags=["health"])
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/api/measure/free-fall",
    response_model=PhysicsCard,
    tags=["measurements"],
)
async def measure_free_fall(
    video: UploadFile = File(...),
    scale_reference_type: ScaleReferenceType = Form(ScaleReferenceType.A4_SHEET),
    real_world_length_m: float = Form(0.297, gt=0, le=10),
    object_prompt: str = Form("ball", min_length=1, max_length=80),
    voice_note: UploadFile | None = File(default=None),
) -> PhysicsCard:
    video_suffix = Path(video.filename or "").suffix.lower()
    if video_suffix not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail="Upload an MP4 or MOV video.",
        )
    if voice_note is not None and Path(voice_note.filename or "").suffix.lower() != ".wav":
        raise HTTPException(status_code=415, detail="Voice note must be a WAV file.")

    video_path: Path | None = None
    voice_path: Path | None = None
    try:
        video_path = await _save_upload(video, video_suffix, MAX_VIDEO_UPLOAD_BYTES)
        if voice_note is not None:
            voice_path = await _save_upload(voice_note, ".wav", MAX_AUDIO_UPLOAD_BYTES)
        return await run_in_threadpool(
            run_free_fall,
            video_path,
            scale_reference_type=scale_reference_type,
            real_world_length_m=real_world_length_m,
            object_prompt=object_prompt,
            voice_path=voice_path,
        )
    except CaptureRejected as error:
        raise HTTPException(
            status_code=422,
            detail={
                "message": str(error),
                "capture_quality": error.quality.model_dump(mode="json", by_alias=True),
            },
        ) from error
    except ProviderError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    finally:
        if video_path is not None:
            video_path.unlink(missing_ok=True)
        if voice_path is not None:
            voice_path.unlink(missing_ok=True)
        await video.close()
        if voice_note is not None:
            await voice_note.close()


@app.get("/api/clips/{video_hash}", tags=["measurements"])
def get_retained_clip(video_hash: str) -> FileResponse:
    if not re.fullmatch(r"[0-9a-f]{64}", video_hash):
        raise HTTPException(status_code=404, detail="Clip not found.")
    clips_directory = BACKEND_ROOT / "data" / "clips"
    for extension in ALLOWED_VIDEO_EXTENSIONS:
        path = clips_directory / f"{video_hash}{extension}"
        if path.is_file():
            return FileResponse(path, media_type=_video_media_type(extension))
    raise HTTPException(status_code=404, detail="Clip not found.")


@app.get("/api/runs/{run_id}/trajectory", tags=["measurements"])
def get_trajectory(run_id: str) -> FileResponse:
    if not re.fullmatch(r"[0-9a-f-]{36}", run_id):
        raise HTTPException(status_code=404, detail="Trajectory not found.")
    path = DATA_ROOT / "runs" / run_id / "trajectory.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Trajectory not found.")
    return FileResponse(path, media_type="application/json")


@app.get("/api/evidence/{video_hash}/{filename}", tags=["measurements"])
def get_evidence_frame(video_hash: str, filename: str) -> FileResponse:
    if not re.fullmatch(r"[0-9a-f]{64}", video_hash) or not re.fullmatch(
        r"keyframe_\d{3}_\d{6}\.jpg", filename
    ):
        raise HTTPException(status_code=404, detail="Evidence frame not found.")
    path = DATA_ROOT / "evidence" / video_hash / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Evidence frame not found.")
    return FileResponse(path, media_type="image/jpeg")


async def _save_upload(
    upload: UploadFile,
    suffix: str,
    maximum_bytes: int,
) -> Path:
    descriptor, filename = tempfile.mkstemp(prefix="phonelab-upload-", suffix=suffix)
    path = Path(filename)
    bytes_written = 0
    try:
        with os.fdopen(descriptor, "wb") as destination:
            while chunk := await upload.read(1024 * 1024):
                bytes_written += len(chunk)
                if bytes_written > maximum_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Upload exceeds the {maximum_bytes // (1024 * 1024)} MB limit.",
                    )
                destination.write(chunk)
        if bytes_written == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        return path
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _video_media_type(extension: str) -> str:
    return "video/quicktime" if extension == ".mov" else "video/mp4"
