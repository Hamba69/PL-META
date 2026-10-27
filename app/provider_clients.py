"""Small, usage-aware clients for the provider endpoints used by PhoneLab."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

import requests

from .cost_tracker import CostTracker

META_BASE_URL = "https://api.meta.ai/v1"
BASETEN_URL = "https://inference.baseten.co/v1"
SPARK_STANDARD_INPUT_PER_MILLION = 1.25
SPARK_STANDARD_OUTPUT_PER_MILLION = 4.25
VOICE_USD_PER_HOUR = 0.18
SAM_IMAGE_USD_PER_THOUSAND = 2.50
SAM_VIDEO_USD_PER_THOUSAND_FRAMES = 0.20
SAM_VIDEO_MINIMUM_BILLABLE_FRAMES = 50


class ProviderError(RuntimeError):
    """A provider failed or returned a response this client cannot use."""


@dataclass(frozen=True)
class TextResult:
    text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    response_status: str


@dataclass(frozen=True)
class TranscriptResult:
    transcript: str
    duration_ms: int
    cost_usd: float


@dataclass(frozen=True)
class TrackBox:
    frame_index: int
    object_index: int
    x1: float
    y1: float
    x2: float
    y2: float
    image_width: int
    image_height: int

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1


class ProviderClients:
    def __init__(
        self,
        cost_tracker: CostTracker,
        *,
        timeout_seconds: tuple[float, float] = (20, 180),
    ) -> None:
        self.cost_tracker = cost_tracker
        self.timeout_seconds = timeout_seconds
        self.meta_key = _required_environment("META_API_KEY")
        self.baseten_key = _optional_environment("BASETEN_API_KEY")

    def transcribe_audio(self, audio_path: str | Path, run_id: str) -> TranscriptResult:
        path = Path(audio_path)
        _validate_pcm_wav(path)
        request_settings = {
            "mode": "PUSH_TO_TALK",
            "model": "muse-voice-transcribe-1.0",
            "audioEncoding": "WAV",
        }
        try:
            with path.open("rb") as audio:
                response = requests.post(
                    f"{META_BASE_URL}/asr/transcribe",
                    headers=self._meta_headers(),
                    files={
                        "request": (None, json.dumps(request_settings), "application/json"),
                        "audio": (path.name, audio, "audio/wav"),
                    },
                    timeout=self.timeout_seconds,
                )
        except requests.RequestException as error:
            raise ProviderError("Meta Voice Transcribe request failed.") from error
        if response.status_code != 200:
            raise _http_error("Meta Voice Transcribe", response)
        try:
            body = response.json()
            transcript = body["transcript"]
            duration_ms = int(body["audioDurationMs"])
        except (ValueError, KeyError, TypeError) as error:
            raise ProviderError("Meta Voice Transcribe returned an invalid response.") from error
        if not isinstance(transcript, str) or not transcript.strip() or duration_ms <= 0:
            raise ProviderError("Meta Voice Transcribe returned no usable transcript.")
        billable_seconds = duration_ms // 1000
        cost = billable_seconds * VOICE_USD_PER_HOUR / 3600
        self.cost_tracker.record_call(
            run_id=run_id,
            model="muse-voice-transcribe-1.0",
            call_kind="transcribe",
            unit_name="audio seconds",
            units=billable_seconds,
            cost_usd=cost,
        )
        return TranscriptResult(transcript, duration_ms, cost)

    def spark_text(
        self,
        prompt: str,
        run_id: str,
        *,
        model: str = "muse-spark-1.3",
        keyframe_paths: list[str | Path] | None = None,
        purpose: str = "classify",
    ) -> TextResult:
        content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
        for path_value in keyframe_paths or []:
            path = Path(path_value)
            mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            content.append(
                {"type": "input_image", "image_url": f"data:{mime};base64,{encoded}"}
            )
        payload = {
            "model": model,
            "input": [{"type": "message", "role": "user", "content": content}],
            "max_output_tokens": 1024,
            "reasoning": {"effort": "low"},
            "store": False,
        }
        response = self._meta_post("/responses", payload)
        result = _parse_responses_text(response)
        cost = _spark_cost(result.input_tokens, result.output_tokens)
        self.cost_tracker.record_call(
            run_id=run_id,
            model=model,
            call_kind=purpose,
            unit_name="tokens",
            units=result.input_tokens + result.output_tokens,
            cost_usd=cost,
        )
        if result.response_status != "completed" or not result.text.strip():
            raise ProviderError("Muse Spark returned an incomplete or empty response.")
        return TextResult(
            result.text,
            result.input_tokens,
            result.output_tokens,
            cost,
            result.response_status,
        )

    def baseten_spark_open(self, prompt: str, run_id: str) -> TextResult:
        if not self.baseten_key:
            raise ProviderError("BASETEN_API_KEY is not configured.")
        payload = {
            "model": "meta-models/muse-spark-open",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 512,
        }
        try:
            response = requests.post(
                f"{BASETEN_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.baseten_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.timeout_seconds,
            )
        except requests.RequestException as error:
            raise ProviderError("Baseten Muse Spark Open request failed.") from error
        if response.status_code != 200:
            raise _http_error("Baseten Muse Spark Open", response)
        try:
            body = response.json()
            usage = body["usage"]
            message = body["choices"][0]["message"]
            text = message["content"]
            input_tokens = int(usage["prompt_tokens"])
            output_tokens = int(usage["completion_tokens"])
        except (ValueError, KeyError, TypeError, IndexError) as error:
            raise ProviderError("Baseten returned an invalid completion response.") from error
        if not isinstance(text, str) or not text.strip():
            raise ProviderError("Baseten Muse Spark Open returned no user-facing text.")
        cost = _spark_cost(input_tokens, output_tokens)
        self.cost_tracker.record_call(
            run_id=run_id,
            model="meta-models/muse-spark-open",
            call_kind="novel_reasoning",
            unit_name="tokens",
            units=input_tokens + output_tokens,
            cost_usd=cost,
        )
        return TextResult(text, input_tokens, output_tokens, cost, "completed")

    def segment_video(
        self,
        video_path: str | Path,
        prompt: str,
        run_id: str,
        *,
        call_kind: str = "track",
    ) -> tuple[list[TrackBox], float]:
        path = Path(video_path)
        file_id = self._upload_video(path)
        try:
            payload = {
                "model": "sam-3.1",
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {"type": "input_video", "file_id": file_id},
                        ],
                    }
                ],
                "stream": True,
            }
            response = self._meta_post("/responses", payload, stream=True)
            output_text = _read_responses_stream(response)
        finally:
            self._delete_uploaded_file(file_id)
        boxes = _parse_sam_boxes(output_text)
        billable_frames = max(
            len(set(re.findall(r"<(\d+)f>", output_text))),
            SAM_VIDEO_MINIMUM_BILLABLE_FRAMES,
        )
        cost = billable_frames * SAM_VIDEO_USD_PER_THOUSAND_FRAMES / 1000
        self.cost_tracker.record_call(
            run_id=run_id,
            model="sam-3.1",
            call_kind=call_kind,
            unit_name="billable video frames",
            units=billable_frames,
            cost_usd=cost,
        )
        if not boxes:
            raise ProviderError("SAM 3.1 returned no tracked object boxes.")
        object_index = _most_frequent_object(boxes)
        selected = [box for box in boxes if box.object_index == object_index]
        return selected, cost

    def _meta_post(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        stream: bool = False,
    ) -> requests.Response:
        headers = self._meta_headers()
        if stream:
            headers["Accept"] = "text/event-stream"
        try:
            response = requests.post(
                f"{META_BASE_URL}{path}",
                headers=headers,
                json=payload,
                stream=stream,
                timeout=self.timeout_seconds,
            )
        except requests.RequestException as error:
            raise ProviderError("Meta Model API request failed.") from error
        if response.status_code != 200:
            raise _http_error("Meta Model API", response)
        return response

    def _upload_video(self, path: Path) -> str:
        if not path.is_file():
            raise FileNotFoundError(f"video does not exist: {path}")
        try:
            media_type = "video/quicktime" if path.suffix.lower() == ".mov" else "video/mp4"
            with path.open("rb") as video:
                response = requests.post(
                    f"{META_BASE_URL}/files",
                    headers=self._meta_headers(),
                    data={"purpose": "user_data"},
                    files={"file": (path.name, video, media_type)},
                    timeout=self.timeout_seconds,
                )
        except requests.RequestException as error:
            raise ProviderError("Meta video upload failed.") from error
        if response.status_code != 200:
            raise _http_error("Meta video upload", response)
        try:
            file_id = response.json()["id"]
        except (ValueError, KeyError, TypeError) as error:
            raise ProviderError("Meta video upload returned no file ID.") from error
        if not isinstance(file_id, str) or not file_id:
            raise ProviderError("Meta video upload returned an invalid file ID.")
        return file_id

    def _delete_uploaded_file(self, file_id: str) -> None:
        try:
            response = requests.delete(
                f"{META_BASE_URL}/files/{file_id}",
                headers=self._meta_headers(),
                timeout=self.timeout_seconds,
            )
        except requests.RequestException:
            return
        if response.status_code not in (200, 204):
            return

    def _meta_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.meta_key}"}


def _parse_responses_text(response: requests.Response) -> TextResult:
    try:
        body = response.json()
        text_parts = [
            part["text"]
            for item in body.get("output", [])
            if isinstance(item, dict)
            for part in item.get("content", [])
            if isinstance(part, dict) and part.get("type") == "output_text"
        ]
        usage = body.get("usage") or {}
        input_tokens = int(usage.get("input_tokens", 0))
        output_tokens = int(usage.get("output_tokens", 0))
        status = str(body.get("status", "unknown"))
    except (ValueError, KeyError, TypeError) as error:
        raise ProviderError("Meta Responses API returned an invalid response.") from error
    return TextResult(
        "\n".join(text_parts),
        input_tokens,
        output_tokens,
        _spark_cost(input_tokens, output_tokens),
        status,
    )


def _read_responses_stream(response: requests.Response) -> str:
    chunks: list[str] = []
    completed = False
    try:
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            raw_event = line[6:]
            if raw_event == "[DONE]":
                break
            event = json.loads(raw_event)
            if event.get("type") == "response.output_text.delta":
                chunks.append(event.get("delta", ""))
            elif event.get("type") == "response.completed":
                completed = True
            elif event.get("type") == "response.failed":
                raise ProviderError("SAM 3.1 stream reported a failed response.")
    except (requests.RequestException, ValueError) as error:
        raise ProviderError("Could not read the SAM 3.1 response stream.") from error
    if not completed:
        raise ProviderError("SAM 3.1 stream ended before response.completed.")
    return "".join(chunks)


def _parse_sam_boxes(output_text: str) -> list[TrackBox]:
    frame_chunks = re.split(r"(?=<\d+f>)", output_text)
    boxes: list[TrackBox] = []
    pattern = re.compile(
        r"(?P<object>\d+)<\|box;"
        r"x1=(?P<x1>\d+(?:\.\d+)?);y1=(?P<y1>\d+(?:\.\d+)?)"
        r";x2=(?P<x2>\d+(?:\.\d+)?);y2=(?P<y2>\d+(?:\.\d+)?)"
        r";w=(?P<width>\d+);h=(?P<height>\d+)\|>"
    )
    for chunk in frame_chunks:
        frame_match = re.match(r"<(?P<frame>\d+)f>", chunk)
        if frame_match is None:
            continue
        frame_index = int(frame_match.group("frame"))
        for match in pattern.finditer(chunk):
            boxes.append(
                TrackBox(
                    frame_index=frame_index,
                    object_index=int(match.group("object")),
                    x1=float(match.group("x1")),
                    y1=float(match.group("y1")),
                    x2=float(match.group("x2")),
                    y2=float(match.group("y2")),
                    image_width=int(match.group("width")),
                    image_height=int(match.group("height")),
                )
            )
    return boxes


def _most_frequent_object(boxes: list[TrackBox]) -> int:
    counts: dict[int, int] = {}
    for box in boxes:
        counts[box.object_index] = counts.get(box.object_index, 0) + 1
    return max(counts, key=counts.get)


def _validate_pcm_wav(path: Path) -> None:
    import wave

    if not path.is_file():
        raise FileNotFoundError(f"audio does not exist: {path}")
    try:
        with wave.open(str(path), "rb") as audio:
            valid = (
                audio.getnchannels() == 1
                and audio.getsampwidth() == 2
                and audio.getframerate() in (16000, 24000)
                and audio.getnframes() > 0
            )
    except (wave.Error, OSError) as error:
        raise ValueError("voice note must be a supported PCM WAV file.") from error
    if not valid:
        raise ValueError(
            "voice note must be mono 16-bit PCM WAV at 16 kHz or 24 kHz."
        )


def _required_environment(name: str) -> str:
    value = _optional_environment(name)
    if not value:
        raise ProviderError(f"{name} is not configured.")
    return value


def _optional_environment(name: str) -> str | None:
    from os import environ

    value = environ.get(name, "").strip()
    return value or None


def _spark_cost(input_tokens: int, output_tokens: int) -> float:
    return (
        input_tokens * SPARK_STANDARD_INPUT_PER_MILLION
        + output_tokens * SPARK_STANDARD_OUTPUT_PER_MILLION
    ) / 1_000_000


def _http_error(provider: str, response: requests.Response) -> ProviderError:
    try:
        body = response.json()
        error = body.get("error", {}) if isinstance(body, dict) else {}
        description = str(error.get("message", ""))[:160]
    except (ValueError, TypeError):
        description = ""
    suffix = f": {description}" if description else ""
    return ProviderError(f"{provider} returned HTTP {response.status_code}{suffix}")
