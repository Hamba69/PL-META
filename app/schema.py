from datetime import datetime
from enum import StrEnum
from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints


class ExperimentFamily(StrEnum):
    FREE_FALL = "free_fall"
    PROJECTILE = "projectile"
    OSCILLATION = "oscillation"
    EXPONENTIAL = "exponential"
    CONSTANT_VELOCITY = "constant_velocity"
    RAMP_FRICTION = "ramp_friction"
    NOVEL = "novel"


class ScaleReferenceType(StrEnum):
    A4_SHEET = "a4_sheet"
    BANKNOTE = "banknote"
    CUSTOM = "custom"


class CheckResult(StrEnum):
    PASS = "pass"
    FAIL = "fail"


class CertificateStatus(StrEnum):
    CERTIFIED = "certified"
    UNCERTIFIED = "uncertified"


class CardModel(BaseModel):
    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        populate_by_name=True,
    )


class ScaleReference(CardModel):
    type: ScaleReferenceType
    real_world_length_m: float = Field(gt=0)
    pixel_length: float = Field(gt=0)


class CardInput(CardModel):
    video_file_hash: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    voice_note_transcript: str
    scale_reference: ScaleReference


class CaptureQuality(CardModel):
    sharpness_score: float
    measured_fps: float = Field(gt=0)
    resolution: tuple[int, int]
    scale_ref_detected: bool
    passed: bool = Field(alias="pass")
    reason_if_failed: str | None


class Tracking(CardModel):
    model: str
    frames_processed: int = Field(gt=0)
    cost_usd: float = Field(ge=0)
    trajectory_px_path: str


class AnalysisMethod(CardModel):
    name: str
    value: float
    unit: str
    uncertainty: float = Field(ge=0)


class Analysis(CardModel):
    method_a: AnalysisMethod
    method_b: AnalysisMethod
    agreement_pct: float = Field(ge=0)
    final_value: float
    final_uncertainty: float = Field(ge=0)
    unit: str


class Certificate(CardModel):
    unit_check: CheckResult
    residual_check: CheckResult
    cross_method_agreement: CheckResult
    overall: CertificateStatus
    notes: str | None


class CostBreakdown(CardModel):
    transcribe: float = Field(ge=0)
    classify: float = Field(ge=0)
    track: float = Field(ge=0)
    analyze: float = Field(ge=0)
    critique_or_report: float = Field(ge=0)


class Cost(CardModel):
    total_usd: float = Field(ge=0)
    breakdown: CostBreakdown


class PhysicsCard(CardModel):
    card_id: UUID
    created_at: datetime
    experiment_family: ExperimentFamily
    input: CardInput
    capture_quality: CaptureQuality
    tracking: Tracking
    analysis: Analysis
    certificate: Certificate
    cost: Cost
    report_markdown: str
    evidence_frame_paths: list[str]
