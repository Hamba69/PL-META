import json
import unittest
from datetime import datetime, timezone
from uuid import uuid4

from pydantic import ValidationError

from backend.app.schema import (
    Analysis,
    AnalysisMethod,
    CaptureQuality,
    CardInput,
    Certificate,
    CertificateStatus,
    CheckResult,
    Cost,
    CostBreakdown,
    ExperimentFamily,
    PhysicsCard,
    ScaleReference,
    ScaleReferenceType,
    Tracking,
)


def example_card() -> PhysicsCard:
    return PhysicsCard(
        card_id=uuid4(),
        created_at=datetime(2026, 10, 26, tzinfo=timezone.utc),
        experiment_family=ExperimentFamily.FREE_FALL,
        input=CardInput(
            video_file_hash="a" * 64,
            voice_note_transcript="Dropped object beside an A4 sheet.",
            scale_reference=ScaleReference(
                type=ScaleReferenceType.A4_SHEET,
                real_world_length_m=0.297,
                pixel_length=812.4,
            ),
        ),
        capture_quality=CaptureQuality(
            sharpness_score=143.2,
            measured_fps=29.8,
            resolution=(1920, 1080),
            scale_ref_detected=True,
            passed=True,
            reason_if_failed=None,
        ),
        tracking=Tracking(
            model="sam-3.1",
            frames_processed=298,
            cost_usd=0.0596,
            trajectory_px_path="trajectories/clip.json",
        ),
        analysis=Analysis(
            method_a=AnalysisMethod(
                name="quadratic fit",
                value=9.74,
                unit="m/s^2",
                uncertainty=0.18,
            ),
            method_b=AnalysisMethod(
                name="endpoint kinematic",
                value=9.69,
                unit="m/s^2",
                uncertainty=0.21,
            ),
            agreement_pct=0.52,
            final_value=9.72,
            final_uncertainty=0.19,
            unit="m/s^2",
        ),
        certificate=Certificate(
            unit_check=CheckResult.PASS,
            residual_check=CheckResult.PASS,
            cross_method_agreement=CheckResult.PASS,
            overall=CertificateStatus.CERTIFIED,
            notes=None,
        ),
        cost=Cost(
            total_usd=0.14,
            breakdown=CostBreakdown(
                transcribe=0.003,
                classify=0.004,
                track=0.0596,
                analyze=0,
                critique_or_report=0.009,
            ),
        ),
        report_markdown="The measured acceleration is 9.72 m/s^2.",
        evidence_frame_paths=["frame_0012.jpg", "frame_0145.jpg"],
    )


class PhysicsCardSchemaTests(unittest.TestCase):
    def test_card_round_trips_through_json(self) -> None:
        original = example_card()
        serialized = original.model_dump_json(by_alias=True)
        restored = PhysicsCard.model_validate(json.loads(serialized))

        self.assertEqual(restored, original)
        self.assertEqual(restored.model_dump_json(by_alias=True), serialized)
        self.assertIn('"pass":true', serialized)

    def test_invalid_video_hash_is_rejected(self) -> None:
        payload = example_card().model_dump(mode="json", by_alias=True)
        payload["input"]["video_file_hash"] = "not-a-sha256-hash"

        with self.assertRaises(ValidationError):
            PhysicsCard.model_validate(payload)

    def test_unknown_card_fields_are_rejected(self) -> None:
        payload = example_card().model_dump(mode="json", by_alias=True)
        payload["unreviewed_field"] = "unexpected"

        with self.assertRaises(ValidationError):
            PhysicsCard.model_validate(payload)

    def test_non_finite_measurements_are_rejected(self) -> None:
        payload = example_card().model_dump(mode="json", by_alias=True)
        payload["analysis"]["final_value"] = float("nan")

        with self.assertRaises(ValidationError):
            PhysicsCard.model_validate(payload)


if __name__ == "__main__":
    unittest.main()
