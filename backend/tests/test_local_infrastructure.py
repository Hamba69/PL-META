import json
import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from backend.app.cache import ContentCache, sha256_file
from backend.app.capture_coach import CaptureCoachConfig, inspect_video
from backend.app.capture_coach import detect_scale_reference
from backend.app.cost_tracker import CostTracker
from backend.app.keyframes import extract_keyframes
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
from eval.run_eval import evaluate_cards, write_csv


class CacheAndCostTests(unittest.TestCase):
    def test_content_cache_uses_hash_step_version_and_parameters(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache = ContentCache(Path(temporary_directory) / "cache.sqlite3")
            video_hash = "a" * 64
            parameters = {"width": 1920, "options": ["target"]}
            self.assertIsNone(cache.get(video_hash, "tracking", "v1", parameters))
            cache_key = cache.put(
                video_hash,
                "tracking",
                "v1",
                parameters,
                {"frames_processed": 12},
            )
            entry = cache.get(video_hash, "tracking", "v1", parameters)

            self.assertEqual(len(cache_key), 64)
            self.assertEqual(entry.value, {"frames_processed": 12})
            self.assertIsNone(
                cache.get(video_hash, "tracking", "v1", {"width": 1280})
            )
            self.assertIsNone(cache.get(video_hash, "tracking", "v2", parameters))

    def test_file_hash_is_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "clip.bin"
            path.write_bytes(b"phonelab")
            self.assertEqual(
                sha256_file(path),
                "a05926fe7c5973c98796e546a936e9ce3ae5881805c13b3d5b4d99db5345586a",
            )

    def test_cost_tracker_records_call_and_daily_total_in_utc(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            tracker = CostTracker(Path(temporary_directory) / "cost.sqlite3")
            tracker.record_call(
                model="muse-spark-1.3",
                call_kind="classify",
                unit_name="tokens",
                units=500,
                cost_usd=0.002,
                run_id="run-1",
                occurred_at=datetime(2026, 10, 26, 23, 30, tzinfo=timezone.utc),
            )
            tracker.record_call(
                model="sam-3.1",
                call_kind="track",
                unit_name="frames",
                units=300,
                cost_usd=0.06,
                run_id="run-1",
                occurred_at=datetime(2026, 10, 27, 1, 30, tzinfo=timezone.utc),
            )

            self.assertAlmostEqual(tracker.daily_total(date(2026, 10, 26)), 0.002)
            self.assertAlmostEqual(tracker.daily_total(date(2026, 10, 27)), 0.06)
            self.assertAlmostEqual(tracker.run_total("run-1"), 0.062)
            self.assertEqual(
                tracker.run_breakdown("run-1"),
                {"classify": 0.002, "track": 0.06},
            )


class VideoUtilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary_directory = tempfile.TemporaryDirectory()
        cls.video_path = Path(cls.temporary_directory.name) / "sample.avi"
        writer = cv2.VideoWriter(
            str(cls.video_path),
            cv2.VideoWriter_fourcc(*"MJPG"),
            30,
            (640, 480),
        )
        if not writer.isOpened():
            cls.temporary_directory.cleanup()
            raise RuntimeError("OpenCV MJPG video writer is unavailable for local tests")
        rng = np.random.default_rng(12)
        for _ in range(30):
            frame = rng.integers(0, 80, size=(480, 640, 3), dtype=np.uint8)
            cv2.rectangle(frame, (200, 70), (400, 353), (255, 255, 255), thickness=-1)
            cv2.rectangle(frame, (200, 70), (400, 353), (0, 0, 0), thickness=4)
            writer.write(frame)
        writer.release()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary_directory.cleanup()

    def test_capture_coach_reads_metadata_sharpness_and_scale(self) -> None:
        result = inspect_video(
            self.video_path,
            claimed_fps=30,
            config=CaptureCoachConfig(minimum_sharpness_variance=10),
        )

        self.assertTrue(result.passed, result.reason_if_failed)
        self.assertEqual(result.measured_fps, 30)
        self.assertEqual(result.resolution, (640, 480))
        self.assertTrue(result.scale_ref_detected)

    def test_capture_coach_reports_specific_bad_capture_reasons(self) -> None:
        result = inspect_video(
            self.video_path,
            claimed_fps=24,
            config=CaptureCoachConfig(
                minimum_resolution=(1280, 720),
                minimum_sharpness_variance=1_000_000,
                claimed_fps_tolerance_fraction=0.05,
            ),
        )

        self.assertFalse(result.passed)
        self.assertIn("blurry", result.reason_if_failed)
        self.assertIn("FPS", result.reason_if_failed)
        self.assertIn("Resolution", result.reason_if_failed)

    def test_capture_coach_rejects_frame_count_over_service_limit(self) -> None:
        result = inspect_video(
            self.video_path,
            config=CaptureCoachConfig(maximum_frames=10),
        )

        self.assertFalse(result.passed)
        self.assertIn("10-frame", result.reason_if_failed)

    def test_scale_detector_rejects_frame_without_reference(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.assertFalse(detect_scale_reference(frame, ScaleReferenceType.A4_SHEET))

    def test_keyframe_extraction_writes_real_decoded_frames(self) -> None:
        with tempfile.TemporaryDirectory() as output_directory:
            paths = extract_keyframes(self.video_path, output_directory, count=5)
            self.assertEqual(len(paths), 5)
            self.assertTrue(all(path.is_file() for path in paths))
            decoded = cv2.imread(str(paths[0]))
            self.assertIsNotNone(decoded)
            self.assertEqual(decoded.shape, (480, 640, 3))

    def test_empty_ground_truth_is_not_reported_as_an_accuracy_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            ground_truth = root / "ground_truth.json"
            ground_truth.write_text(json.dumps({"measurements": []}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "ground-truth set is empty"):
                evaluate_cards(ground_truth, root / "cards")

    def test_eval_compares_real_card_and_writes_accuracy_table(self) -> None:
        video_hash = "b" * 64
        card = PhysicsCard(
            card_id=uuid4(),
            created_at=datetime(2026, 10, 26, tzinfo=timezone.utc),
            experiment_family=ExperimentFamily.FREE_FALL,
            input=CardInput(
                video_file_hash=video_hash,
                voice_note_transcript="Test clip",
                scale_reference=ScaleReference(
                    type=ScaleReferenceType.A4_SHEET,
                    real_world_length_m=0.297,
                    pixel_length=800,
                ),
            ),
            capture_quality=CaptureQuality(
                sharpness_score=140,
                measured_fps=30,
                resolution=(1280, 720),
                scale_ref_detected=True,
                passed=True,
                reason_if_failed=None,
            ),
            tracking=Tracking(
                model="sam-3.1",
                frames_processed=100,
                cost_usd=0.02,
                trajectory_px_path="trajectory.json",
            ),
            analysis=Analysis(
                method_a=AnalysisMethod(
                    name="quadratic fit",
                    value=9.8,
                    unit="m/s^2",
                    uncertainty=0.1,
                ),
                method_b=AnalysisMethod(
                    name="endpoint kinematic",
                    value=9.8,
                    unit="m/s^2",
                    uncertainty=0.1,
                ),
                agreement_pct=0,
                final_value=9.8,
                final_uncertainty=0.1,
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
                total_usd=0.02,
                breakdown=CostBreakdown(
                    transcribe=0,
                    classify=0,
                    track=0.02,
                    analyze=0,
                    critique_or_report=0,
                ),
            ),
            report_markdown="Measured acceleration: 9.8 m/s^2.",
            evidence_frame_paths=["frame_001.jpg"],
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            cards_directory = root / "cards"
            cards_directory.mkdir()
            (cards_directory / f"{video_hash}.json").write_text(
                card.model_dump_json(by_alias=True),
                encoding="utf-8",
            )
            ground_truth_path = root / "ground_truth.json"
            ground_truth_path.write_text(
                json.dumps(
                    {
                        "measurements": [
                            {
                                "video_file_hash": video_hash,
                                "experiment_family": "free_fall",
                                "value": 9.81,
                                "unit": "m/s^2",
                                "tolerance_abs": 0.02,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            rows = evaluate_cards(ground_truth_path, cards_directory)
            output_path = root / "results.csv"
            write_csv(rows, output_path)

            self.assertTrue(rows[0]["within_tolerance"])
            self.assertTrue(rows[0]["certified"])
            self.assertIn("ground_truth", output_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
