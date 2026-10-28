import argparse
import csv
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from backend.app.schema import CertificateStatus, ExperimentFamily, PhysicsCard


class GroundTruthMeasurement(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")

    video_file_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    experiment_family: ExperimentFamily
    value: float
    unit: str
    tolerance_abs: float | None = Field(default=None, ge=0)
    tolerance_pct: float | None = Field(default=None, ge=0)


class GroundTruthFile(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")

    measurements: list[GroundTruthMeasurement]


def evaluate_cards(
    ground_truth_path: Path,
    cards_directory: Path,
) -> list[dict[str, Any]]:
    try:
        ground_truth = GroundTruthFile.model_validate_json(
            ground_truth_path.read_text(encoding="utf-8")
        )
    except (OSError, ValidationError) as error:
        raise ValueError(f"invalid or unreadable ground-truth file: {error}") from error
    if not ground_truth.measurements:
        raise ValueError(
            "ground-truth set is empty; add independently measured real clips before evaluation"
        )

    rows: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    for measurement in ground_truth.measurements:
        if measurement.video_file_hash in seen_hashes:
            raise ValueError(
                f"duplicate ground-truth video hash: {measurement.video_file_hash}"
            )
        seen_hashes.add(measurement.video_file_hash)
        if measurement.tolerance_abs is None and measurement.tolerance_pct is None:
            raise ValueError(
                f"ground truth needs an absolute or percentage tolerance: "
                f"{measurement.video_file_hash}"
            )
        card_path = cards_directory / f"{measurement.video_file_hash}.json"
        try:
            card = PhysicsCard.model_validate_json(card_path.read_text(encoding="utf-8"))
        except OSError as error:
            raise ValueError(
                f"missing result card for {measurement.video_file_hash}: {card_path}"
            ) from error
        except ValidationError as error:
            raise ValueError(f"invalid physics card {card_path}: {error}") from error
        if card.input.video_file_hash != measurement.video_file_hash:
            raise ValueError(f"card video hash does not match ground truth: {card_path}")
        if card.experiment_family != measurement.experiment_family:
            raise ValueError(f"experiment family mismatch for {measurement.video_file_hash}")
        if card.analysis.unit != measurement.unit:
            raise ValueError(f"unit mismatch for {measurement.video_file_hash}")

        absolute_error = abs(card.analysis.final_value - measurement.value)
        allowed_error = max(
            measurement.tolerance_abs or 0.0,
            abs(measurement.value) * (measurement.tolerance_pct or 0.0) / 100,
        )
        rows.append(
            {
                "video_file_hash": measurement.video_file_hash,
                "experiment_family": measurement.experiment_family.value,
                "measured_value": card.analysis.final_value,
                "uncertainty": card.analysis.final_uncertainty,
                "ground_truth": measurement.value,
                "unit": measurement.unit,
                "absolute_error": absolute_error,
                "allowed_error": allowed_error,
                "within_tolerance": absolute_error <= allowed_error,
                "certified": card.certificate.overall is CertificateStatus.CERTIFIED,
            }
        )
    return rows


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    if not rows:
        raise ValueError("refusing to write an empty evaluation table")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare actual physics cards against independently measured ground truth."
    )
    parser.add_argument("--ground-truth", type=Path, default=Path("eval/ground_truth.json"))
    parser.add_argument("--cards", type=Path, default=Path("eval/cards"))
    parser.add_argument("--output", type=Path, default=Path("eval/results.csv"))
    arguments = parser.parse_args()
    rows = evaluate_cards(arguments.ground_truth, arguments.cards)
    write_csv(rows, arguments.output)
    passed = sum(row["within_tolerance"] and row["certified"] for row in rows)
    print(f"Evaluated {len(rows)} real result cards; {passed} passed tolerance and certification.")
    print(f"Accuracy table written to {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
