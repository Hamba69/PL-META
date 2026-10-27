from dataclasses import dataclass
from math import isfinite

from pydantic import BaseModel, ConfigDict, Field

from .schema import (
    Analysis,
    Certificate,
    CertificateStatus,
    CheckResult,
    ExperimentFamily,
)


@dataclass(frozen=True)
class CertificateThresholds:
    cross_method_agreement_pct: float = 10.0
    normalized_rmse: float = 0.1
    minimum_r_squared: float = 0.95

    def __post_init__(self) -> None:
        if not isfinite(self.cross_method_agreement_pct) or self.cross_method_agreement_pct < 0:
            raise ValueError("agreement threshold must be finite and non-negative")
        if not isfinite(self.normalized_rmse) or self.normalized_rmse < 0:
            raise ValueError("residual threshold must be finite and non-negative")
        if not isfinite(self.minimum_r_squared) or not 0 <= self.minimum_r_squared <= 1:
            raise ValueError("minimum R-squared threshold must be in [0, 1]")


class CertificateEvaluation(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")

    certificate: Certificate
    reason_codes: list[str]
    residual_metric: float = Field(ge=0)


_EXPECTED_UNITS: dict[ExperimentFamily, str] = {
    ExperimentFamily.FREE_FALL: "m/s^2",
    ExperimentFamily.PROJECTILE: "m/s^2",
    ExperimentFamily.OSCILLATION: "s",
    ExperimentFamily.EXPONENTIAL: "1/s",
    ExperimentFamily.CONSTANT_VELOCITY: "m/s",
    ExperimentFamily.RAMP_FRICTION: "1",
}


def evaluate(
    family: ExperimentFamily,
    analysis: Analysis,
    residual_metric: float,
    *,
    fit_r_squared: float | None = None,
    thresholds: CertificateThresholds = CertificateThresholds(),
) -> CertificateEvaluation:
    if family is ExperimentFamily.NOVEL:
        raise ValueError("novel experiments do not have a certified family template")
    if not isfinite(residual_metric) or residual_metric < 0:
        raise ValueError("residual metric must be finite and non-negative")

    expected_unit = _EXPECTED_UNITS[family]
    values = (analysis.method_a.value, analysis.method_b.value, analysis.final_value)
    unit_pass = (
        analysis.method_a.unit == expected_unit
        and analysis.method_b.unit == expected_unit
        and analysis.unit == expected_unit
        and all(isfinite(value) and _plausible(family, value) for value in values)
    )
    residual_pass = residual_metric <= thresholds.normalized_rmse
    if family is ExperimentFamily.CONSTANT_VELOCITY:
        residual_pass = (
            residual_pass
            and fit_r_squared is not None
            and isfinite(fit_r_squared)
            and fit_r_squared >= thresholds.minimum_r_squared
        )
    agreement_denominator = (
        abs(analysis.method_a.value) + abs(analysis.method_b.value)
    ) / 2
    measured_agreement_pct = (
        0.0
        if agreement_denominator == 0
        else abs(analysis.method_a.value - analysis.method_b.value)
        / agreement_denominator
        * 100
    )
    agreement_pass = (
        isfinite(analysis.agreement_pct)
        and abs(analysis.agreement_pct - measured_agreement_pct) <= 1e-9
        and measured_agreement_pct <= thresholds.cross_method_agreement_pct
    )
    checks = {
        "unit_check": unit_pass,
        "residual_check": residual_pass,
        "cross_method_agreement": agreement_pass,
    }
    reasons = [f"{name.upper()}_FAILED" for name, passed in checks.items() if not passed]
    overall = (
        CertificateStatus.CERTIFIED
        if all(checks.values())
        else CertificateStatus.UNCERTIFIED
    )
    notes = None
    if reasons:
        notes = "Certificate failed: " + ", ".join(reasons).lower().replace("_", " ") + "."
    certificate = Certificate(
        unit_check=CheckResult.PASS if unit_pass else CheckResult.FAIL,
        residual_check=CheckResult.PASS if residual_pass else CheckResult.FAIL,
        cross_method_agreement=(
            CheckResult.PASS if agreement_pass else CheckResult.FAIL
        ),
        overall=overall,
        notes=notes,
    )
    return CertificateEvaluation(
        certificate=certificate,
        reason_codes=reasons,
        residual_metric=residual_metric,
    )


def _plausible(family: ExperimentFamily, value: float) -> bool:
    if family in (ExperimentFamily.FREE_FALL, ExperimentFamily.PROJECTILE):
        return 9.0 <= value <= 10.5
    if family is ExperimentFamily.OSCILLATION:
        return value > 0
    if family is ExperimentFamily.EXPONENTIAL:
        return value != 0
    if family is ExperimentFamily.CONSTANT_VELOCITY:
        return isfinite(value)
    if family is ExperimentFamily.RAMP_FRICTION:
        return 0 <= value <= 2
    return False
