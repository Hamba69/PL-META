import unittest

import numpy as np

from backend.app.certificate import CertificateThresholds, evaluate
from backend.app.families.constant_velocity import analyze as analyze_constant_velocity
from backend.app.families.exponential import analyze as analyze_exponential
from backend.app.families.free_fall import analyze as analyze_free_fall
from backend.app.families.oscillation import analyze as analyze_oscillation
from backend.app.families.oscillation import pendulum_gravity
from backend.app.families.projectile import analyze as analyze_projectile
from backend.app.families.ramp_friction import analyze as analyze_ramp_friction
from backend.app.schema import (
    Analysis,
    AnalysisMethod,
    ExperimentFamily,
)


class DeterministicAnalysisTests(unittest.TestCase):
    def test_free_fall_methods_recover_textbook_gravity(self) -> None:
        time_s = np.linspace(0, 1.5, 61)
        vertical_position = 4 - 0.5 * 9.81 * time_s**2

        analysis, diagnostics = analyze_free_fall(time_s, vertical_position)

        self.assertAlmostEqual(analysis.method_a.value, 9.81, places=10)
        self.assertAlmostEqual(analysis.method_b.value, 9.81, places=10)
        self.assertAlmostEqual(analysis.final_value, 9.81, places=10)
        self.assertLess(diagnostics.normalized_rmse, 1e-12)

    def test_constant_velocity_methods_match_linear_trajectory(self) -> None:
        time_s = np.linspace(0, 10, 51)
        position_m = 0.4 + 1.25 * time_s

        analysis, diagnostics = analyze_constant_velocity(time_s, position_m)

        self.assertAlmostEqual(analysis.method_a.value, 1.25, places=10)
        self.assertAlmostEqual(analysis.method_b.value, 1.25, places=10)
        self.assertAlmostEqual(diagnostics.r_squared, 1.0, places=12)

    def test_exponential_growth_methods_recover_growth_rate(self) -> None:
        time_s = np.linspace(0, 4, 81)
        magnitude = 2 * np.exp(0.25 * time_s)

        analysis, diagnostics = analyze_exponential(time_s, magnitude)

        self.assertAlmostEqual(analysis.method_a.value, 0.25, places=10)
        self.assertAlmostEqual(analysis.method_b.value, 0.25, delta=0.001)
        self.assertLess(diagnostics.normalized_rmse, 1e-12)

    def test_exponential_decay_methods_recover_decay_rate(self) -> None:
        time_s = np.linspace(0, 4, 81)
        magnitude = 8 * np.exp(-0.3 * time_s)

        analysis, _ = analyze_exponential(time_s, magnitude)

        self.assertAlmostEqual(analysis.method_a.value, -0.3, places=10)
        self.assertAlmostEqual(analysis.method_b.value, -0.3, delta=0.001)

    def test_oscillation_methods_recover_period(self) -> None:
        time_s = np.arange(0, 8, 0.01)
        position = 3 + 2 * np.cos(2 * np.pi * time_s / 1.6 + 0.2)

        analysis, diagnostics = analyze_oscillation(time_s, position, bootstrap_samples=30)

        self.assertAlmostEqual(analysis.method_a.value, 1.6, delta=0.002)
        self.assertAlmostEqual(analysis.method_b.value, 1.6, delta=0.011)
        self.assertLess(diagnostics.normalized_rmse, 1e-10)

    def test_pendulum_periods_produce_gravity_cross_check(self) -> None:
        periods, _ = analyze_oscillation(
            np.arange(0, 24, 0.01),
            1.2 + 0.3 * np.cos(2 * np.pi * np.arange(0, 24, 0.01) / 2),
            bootstrap_samples=20,
        )

        gravity = pendulum_gravity(periods, 9.80665 * 2**2 / (4 * np.pi**2))

        self.assertAlmostEqual(gravity.method_a.value, 9.80665, delta=1e-4)
        self.assertAlmostEqual(gravity.method_b.value, 9.80665, delta=0.01)
        self.assertEqual(gravity.unit, "m/s^2")

    def test_projectile_analyzes_vertical_and_horizontal_axes_independently(self) -> None:
        time_s = np.linspace(0, 1, 41)
        horizontal = 2 + 3.5 * time_s
        vertical = 5 - 0.5 * 9.81 * time_s**2

        gravity, vertical_fit, horizontal_velocity, horizontal_fit = analyze_projectile(
            time_s,
            horizontal,
            vertical,
        )

        self.assertAlmostEqual(gravity.final_value, 9.81, places=10)
        self.assertAlmostEqual(horizontal_velocity.final_value, 3.5, places=10)
        self.assertAlmostEqual(vertical_fit.r_squared, 1, places=10)
        self.assertAlmostEqual(horizontal_fit.r_squared, 1, places=10)

    def test_ramp_friction_methods_recover_coefficient(self) -> None:
        time_s = np.linspace(0, 1.2, 61)
        angle = np.deg2rad(30)
        acceleration = 9.80665 * (np.sin(angle) - 0.2 * np.cos(angle))
        distance = 0.5 * acceleration * time_s**2

        analysis, diagnostics = analyze_ramp_friction(time_s, distance, angle)

        self.assertAlmostEqual(analysis.method_a.value, 0.2, places=10)
        self.assertAlmostEqual(analysis.method_b.value, 0.2, delta=1e-10)
        self.assertLess(diagnostics.normalized_rmse, 1e-12)

    def test_analysis_rejects_non_increasing_time(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            analyze_constant_velocity([0, 1, 1], [0, 2, 3])


class CertificateTests(unittest.TestCase):
    def test_certifies_agreeing_plausible_free_fall_result(self) -> None:
        analysis, diagnostics = analyze_free_fall(
            np.linspace(0, 1, 31),
            1 - 0.5 * 9.81 * np.linspace(0, 1, 31) ** 2,
        )

        result = evaluate(
            ExperimentFamily.FREE_FALL,
            analysis,
            diagnostics.normalized_rmse,
        )

        self.assertEqual(result.certificate.overall.value, "certified")
        self.assertEqual(result.reason_codes, [])

    def test_reports_specific_failure_codes(self) -> None:
        analysis = Analysis(
            method_a=AnalysisMethod(name="fit", value=12, unit="m/s^2", uncertainty=0.1),
            method_b=AnalysisMethod(name="endpoint", value=8, unit="m/s^2", uncertainty=0.1),
            agreement_pct=40,
            final_value=10,
            final_uncertainty=0.1,
            unit="m/s^2",
        )

        result = evaluate(
            ExperimentFamily.FREE_FALL,
            analysis,
            0.5,
            thresholds=CertificateThresholds(
                cross_method_agreement_pct=10,
                normalized_rmse=0.1,
            ),
        )

        self.assertEqual(
            result.reason_codes,
            ["UNIT_CHECK_FAILED", "RESIDUAL_CHECK_FAILED", "CROSS_METHOD_AGREEMENT_FAILED"],
        )
        self.assertEqual(result.certificate.overall.value, "uncertified")
        self.assertIn("unit check failed", result.certificate.notes)

    def test_recomputes_method_agreement_instead_of_trusting_reported_metric(self) -> None:
        analysis = Analysis(
            method_a=AnalysisMethod(name="fit", value=9, unit="m/s^2", uncertainty=0.1),
            method_b=AnalysisMethod(name="endpoint", value=10, unit="m/s^2", uncertainty=0.1),
            agreement_pct=0,
            final_value=9.5,
            final_uncertainty=0.1,
            unit="m/s^2",
        )

        result = evaluate(ExperimentFamily.FREE_FALL, analysis, 0)

        self.assertEqual(result.certificate.cross_method_agreement.value, "fail")
        self.assertIn("CROSS_METHOD_AGREEMENT_FAILED", result.reason_codes)

    def test_novel_family_is_not_silently_certified(self) -> None:
        analysis = Analysis(
            method_a=AnalysisMethod(name="a", value=1, unit="1", uncertainty=0),
            method_b=AnalysisMethod(name="b", value=1, unit="1", uncertainty=0),
            agreement_pct=0,
            final_value=1,
            final_uncertainty=0,
            unit="1",
        )
        with self.assertRaisesRegex(ValueError, "novel experiments"):
            evaluate(ExperimentFamily.NOVEL, analysis, 0)

    def test_constant_velocity_requires_high_fit_r_squared(self) -> None:
        analysis = Analysis(
            method_a=AnalysisMethod(name="fit", value=2, unit="m/s", uncertainty=0),
            method_b=AnalysisMethod(name="endpoint", value=2, unit="m/s", uncertainty=0),
            agreement_pct=0,
            final_value=2,
            final_uncertainty=0,
            unit="m/s",
        )

        result = evaluate(
            ExperimentFamily.CONSTANT_VELOCITY,
            analysis,
            0,
            fit_r_squared=0.8,
        )

        self.assertEqual(result.certificate.residual_check.value, "fail")


if __name__ == "__main__":
    unittest.main()
