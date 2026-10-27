"""Run the deterministic free-fall calculation in an isolated Modal sandbox."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path

from .certificate import evaluate
from .families import FitDiagnostics
from .schema import Analysis, ExperimentFamily

BACKEND_ROOT = Path(__file__).resolve().parents[1]
APP_SOURCE = BACKEND_ROOT / "app"


class SandboxExecutionError(RuntimeError):
    """The remote deterministic calculation could not be completed."""


def analyze_free_fall_in_sandbox(
    time_s: list[float],
    vertical_position_m: list[float],
    *,
    bootstrap_samples: int = 200,
    seed: int = 0,
) -> tuple[Analysis, FitDiagnostics]:
    if not os.environ.get("MODAL_TOKEN_ID") or not os.environ.get("MODAL_TOKEN_SECRET"):
        raise SandboxExecutionError("Modal sandbox credentials are not configured.")
    try:
        import modal
    except ImportError as error:
        raise SandboxExecutionError("The Modal Python package is not installed.") from error

    payload = {
        "time_s": time_s,
        "vertical_position_m": vertical_position_m,
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
    }
    encoded_payload = base64.b64encode(
        json.dumps(payload, allow_nan=False, separators=(",", ":")).encode("utf-8")
    ).decode("ascii")
    driver = (
        "import base64,json,sys\n"
        "sys.path.insert(0, '/opt/phonelab')\n"
        "from app.families.free_fall import analyze\n"
        "data=json.loads(base64.b64decode(sys.argv[1]))\n"
        "analysis,diagnostics=analyze(data['time_s'],data['vertical_position_m'],"
        "bootstrap_samples=data['bootstrap_samples'],seed=data['seed'])\n"
        "print(json.dumps({'analysis':analysis.model_dump(mode='json'),"
        "'diagnostics':{'r_squared':diagnostics.r_squared,"
        "'normalized_rmse':diagnostics.normalized_rmse}},allow_nan=False))\n"
    )
    sandbox = None
    try:
        app = modal.App.lookup("phonelab-analysis", create_if_missing=True)
        image = (
            modal.Image.debian_slim(python_version="3.12")
            .pip_install(
                "numpy>=2.0,<3.0",
                "scipy>=1.14,<2.0",
                "pydantic>=2.8,<3.0",
            )
            .add_local_dir(str(APP_SOURCE), remote_path="/opt/phonelab/app")
        )
        sandbox = modal.Sandbox.create(
            app=app,
            image=image,
            timeout=180,
            idle_timeout=20,
            cpu=1,
            memory=2048,
            block_network=True,
        )
        process = sandbox.exec(
            "python",
            "-c",
            driver,
            encoded_payload,
            timeout=120,
        )
        stdout = process.stdout.read()
        stderr = process.stderr.read()
        if not stdout.strip():
            raise SandboxExecutionError("Modal sandbox returned no analysis output.")
        result = json.loads(stdout.strip().splitlines()[-1])
        analysis = Analysis.model_validate(result["analysis"])
        diagnostics = FitDiagnostics(**result["diagnostics"])
        return analysis, diagnostics
    except SandboxExecutionError:
        raise
    except Exception as error:
        # Provider exceptions can contain remote request details. Keep them out
        # of API responses and logs; the local server only needs a safe summary.
        raise SandboxExecutionError("Modal sandbox analysis failed.") from error
    finally:
        if sandbox is not None:
            try:
                sandbox.terminate()
            except Exception:
                pass
