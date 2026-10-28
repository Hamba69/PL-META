import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app import lab
from backend.app.main import app
from backend.app.provider_clients import ProviderError, TextResult


@pytest.fixture
def client():
    with tempfile.TemporaryDirectory() as directory, patch.object(lab, "RUNS_DIR", Path(directory)):
        with TestClient(app) as test_client:
            yield test_client


def payload(family="constant_velocity", source="imported_observations", speed=0.2):
    times = np.linspace(0, 4, 41)
    values = speed * times
    if family == "free_fall":
        times = np.linspace(0, .4, 41)
        values = 1 - .5 * 9.81 * times**2
    elif family == "oscillation":
        times = np.linspace(0, 12, 121)
        values = .1 * np.cos(np.pi * times)
    elif family == "ramp_friction":
        values = .5 * 9.80665 * (.5 - .2 * np.cos(np.pi / 6)) * times**2
    elif family == "exponential":
        values = 10 * np.exp(-.3 * times)
    return dict(family=family, title="Controlled trial", question="Does the condition change the result?",
                condition="baseline", source=source, source_name="test fixture",
                notes="Mathematical test fixture; not validation evidence.", assumptions_confirmed=True,
                angle_degrees=30 if family == "ramp_friction" else None,
                observations=[dict(time_s=float(t), value=float(v)) for t,v in zip(times,values)])


@pytest.mark.parametrize("family,expected", [("constant_velocity", .2), ("free_fall", 9.81),
    ("oscillation", 2), ("ramp_friction", .2), ("exponential", -.3)])
def test_supported_families_round_trip(client, family, expected):
    request = payload(family)
    response = client.post("/api/lab/trials", json=request)
    assert response.status_code == 200, response.text
    trial = response.json()
    assert trial["analysis"]["final_value"] == pytest.approx(expected, rel=.001)
    assert trial["status"] == "checks_passed"
    assert trial["externally_validated"] is False
    assert trial["input"] == lab.TrialRequest.model_validate(request).model_dump(mode="json")
    assert client.get(f'/api/lab/trials/{trial["id"]}').json() == trial
    assert len(client.get("/api/lab/trials").json()) == 1


def test_synthetic_results_cannot_be_evidence_or_compared(client):
    a = client.post("/api/lab/trials", json=payload(source="synthetic_demo")).json()
    b = client.post("/api/lab/trials", json=payload()).json()
    assert a["status"] == "review_required"
    assert "SYNTHETIC_DATA" in a["reason_codes"]
    assert client.post("/api/lab/compare", json={"baseline_id":a["id"], "candidate_id":b["id"]}).status_code == 422


def test_comparison_checks_compatibility_and_calculates_delta(client):
    a = client.post("/api/lab/trials", json=payload()).json()
    b = client.post("/api/lab/trials", json=payload(speed=.3)).json()
    c = client.post("/api/lab/trials", json=payload("free_fall")).json()
    result = client.post("/api/lab/compare", json={"baseline_id":a["id"], "candidate_id":b["id"]})
    assert result.json()["difference"] == pytest.approx(.1)
    assert result.json()["relative_change_pct"] == pytest.approx(50)
    for other in [a,c]:
        assert client.post("/api/lab/compare", json={"baseline_id":a["id"], "candidate_id":other["id"]}).status_code == 422


@pytest.mark.parametrize("mutation", [
    lambda p: p.update(angle_degrees=None),
    lambda p: p["observations"][1].update(time_s=0),
    lambda p: p.update(observations=p["observations"][:3]),
    lambda p: p.update(source="manual_video", source_sha256=None),
    lambda p: p.update(family="novel"),
    lambda p: p.update(observations=[dict(time_s=i, value=1) for i in range(10)]),
])
def test_invalid_inputs_are_rejected_without_saving(client, mutation):
    request=payload("ramp_friction")
    mutation(request)
    assert client.post("/api/lab/trials", json=request).status_code == 422
    assert client.get("/api/lab/trials").json() == []


def test_bad_physics_and_unconfirmed_setup_do_not_pass(client):
    request=payload("free_fall")
    for observation in request["observations"]:
        observation["value"] += 2 * observation["time_s"]  # thrown, not released
    trial=client.post("/api/lab/trials", json=request).json()
    assert trial["status"] == "review_required"
    request=payload()
    request["assumptions_confirmed"]=False
    trial=client.post("/api/lab/trials", json=request).json()
    assert "ASSUMPTIONS_NOT_CONFIRMED" in trial["reason_codes"]


def test_fingerprint_covers_setup_not_only_observations(client):
    a=payload()
    one=client.post("/api/lab/trials",json=a).json()
    two=client.post("/api/lab/trials",json=a).json()
    assert one["input_sha256"] == two["input_sha256"]
    a["notes"]="Different calibration"
    three=client.post("/api/lab/trials",json=a).json()
    assert three["input_sha256"] != one["input_sha256"]


def test_uncertainty_does_not_assume_independent_methods(client):
    request=payload()
    for i, row in enumerate(request["observations"]):
        row["value"] += .001 * np.sin(i)
    result=client.post("/api/lab/trials",json=request).json()["analysis"]
    assert result["final_uncertainty"] == max(result["method_a"]["uncertainty"],result["method_b"]["uncertainty"])


def test_planner_accepts_only_catalog_selection(client):
    with patch.object(lab, "ProviderClients") as provider:
        provider.return_value.spark_text.return_value=TextResult('{"family":"ramp_friction"}',10,10,.0001,"completed")
        response=client.post("/api/lab/plan",json={"question":"Which surface makes a block slide more slowly?"})
        assert response.json()["protocol"] == lab.PROTOCOLS["ramp_friction"]
        provider.return_value.spark_text.return_value=TextResult('{"family":"free_fall","gravity":9.81}',10,10,.0001,"completed")
        assert client.post("/api/lab/plan",json={"question":"Measure gravity for me please"}).status_code == 502
        provider.return_value.spark_text.return_value=TextResult('{"family":"unsupported"}',10,10,.0001,"completed")
        assert client.post("/api/lab/plan",json={"question":"Measure water purity for drinking"}).json()["protocol"] is None
        provider.side_effect=ProviderError("unavailable")
        assert client.post("/api/lab/plan",json={"question":"Measure gravity for me please"}).status_code == 503


def test_workbench_assets_and_invalid_run(client):
    assert client.get("/").status_code == 200
    assert client.get("/static/lab.js").status_code == 200
    assert len(client.get("/api/lab/protocols").json()["protocols"]) == 5
    assert client.get("/api/lab/trials/not-a-uuid").status_code == 404
    assert client.get("/healthz").json() == {"status":"ok"}
