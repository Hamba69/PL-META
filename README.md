# PhoneLab

**A phone, a question, a repeatable experiment.**

A local experiment workbench for classrooms and small workshops. Choose a protocol, calibrate video or import observations, calculate with two numerical methods, inspect evidence, and compare trials. The strongest first use: one sliding block, two surfaces, one measured ramp angle.

## Run

From the project root, with Python 3.11+:

```powershell
python -m pip install -r backend/requirements.txt
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000**. API documentation: `/docs`. Alternatively: `docker compose --env-file .env.local up --build`.

Manual analysis needs no provider credentials. Optional model planning and automatic free-fall use the private provider settings in `.env.local` (see `.env.example`); automatic free-fall also needs Modal. This is a local single-user prototype without authentication or multi-user isolation.

## Five benches

| Bench | Result | Constraint |
| --- | --- | --- |
| Motion and transport | Signed velocity | Straight, steady motion; calibrated position |
| Pendulum and vibration | Period | Uniform sampling and enough cycles |
| Gravity | Downward acceleration | Released from rest; positive-up coordinates |
| Surfaces and sliding | Kinetic friction coefficient | Sliding block, measured angle, release from rest |
| Decay and recovery | Signed exponential rate | Positive baseline-corrected signal; full halving or doubling |

General projectiles, rolling friction, volumetric flow and arbitrary chemistry are not supported measurements.

## Workflow

1. Choose a bench, or optionally ask Muse Spark to select an approved protocol.
2. Name the trial and condition changed. Read the capture steps and assumptions.
3. Paste `time, value` rows, or choose **Mark a local video**. Pause and capture a frame, click both ends of a known reference, then mark the target on successive captured frames. Reference direction defines the signed axis. Correct the real-time/playback factor for slow motion.
4. Record calibration notes and confirm setup assumptions. Calculate and inspect both methods, the plot and checks.
5. Download a receipt, repeat under another condition, and compare compatible trials.

Manual video stays in the browser. Observations, notes and a video fingerprint are sent to the local server. The separate automatic free-fall uploader uses external providers and incurs charges.

**Explore a synthetic example** fills a labelled mathematical fixture. It is not experimental evidence. Select a bench again to start a real trial.

## Evidence boundaries

`checks_passed` means numerical checks passed for supplied observations and declared assumptions. It does **not** mean independently validated or physically accurate. Both methods share observations and systematic errors. Uncertainty is fit variability only: scale, timing, perspective and angle errors are not included.

Receipts contain observations, provenance notes, input SHA-256, calculation version, method results, diagnostics and limitations. Trials live in `backend/data/lab/`. Comparison reports descriptive differences, not statistical significance or causality. Synthetic and rejected trials are excluded.

Muse Spark selects approved protocol IDs; it cannot supply measurements. The existing automatic free-fall pipeline uses SAM and Modal, with optional voice transcription. Local analysis runs fixed NumPy/SciPy routines. Provider cost estimates use existing client rate assumptions; provider billing is authoritative.

A live Spark protocol-selection call succeeded on 28 October 2026. Spark Open returned no usable design-review text. Historical service smoke tests and the existing failed filmed run do not establish physical validation of the new bench.

## Verify

From the project root:

```powershell
python -m pip install pytest httpx
python -m pytest backend/tests -q
node --check backend/app/static/lab.js
```

Tests cover numerical references, invalid inputs, persistence, provenance, comparison gates and planner restrictions. Mathematical fixtures and browser examples do not establish real-world accuracy; independent physical evaluation remains incomplete.

Read [the critical rethink](docs/PHONELAB_RETHINK.md) for alternatives, model roles, organiser-theme alignment, scientific limits and expansion candidates. [Architecture](docs/architecture.md) includes current and historical implementation evidence.

The official [event page](https://cerebralvalley.ai/e/meta-ai-developer-hackathon) includes scientific experimentation and physical systems in its themes. Private challenge statements and final judging weights remain unverified.

Public deployment, MCP integration, a completed real-footage accuracy study and the final hackathon submission are not included.
