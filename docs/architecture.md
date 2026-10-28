# PhoneLab architecture

## Current workbench update — 28 October 2026

The runnable entry point is now a responsive static workbench at `/`, served by
the existing FastAPI process. `backend/app/lab.py` exposes protocol selection,
bounded observation analysis for five families, local trial persistence, receipt
retrieval and descriptive trial comparison under `/api/lab`. Its fixed solvers
run locally; only the optional planner calls Muse Spark. Video point marking
and calibration happen in the browser without uploading footage. The separate
legacy free-fall uploader still uses SAM and Modal through the existing pipeline.

New bench receipts distinguish internal checks from independent validation and
label synthetic examples. See [the rethink decision record](PHONELAB_RETHINK.md)
for scope, scientific assumptions and current limits. The phase table and smoke
records below are historical records from the earlier implementation; their
"not started" UI entries do not describe the new workbench. The physical-data
validation gates remain open.

PhoneLab is planned as a trust layer between phone video and people or agents
that need a physical measurement. The intended flow is:

1. A local capture-quality gate rejects unsuitable footage before paid calls.
2. Voice transcription and keyframe classification identify the setup and
   experiment family.
3. A tracking service returns image-space trajectories.
4. Deterministic family-specific code computes results and uncertainty.
5. A certificate gate checks units, fit residuals, and agreement between two
   independent methods.
6. The system returns one shared physics-card schema to the frontend and MCP
   clients.

Models may classify, plan, critique, or narrate; they must not supply measured
values. Numerical analysis belongs in deterministic code.

## Current implementation

Implemented local modules:

- `backend/app/main.py` exposes `GET /healthz`.
- `backend/app/schema.py` defines the shared, validated `PhysicsCard` contract.
- `backend/app/families/` implements two independent numerical methods per
  family, including independent horizontal/vertical fits for projectile
  motion, pendulum-derived gravity, residual diagnostics, and residual-bootstrap
  uncertainty estimates.
- `backend/app/capture_coach.py` reads local video metadata, samples sharpness,
  checks frame-rate claims and resolution, enforces the SAM frame limit, and
  heuristically looks for the selected rectangular scale reference.
- `backend/app/keyframes.py` extracts evenly spaced local keyframes.
- `backend/app/certificate.py` checks family units/ranges, fit residuals, and
  independently recomputed cross-method agreement.
- `backend/app/cache.py` stores JSON-serializable pipeline results by video
  content hash, step, implementation version, and parameters.
- `backend/app/cost_tracker.py` records model name, consumed units, call type,
  run ID, cost, UTC daily totals, and per-run breakdowns.
- `eval/run_eval.py` compares saved physics cards with independent ground
  truth; it rejects empty datasets rather than presenting invented results.
- Docker Compose runs the backend and checks its health endpoint.

The family analyses are local pure functions and do not use a model to generate
numbers. Their coordinate conventions are explicit: free-fall vertical
position is positive upward; ramp distance is positive down the incline; angles
are radians. Oscillation and ramp fitting use SciPy; numerical operations use
NumPy.

## Gated work and validation status

The manual requires phase gates to pass on real data before later phases start.
The following items remain open:

| Gate or phase | Status |
| --- | --- |
| Phase 0: live SAM 3.1 image/video, Muse Spark, Voice Transcribe, and one open-weight smoke test | Service smoke tests passed; see the record below. Modal authentication passed; no Modal sandbox was launched |
| Capture coach: at least five real phone clips, including underlit and shaky footage | Not run; automated tests use a generated local video fixture only |
| Phase 1: end-to-end certified free-fall card and logged real-run cost | API path implemented, including SAM tracking and Modal analysis; not run on a PhoneLab clip. Real footage and independent ground truth are absent |
| Phase 2: local family methods | Implemented and unit-tested against hand-computed references |
| Phase 2: classification/critique loop and real cards for three families | Not run; provider integration and real clips remain gated |
| Phase 3: frontend, real evaluation set, open-weight path | Not started; no placeholder UI or accuracy numbers were added |
| Phase 4: Modal sandbox, MCP client test, real demo recording | Modal sandbox runner implemented and smoke-tested with a hand-computed trajectory; MCP client and real demo are not started |
| Phase 5: final submission README/NEW_WORK, public-repository and submission checks | Not started |

The residual and agreement thresholds in `backend/app/certificate.py` are
initial values only. Constant-velocity certificates additionally require a
high fit R-squared. These thresholds are not calibrated; tune them against real
evaluation clips before treating a card as certified in a demo or downstream
workflow.

## Provider access and project workflow

The playbook prioritizes resolving eligibility questions, confirming the actual
Meta API credit balance, provisioning a partner account, and checking the
official rules before investing in the full event build. Those are account- or
organizer-dependent actions and have not been performed or asserted here.
The live Phase 0 requests and observed response shapes are recorded below.
The first free-fall API path is implemented, but later phase work remains gated
on its real-footage validation. Use real clips/cards for evaluation and demos,
and keep a dated project history once this workspace is connected to its
public repository.

## Provider smoke-test record

Phase 0 endpoint smoke tests were run on 27 October 2026. These verify live
authentication and request/response shapes only; they are not PhoneLab
measurements, real-phone validation, or grounds to claim the pipeline works.

| Service | Live smoke result | Observed request and response |
| --- | --- | --- |
| Meta Model API authentication | Passed; `GET /v1/models` returned HTTP 200 and included Muse Spark 1.3 and SAM 3.1 | Bearer authentication with the existing local key |
| Muse Spark 1.3 text | Passed; Responses API returned `status=completed` and the requested one-word output | `POST /v1/responses`; `model` and `input`; output contains reasoning and assistant message items, with user-facing text in an `output_text` content part |
| Muse Spark 1.3 video | Passed on the 60-frame public test clip; response completed and described the time ball rising | `POST /v1/files` with `purpose=user_data`, then `POST /v1/responses` with an `input_video` part using `file_id`; successful response reported 538 input tokens and 480 output tokens |
| SAM 3.1 image | Passed on a Wikimedia Commons soccer-ball still; one box and one mask returned | `POST /v1/responses`, `model=sam-3.1`, `input_text` plus base64 data-URL `input_image`, streaming SSE; output arrives as `response.output_text.delta` containing SAM box/mask tokens and ends with `response.completed` |
| SAM 3.1 video | Passed on the first 60 frames of a public time-ball clip; 24 frame records, boxes, and masks returned | Upload MP4 through `POST /v1/files`, then stream a Responses request with `input_video.file_id`; SSE output contains one output-text record per returned frame |
| Muse Voice Transcribe | Passed on a 4-second locally synthesized speech sample; transcript was non-empty and matched the spoken phrase | Multipart `POST /v1/asr/transcribe` with JSON `request` and mono 24 kHz, 16-bit PCM WAV `audio`; JSON response has `transcript`, `audioDurationMs`, `turns`, and `sessionId` |
| Muse Spark Open via Baseten | Passed; model listing returned HTTP 200 and one request to `meta-models/muse-spark-open` returned a completion | `POST https://inference.baseten.co/v1/chat/completions`; standard bearer key and chat-completion request; response included `choices`, `usage`, and a string `message.content` |
| Modal credentials and sandbox | Passed; `modal token info` exited successfully, and one hand-computed trajectory returned 9.81 m/s^2 from a short-lived sandbox | Local analysis code was mounted into the image, network access was disabled, numeric input was passed as encoded JSON, and analysis plus fit diagnostics were returned as JSON; no secrets were injected |

Test media: the still was the CC0-licensed [Soccer ball on ground](https://commons.wikimedia.org/wiki/File:Soccer_ball_on_ground.jpg).
The video was [Time ball drop 10x speed](https://commons.wikimedia.org/wiki/File:Time_ball_drop_10x_speed.webm),
licensed CC BY-SA 4.0; it was transcoded to a short MP4 for the API smoke tests.
The transcription sample was generated locally with speech synthesis. Temporary
media was deleted after each test. None of these assets is a PhoneLab ground-
truth clip.

Cost evidence from these smoke tests: the single image costs $0.0025 at the
published SAM rate; the 4-second transcription costs about $0.0002; the
completed Spark video call used 538 input and 480 output tokens, approximately
$0.00271 at Standard pricing. A 60-frame SAM request should cost about $0.01
at the 50-frame minimum. One intentionally under-budgeted Spark video call
returned an incomplete result after 540 input and 64 output tokens,
approximately $0.00095. Some early text and Baseten access calls did not have
their token usage copied into the local output; their exact cost is unknown
and must be reconciled in the provider dashboard. The Modal sandbox smoke
test's compute charge is also not available from the local SDK response.
Future model clients persist response usage and estimated model cost before
returning; check provider billing dashboards for final charged amounts.

Phase 0's service smoke checks are complete. The real-clip gates remain open:
no PhoneLab footage, voice note, scale reference, or independently measured
ground truth is present in this workspace. Do not treat the public test clip or
the synthesized voice sample as passing the capture-coach, free-fall, or
evaluation gates.

## Physics card contract

`backend/app/schema.py` is the single source of truth for card fields. Its
models map to the example in the build manual, serialize enum values and
timestamps as JSON, and reject unknown fields. The `capture_quality.passed`
Python field serializes as the manual's JSON key `"pass"`.
