# Workbench validation — 28 October 2026

Scope: local implementation and integration checks, not physical accuracy.

- Original baseline: 31 tests passed when run from the project root.
- Final suite: `python -m pytest backend/tests -q` — 50 tests passed. One
  dependency deprecation warning from Starlette's HTTPX test client.
- `node --check backend/app/static/lab.js` passed.
- `git -C backend diff --check` passed.
- Chromium, 1440 × 1100: all five synthetic example workflows calculated and
  saved; five downloaded JSON receipts retained synthetic provenance and
  `externally_validated: false`. No browser JavaScript errors observed.
- Chromium, 390 × 844: no horizontal page overflow. Desktop and mobile rendered
  screenshots were inspected. Screenshots are local artefacts under `backend/data/`.
- Browser loaded the existing `newvid.mp4` locally, computed its fingerprint,
  accepted two calibration endpoints, marked eight points at 0.1 s playback
  intervals, undid one point and cleared observations after a scale change.
  These clicks tested UI mechanics; they were not measurements of the clip's
  actual object and were not submitted as physical evidence.
- Live Muse Spark routing selected `ramp_friction` for a sliding-surface question.
  Run: `9e2d8720-9581-4c8b-a199-caa0913c5966`; estimated cost $0.00208875.
- Baseten Spark Open returned no user-facing review text. See
  `model_design_review.json`. No successful second-model design review is claimed.
- An empty-response cost-logging regression was corrected and tested: returned
  usage is now recorded before rejecting empty Baseten text.

API regression cases include solver reference results for five families,
persisted-receipt round trips, incorrect chronology, insufficient observations,
missing ramp geometry, constant observations, unsupported family, missing video
fingerprint, synthetic comparison rejection, incompatible or duplicate trial
comparison rejection, unconfirmed assumptions, thrown-versus-released free fall,
input hashes covering setup changes, correlated-method uncertainty handling,
invalid planner output, unsupported questions and provider unavailability.

Not established: five-family automatic video tracking, complete uncertainty
propagation, real-world accuracy, repeated physical trials, calibrated certificate
thresholds, public deployment or hackathon submission. The older real-data gates
remain open. Synthetic examples remain visibly labelled in the local notebook.
