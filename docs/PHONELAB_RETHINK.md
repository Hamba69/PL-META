# PhoneLab: a repeatable experiment bench

Decision and implementation record — 28 October 2026.

## The strongest idea

Build a phone-based **experiment workbench for schools and small workshops**.
The unit of value is a controlled experiment, not an impressive number extracted
from a clip. The loop is: question → approved protocol → calibration → observations
→ two computational cross-checks → inspect evidence → repeat with one changed
condition → compare → export a reproducible receipt.

The opening problem: a learner or small workshop can film motion but often lacks
the instruments, protocol, and analysis workflow to turn it into a defensible
comparison. A phone can lower that barrier for selected observable processes.
It cannot replace instruments for quantities its camera does not measure.
This is a product hypothesis, not a claim backed by user interviews or field trials.

The best demonstration is **one sliding block, two surfaces, one measured ramp
angle**. Ask whether changing the surface changes kinetic friction. Show the
calibration, inspect points, calculate both estimates, repeat each condition, and
compare the receipts. Include an intentionally pushed or rolling object as a
failure case. Do not label the surface safe or unsafe from this experiment.

## What the organisers actually say

The official [Meta/Cerebral Valley event page](https://cerebralvalley.ai/e/meta-ai-developer-hackathon),
checked on 28 October 2026, describes two themes: solving a problem in a domain
you know and extending what models can do with physical systems and scientific
experimentation. It describes the Indie track in terms of creativity, technical
ambition, and working execution. These are broad themes, not a supplied list of
specific challenge statements. No additional committee problem-statement document
was attached to this task or found in the project.

PhoneLab maps to those themes through affordable practical experimentation,
visible model-assisted protocol selection, and reproducible numerical outputs.
The exact final judging weights and any private Discord challenges remain
unverified. This design does not depend on the secondary-source weight split in
the older playbook. That playbook also contains historical provider and credit
information; it should not be treated as current account or pricing evidence.

## Alternatives considered

| Direction | Strength | Critical weakness | Decision |
| --- | --- | --- | --- |
| Universal camera scientist | Broad and visually exciting | Underdetermined measurements; encourages made-up quantities and unsafe extrapolation | Reject as the core promise |
| A polished free-fall demo | Easy to explain and narrowly test | Small practical scope; existing filmed run failed its gates | Retain as one bench |
| Automated industrial inspection | Potential economic value | Needs domain tolerances, representative data, calibration and deployment evidence | Later, with a specific partner |
| Protocol-led multi-experiment bench | Reuses working solvers; combines accessible capture with reproducibility and comparative questions | Manual capture is slower; real validation still needed | Implement now |

The creative step is **changing one condition and following the evidence**, not
adding a long menu of unsupported visual calculations. A useful later feature is
an experimental-design assistant that asks which next observation could distinguish
two explanations. It must work within approved protocols and cannot invent a
physical result or automatically certify its own proposed method.

## The scientific boundaries that change the design

1. Two estimators on one trajectory share camera, calibration, tracking, and timing
   errors. Agreement is an internal consistency check, not independent validation.
   New bench results say `checks_passed` or `review_required`, never externally
   validated. The legacy free-fall card retains its existing schema and terminology.
2. Existing residual-bootstrap uncertainties describe fit variability, not the
   uncertainty of the entire measurement. The new bench conservatively retains
   the larger method uncertainty instead of shrinking it as if the methods were
   independent. This still does not include calibration, perspective, angle or
   timebase uncertainty. Even zero residual scatter does not mean perfect accuracy.
3. Free-fall endpoint kinematics assumes release from rest at the first observation.
   The existing projectile module inherits that assumption and is therefore not
   exposed as a general projectile instrument. It needs an independently checked
   initial-velocity treatment before that claim is made.
4. A scale must be in the same plane as the tracked point. A bounding-box long edge
   is not automatically the physical length of a tilted reference. The manual
   bench allows explicit scale endpoints; it cannot infer coplanarity from a click.
5. Playback seconds may differ from physical seconds in slow-motion footage. The
   marking UI records an explicit time factor, with a reminder to use capture time.
6. A sliding block and a rolling ball require different dynamics. Friction analysis
   requires measured angle, positive-down-ramp displacement and release from rest.
7. A falling liquid level is not generally exponential, and a moving particle does
   not determine volumetric flow without geometry and velocity-profile assumptions.
   Exponential analysis accepts positive baseline-corrected measured signals only;
   it requires a halving or doubling and does not claim concentration or temperature.
8. Passing thresholds is not a field-validation study. The repository's prior
   `newvid_clips/provider_measurement_report.md` reports failed free-fall checks;
   it is useful failure evidence, not a successful lab measurement.

## Available model resources: useful roles and observed limits

| Resource | Appropriate role | Current evidence / decision |
| --- | --- | --- |
| Muse Spark 1.3 | Route a question to a finite, reviewed protocol catalogue | New live `/api/lab/plan` call selected `ramp_friction`; run `9e2d8720-9581-4c8b-a199-caa0913c5966`; estimated cost $0.00208875 using the existing client rate assumptions |
| SAM | Extract target/reference observations for review | Existing automatic free-fall path retained; no new five-family automatic-tracking claim |
| Voice transcription | Reduce typing burden for setup descriptions | Existing optional backend path retained; not necessary for the offline bench |
| Spark Open via Baseten | Optional critical review / planning alternative | Attempt returned no user-facing text; recorded in `model_design_review.json`. No successful review claimed |
| Modal | Isolate execution of analysis in the provider pipeline | Existing automatic path retained; new bench executes only fixed local solvers, never model-generated code |
| Wonder / additional models | Potential investigation of new protocols | Not needed to prove this prototype; do not add cost or an unreviewed numerical path merely to use more models |

The failed Open-model call did not yield usable review text. Its charge was not
recoverable from the old client after the error; provider billing is authoritative.
The client now logs returned token usage before rejecting an empty response so
future failures remain cost-visible. No current provider-price verification is
claimed. The successful Spark planner selects an allowlisted ID only: it cannot
insert invented measurements, code, or extra protocol fields into the result.
If planning fails, the catalogue and local analysis remain available.

## Implemented scope

- A responsive browser workbench served by FastAPI at `/`, with no extra frontend
  service, CDN, account setup, or JavaScript package install.
- Five protocols with questions, acquisition steps, assumptions, use cases and limits.
- Optional model routing from a question; unsupported questions return no protocol.
- Local video playback, directional scale calibration and manual point marking.
  Video bytes stay in the browser for this workflow. The source SHA-256, calibration
  notes and numeric observations accompany a saved trial.
- Pasted observation input for independently measured values. Browser/manual
  provenance is user-supplied; it is not a tamper-proof chain of custody.
- Fixed numerical solvers, diagnostics, method comparison, fit variability and
  explicit review flags. Finite values, bounds, chronology and geometry are checked.
- Local notebook, per-trial JSON receipts, observed-data plots and descriptive
  comparisons of compatible passing trials. Synthetic and rejected trials cannot
  support comparison. Differences are not statistical significance claims.
- Explicit synthetic examples to explore the UI without credentials or footage.
- The existing provider-based free-fall upload remains accessible from a separate
  section; that action clearly identifies upload and provider usage.

The bench is a local, single-user prototype. No authentication, public hosting,
multi-user isolation, MCP integration, automatic general video analysis, final
submission, or completed accuracy evaluation is implied.

## How extensive it can responsibly become

**Near-term, after real footage validation:** reviewed SAM observations across the
five protocols, frame-timestamp preservation, tracking edits, repeated-trial
summaries, full calibration uncertainty, residual plots and comparison against
independent instruments. Teacher-authored protocols can become reusable practicals.

**Next protocols, each with its own validation:** spring stiffness with known loads
and calibrated static displacement; restitution from clean pre/post-impact tracks;
volumetric fill rate from calibrated vessel geometry; damping from an observed
envelope over enough cycles; multi-object momentum with measured masses. These
are candidates, not implemented features or quantities a vision model may guess.

**Further instrument-assisted work:** colorimetry with reference standards and
controlled lighting; microscopy with known optics and scale; thermal experiments
with independent temperature sensors. A phone becomes the capture, coordination,
analysis and notebook interface, while the needed sensor remains explicit.

Avoid universal chemistry, water-potability, clinical diagnosis, structural-safety
or product-certification claims from unconstrained video. These are different
measurement and validation problems, not additional prompts.

## Validation and demonstration plan

Automated solver/API tests check implementation behaviour against mathematical
fixtures. Browser tests exercise example calculations and receipt downloads.
A live Spark call checks provider integration. None replaces physical validation.

For a credible real demo: capture at least five varied clips, including poor light
and camera movement; build an independently measured set of 15–20 clips; record
reference instrument, uncertainty, original capture timing, scale, release point
and exclusions; report failure rate as well as error. For the ramp story, repeat
each surface condition and show the data before making a conclusion. Thresholds
must be calibrated on representative footage, not tuned to make a single clip pass.

The new implementation proceeds under the user's explicit request to rethink and
execute. It does not mark the older manual's physical-data gates complete.
