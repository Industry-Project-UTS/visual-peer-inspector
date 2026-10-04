# Web integration verification

## Private SageMaker readiness checks

The application is installed in a persistent SageMaker Studio Space on a T4.
LocateAnything and SAM run in an isolated CUDA environment; DINO uses the exact
CPU environment required by the frozen encoder contract. The Space and app
server must both run. The authenticated Studio URL is private.

Current local validation: **144 passed, 1 skipped, 12 subtests passed**. The
skipped check requires a private feature cache. Deployment preflight passed;
its model/GPU fields describe the local code check, not the hosted model test.
The pipeline, scorer, model configuration and calibrated review policy remain
byte-identical to the base revision.

Hosted checks completed on selected inputs:

- Real bottles, two views: fresh Automatic selected the uncapped bottle and
  linked it in 2/2, confidence index 0.500. Confirmed Assisted selected the same
  bottle C in 2/2, confidence 0.286, reusing all six candidates without encoding.
  Both remained Review required. The review table's upload-order View labels
  and cleared stale export links were verified in this fresh hosted flow.
- Real bottles, three views: Automatic selected the uncapped bottle but split
  its evidence across tracks, with selected support 2/3. Assisted confirmation
  linked the same physical bottle in 3/3 and selected it; reranking reused all
  nine candidates and encoded none.
- Real scrub pads, three views: fresh Automatic inference selected the cut pad
  and linked it in 3/3, with confidence index 0.535 and Review required.
  Confirmed Assisted also selected that pad in 3/3, confidence 0.137, reusing
  all 15 candidates with no encoding.
- Curated apples, three views: Automatic selected the broken apple but linked
  it in 1/3. Correctly confirmed Assisted selected it in 3/3, confidence 0.145,
  Review required. An earlier incorrect correspondence entry produced a wrong
  selection; it is retained in the private notes and is not a valid scoring check.
- A previously used development scene completed fresh 2-, 3- and 5-view
  diagnostic checks. These checks establish execution, not correct selection.
- Cancellation preserved the uploads and restored controls. Separate browser
  sessions and disposable local tests checked session/cache isolation.
- Case ZIP, HTML and JSON downloads worked through HTTPS Studio links.
  Reopened case masks and reconstructed heatmaps matched original image sizes.
- Observed peak GPU allocation was 7,849 MiB for the initial
  three-view check; exact latency and maximum-upload capacity are unmeasured.

Interface fixes retain inactive tab controls and correction inputs during
reranking, leave the tab container unchanged by model callbacks, return one
complete result update, expose global progress and cancellation before work,
clear stale export links during work, use upload-order View labels in the
correction table, and restore inputs after success, cancellation or failure. Groq's optional
panel is removed. Restart helpers preserve an existing app, refuse foreign
listeners and refuse to stop active inference. Details and private report
screenshots remain in the working project's `development/sagemaker-deployment`.

These are selected engineering and demonstration checks, not a new accuracy
cohort. Confidence is a routing index, masks cover objects, and heatmaps are
coarse evidence. Automatic matching still needs review; assisted matching
requires explicit confirmation. No threshold was changed to relabel a weak
result as ready.

The curated demo inputs are two-view bottles and three-view scrub pads for both
modes, and three-view apples for Assisted. They were selected after observing
demo outcomes and must not be presented as an independent accuracy sample.
Compact aggregate evidence is recorded in
[`sagemaker-readiness.json`](../../reports/web-integration/safe/sagemaker-readiness.json).


## Earlier inspection case workflow checks

This update adds case context, an object board, an operator decision step kept
apart from the model result, portable case files with read-only reopening, and
linked follow-up inspections. Models, preprocessing, scorers, thresholds, the
review policy and evaluations are unchanged: `pipeline.py`, `decision.py`,
`frozen_policy.py` and the sealed policy were not edited, and the deployment
preflight reports the same pipeline fingerprint.

Checked locally:

- The focused web suite, including the new `tests/test_web_case.py`, passed
  **128 tests** with **1 skipped** (the private cached parity check). The new
  tests cover case creation and validation, decision validation and separation,
  no automatic approval, all-normal decisions, missing links, ties, unlinked
  observations, save/reopen fidelity and privacy, 19 malformed case files, size
  limits, rejection of a too-large export without replacing earlier downloads,
  follow-up linking without score comparison, report/JSON/case-file agreement,
  data-origin labels, and the UI callbacks end to end. Pinned pytest and scipy
  ran from a temporary environment layered over the app `.venv`, which was not
  modified.
- Offline deployment preflight passed.
- Browser run against a scratchpad-only backend that runs the real upload
  pipeline and frozen decision code on stand-in detection, masks and features.
  Every result was labelled "Synthetic test data". Three generated views went
  through Assisted suggestions, confirmed reranking (Result ready, one object
  highest), a correct/rework decision, saving and read-only reopening. A
  follow-up recapture in which all items matched produced tied scores, "No unique
  selection", Review required and an open case, which was saved without a
  decision. The saved reports kept the model and operator sections distinct.
- Real demo photos (bottles and scrub pads, three views each, duplicate excluded)
  passed upload validation, ordering and preview in the preview-only app. No
  inference ran, no result was produced, and the source hashes matched their
  manifest before and after.
- At 375 pixels the checked tabs had no document horizontal overflow; the
  machine and operator record cards stack.

Gradio 6 behaviour found during browser checks:

- Blur events on the case fields were not always delivered, so upload, inspect,
  decision, save, new-case and follow-up actions read the case fields directly.
- Starting a follow-up from **Saved cases** while switching tabs left later result
  updates unapplied. The follow-up now stays on **Saved cases** and tells the
  operator to open **Images**.
- `visible=False` on a Markdown notice in a tab not yet opened was not applied.
  The Decision notice changes its content instead; the existing Review notice
  shows the same behaviour and was left as it was.

At that earlier checkpoint, fresh GPU inference was unverified. The current
SageMaker checks above supersede that limitation for the selected photos only.
Uploads near the 100 MB case-file limit and operator usability remain unverified.

## Frozen policy and evidence update

The web app now bundles the byte-identical 09B `calibration-selective-v1`
policy. The package check verifies its sealed SHA-256. Synthetic 2/3/5-view
tests check automatic review routing, confirmed assisted reranking from cached
features, ties, support, thresholds and confidence parity with the evaluator
formula. JSON and HTML exports include the review signals, selected locations
and missing-link explanations. The focused local web suite passed **85 tests**
with **1 skipped** and **9 subtests passed**; offline deployment preflight
passed. These checks use generated images and do not establish new-upload
accuracy or hosted GPU readiness.

## Verified locally

- **71 app tests passed**, with the cached scientific parity test explicitly
  enabled: UI, transport, both decision modes, ties, identity constraints,
  candidate retention, original-image coordinates, session isolation, corruption,
  expiry, per-candidate invalidation, cancellation, runtime separation and exports.
- **36 remaining-workflow/notebook checks passed**, including original membership
  hashing, synthetic full-size 1000/600/400 caches, protected-stage guards,
  checkpoint resume and publication boundaries. Guard helpers use isolated
  fixtures; the 09E builder leaves future notebooks unchanged.
- **185 prior-stage and publication checks passed** for the included 08N/08O/08P
  code, including the configured sealed-pilot check and a real Jupyter kernel
  smoke test. The kernel test passed after allowing its local socket binding.
- Recorded model-stage replay exercised 2/3/5 views of one fixed development
  scene. It retained 10/14/24 candidates. Identity-only reranking made **zero**
  additional proposal, segmentation or encoding calls at all three view counts.
- A real four-crop CPU DINO subprocess produced exactly the same global and
  patch tensors as the sealed 08N cache (maximum absolute difference 0).
  The parent app environment had no torch installation. This checks the actual
  process boundary with the verified local snapshot, not model download behavior.
- Browser checks covered upload, Automatic inspection, fresh Assisted preparation,
  seeded neutral-ID suggestions, explicit confirmation, cached reranking,
  black-background evidence and HTML/JSON export. Recorded inputs were labelled.
- Desktop/narrow layouts were inspected; the 390-pixel layout had no document
  horizontal overflow. Oversized labels were replaced with compact consistent
  aliases. Full IDs and alias legends remain in tables/reports.
- The CPU automatic adapter was re-evaluated once on the existing 100 development
  scenes for the fixed primary 2/3/5 schedules. Predictions were sealed before
  opening the already-used evaluator. No new model inference or scorer search.
- Deployment preflight and the isolated packaged-decision import passed. The
  latter executes without importing torch, Transformers or SAM. The source
  package includes the selected scorer dependency tree and excludes QA replay
  adapters, datasets, checkpoints, private evidence and tests.

The combined PR validation total is **292 passed** across these three suites.
The tests use generated fixtures plus explicitly configured private cache checks; synthetic cohort-sized tests
are not real 2000-scene model evaluations. Earlier verification counts refer to
older source revisions and are not added to this total.

## Evidence

- [CPU development metrics](../../reports/web-integration/safe/cpu-primary-2-3-5/summary.md)
- [Real CPU encoder receipt](../../reports/web-integration/safe/real-cpu-dino-smoke.json)
- [Selected-route comparison](../../reports/web-integration/safe/selected_primary_top1.png)
- [Verified ClearML publication](../../reports/web-integration/clearml-publication.json)
- [Integration and next milestones](../../docs/progress/multiview-web-integration.md)

Private replay reports, model-stage call counts, per-scene predictions, evaluator
seals and browser-export checks remain under `.local/app-qa/`. The local replay
server is operator-configured QA and is excluded from deployment; it accepts
only its exact recorded development images.

## Remaining validation boundaries

- Linux Docker image execution: a Docker binary exists locally, but its daemon
  was unavailable.
- Uploads at maximum resolution/size, simultaneous-user load and exact latency benchmarks.
- Human-confirmed matching accuracy, arbitrary user-image accuracy and an
  independent protected-cohort result.

The default local app remains a preview unless an inference adapter is enabled.
The private SageMaker deployment explicitly enables inference. Selected hosted
smoke checks do not establish arbitrary-image accuracy or public access.
