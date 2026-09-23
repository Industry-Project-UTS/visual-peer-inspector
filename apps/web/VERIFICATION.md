# Web integration verification

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

## Not yet verified

- Linux Docker image execution: a Docker binary exists locally, but its daemon
  was unavailable.
- Fresh LocateAnything/SAM GPU inference through the enhanced upload adapter.
- GPU memory, full-pipeline latency, hosted access and restart behavior.
- Human-confirmed matching accuracy, arbitrary user-image accuracy and an
  independent protected-cohort result.

The default app still opens as an interface preview unless an inference adapter
is explicitly enabled. Passing CPU tests or publishing ClearML artifacts does
not establish those remaining results.
