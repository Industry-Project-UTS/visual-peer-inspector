# Full development results

The automatic model stack is frozen: LocateAnything-3B, SAM 2.1 Hiera-large,
DINOv2-small and P1. Notebook 08E Version 5 processed the full development
role, which contains 1,000 scenes with 20 views each. The primary experiment
still contains 2,000 scenes, split into 1,000 development, 600 calibration and
400 internal-lock scenes. No split or model setting changed during recovery.

## Completed evidence

- Notebook 08D Version 2 released the remaining development RGB inputs.
- Notebook 08E Version 5 completed in 3 hours, 6 minutes and 31 seconds.
- All 1,007 registered output files passed downloaded SHA-256 checks.
- Three prediction locks cover all 20,000 development views.
- Ninety review figures contain predicted boxes, SAM outlines, binary masks,
  masked objects on black, and P1 heatmaps projected into scene coordinates.
- The downloaded executed 08D and 08E notebooks match their original cell
  sources and contain no error outputs.
- Three ClearML tasks were repaired and verified after the run. This did
  not repeat model inference.

Completion means that every view has a recorded outcome. It does not mean
every object was detected correctly. The screen run contains 17 zero-proposal
views and the remainder run contains 69. The confirmation feature stage
propagates 38 zero-proposal outcomes from its reused detections. The remainder
DINO stage also records two stage failures. Preserve these in the analysis.

## What the images show

Some cupcake examples have plausible boxes and masks, but the highest-ranked
object changes between views. Other examples include background proposals or
missed objects. Binary SAM masks and objects on black are present. These are
prediction-only diagnostic examples, not a ground-truth accuracy result.

Notebook 08F now produces cross-view odd-one-out results. Its images also show
why an automatic result must not be confused with a correct result. A chosen
identity can appear in only one view, leaving the other mask panels empty.
The separate SAM evidence panels still show all detected objects in every
view. Missing winner panels are not missing downloads.

## Identity release and comparison

08E.1 Version 4 completed the reference release for all 1,000 development
scenes. It contains 80,954 predicted-object observations, of which 74,483
have a reference match and 6,471 do not. This is reference matching coverage,
not detection recall. The corrected decoder uses the established grayscale
threshold of 128. All 23,179 correspondence records covered by the earlier
300-scene reference package passed an independent comparison. Version 3 is
rejected diagnostic evidence and must not be used for evaluation.

08F Version 1 completed on CPU using the saved predictions and embeddings.
It produced 6,000 identity predictions, 2,000 additional missing-view cases,
3,000 application results, and 30 review figures for 15 scenes. All 12
structural gates passed. All 16,056 archived files were verified, and 40,416
application mask records were decoded and checked against their hashes.

The selected development baseline is embedding-only matching. Adding the
tested normalized-geometry term reduced ambiguity but reduced coverage and
increased unmatched observations. It did not justify the added method.

| Views | Mean pairwise identity accuracy | 95% clustered interval | Eligible scenes |
| --- | --- | --- | --- |
| 2 | 36.44% | 34.02% to 38.98% | 932 |
| 3 | 33.60% | 32.11% to 35.09% | 974 |
| 5 | 33.04% | 31.92% to 34.16% | 989 |

These are averages of per-scene link accuracy over scenes with evaluable
links. They are not odd-one-out accuracy, scene AP, or accuracy over every
possible object pair. Uncertainty is clustered by base asset.

The five-view result has 93.29% mapped-observation linking coverage, but only
33.04% link accuracy. Coverage does not establish that the links are correct.
Reversed-order checks, duplicate rejection and the missing-view tests were
stable, but stability also does not establish correctness.

In 413 of the 1,000 five-view results, the chosen identity appears in only one
view. Of those, 336 still have agreement equal to 1.0 because agreement is
calculated over the views supporting that identity. It is not a confidence
probability or evidence that all five views agree. The current provisional
review state recommends review for 900 five-view results.

The method and output-contract artifacts are frozen for reproducibility.
They do not establish readiness for a reliable fully automatic web app.
The current implementation can run and produce useful diagnostics, but
physical-object identity matching remains a clear limitation.

## Saved evidence and tracking

The original executed notebooks for 08D V2, 08E V5, 08E.1 V4 and 08F V1
match their saved code snapshots and contain no error outputs. Source
snapshots 55, 59 and 65 are preserved. The repository copies retain aggregate
tables and charts while private scene images remain in the review package.

ClearML has six completed tasks: one primary release, three development
cohorts, one identity-reference release and one identity comparison. Tables,
charts and run records were uploaded after computation where tracking had
been disabled. Remote artifacts were checked. No inference was repeated
for tracking.

## Runtime corrections

The shared code now handles nested Kaggle mounts, preserved recovery ZIP
bytes, ZIP and extracted source layouts, bounded post-run tracking, atomic
scene records and recovery after infrastructure failures. Source checks
reject changed indexes, corrupt members and unsafe paths. Corrected
dependency metadata uses the notebook URL returned by Kaggle.
Resource discovery skips the full raw-image mount and saved checkpoints.
The identity workflow reports progress every 25 scenes and saves a verified
download archive after completion.

The corrected implementation passed 548 tests and 29 subtests.
Ten actual source images also passed the extracted-layout check. Tests reduce
repeat failures; they cannot guarantee that an external service will never
fail.

## Next steps

1. Review the low identity accuracy and single-view winner cases with the
   team. Keep the completed comparison as evidence, including its failures.
2. If matching or fusion changes, test it only on development data and create
   a new versioned method lock. Do not tune using protected labels.
3. After the method decision, continue with 08F.1, 09A and 09B for calibration
   inputs, frozen inference and review-policy calibration.
4. Continue with 09B.1, 09C and 09D for the one-shot internal-lock evaluation,
   then the existing official-evaluation sequence.

Calibration and internal-lock resources remain closed. The transfer contains
the completed development set, not completed protected evaluation or a web
deployment. No new PR is part of this packaging step.
