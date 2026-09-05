# DS1 prompt freeze and DS2 confirmation runbook

## Purpose

This runbook completes the next evidence stage without changing the frozen
2,000-scene selection. The DS1 screen is used for one bounded two-prompt
LocateAnything comparison. The untouched 300-scene DS2 cohort then confirms the full
RGB-to-result pipeline.

There is no hard proposal-accuracy cutoff. The decision reports the complete
metric profile and qualitative failures.

The model-facing confirmation path must include all three routes:

- supplied masks as the scientific upper-bound route;
- reference or user-confirmed boxes passed to SAM 2.1 as the assisted route;
- LocateAnything boxes passed to SAM 2.1 as the automatic route.

DINOv2-small, B0 and P1 remain fixed during DS2 confirmation. Three-view
mean-rank fusion is a protocol-declared application policy, not a DS1-selected
result. Other view counts and visibility weighting are descriptive ablations.
DS2 results must not be used to silently promote the best DS2 setting.

## Run order

Before a Kaggle run, build a code-only snapshot with
`scripts/build_kaggle_evaluation_snapshot.py`. Until the source is committed,
use `--allow-dirty` and label every result `precommit_candidate`. A clean final
evidence run can be produced later from the committed source.

### 1. Notebook 04.3: bounded DS1 LocateAnything prompt and policy freeze

Run on one Kaggle T4 GPU using the existing DS1 model-safe RGB package and the
DS1 evaluator package. Prompts and generation settings must be declared before
labels are opened. Load LocateAnything once and reuse the loaded model for the
additional prompt. Compare `object` and `individual object`. A prompt that
fails the one-image gate must stop without consuming the full screen.

Required stages:

1. verify snapshot and package hashes;
2. record the Kaggle image, Python, CUDA and GPU versions;
3. load the pinned LocateAnything revision with visible checkpoints;
4. complete one-image and 20-image gates;
5. generate and hash predictions for the complete DS1 screen;
6. open DS1 evaluator data only after prediction hashes are frozen;
7. compare precision, recall, F1, F2, complete-image rate, matched and
   detection-aware IoU, misses, extras, runtime, and representative successes
   and failures; and
8. save one frozen prompt and post-processing decision with its configuration
   hash.

There is no hard numerical acceptance gate. Prefer the most useful deployable
trade-off and record the assisted correction fallback when misses remain. The
candidate policies are raw proposals, area filtering with NMS at 0.75, and the
same cleanup followed by a label-free eight-proposal outlier safeguard.

### 2. Notebook 08A: DS2 confirmation release

The completed Kaggle Version 5 release has already verified 300 scenes, 20
views per scene and 6,000 RGB images. Reuse its private RGB, geometry and
evaluator resources after checking their package hashes. Do not rerun the full
archive scan unless a published package is missing or its identity differs.

Notebook 08B must not receive the evaluator-only resource.

### 3. Notebook 08B: complete frozen model-facing pipeline

Run on one Kaggle T4 GPU with only code, the frozen prompt decision, RGB and
supplied geometry attached. Internet is required for pinned model downloads.
The model revision, prompt and post-processing remain the DS1 freeze. The DS2
operational policy is separately versioned and hashed: 4,096 new tokens, at
most two attempts per image, a 75-second timeout per attempt, and a terminal
failure carried forward as an explicit zero-proposal row.

Run the GPU models sequentially so they do not compete for memory:

1. load LocateAnything once and write one terminal result for every image;
2. release the LocateAnything model and GPU cache;
3. load SAM 2.1 once, pass its one-image gate and generate automatic-box and
   reference-box masks;
4. release SAM 2.1 and the GPU cache;
5. load the pinned DINOv2-small revision and calculate B0 and P1 for supplied,
   assisted and automatic routes; and
6. save restartable shards, aggregate tables, report charts, box examples,
   mask overlays, P1 heatmaps and sanitized failure evidence.

An image is covered when it has a terminal row, not only when it has a box.
Failed generations count as misses in Notebook 08C. Every manifest must record
the original DS1 freeze hash, original configuration hash, effective DS2
configuration hash and effective operational-policy hash.

Do not reinstall Kaggle's PyTorch, NumPy, Pillow, pandas, matplotlib or SciPy.
Install only `requirements/confirmation-lock.txt` and record the container
versions that were actually used.

### 4. Notebook 08C: protected confirmation and multi-view evaluation

Run on CPU with the completed Notebook 08B version attached as a Kaggle kernel
source. Notebook 08C verifies the producer checksums before extracting the
three downstream archives. Attach evaluator-only attributes only to this
notebook. A separate prediction-output dataset is not required.

The evaluator verifies frozen prediction hashes before opening labels. It then
reports:

- LocateAnything precision, recall, F1, complete-image rate, misses and extras;
- SAM mean IoU, Dice, boundary F1 and recall at IoU 0.75;
- B0 and P1 for supplied, assisted and automatic routes;
- paired, asset-clustered confidence intervals;
- 1, 2, 3, 5, 10 and 20 camera-spread view results;
- mean-rank and visibility-weighted fusion;
- one-anomaly and two-anomaly measures;
- category, anomaly-family, object-count and anomaly-count subgroups;
- rank disagreement and confidence margin; and
- private success and failure panels.

The confirmation record states the automatic route, assisted fallback,
encoder, scorer, prompt, post-processing, primary view count and fusion rule
that were declared before evaluator access. Automatic cross-view association
in 08C is evaluator-supported after prediction lock. It does not validate the
label-free identity method required by the web application. A negative result
must be retained and explained.

## Kaggle execution safeguards

- Discover inputs only through known Kaggle mount shapes and marker files.
- Never recursively scan `/kaggle/input`.
- Print checkpoints for dependency setup, model download, model load, GPU
  transfer, one-image inference, pilot completion and every fixed batch.
- Reuse the worker loaded for the one-image gate. Do not start another process
  that reloads the same model for the full run.
- Use one T4 unless multi-GPU execution has been implemented and tested.
- Write heavy intermediate outputs below `/kaggle/temp`.
- Use deterministic shards, atomic writes and a completed-item manifest.
- Resume only from shards whose input and configuration hashes still match.
- Stop on a package, membership, checksum, prediction-lock or privacy failure.
- Do not stop only because the Kaggle output counter still shows zero bytes.

## Compact output contract

Run `scripts/package_kaggle_stage_output.py` only after the stage verifier and
artifact manifests pass. Put its output in `/kaggle/working`. Each stage should
persist only a small set of files:

- `<stage>-repository-evidence.zip` for aggregate GitHub candidates;
- `<stage>-review-evidence.zip` for explicitly selected private figures and
  executed evidence;
- `<stage>-downstream-input.zip` only when another notebook needs full output;
- `<stage>-verification.json`; and
- `SHA256SUMS`.

Keeping the expanded result tree in `/kaggle/temp` prevents the Kaggle CLI from
enumerating thousands of small files.

After a completed version, inspect and download only the compact files:

```bash
kaggle kernels status OWNER/KERNEL/VERSION
kaggle kernels files OWNER/KERNEL/VERSION --format json --page-size 200
kaggle kernels output OWNER/KERNEL/VERSION \
  --path LOCAL_OUTPUT_DIRECTORY \
  --file-pattern '(^|/)(.*-repository-evidence\.zip|.*-review-evidence\.zip|.*-downstream-input\.zip|.*-verification\.json|SHA256SUMS)$' \
  --page-size 200
kaggle kernels pull OWNER/KERNEL/VERSION \
  --path EXECUTED_NOTEBOOK_DIRECTORY --metadata
```

Verify `SHA256SUMS`, test every archive and run the stage verifier locally
before using an output downstream or preparing a transfer package.

## ClearML boundary

ClearML starts only after the scientific files, manifests and compact bundles
are complete. The optional dependency is pinned separately in
`requirements/confirmation-tracking-lock.txt`.

The isolated finalizer receives only the validated `safe` directory. It may
log aggregate scalars, CSV tables, PNG charts, safe JSON records, configuration
hashes and artifact hashes. It must not receive RGB images, masks, boxes,
object IDs, scene IDs, per-scene scores, labels, paths or credentials.

ClearML installation, initialization and uploads have bounded timeouts. A
tracking failure writes a sanitized receipt but does not change a completed
scientific result into a model failure. The same safe folder may be finalized
locally later if the Kaggle secrets or tracking service are unavailable.

## Local verification and transfer boundary

Inspect at least 12 varied qualitative examples across categories, view
positions, crowded scenes, small objects, successes and failures. Verify all
counts, hashes, schemas, run records, figures and ClearML receipts.

Later, one combined transfer package may contain:

- `repository-files/` with clean notebook sources, reusable code, tests,
  dependency locks, progress notes and safe aggregate reports;
- `review-only/` with executed notebooks, logs, selected private panels,
  verification notes and ClearML receipts;
- `LOCAL_UPLOAD_STEPS.md`; and
- `CHECKSUMS.sha256`.

Raw data, complete masks, per-image predictions, credentials, model caches,
downstream archives, instruction files and the transfer ZIP stay outside the
GitHub repository.

## Completed run record

- Notebook 04.3 Version 4 froze `individual object` with
  `area_nms_075_cap8`.
- Notebook 08A Version 5 verified the 300-scene DS2 release.
- Notebook 08B Version 4 completed the frozen model pipeline on one Kaggle T4.
  It processed 6,000 images, retained 23,180 trusted proposals, produced both
  SAM routes, and wrote 138,760 B0 and P1 score rows.
- Notebook 08C Version 1 completed the protected CPU evaluation in 955 seconds.
  It kept the declared DINOv2-small, P1, three-view, and mean-rank policy.
- Every stage bundle passed its SHA-256 inventory and ZIP integrity check after
  download.
- ClearML finalisation ran only from verified aggregate safe folders. Private
  examples stayed in review-only evidence.

Notebook 08C is a development confirmation result. It did not open the
calibration or internal-lock roles and it was not used to retune the frozen
models.

## Evidence for the next supervisor update

Use completed Kaggle evidence, not local contract values. Include the DS1
prompt comparison, DS2 integrity table, automatic proposal results, assisted
and automatic SAM quality, B0 versus P1, route comparison, view-count curve,
fusion comparison, lowest supported subgroups, real success and failure cases,
runtime evidence and the final pipeline decision.
