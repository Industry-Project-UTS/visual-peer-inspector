# ToysAD-8K 2,000-scene selection audit

This folder preserves the checked outputs from the controlled 2,000-scene
selection audit. The audit was run after the full-dataset EDA and before any
model-facing package was created.

The completed private Kaggle notebook is:

<https://www.kaggle.com/code/esreekarreddy/toysad-8k-2-000-scene-selection-audit?scriptVersionId=344254160>

## What was checked

- 4,107 leakage-safe, train-derived eligible scenes
- 2,000 selected scenes across 545 base assets
- 1,000 development, 600 calibration, and 400 internal-lock scenes
- no official seen-test or unseen-test scene in the selection
- no base asset shared across experiment roles
- all five object and anomaly strata represented in each role
- all 39 eligible categories represented overall
- 238,000 required source paths indexed and non-empty
- byte-identical repeated selection build

The audit passed every declared gate. The 2,000 scenes remain the primary
experiment. The other eligible scenes are reserved for optional post-freeze
sensitivity work.

## Run verification

Kaggle Version 4 completed successfully. The deterministic aggregate outputs
matched the verified local run. The clean and executed notebook sources also
matched.

The accepted ClearML task is `selection-audit-kaggle-v4`, with task ID
`ce9b220344b2435fac60d8e180c37fdd`. ClearML contains safe aggregate metrics,
configuration details, and hashes only.

## Folder contents

The `kaggle-v4` folder contains the reviewed tables, figures, aggregate result,
run record, and artifact manifest from the accepted run.

Useful entry points are:

- `selection_aggregate.json`: headline counts and gate result
- `selection_gate_status.csv`: individual gate outcomes
- `selection_summary.csv`: eligible and selected corpus totals
- `role_counts.csv`: development, calibration, and internal-lock sizes
- `strata_by_role.csv`: object and anomaly strata by role
- `development_pilot_counts.csv`: the 100, 300, and remaining development sets
- `selected_media_inventory.csv`: indexed selected-media totals
- `artifact_manifest.csv`: output sizes and SHA-256 values
- `run_record.json`: environment, provenance, and tracking record

## Data boundary

This repository stores code, tests, notebook evidence, compact tables, figures,
and hashes. It does not store the raw ToysAD archive, extracted images, private
crosswalks, source identifiers, calibration labels, internal-lock labels,
credentials, model weights, or feature caches.

Notebook 02 is a trusted preparation notebook. Its inputs and report tables
must not be attached to LocateAnything, SAM, or representation jobs. Those jobs
receive a separate opaque media package with no labels or source records.
