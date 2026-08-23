# Week 4 working record: selection audit and tracking

**Dates:** 17 to 23 August 2026  
**Status:** The 2,000-scene selection audit is complete. Model execution has not started from this notebook.

## Aim for this week

The aim was to turn the agreed 2,000-scene experiment into a checked and
portable data-preparation stage. The same notebook needed to run locally and
on Kaggle, keep protected records away from model code, and save a small
ClearML record without uploading private data.

## Work completed

- Kept the controlled 2,000-scene experiment instead of replacing it with the
  full 4,107-scene eligible pool.
- Confirmed the 1,000 development, 600 calibration, and 400 internal-lock
  roles.
- Checked that one base asset does not cross experiment roles.
- Checked the nested 100-scene screen and separate 300-scene confirmation set.
- Ran the clean notebook locally and on Kaggle Version 4.
- Compared the deterministic local and Kaggle outputs.
- Saved the clean notebook, executed notebook, tables, figures, run record,
  artifact hashes, and Kaggle metadata.
- Recorded the accepted Kaggle run in ClearML using aggregate evidence only.

## Main results

| Measure | Result |
| --- | ---: |
| Eligible scenes | 4,107 |
| Selected scenes | 2,000 |
| Selected base assets | 545 |
| Development scenes | 1,000 |
| Calibration scenes | 600 |
| Internal-lock scenes | 400 |
| Development screen | 100 |
| Development confirmation | 300 |
| Views per scene | 20 |
| Indexed selected source paths | 238,000 |
| Official test scenes selected | 0 |
| Base assets shared across roles | 0 |

All audit gates passed. The repeated selection build was byte-identical. The
accepted local and Kaggle aggregate files matched.

## ClearML record

The accepted task is `selection-audit-kaggle-v4`, with task ID
`ce9b220344b2435fac60d8e180c37fdd`.

ClearML stores the safe run configuration, aggregate metrics, and hashes. It
does not store raw images, masks, source paths, source identifiers, private
labels, crosswalks, model weights, or credentials.

## Evidence saved in the repository

| Location | Purpose |
| --- | --- |
| `notebooks/data-preparation/toysad8k-selection-audit/toysad8k-2000-selection-audit.ipynb` | Clean Kaggle source notebook |
| `notebooks/data-preparation/toysad8k-selection-audit/toysad8k-2000-selection-audit-executed.ipynb` | Saved Kaggle notebook with outputs |
| `notebooks/data-preparation/toysad8k-selection-audit/kernel-metadata.json` | Kaggle notebook and input metadata |
| `reports/data-preparation/toysad8k-selection-audit/kaggle-v4/` | Reviewed output bundle from the accepted run |
| `reports/data-preparation/toysad8k-selection-audit/README.md` | Short guide to the audit and its data boundary |
| `requirements/selection-lock.txt` | Selection code dependencies |
| `requirements/selection-notebook-lock.txt` | Notebook dependencies |
| `requirements/tracking-lock.txt` | ClearML dependency |

## What this audit does not prove

- It does not prove that LocateAnything finds every object.
- It does not prove that SAM 2.1 produces accurate masks from predicted boxes.
- It does not select the final visual encoder or anomaly scorer.
- It does not report final anomaly-detection accuracy.
- It must not be attached to model-facing Kaggle jobs because it uses trusted
  preparation records.

The next controlled stage is a separate opaque 20-scene, all-view package.
Only that model-safe package can be mounted by the LocateAnything notebook.
