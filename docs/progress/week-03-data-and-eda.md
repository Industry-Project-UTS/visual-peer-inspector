# Week 3 working record: data and exploratory analysis

**Dates:** 10 to 16 August 2026
**Status:** Full-dataset EDA completed. Model pipeline gates remain open.

## Aim for this week

The aim was to understand the complete ToysAD-8K dataset before choosing the
scenes used for model development. We also wanted one shared Kaggle source that
both team members could access and a reproducible record of the analysis.

## Work completed

- Uploaded the complete private ToysAD-8K source to the shared Kaggle group.
- Ran the full EDA notebook on Kaggle using CPU execution with internet
  disabled.
- Audited all 7,396 catalogue scenes and all 1,161,604 indexed files.
- Checked the headers of all 1,139,416 indexed PNG and EXR members.
- Saved the clean notebook, executed notebook, output tables, figures, run
  record, artifact hashes, and execution log.
- Compared the Kaggle results with the earlier local reference. All 12 count
  checks passed and 20 directly comparable CSV files were byte-identical.
- Retained the earlier dataset baseline as historical work. It is not the
  current project method.

## Main EDA results

| Measure | Result |
| --- | ---: |
| Catalogue scenes | 7,396 |
| Categories | 51 |
| Base assets | 974 |
| Objects | 27,290 |
| Anomalous objects | 8,482 |
| Availability-ready scenes | 7,373 |
| Indexed files | 1,161,604 |
| PNG and EXR members checked | 1,139,416 |
| Valid image headers | 1,139,405 |
| Zero-byte image files | 11 |
| Eligible train-derived scenes | 4,107 |
| Eligible base assets | 545 |
| Eligible categories | 39 |

The full Kaggle run took about 3 hours and 16 minutes. Most of the runtime was
spent opening and checking image files on the Kaggle dataset mount.

## Important findings

- Every catalogue scene contains at least one anomalous object. ToysAD-8K does
  not provide a native all-normal scene set.
- Anomalous objects appear first in the source object order. Object order,
  source names, and source paths would leak the answer and must not reach the
  model.
- The supplied segmentations are whole-object masks. They are not pixel-level
  annotations of the anomalous part.
- Each scene has multiple camera views. These are repeated views of the same
  scene and must not be counted as independent samples.
- Official test pixels were not displayed or interpreted during EDA. Their
  file headers and aggregate storage counts were included in the audit.

## Decisions carried forward

1. The EDA describes the full 7,396-scene catalogue.
2. The 4,107 scenes are the complete leakage-safe, train-derived eligible pool.
3. The primary controlled experiment remains the planned 2,000 scenes. The
   availability of 4,107 eligible scenes does not by itself justify changing
   the agreed experiment.
4. The 2,000 scenes must remain grouped by base asset and divided into
   development, calibration, and internal-lock roles.
5. The earlier 40-scene, one-view DINOv2 baseline remains useful as a regression
   check only. It does not decide the final backbone or pipeline.
6. LocateAnything can be evaluated as a box-proposal stage before prompted SAM
   2.1. Its output must be tested together with SAM before it becomes part of
   the fixed architecture.

## Evidence saved in the repository

| Location | Purpose |
| --- | --- |
| notebooks/eda/toysad8k-full-dataset/toysad-8k-full-dataset-eda.ipynb | Clean Kaggle source notebook |
| notebooks/eda/toysad8k-full-dataset/toysad-8k-full-dataset-eda-executed.ipynb | Saved Kaggle notebook with outputs |
| notebooks/eda/toysad8k-full-dataset/kernel-metadata.json | Kaggle notebook and input metadata |
| reports/eda/toysad8k-full-dataset/kaggle-v1/ | Complete 36-file Kaggle output bundle and execution log |
| reports/eda/toysad8k-full-dataset/README.md | Short guide to the results and data boundary |
| requirements/eda-kaggle.txt | Core package versions recorded for this analysis |

Raw ToysAD-8K files, private labels, credentials, model weights, and feature
caches are not stored in GitHub.

## What this EDA does not prove

- It does not prove that LocateAnything finds every object.
- It does not prove that SAM 2.1 produces accurate masks from predicted boxes.
- It does not select the final DINO-family backbone.
- It does not report anomaly-detection accuracy.
- It does not replace the controlled 2,000-scene selection and evaluation
  protocol.

## Next working steps

1. Use the verified EDA tables and figures to prepare the supervisor update.
2. Rebuild or import the controlled 2,000-scene manifest and run leakage checks.
3. Define nested 20-scene, 100-scene, and 400-scene gates inside development.
4. Run the 20-scene all-view data-loader and resume smoke test.
5. Pin the LocateAnything and SAM 2.1 model revisions, prompts, preprocessing,
   and random seeds.
6. Measure the full LocateAnything-to-SAM path against supplied whole-object
   masks before choosing the segmentation stack.
7. Screen a small, fixed set of feature backbones using supplied masks so that
   segmentation errors do not affect the backbone comparison.

## Short supervisor update

This week we completed the full ToysAD-8K exploratory analysis on Kaggle. The
analysis covers 7,396 scenes, 51 categories, 974 base assets, and every indexed
PNG and EXR header. It identified a leakage-safe train-derived pool of 4,107
scenes, but we are retaining the planned controlled 2,000-scene experiment.
The next implementation step is to validate the automatic box-to-mask path on
small all-view gates, then compare feature backbones before running the larger
experiment.
