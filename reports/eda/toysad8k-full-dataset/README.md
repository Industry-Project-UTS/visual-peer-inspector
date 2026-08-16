# ToysAD-8K full dataset EDA

This folder preserves the outputs from the completed ToysAD-8K exploratory
data analysis run on Kaggle. The purpose of the run was to understand the full
dataset before choosing scenes for modelling.

The saved Kaggle notebook is private and shared with the project group:

<https://www.kaggle.com/code/esreekarreddy/toysad-8k-full-dataset-eda>

## What was checked

The notebook analysed the complete catalogue and inspected every indexed PNG
and EXR header. It did not train an anomaly detector.

- 7,396 catalogue scenes
- 51 categories
- 974 base assets
- 27,290 objects
- 8,482 anomalous objects
- 7,373 availability-ready scenes
- 1,161,604 indexed non-directory files
- 1,139,416 PNG and EXR members checked
- 1,139,405 valid image headers
- 11 zero-byte image files
- 4,107 leakage-safe, train-derived eligible scenes
- 545 eligible base assets
- 39 eligible categories

The 4,107 scenes are the complete eligible pool. They do not replace the
project's controlled 2,000-scene primary experiment. The primary experiment
and its development, calibration, and internal-lock roles are defined
separately.

## Important findings

- Every catalogue scene contains at least one anomalous object. The dataset
  does not provide a native all-normal scene set.
- Anomalous objects occur first in the source object order. Object order,
  source names, and source paths must not be used as model inputs.
- The supplied segmentations are whole-object references. They are not
  pixel-level defect masks.
- Views are repeated observations of a scene. Twenty views do not count as
  twenty independent scenes.
- Official test pixels were not displayed or interpreted. Their file headers
  and aggregate counts were included in the storage audit.

## Run verification

The Kaggle run completed on 16 August 2026 using CPU execution with internet
disabled. The full notebook took about 3 hours and 16 minutes. Most of that
time was spent opening 1,139,416 files on the Kaggle dataset mount.

Verification after download found:

- all 12 cross-environment reference checks passed;
- all 35 artifacts listed in `eda_artifact_manifest.csv` matched their recorded
  SHA-256 values;
- all 20 directly comparable data tables were byte-identical to the earlier
  local run; and
- the generated figures were opened and visually checked.

The plotting libraries differ slightly between the local and Kaggle
environments, so rendered PNG files are not expected to have identical byte
hashes. The underlying tables and headline counts agree.

## Folder contents

The `kaggle-v1` folder contains the complete 36-file output bundle and the
downloaded execution log.

Useful entry points are:

- `run_record.json`: source hashes, environment, counts, and protocol notes
- `cross_environment_equivalence.csv`: local-versus-Kaggle count checks
- `eda_artifact_manifest.csv`: output names, sizes, and SHA-256 values
- `dataset_overview.csv`: headline dataset totals
- `split_summary.csv`: official split summary
- `eligibility_bridge.csv`: full catalogue to eligible-pool filtering
- `training_anomaly_examples.png`: representative training examples
- `training_scene_twenty_views.png`: all twenty views of one scene
- `training_mask_semantics.png`: supplied mask interpretation
- `toysad-8k-full-dataset-eda.log`: Kaggle execution log

The notebook folder contains two deliberately separate files:

- `toysad-8k-full-dataset-eda.ipynb` is the clean Kaggle source notebook.
- `toysad-8k-full-dataset-eda-executed.ipynb` is the saved Kaggle version
  with all 18 code cells and their outputs.

The Kaggle metadata is stored beside them under
`notebooks/eda/toysad8k-full-dataset/`.

## Data boundary

This repository stores the notebook, summaries, figures, hashes, and run log.
It does not store the raw ToysAD-8K archive, extracted images, model weights,
feature caches, or private evaluator labels.

The CSV files in this report folder are reporting evidence. They must not be
used as model-facing manifests. In particular, model code must not receive
source object order, anomaly counts, labels, raw paths, or source identifiers.
