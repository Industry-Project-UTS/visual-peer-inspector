# Data handling

Raw ToysAD-8K files are not stored in this repository. The tracked `manifests`
directory contains only the information needed to verify the source archive and
reproduce the historical 40-scene baseline.

## Active experiment protocol

The current primary experiment uses a private, leakage-safe selection of 2,000
eligible scenes:

| Role | Scenes | Purpose |
| --- | ---: | --- |
| Development | 1,000 | Method development and controlled comparisons |
| Calibration | 600 | Threshold and operating-point selection |
| Internal lock | 400 | Final internal confirmation after the method is frozen |

The private crosswalk and protected labels are kept outside GitHub. Model-facing
jobs receive opaque scene identifiers, selected RGB views, and only the inputs
required by that stage.

## Evaluation gates

- DS0 uses 20 scenes and all 20 views per scene for end-to-end smoke testing.
- DS1 uses 100 scenes for the first method and encoder screen.
- DS2 is a separate 300-scene confirmation gate.
- Wider runs use the remaining eligible scenes only after the method is frozen.

These gates do not replace the development, calibration, and internal-lock
roles. They control when a method is allowed to move to a larger evaluation.

## Local storage

Place local extractions and generated private material under ignored locations
such as `data/local/`, or outside the repository. Do not commit:

- raw images, masks, or extracted archives
- private scene crosswalks or protected labels
- model weights, feature caches, or bulk predictions
- credentials, Kaggle logs, or ClearML configuration files

Trusted audit resources must not be attached to model-facing Kaggle jobs. Safe
aggregate tables, charts, hashes, and sanitized run records may be committed
under `reports/` after review.

See [Dataset and split protocol](../docs/dataset.md) for the dataset findings and
[System architecture](../docs/architecture.md) for the evaluator boundary.
