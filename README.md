# Visual Peer Inspector

Visual Peer Inspector compares similar objects visible in the same scene and
ranks the objects that look most unusual relative to their peers. The project
uses multiple RGB views from ToysAD-8K to develop a reproducible object
proposal, segmentation, representation, peer-scoring, and evaluation pipeline.

This repository contains code, tests, notebook sources, compact experiment
evidence, and documentation for the UTS 32040 Industry Project.

## Current status

The controlled 2,000-scene experiment is fixed and leakage checked. Verified
development evidence is complete through Notebook 08H. The remaining protected
and official notebook sequence is built and locally checked, but has not yet
been run on Kaggle.

| Stage | Result |
| --- | --- |
| 01 Full Dataset EDA | Complete on the 7,396-scene catalogue |
| 02 Selection Audit | Complete for 1,000 development, 600 calibration, and 400 internal-lock scenes |
| 03 Opaque Package Smoke | Complete for 20 scenes and 400 images |
| 04 LocateAnything Proposals | Complete for all 400 DS0 images |
| 05 SAM 2.1 Segmentation | Complete for reference-box and automatic-box routes |
| 06 Proposal and Mask Evaluation | Complete; all eight declared DS0 gates passed |
| 07 to 07.1 Representation and Peer Scores | Complete candidate evidence; DINOv2-small with P1 selected for the development freeze |
| 08A to 08C DS2 Confirmation | Complete development confirmation evidence |
| 08D to 08F Development Freeze | Four stages complete; 20,000 views locked and 3,000 application results verified; cross-view identity remains weak |
| 08G to 08H Development Scoring Comparison | Nine scoring methods compared on the existing 100-scene development screen; automatic-route accuracy remains limited by predicted identity |
| 08F.1 to 10B Protected and Official Evaluation | Built and locally checked; Kaggle scientific evidence pending |
| 11R to 11B Full-pool Sensitivity | Three optional post-official stages built and locally checked; execution remains optional |

The working route is LocateAnything-3B with the `individual object` prompt and
`area_nms_075_cap8` cleanup, SAM 2.1 Hiera-large, DINOv2-small, P1, label-free
cross-view identity, and mean normalized-rank fusion. This is a development
freeze for the protected evaluation sequence. It is not final official
performance or a production deployment claim. Notebook 08F retained
embedding-only matching, but its five-view mean pairwise identity accuracy is
33.04%. This measures object linking, not odd-one-out accuracy. Notebooks 08G
and 08H then reused the fixed DINOv2-small features to compare P1 with
directional, mutual-support, geometry, consensus, and out-of-fold context
scorers. The best automatic three-view result reached 0.34165 scene average
precision and 0.24 top-1 accuracy. With supplied masks and reference identity,
the same feature pipeline reached up to 0.66 top-1 accuracy. This shows that
cross-view identity is the main next development problem. See the [development
scoring comparison](docs/progress/development-scoring-comparison.md) before
starting protected calibration or making a web-app readiness claim.

## Pipeline

The current system uses LocateAnything to propose object boxes and SAM 2.1 to
produce prompted whole-object masks. A frozen visual encoder then represents
each object for B0, B1, and P1 peer comparison. Geometry and cross-view fusion
remain separate parts of the method.

See [System architecture](docs/architecture.md) for the research pipeline,
application pipeline, data boundaries, route names, and claim limits.

## Repository layout

The independent [web app](apps/web/README.md) provides the upload, inspection,
review and report interface. It starts without GPU dependencies; live inference
is connected through an optional backend adapter.

```text
apps/web/             Independent inspection web app and launch instructions
data/manifests/       Public-safe dataset and historical cohort records
docs/                 Architecture, method, dataset, experiment, and progress notes
notebooks/            Ordered Kaggle and experiment entry points
reports/              Reviewed aggregate evidence and compact figures
requirements/         Stage-specific dependency locks
scripts/              Verification, preparation, execution, and finalisation commands
src/                  Reusable data, model, workflow, and evaluation code
tests/                Unit and integration tests
```

Large private datasets, model inputs, model outputs, feature caches, masks,
weights, credentials, and evaluator crosswalks are intentionally excluded.

## Run the checks

The core checks do not require the large ToysAD source:

```bash
python3 -m unittest discover -s tests -v
```

Stage-specific dependencies are pinned under `requirements/`. The ordered
notebook responsibilities and upstream inputs are documented in
[notebooks/README.md](notebooks/README.md).

The earlier one-view B0 and B1 commands remain available as historical
regression checks:

```bash
python3 -m pip install -e '.[baseline]'
python3 scripts/run_baseline.py b0 data/local/development-40 outputs/b0-development.json
python3 scripts/run_baseline.py b1 data/local/development-40 outputs/b1-development.json
```

These 40-scene results are not the current multi-view architecture or final
evaluation.

## Data and evaluation boundary

The project uses three separate data zones:

1. Trusted preparation may read private source metadata and protocol records.
2. Model execution receives opaque IDs and only the media or prompts required
   by that stage.
3. Evaluator-only code joins frozen predictions with protected references and
   labels after inference is complete.

Object order, source names, source paths, labels, categories, base-asset
groups, role assignments, and official-test information must not enter model
inputs. ClearML stores approved aggregate evidence and provenance, not private
data or credentials.

## Key documents

- [System architecture](docs/architecture.md)
- [Notebook sequence](notebooks/README.md)
- [Dataset and protocol checks](docs/dataset.md)
- [Problem and method](docs/problem-and-method.md)
- [Progress index](docs/progress/README.md)
- [Full ToysAD-8K EDA](reports/eda/toysad8k-full-dataset/README.md)
- [DS0 LocateAnything evidence](reports/detection-and-segmentation/toysad8k-ds0-locateanything/README.md)
- [DS0 SAM 2.1 evidence](reports/detection-and-segmentation/toysad8k-ds0-sam21/README.md)
- [DS0 proposal and mask evaluation](reports/detection-and-segmentation/toysad8k-ds0-mask-evaluation/README.md)
- [Contribution rules](CONTRIBUTING.md)

## Claim boundary

ToysAD-8K catalogue scenes contain at least one anomalous object. The project
therefore evaluates relative peer ranking, not a general normal-versus-anomaly
classifier. Supplied and predicted masks are whole-object masks, not
pixel-accurate annotations of the defective region. P1 heatmaps will be
reported as coarse supporting evidence only.

## Team

- Sreekar Reddy
- Vaibhav Bairathi

UTS 32040 Industry Project, Spring 2026.
