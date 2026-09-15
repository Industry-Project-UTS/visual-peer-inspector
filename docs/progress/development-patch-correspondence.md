# Notebook 08K Kaggle verification

Date: 14 September 2026

## Run

- Kaggle notebook: `esreekarreddy/toysad-8k-development-patch-correspondence`
- Kaggle version: 3
- Status: complete
- Accelerator: GPU T4 x2 available, one GPU used by the runner
- Development scenes: 100
- Views available per scene: 8
- Candidate configurations per scene: 12
- Protected data used: no
- Split changed: no

## Validation

- Kaggle focused tests: 13 passed
- Prediction scenes: 100 of 100
- Evaluation scenes: 100 of 100
- Result files: 241
- ZIP integrity: passed
- Prediction seal: passed after download
- Evaluation seal: passed after download
- Executed notebook: 24 cells, 10 of 10 code cells executed, no error outputs
- Private gallery: all 12 examples inspected
- Shareable figures: all 3 figures inspected

## Main result

The foreground patch methods did not pass the joint identity, ranking and anti-fragmentation promotion rule. The retained development configuration remains `current_hungarian_035` with DINOv2-small and P1 for the three-view automatic route.

Three-view P1 results:

| Configuration | AP | Top-1 | Identity link F1 | Singleton fraction |
| --- | ---: | ---: | ---: | ---: |
| current_hungarian_035 | 0.3200 | 0.23 | 0.1888 | 0.2317 |
| reciprocal_hungarian_035 | 0.2923 | 0.22 | 0.1493 | 0.3978 |
| sinkhorn_hungarian_035 | 0.2906 | 0.20 | 0.1647 | 0.3631 |
| sinkhorn_consensus_035 | 0.2903 | 0.20 | 0.1564 | 0.3063 |

The patch methods produced some isolated two-view ranking gains, but they reduced identity link quality and did not generalise to the primary three-view route. They are not promoted.

## ClearML

- Task ID: `9a43c859d1b34ed0929d086636a4e155`
- Status read back from ClearML: completed
- Safe artifacts read back: 12
- Scalar groups read back: 8
- Final tags: `development-only`, `dinov2-small`, `patch-correspondence`, `verified`, `final`, `retain-current`
- Task URL: https://app.clear.ml/projects/cae39970e23e4e7eaac2f5c70617d1b1/experiments/9a43c859d1b34ed0929d086636a4e155/output/log

## Local files

- `identity-patch-study-results.zip`
- `toysad-8k-development-patch-correspondence.executed.ipynb`
- `toysad-8k-development-patch-correspondence.log`
- `extracted/identity-patch-predictions/`
- `extracted/identity-patch-evaluation/`

Result ZIP SHA-256: `2185fb9483347385949e3cc28c2e55e0e9dfb009eebf7319a8dc89e2ef1617c1`

Executed notebook SHA-256: `de144b50a36f8d39cfacf811369f419fcdaa7750fed5d1205e53f1cf517153c8`

Nothing in this run folder has been committed.
