# Notebook 08J Kaggle Run Summary

Notebook 08J completed on Kaggle and its outputs were downloaded to this local development workspace. No files were committed to Git.

## Experiment boundary

- Role: development only
- Scenes: 100
- Asset groups: 27
- View settings: 2, 3 and 5 views
- Frozen representation: DINOv2-small
- Frozen scoring method: `forward_02_cov70`
- Protected calibration and internal-lock roles used: no
- Neural inference repeated: no

The notebook reused the sealed Notebook 08G embeddings and scores. It compared five predeclared cross-view association strategies. Predictions were sealed before evaluator labels were opened.

## Result

The current Hungarian method remains selected.

| Strategy | AP | Top-1 | Recall@2 | Identity F1 | Linked nodes | Singletons |
|---|---:|---:|---:|---:|---:|---:|
| Current Hungarian 0.35 | 0.340 | 0.240 | 0.355 | 0.189 | 0.868 | 0.232 |
| Reciprocal 0.20 | 0.445 | 0.360 | 0.420 | 0.098 | 0.508 | 0.657 |
| Reciprocal 0.35 | 0.455 | 0.380 | 0.430 | 0.133 | 0.642 | 0.546 |
| Support consensus 0.35 | 0.455 | 0.380 | 0.430 | 0.133 | 0.642 | 0.546 |
| Cycle pruned 0.35 | 0.563 | 0.520 | 0.465 | 0.059 | 0.153 | 0.933 |

Cycle pruning raised scene ranking metrics, but it did so by breaking most proposals into singletons. Its identity F1 fell from 0.189 to 0.059. No candidate improved identity quality and anomaly ranking together, so none was promoted.

## Verification

- Kaggle kernel: `esreekarreddy/toysad-8k-development-identity-association`
- Successful run: Version 2
- Prediction scene files and receipts: 200
- Evaluation files: 29
- Prediction lock SHA-256: `c939d63c9ade9f11bc5085d82d750730ec1787302160e92bac75034a3e8c5fb8`
- Evaluation lock SHA-256: `9e89f830b974868ff876a544430fa617c43cfec4931d8f18815bc137d4b847cb`
- Pulled Kaggle notebook SHA-256: `20e1af3d44cab08b1129b3749cd3d6b20dca4e033f9225e2e19ca2b7a931ce33`
- Local and Kaggle result tables: identical
- Tracebacks or fatal errors in Version 2 logs: none
- ClearML task: `222194e65f8d40efba0a2a3b02599fc8`
- ClearML remote status: completed

Version 1 failed because Kaggle expanded the source archive before the notebook mounted it. The notebook now checks for an expanded source tree first and uses the archive only as a fallback. This recovery rule is recorded in `KAGGLE_ISSUES.md`.

## Next stage

Identity association should now be frozen at the current Hungarian method. The next notebook should use the calibration role to set review and uncertainty thresholds for the complete frozen pipeline. The 400-scene internal-lock role must remain unopened until that calibration policy and all model settings are frozen.
