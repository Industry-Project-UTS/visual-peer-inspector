# Development identity refinement

This folder contains GitHub-safe aggregate evidence from Notebook 08I Kaggle
Version 3. The run reused the existing 100-scene development screen and the
fixed DINOv2-small revision. It did not change the 2,000-scene split and did
not open calibration or internal-lock data.

The experiment compared the existing global descriptor with foreground patch
pooling and hybrid descriptors. Each descriptor was evaluated with three
unmatched costs, two, three and five views, and the existing peer scorers.

## Primary three-view result

| Measure | Existing baseline | Best development candidate | Difference |
| --- | ---: | ---: | ---: |
| Scene average precision | 0.339835 | 0.375897 | +0.036062 |
| Top-1 accuracy | 0.24 | 0.31 | +0.07 |
| Recall at 2 | 0.355 | 0.335 | -0.02 |
| Identity-link F1 | 0.188789 | 0.139612 | -0.049177 |
| Track purity | 0.687222 | 0.842944 | +0.155722 |

The candidate is `hybrid50_cov25__u020__v3` with
`forward_02_cov70`. Its top-1 interval was positive, but its identity-link F1
fell and the average-precision interval crossed zero. The candidate therefore
was not promoted. The current global identity method remains the stable
default.

## Evidence boundary

This folder includes aggregate tables, two figures, the completed run status,
the evaluation binding, the verified ClearML receipt, and a sanitized execution
transcript. A repository-safe saved-output notebook is stored under
`notebooks/executed/development-identity/`. Its cell sources match the clean
notebook, all 11 code cells completed, and private visuals were removed.

Per-scene evaluator rows, private galleries, RGB images, masks, prediction
shards, feature caches, recovery archives, raw Kaggle logs, credentials, and
protected identifiers are excluded.
