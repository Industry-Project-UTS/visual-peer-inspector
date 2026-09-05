# Notebook run evidence

This folder is the review index for the completed ToysAD development runs. It
connects each clean notebook to its saved-output evidence, execution transcript,
aggregate reports, and ClearML record.

The clean notebook is always the runnable source. An `-executed.ipynb` file is
a publication copy of the completed Kaggle run. Its cell source is verified
against the clean notebook. Private scene images and protected identifiers are
removed, while safe aggregate tables, charts, execution counts, and console
output are retained.

## Evidence coverage

| Stage | Saved-output notebook | Transcript | Aggregate evidence |
| --- | --- | --- | --- |
| 01 Full EDA | [Executed notebook](../../notebooks/eda/toysad8k-full-dataset/toysad-8k-full-dataset-eda-executed.ipynb) | Output is retained in the notebook | [EDA report](../eda/toysad8k-full-dataset/kaggle-v1/) |
| 02 Selection audit | [Executed notebook](../../notebooks/data-preparation/toysad8k-selection-audit/toysad8k-2000-selection-audit-executed.ipynb) | Output is retained in the notebook | [Selection report](../data-preparation/toysad8k-selection-audit/kaggle-v4/) |
| 03 DS0 package smoke | Saved-output copy was not retained | [Run record](../data-preparation/toysad8k-opaque-package-smoke/kaggle-v5-candidate/run_record.json) | [Package report](../data-preparation/toysad8k-opaque-package-smoke/kaggle-v5-candidate/) |
| 04 DS0 LocateAnything | Saved-output copy was not retained | [Run record](../detection-and-segmentation/toysad8k-ds0-locateanything/kaggle-v13-candidate/run_record.json) | [Proposal report](../detection-and-segmentation/toysad8k-ds0-locateanything/kaggle-v13-candidate/) |
| 05 DS0 SAM 2.1 | Saved-output copy was not retained | [Run record](../detection-and-segmentation/toysad8k-ds0-sam21/kaggle-v5-candidate/run_record.json) | [Segmentation report](../detection-and-segmentation/toysad8k-ds0-sam21/kaggle-v5-candidate/) |
| 06 DS0 evaluation | Saved-output copy was not retained | [Run record](../detection-and-segmentation/toysad8k-ds0-mask-evaluation/kaggle-v6-candidate/run_record.json) | [Evaluation report](../detection-and-segmentation/toysad8k-ds0-mask-evaluation/kaggle-v6-candidate/) |
| 07 DS0 peer scores | Saved-output copy was not retained | [Run record](../representation-and-scoring/toysad8k-ds0-peer-scores/kaggle-v5-candidate/run_record.json) | [Representation report](../representation-and-scoring/toysad8k-ds0-peer-scores/kaggle-v5-candidate/) |
| 03.1 DS1 package validation | [Executed notebook](../../notebooks/data-preparation/toysad8k-ds1-opaque-package/03.1-toysad8k-ds1-opaque-package-executed.ipynb) | [Log](logs/03-1-ds1-opaque-package.log) | [Package report](../data-preparation/toysad8k-ds1-opaque-package/kaggle-v3/) |
| 04.1 DS1 LocateAnything | [Executed notebook](../../notebooks/detection-and-segmentation/toysad8k-ds1-locateanything/04.1-toysad8k-ds1-locateanything-screen-executed.ipynb) | [Log](logs/04-1-ds1-locateanything-screen.log) | [Proposal report](../detection-and-segmentation/toysad8k-ds1-locateanything/kaggle-v4/) |
| 04.2 DS1 proposal refinement | [Executed notebook](../../notebooks/detection-and-segmentation/toysad8k-ds1-proposal-refinement/04.2-toysad8k-ds1-proposal-refinement-executed.ipynb) | [Log](logs/04-2-ds1-proposal-refinement.log) | [Refinement report](../detection-and-segmentation/toysad8k-ds1-proposal-refinement/kaggle-v2-candidate/) |
| 04.3 DS1 prompt freeze | [Executed notebook](../../notebooks/detection-and-segmentation/toysad8k-ds1-locateanything-prompt-freeze/04.3-toysad8k-ds1-locateanything-prompt-freeze-executed.ipynb) | [Log](logs/04-3-ds1-locateanything-prompt-freeze.log) | [Prompt report](../confirmation-and-multiview/ds1-locateanything-prompt-freeze/kaggle-v4-candidate/) |
| 05.1 DS1 SAM 2.1 | [Executed notebook](../../notebooks/detection-and-segmentation/toysad8k-ds1-sam21/05.1-toysad8k-ds1-sam21-segmentation-executed.ipynb) | [Log](logs/05-1-ds1-sam21-segmentation.log) | [Segmentation report](../detection-and-segmentation/toysad8k-ds1-sam21/kaggle-v2/) |
| 06.1 DS1 evaluation | [Executed notebook](../../notebooks/detection-and-segmentation/toysad8k-ds1-mask-evaluation/06.1-toysad8k-ds1-proposal-and-mask-evaluation-executed.ipynb) | [Log](logs/06-1-ds1-proposal-and-mask-evaluation.log) | [Evaluation report](../detection-and-segmentation/toysad8k-ds1-mask-evaluation/kaggle-v3/) |
| 07.1 DS1 peer scores | [Executed notebook](../../notebooks/representation-and-scoring/toysad8k-ds1-peer-scores/07.1-toysad8k-ds1-representation-and-peer-scores-executed.ipynb) | [Log](logs/07-1-ds1-representation-and-peer-scores.log) | [Representation report](../representation-and-scoring/toysad8k-ds1-peer-scores/kaggle-v3-candidate/) |
| 08A DS2 release | [Executed notebook](../../notebooks/confirmation-and-multiview/08a-toysad8k-ds2-confirmation-release-executed.ipynb) | [Log](logs/08a-ds2-confirmation-release.log) | [Release report](../confirmation-and-multiview/ds2-confirmation-release/kaggle-v5-candidate/) |
| 08B DS2 frozen pipeline | [Executed notebook](../../notebooks/confirmation-and-multiview/08b-toysad8k-ds2-frozen-pipeline-executed.ipynb) | [Log](logs/08b-ds2-frozen-pipeline.log) | [LocateAnything](../confirmation-and-multiview/ds2-locateanything/kaggle-v4-candidate/), [reference SAM](../confirmation-and-multiview/ds2-reference-sam21/kaggle-v4-candidate/), [automatic SAM](../confirmation-and-multiview/ds2-automatic-sam21/kaggle-v4-candidate/), and [DINOv2/P1](../confirmation-and-multiview/ds2-dinov2-p1/kaggle-v4-candidate/) |
| 08C DS2 confirmation | [Executed notebook](../../notebooks/confirmation-and-multiview/08c-toysad8k-ds2-confirmation-and-multiview-executed.ipynb) | [Log](logs/08c-ds2-confirmation-and-multiview.log) | [Confirmation report](../confirmation-and-multiview/ds2-confirmation-and-multiview/kaggle-v1-candidate/) |

The DS0 rows remain valid because their run records, manifests, tables, and
figures were retained. A missing saved-output copy is reported honestly rather
than reconstructed after the run.

## ClearML records

The DS1 prompt freeze and DS2 runs have repository-safe receipts in
[`clearml/`](clearml/). The task identifiers prove that the aggregate evidence
was uploaded to ClearML. Existing DS1 proposal-refinement and representation
receipts remain beside their stage reports.

| Run | Task ID |
| --- | --- |
| DS1 LocateAnything prompt freeze | `7ea2f4401c9544bab2ddfd8fe00d8cdd` |
| DS2 confirmation release | `bee6185ad67b44b5b5c068312a744f30` |
| DS2 LocateAnything | `18f9687457a74fb09a15569cbcc4403b` |
| DS2 reference-box SAM 2.1 | `961d126a698245409ad90f1d1796e165` |
| DS2 automatic-box SAM 2.1 | `569e2e3905e043b6ba12a0f27e40f9eb` |
| DS2 DINOv2-small and P1 | `30aaa95dcdb2413783e72abaa78c148b` |
| DS2 confirmation and multi-view evaluation | `f42ebb3192474535933bb950ca77ea74` |

## Main result figures

- [DS2 LocateAnything proposal quality](../confirmation-and-multiview/ds2-confirmation-and-multiview/kaggle-v1-candidate/locateanything_proposal_quality.png)
- [DS2 SAM mask quality](../confirmation-and-multiview/ds2-confirmation-and-multiview/kaggle-v1-candidate/sam_mask_quality.png)
- [DS2 route average precision](../confirmation-and-multiview/ds2-confirmation-and-multiview/kaggle-v1-candidate/route_average_precision.png)
- [DS2 three-view primary metrics](../confirmation-and-multiview/ds2-confirmation-and-multiview/kaggle-v1-candidate/primary_three_view_metrics.png)
- [DS2 multi-view ablation](../confirmation-and-multiview/ds2-confirmation-and-multiview/kaggle-v1-candidate/multiview_ablation.png)

The stage artifact manifests provide SHA-256 checksums for the figures and
tables. [`evidence_manifest.json`](evidence_manifest.json) separately hashes
the publication notebooks, logs, ClearML receipts, and publication receipts.

## Publication boundary

GitHub contains aggregate evidence needed to understand and review the work.
It does not contain raw ToysAD RGB images, per-object masks, model-facing
records, protected evaluator labels, credentials, private example panels, or
unredacted platform logs. Those files are kept in the verified local evidence
archive. This prevents leakage while keeping the scientific results auditable.

## Verification

Run:

```bash
python3 -m unittest discover -s tests -q
python3 scripts/verify_repository_run_evidence.py
git diff --check
```

The verifier checks notebook execution status, clean-source identity,
publication receipts, file hashes, and forbidden private material.
