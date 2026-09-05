# Notebook sequence

The notebooks are ordered experiment entry points. Reusable preparation,
model, scoring, evaluation, and tracking logic belongs in tested modules under
`src/visual_peer_inspector/`.

## Current sequence

| Notebook | Responsibility | Status |
| --- | --- | --- |
| 01 Full Dataset EDA | Audits the full catalogue, archive inventory, labels, views, data quality, and leakage risks | Complete |
| 02 Selection Audit | Verifies the controlled 2,000-scene roles, coverage, base-asset separation, and release evidence | Complete |
| 03 Opaque Package Smoke | Builds and verifies the model-safe 20-scene, 400-image DS0 RGB package | Complete candidate evidence |
| 04 Box Proposals | Runs pinned LocateAnything on DS0 and records frozen proposal evidence | Complete candidate evidence |
| 05 Segmentation | Runs pinned SAM 2.1 for the reference-box and automatic-box routes | Complete candidate evidence |
| 06 Proposal and Mask Evaluation | Compares proposals and masks with protected references inside evaluator-only code | Complete candidate evidence |
| 07 Representation and Peer Scores | Adds feature caching, encoder screening, B0, B1, P1, geometry, and heatmaps | Kaggle DS0 candidate evidence complete |
| 03.1 DS1 Opaque Package | Verifies the label-free 100-scene, 2,000-image DS1 input package | Complete clean-snapshot evidence |
| 04.1 DS1 Box Proposals | Runs pinned LocateAnything on all 2,000 DS1 images | Complete clean-snapshot evidence |
| 05.1 DS1 Segmentation | Runs pinned SAM 2.1 for supplied and automatic boxes | Complete clean-snapshot evidence |
| 06.1 DS1 Proposal and Mask Evaluation | Measures DS1 proposal and mask quality with protected references | Complete; refine or replace LocateAnything |
| 04.2 DS1 Proposal Refinement | Tests fixed label-free filters on the frozen LocateAnything proposals | Complete candidate evidence; precision improved but misses remain |
| 07.1 DS1 Representation and Peer Scores | Screens encoders and B0, B1 and P1 on all 100 DS1 scenes | Complete candidate evidence; DINOv2-small with P1 selected |
| 04.3 DS1 LocateAnything Prompt Freeze | Compares two supported generic prompts and freezes one prompt and post-processing policy | Kaggle Version 4 verified; `individual object` with `area_nms_075_cap8` selected |
| 08A DS2 Confirmation Release | Builds the untouched 300-scene RGB, geometry, and protected evaluator resources | Kaggle Version 5 verified; three private resources published and checked |
| 08B DS2 Frozen Pipeline | Runs LocateAnything, reference-box and automatic-box SAM 2.1, and frozen DINOv2-small with B0 and P1 on all 6,000 views | Kaggle Version 4 complete; compact bundles verified |
| 08C DS2 Confirmation and Multi-view | Measures proposals, masks, B0/P1 rankings, view counts, fusion, subgroups, and confirmation status | Kaggle Version 1 complete; development confirmation measured |
| 09 Frozen Evaluation | Opens internal-lock and official evaluation only after settings and predictions are frozen | Planned |
| 10 Full-Pool Sensitivity | Runs the optional role-preserving 4,107-scene analysis after the primary method freezes | Optional |

Candidate evidence is valid implementation evidence from a verified
pre-commit source snapshot that recorded `git_dirty: true`. The original run
record must remain unchanged. A clean-snapshot rerun may later replace a
candidate result.

The current DS2 decision keeps LocateAnything with an assisted correction
route, SAM 2.1 Hiera-large, DINOv2-small, P1, three declared views, and
mean-rank fusion. B0 remains the baseline. This is a development freeze for
the next calibration stage, not a final production claim.

## Dependency flow

```mermaid
flowchart LR
    N1["01 EDA"] --> N2["02 Selection"]
    N2 --> N3["03 Opaque package"]
    N3 --> N4["04 LocateAnything"]
    N4 --> N5["05 SAM 2.1"]
    N3 --> N5
    N4 --> N6["06 Proposal and mask evaluation"]
    N5 --> N6
    N6 --> N7["07 Representation and peer scores"]
    N7 --> N8A["08A DS2 confirmation release"]
    N8A --> N8B["08B Frozen pipeline"]
    N8B --> N8C["08C Confirmation and multi-view"]
    N8C --> N9["09 Frozen evaluation"]
    N9 --> N10["10 Optional sensitivity"]
    N2 --> N31["03.1 DS1 opaque package"]
    N31 --> N41["04.1 DS1 LocateAnything"]
    N31 --> N51["05.1 DS1 SAM"]
    N41 --> N51
    N41 --> N61["06.1 DS1 evaluation"]
    N51 --> N61
    N61 --> N42["04.2 DS1 proposal refinement"]
    N42 --> N43["04.3 DS1 prompt and policy freeze"]
    N43 --> N8B
    N61 --> N71["07.1 DS1 representation screen"]
```

## Notebook locations

- Notebook 01:
  `eda/toysad8k-full-dataset/toysad-8k-full-dataset-eda.ipynb`
- Notebook 02:
  `data-preparation/toysad8k-selection-audit/toysad8k-2000-selection-audit.ipynb`
- Notebook 03:
  `data-preparation/toysad8k-opaque-package-smoke/toysad8k-opaque-package-smoke.ipynb`
- Notebook 04:
  `detection-and-segmentation/toysad8k-ds0-locateanything/toysad8k-ds0-locateanything-proposals.ipynb`
- Notebook 05:
  `detection-and-segmentation/toysad8k-ds0-sam21/05-toysad8k-ds0-sam21-segmentation.ipynb`
- Notebook 06:
  `detection-and-segmentation/toysad8k-ds0-mask-evaluation/06-toysad8k-ds0-mask-evaluation.ipynb`
- Notebook 07:
  `representation-and-scoring/toysad8k-ds0-peer-scores/07-toysad8k-ds0-representation-and-peer-scores.ipynb`
- Notebook 03.1:
  `data-preparation/toysad8k-ds1-opaque-package/03.1-toysad8k-ds1-opaque-package.ipynb`
- Notebook 04.1:
  `detection-and-segmentation/toysad8k-ds1-locateanything/04.1-toysad8k-ds1-locateanything-screen.ipynb`
- Notebook 05.1:
  `detection-and-segmentation/toysad8k-ds1-sam21/05.1-toysad8k-ds1-sam21-segmentation.ipynb`
- Notebook 06.1:
  `detection-and-segmentation/toysad8k-ds1-mask-evaluation/06.1-toysad8k-ds1-proposal-and-mask-evaluation.ipynb`
- Notebook 04.2:
  `detection-and-segmentation/toysad8k-ds1-proposal-refinement/04.2-toysad8k-ds1-proposal-refinement.ipynb`
- Notebook 07.1:
  `representation-and-scoring/toysad8k-ds1-peer-scores/07.1-toysad8k-ds1-representation-and-peer-scores.ipynb`
- Notebook 04.3:
  `detection-and-segmentation/toysad8k-ds1-locateanything-prompt-freeze/04.3-toysad8k-ds1-locateanything-prompt-freeze.ipynb`
- Notebook 08A:
  `confirmation-and-multiview/08a-toysad8k-ds2-confirmation-release.ipynb`
- Notebook 08B:
  `confirmation-and-multiview/08b-toysad8k-ds2-frozen-pipeline.ipynb`
- Notebook 08C:
  `confirmation-and-multiview/08c-toysad8k-ds2-confirmation-and-multiview.ipynb`

Planned notebook directories are created only when their design and required
inputs are ready.

## Acceptance rules

An accepted notebook stage must:

1. use pinned and hashed inputs, code, models, and configuration;
2. keep source identifiers, labels, roles, and private protocol fields outside
   model-facing resources;
3. support deterministic restart or resume behaviour where applicable;
4. produce a clean notebook source with no saved execution outputs;
5. store large or private outputs outside GitHub, with a separate sanitised
   saved-output copy allowed for repository evidence;
6. retain a sanitized run record, aggregate tables, figures, and checksums;
7. record approved aggregate evidence in ClearML or preserve a synchronized
   offline record when ClearML is unavailable;
8. include failures and negative results, not only successful examples; and
9. pass the repository tests and `git diff --check` before review.

## Notebook 07 entry conditions

Notebook 07 must consume the frozen, label-free mask-route package created by
Notebook 06. The first run should validate B0, B1, and P1 on DS0 without
changing proposal or mask settings. The 100-scene DS1 screen should then:

- screen the bounded encoder candidates on supplied masks;
- compare downstream results across the three mask routes;
- add geometry without mixing it into the appearance encoder;
- report one-anomaly and two-anomaly ranking measures separately;
- use paired uncertainty grouped by base asset;
- store feature caches outside GitHub and ClearML; and
- save report-ready score tables, heatmaps, failure examples, runtime, memory,
  and a clear promote-or-stop decision.
