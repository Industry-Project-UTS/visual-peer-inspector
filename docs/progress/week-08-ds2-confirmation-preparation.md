# Week 8 progress: DS2 confirmation

**Status:** Complete development confirmation.

## What is ready

- The frozen 300-scene DS2 cohort contract is registered.
- Kaggle Notebook 08A Version 5 verified 300 scenes and 6,000 RGB images.
- Data preparation writes RGB, supplied geometry, and protected attributes to
  separate resources.
- Reference-box SAM 2.1 has a repeatability pilot and restartable image shards.
- DINOv2-small uses the immutable revision already tested on DS1.
- B0 and P1 use the same fixed crop and scoring settings as the DS1 decision.
- The evaluator compares mask routes, six view counts, and two fusion rules.
- Aggregate tables and figures have a separate ClearML finalisation step.
- Real mask overlays, P1 heatmaps, and success and failure panels stay in the
  private output for reporting.
- Notebook 04.3 Version 4 compared the two prompts that passed the one-image
  gate and verified `individual object` with `area_nms_075_cap8`. It does not
  use a hard accuracy cutoff.

## Prepared notebook set

- `08a-toysad8k-ds2-confirmation-release.ipynb` prepares and verifies the
  300-scene confirmation release.
- `08b-toysad8k-ds2-frozen-pipeline.ipynb` runs LocateAnything, reference-box
  and automatic-box SAM 2.1, DINOv2-small, B0, and P1 without evaluator labels.
- `08c-toysad8k-ds2-confirmation-and-multiview.ipynb` opens the protected
  evaluator after prediction hashes are frozen. It measures mask quality,
  scoring, view count, fusion, subgroups, and difficult cases.
- Each notebook has private Kaggle metadata, a pinned requirements file, and a
  bounded ClearML finalisation step.
- The code-only snapshot builder now includes all three notebooks and their
  supporting modules, scripts, requirements, and tests.

## Completed runs

- Notebook 04.3 Version 4 froze `individual object` with
  `area_nms_075_cap8` on DS1.
- Notebook 08A Version 5 verified 300 DS2 scenes, 6,000 RGB images, 82 asset
  groups, and 23,100 visible object masks.
- Notebook 08B Version 4 completed all frozen model stages. LocateAnything
  produced 23,180 trusted proposals. Reference-box SAM produced 23,100 masks,
  automatic-box SAM produced 23,180 masks, and DINOv2-small produced 138,760
  B0 and P1 score rows.
- Notebook 08C Version 1 evaluated the frozen predictions on 300 scenes. It
  produced 21,600 scene metric rows across three routes, two methods, six view
  counts, and two fusion rules.
- All compact bundles passed SHA-256 and ZIP checks after download.
- The full repository suite passed 330 tests and 13 subtests before the final
  Kaggle run.
- Aggregate evidence from the five DS2 stages was finalised in ClearML after
  the scientific bundles were verified.

## Main results

- LocateAnything reached 0.9350 precision, 0.9383 recall, 0.9366 F1, and an
  0.8792 complete-image rate at box IoU 0.50.
- Reference-box SAM reached 0.9478 mean mask IoU and 0.9836 recall at mask IoU
  0.75.
- Automatic-box SAM reached 0.8834 mean mask IoU and 0.9031 recall at mask IoU
  0.75.
- P1 improved supplied-mask 20-view scene average precision over B0 by 0.0505.
  The paired 95 percent interval was 0.0249 to 0.0777.
- The declared three-view automatic route with P1 and mean-rank fusion reached
  0.6531 scene average precision, 0.4633 top-1 accuracy, and 0.6533 recall at
  two.
- The 20-view automatic route reached 0.6954 scene average precision. The
  ten-view visibility-weighted result was higher, but DS2 was not used to
  retune the declared three-view policy.

## Decision and limits

Keep LocateAnything, SAM 2.1 Hiera-large, DINOv2-small, and P1 as the working
pipeline. Keep B0 as the baseline and keep B1 stopped. The automatic route is
useful, but it missed 1,426 visible object instances and did not completely
detect every object in 725 images. A user-confirmed box route remains required.

These results are development confirmation. The calibration and internal-lock
roles remain unopened. The evaluator used protected identity correspondence
after prediction lock, so label-free cross-view identity matching is still an
application task. ToysAD-8K cannot measure all-normal false alarms because the
audited release contains no all-normal scenes.

## ClearML records

| Evidence stage | Task ID |
| --- | --- |
| DS1 prompt freeze | `7ea2f4401c9544bab2ddfd8fe00d8cdd` |
| DS2 confirmation release | `bee6185ad67b44b5b5c068312a744f30` |
| DS2 LocateAnything | `18f9687457a74fb09a15569cbcc4403b` |
| DS2 reference-box SAM 2.1 | `961d126a698245409ad90f1d1796e165` |
| DS2 automatic-box SAM 2.1 | `569e2e3905e043b6ba12a0f27e40f9eb` |
| DS2 DINOv2-small and P1 | `30aaa95dcdb2413783e72abaa78c148b` |
| DS2 confirmation and multi-view evaluation | `f42ebb3192474535933bb950ca77ea74` |

## Next work

1. Prepare the clean Week 8 repository transfer and review the pull request.
2. Build label-free cross-view identity matching with a user confirmation
   fallback.
3. Use the 600 calibration scenes to freeze review thresholds and uncertain
   states without changing the selected models.
4. Run the 400-scene internal lock once after the configuration and predictions
   are saved.
5. Start the web application around the frozen automatic and assisted routes.
