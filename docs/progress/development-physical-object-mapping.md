# Physical-object mapping: result and stopping decision

## Decision

The joint mapper failed the predefined promotion gate. Stop this matching/scoring search. Retain winner consensus as the stronger automatic development baseline and `object_bank_top2` with supplied correspondence as the assisted development fallback. Neither method is yet integrated into the upload app's older P1 runtime.

## Results on the same scenes

Tie-aware scene top-1, primary schedules, 100 development scenes and 27 underlying asset groups:

| Method | 2 views | 3 views | 5 views |
| --- | ---: | ---: | ---: |
| Existing all-candidate map + bank scorer | 25.3% | 20.4% | 18.7% |
| Joint partial assignment + bank scorer | 25.3% | 23.7% | 18.0% |
| Previous winner consensus | **39.0%** | **37.7%** | 18.9% |
| Predicted masks + supplied correspondence + bank scorer | **57.7%** | **62.8%** | **69.0%** |

The last row is dataset-assisted performance. Supplying neutral object correspondence does not supply the anomaly answer, but it does resolve identity ambiguity unavailable to the automatic route. It is not measured human-confirmation accuracy or ordinary RGB-only upload accuracy.

The joint method trails winner consensus by **14.0 percentage points** at primary three views, with a paired asset-bootstrap 95% interval of **−23.1 to −4.6 points**. Across the five predefined three-view schedules, mean top-1 is 22.1% for joint mapping, 21.3% for the existing all-candidate map, 30.1% for winner consensus and 62.5% for supplied correspondence. Each scene is averaged across schedules before uncertainty estimation.

Identity link F1 changes from 0.1856 to 0.1781 against the existing all-candidate map; fragmentation increases from 1.4122 to 1.4318 excess tracks per reference, averaged by scene. This is not a useful physical-identity improvement. All 100 scenes retain an anomaly candidate somewhere in the primary selected views, but only 67 remain rankable under the new map. Candidate presence does not establish correct cross-view identity.

## What the experiment establishes

The code solves the declared appearance-based map objective: all 1,900 scene/schedule settings obtained optimal solver status, and no primary fallback was needed. That does not establish correct real-world identity. Similar objects and viewpoint changes still produce mixed or fragmented tracks. Among the retained qualitative examples, even a ranking success selects an object linked in only one view. These limitations are shown rather than counted as complete multi-view highlighting success.

The final comparison holds the neural outputs and scorer fixed. The scorer's stronger assisted result does not survive the tested automatic identity mappings. A further weight or backbone search is outside the stopping decision.

## Executed evidence

- Notebook 08P executed locally: seven code cells completed with unchanged sources.
- Final mapping and scoring took about 97 seconds on CPU; evaluation took about 1.6 seconds. No LocateAnything, SAM or DINO inference was repeated.
- 7,600 scene/method/schedule metric rows retain all 100 scenes and all 27 asset groups.
- 128 focused tests passed, including 57 new mapping/evaluation/prediction tests and 71 reused scoring/checkpoint tests. The actual first cached scene was included.
- Scientific, evaluation and report seals passed verification. A pre-evaluation validation refinement has its earlier source and outputs retained privately; it did not change vision inference or permit label-based tuning.
- Two aggregate figures and two private original-coordinate galleries were visually inspected. The sanitized executed notebook retains the aggregate figures and removes private gallery images.
- Eleven aggregate artifacts were uploaded and verified in [ClearML](https://app.clear.ml/projects/62a09a229f184cb7bbe50da59b4226e7/experiments/bbc36baf29bd419197f0671ce3e27894/output/log). Private galleries and per-scene records were excluded.

## Next implementation work

The next milestone is application integration, not a new scorer experiment:

1. Add the frozen assisted bank scorer to the existing 2–5-view upload workflow. Record correspondence provenance as predicted, user-confirmed or dataset-reference.
2. Preserve per-view masks, patch features and crop transforms in a session cache. Identity corrections should rerun CPU mapping/scoring only.
3. Show one combined physical-object ranking. Ties must produce no unique selection; missing or uncertain links must remain visible. A box should appear only on an observation belonging to the selected map.
4. Keep the stronger automatic experimental baseline separate. The current app still uses P1, so winner consensus also needs an exact versioned adapter before that performance can be attributed to the app.
5. Verify a controlled 2–3-view end-to-end demonstration. Dataset-assisted replay, human-confirmed uploads and automatic uploads require separate accuracy and runtime claims.

No promise of reliable arbitrary RGB-only uploads follows from this stage.

## Exposure and submitted schedule

All 1,000 development scenes have already been model-processed and identity-evaluated. The 600-scene development remainder can support a later frozen-scorer check, but cannot be called globally untouched. The resolved exploratory asset-exposure ledger was not found in the targeted audit. Protected cohorts therefore remain closed until exposure and confirmation membership are reconciled; their unaffected counts are unknown.

The submitted semester schedule sets method/calibration freeze for 20 September, stable app/results for 4 October, reports for 18 October and video/peer assessment for 25 October. Calibration by 20 September depends on exposure reconciliation. Preserve the 4 October application milestone by ending optional scorer work now. The 15 September draft date is an internal target; retained assessment instructions specify Week 8.

## Repository preservation

Completed 08N/08O source, notebooks and documentation were committed only in the private working-project repository as `3c33d278`. Its working tree is clean; ignored private outputs, caches and archives remain in place.

The team repository's primary checkout remains at `e4d2c3b`. The separate `multiview-physical-objects` branch contains the 42 byte-verified 08N/08O overlay files plus the 08P source, tests and evidence for combined review. Ongoing evaluation work is developed in the separate working project before selected completed files enter this repository.
