# Fixed multi-view Kaggle execution plan

## Prepared sequence

The remaining sequence has executable input producers, verified feature reuse, typed evaluator conversion, runtime setup, checkpoints and report entry points. The first Kaggle jobs prepare the target environments; scientific jobs follow their retained setup receipts. No dataset experiment was run while preparing this source.

Models stay fixed: LocateAnything-3B → SAM 2.1 → DINOv2-small. Automatic prediction uses the selected winner consensus. Assisted prediction uses the foreground patch bank with strongest-two fusion. Both use nineteen fixed schedules: 2/3/5 views are application settings and 1 view is diagnostic. This is a fixed-method evaluation, not another model or weight search.

## Reuse uploaded data

| Existing Kaggle resource | Use |
| --- | --- |
| toysad8k-full-private-eda | Indexed raw data for missing role resources |
| vpi-toysad-selection-private | Original opaque 2,000-scene assignment and administrative mapping |
| vpi-toysad-ds1-screen100-model-input | Existing 100-scene RGB |
| vpi-toysad-development-remainder-model-input | Existing 600-scene RGB |
| vpi-toysad-ds2-confirmation-rgb | Existing 300-scene RGB |
| toysad-8k-frozen-development-pipeline output | Boxes/masks for 1,000 development scenes and 20,000 views |

No full raw-dataset or 2,000-scene RGB re-upload is required. New uploads contain source, small support/configuration files and the derived development 100 feature resource. Later jobs attach retained upstream outputs.

The real lineage audit matched all 2,000 existing 08N automatic-view units and 7,594 candidates to the native development source. The 2.796 GB prepared reuse resource preserves NPZ bytes, tensor indices and the original feature lock. Its universe is 1,000 scenes; 18,000 feature units remain absent. It is a partial feature resource, not completed 1,000-scene evidence. Historical packed-mask hashes and raw-pixel hashes are separately verified before normalization.

## Implemented input producers

1. `native_rgb.prepare_rgb_index` verifies exact original/official membership and indexes existing archive members without copying RGB. Its private source-path index cannot substitute for a model input.
2. `native_rgb.release_rgb` builds a canonical model-only package for a role that lacks one, using the existing indexed archive and atomic scene checkpoints.
3. `native_regions` runs only LocateAnything and SAM for missing phases/views. It checks pinned assets and the separate GPU profile, preserves empty/terminal cases, and reuses valid phase checkpoints. It never runs DINO or P1.
4. `feature_remap` verifies and rebinds the compatible 08N units. `native_cache` extracts only missing DINO crops on the pinned CPU FP32 runtime, then regenerates current score components from saved grids.
5. Neural and cache preparation use deterministic scene partitions. Retain full-grid feature shards separately from foreground-cache shards. Merging requires exact full-cohort membership and rejects overlap/missing scenes. Shards are resource chunks, not independent evaluation samples.
6. `raw_reference.prepare_raw_geometry` creates neutral supplied boxes/IDs and the reviewed scene-to-asset index. `native_evaluator.prepare_neutral` applies the existing one-to-one box-IoU≥0.5 policy independently per view. Unmatched candidates remain null. No anomaly answer chooses a match.
7. `convert_raw_evaluator` validates complete automatic and assisted predictions before opening normalized anomaly flags. The original opaque-object permutation is checked before labels are read. Typed truth binds the exact cache, geometry, source, membership, predictions and method freeze.

Existing native crosswalk/geometry/label packages are also supported. Neutral conversion reads registered geometry/crosswalk members only; outcome conversion is a later operation. Dataset-reference assistance remains distinct from human-confirmed assistance.

## Exposure and claims

Original roles remain 1,000 development, 600 calibration and 400 internal-lock, with 273/163/109 asset groups. Development reuse is documented; exact earlier exploratory access to reserved assets remains incomplete.

The prepared audit therefore declares **exposure-qualified fixed-method evaluation**. Assets with unknown history remain potentially exposed. Do not call this untouched or independent confirmation, and do not reshuffle it into a claimed clean split. The historical strict gate remains unchanged. Role names, sizes and exposure qualifications remain in reports.

Official seen 715 scenes / 113 assets and unseen 2,437 scenes / 316 assets are separate roles. The queue builds their neutral indexes and separate exposure-qualified audit. The original 2,000-scene audit cannot substitute for that official audit.

## Execution order

Generate the machine-readable queue with `scripts/build_current_launch_plan.py`. `scripts/current_launch.py` resolves declared existing Kaggle resources and retained producer outputs. It reads actual upstream hashes when configuring a dependent job; it never invents future output hashes.

1. Setup-only CPU, DINO, GPU and ClearML jobs build verified wheels, record the actual Linux Python patch and prepare pinned model bytes. They run no dataset experiment.
2. Create the current method/source/runtime freeze from the completed 09E decision.
3. Reuse development 100 features, prepare the remaining development cache, and run the fixed 1,000-scene development prediction/evaluation.
4. Prepare calibration resources. 08F1 releases the current cache; 09A predicts; truth conversion precedes 09B evaluation.
5. After 09B, 09B1 releases internal resources; 09C predicts; truth conversion and 09D follow.
6. After 09D, 10R releases each official cache; 10A/10A1 predict each mode; finalizers seal seen/unseen together per mode; 10B evaluates the roles separately.
7. Verify downloads, render/repair reports, publish bounded ClearML artifacts and select a sanitized GitHub overlay later.

Neutral metadata preparation can precede scoring. Model/feature jobs cannot bypass required freeze, exposure and prior protocol receipts. A checkpointed job is incomplete; downstream jobs require the corresponding final seal.

## Metrics and report material

Report tie-aware top-1, Recall@2, AP, MRR, candidate coverage and loss breakdown, identity purity/precision/recall/fragmentation, per-schedule/view-count results, runtime and resource receipts. Bootstrap uses 2,000 asset-group draws with seed 32040. Average schedules per scene before uncertainty calculations. Missing/unrankable cases remain in the denominator.

Private exports include RGB, boxes, SAM overlays, objects on black, consistent IDs, complete per-object/per-view scores, final rankings, fusion contributors and selected-object boxes. Coarse heatmaps require verified crop-to-image geometry. Missing geometry and unmatched views are explicit. Deterministic examples and labelled post-hoc failures illustrate full results; they do not define the accuracy subset.

09B measures fixed rankings. It does not fit calibrated probabilities or a new readiness threshold. The app retains provisional review status. Accuracy must name technique, view count, assistance provenance and exposure-qualified cohort. 60% remains a target, not a promised result.

## Reliability

The source bundle includes the complete required Python tree. Bootstrap verifies source/profile/wheel hashes before scientific imports. CPU scoring and DINO use the pinned scientific package versions; LocateAnything/SAM use a separate Torch/Transformers environment. Exact source/runtime changes after freezing fail.

Atomic checkpoints, terminal failure records, bounded downloads/workers and explicit recovery are implemented. Recovery survives a lost session only if files were retained or downloaded. Resume matching inputs; do not rerun models for figures, tracking or download repair. Storage limits use actual produced shard sizes and stop before exceeding the configured volume.

Local tests and real feature-file checks establish preparation behavior. Actual Kaggle GPU inference, future scientific results and external ClearML publication require their own execution/download receipts. They are not claimed by this preparation.
