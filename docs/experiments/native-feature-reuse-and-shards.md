# Native feature reuse and bounded cache preparation

This is input preparation for the frozen two-mode method. No new scoring choice or evaluation split is introduced. Every prediction cohort retains its original denominator.

## Verified source formats

`native_cache.native_regions` accepts the sealed08G comparison package, a native08E/protected prediction manifest plus its verified RGB package, the exact100+600+300 development component union, or the current native-region producer's sealed output. Historical masks use a shape-and-packed-bits hash. The importer verifies that hash before producing the normalized raw-uint8 hash used by08N. Historical P1 scores and float16 global vectors are ignored.

`feature_remap.audit_remap(config)` verifies existing08N CPU features against the original full-development lock and all three component prediction locks. It maps each automatic hashed candidate ID back to the native observation ID using the original hash function, verifies RGB bytes against both sealed sources, verifies decoded masks and both hash formats, and recomputes crop/foreground geometry. It validates FP32 tensors and array indices without running a model or reading anomaly labels. The source proposal box can differ from the tight mask box; the encoder crop must match exactly.

Audit config fields:

- `source_region_root`, `expected_source_region_lock_sha256`.
- `source_feature_root`, `expected_source_feature_lock_sha256`.
- `full_development_lock`, `expected_region_lock_sha256`.
- `component_roots`: exact three native cohort IDs mapped to their model-safe directories.
- `remap_root`: new writable evidence directory.
- Optional `selected_views`: explicit subset of the frozen views; default all20.

`feature_remap.import_remapped(config)` takes `source_feature_root`, `remap_root`, `feature_root`, and optional positive `max_new_units`. It copies selected automatic NPZ files byte-for-byte, preserving array names and indices. Only the candidate-ID metadata changes. Its new feature binding names the full native region lock. The source08N directory is never modified. The imported subset has a `feature_shard_lock.json`, not a misleading complete-cohort feature lock; the audited crosswalk is included as sealed provenance.

CLI equivalents are `scripts/remap_native_features.py audit|import --config CONFIG.json`.

## One bounded preparation job

`native_cache.prepare_shard(config, source_root)` runs missing CPU FP32 DINO extraction and then cache assembly for one deterministic lexical scene partition. Use:

- `shard_count` and zero-based `shard_index`; omitted values mean the complete cohort.
- A writable `feature_root` for newly extracted units.
- Optional `feature_roots` containing immutable, compatible mounted reuse shards. Imported08N units are skipped without loading the encoder.
- `output_root` for the assembled scene shard.
- `model_source` for the verified fixed DINO snapshot; execution requires its separate pinned CPU environment.
- The usual exact region, neutral-correspondence, method-freeze and exposure/prerequisite bindings.
- Optional `max_new_units` and `max_new_scenes` for bounded continuation; rerun with identical scientific inputs.
- `maximum_output_bytes` reflecting retained disk space. Estimates use foreground counts for assembly and include checkpoint/released copies.

A feature shard can be mounted directly; do not copy all full-grid tensors into a merged working directory. Finished cache shards contain `cache_shard_lock.json` and `shard_manifest.json`. They are not accepted as complete prediction cohorts. Partial runs retain atomic units and record `checkpointed`; corrupt or incompatible units stop.

For practical queue sizing,100 scenes per job means10 development,6 calibration,4 internal-lock,8 official-seen and25 official-unseen chunks. Verify runtime/output size with the configured pilot before launching the remaining chunks. Merge input mounts must fit the current Kaggle limit; the launch manifest is authoritative for actual counts.

## Full-cohort merge

`cache_shards.merge_cache_shards(config, source_root)` takes the normal `role` and freeze/exposure/prerequisite fields, `cache_shards`, `output_root`, and optional `maximum_output_bytes`. It verifies every shard and exact input bindings, rejects overlaps, and requires the exact full cohort membership before publishing a cache seal. It copies only released RGB, foreground NPZ and scene metadata. Full-grid feature files and private checkpoints are not copied.

CLI: `scripts/merge_native_cache_shards.py --config CONFIG.json --source-root SOURCE_ROOT`.

Only the final merged cache can enter09A/09C/10A/10A1. Neutral correspondence stays explicitly assisted; anomaly labels remain in a separate evaluator resource until predictions are sealed. Preparing these scripts/notebooks is not evidence that Kaggle inference or evaluation has run.

## Bounded missing-region inference

`native_regions.extract_regions(config, source_root)` also accepts `shard_count` and zero-based `shard_index`, using the same lexical scene partition rule. `maximum_new_views=2000` bounds a100-scene job over20views. LocateAnything proposal checkpoints and SAM region checkpoints remain separate, so a SAM interruption reuses completed proposals. Only selected partition images are passed to workers; the complete canonical RGB package is verified and remains read-only.

A finished partition emits `region_shard_manifest.json` and `region_shard_lock.json`, with the full cohort scene/image denominator recorded separately from completed partition counts. Unprocessed views are absent, never fabricated as empty detections. Empty detections are retained only for views that actually reached the explicit terminal policy. Every region is projected to the normalized geometry allowlist; inherited scores or embeddings do not enter this resource.

`region_shards.merge_region_shards(config, source_root)` takes `region_shards`, `rgb_root`, the normal role/gate fields, `output_root` and optional `maximum_output_bytes`. It requires all nonoverlapping declared partitions, verifies the exact RGB/vision/runtime bindings, and copies only region JSON. Its final `region_manifest.json`/`artifact_lock.json` resource is accepted by `region_format=current-native-regions`. The merge does not copy RGB or run any model.
