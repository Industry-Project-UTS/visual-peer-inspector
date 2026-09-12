# Notebook runtime and recovery checks

The experiment uses the existing 2,000-scene split and frozen model settings. Runtime repairs do not authorise a new prompt, backbone, view set or evaluation threshold.

## Checks before the next run

The source tree contains current notebooks, earlier experiments and executed evidence. The 18 remaining stage contracts are checked separately. Saved evidence pairs are excluded from current-source discovery. The regression suite covers recovery, transport, source integrity, reporting and model/evaluator separation.

| Previous failure or risk | Required behaviour | Verification |
| --- | --- | --- |
| Flat and nested Kaggle mount paths | Resolve the dataset or notebook mount, not the parent container | Generated helper tests and stage structure tests |
| Kaggle extracts a checkpoint ZIP | Accept the byte-preserved .zip.bin upload and its manifest | Archive discovery and recovery tests |
| Kaggle extracts the original dataset ZIP | Locate the archive index; verify identity files and every requested member's length and CRC | Equivalent ZIP/extracted fixture tests and ten actual source image reads |
| Kaggle creates a title-derived notebook URL | Use the canonical URL in metadata and downstream notebook inputs | Metadata contracts and the actual 08E.1 launch response |
| Missing or conflicting recovery inputs | Stop before inference; never guess an input version | Recovery identity and duplicate-input tests |
| Changed prompt, model, split or saved file | Reject mismatched hashes | Frozen identity, prediction-lock and shard tests |
| A crop heatmap is treated as a scene image | Project through the actual padded crop and object mask | Offset-mask, NaN and crop-projection tests |
| A completed shard is processed again | Reuse only a verified receipt and all its artifacts | Exact-resume tests |
| GPU memory, disk or permission failure becomes an image result | Stop; keep the completed shards and preserve the original error | Injected infrastructure-failure tests |
| CPU release fails between scenes | Retain atomic scene records and export a recovery archive | Interrupted-release and restore tests |
| Reporting fails after protected inference | Export completed inference state without pruning | Protected runner recovery path |
| ClearML stalls or dependencies are unavailable | Run tracking after saved results, with a bounded subprocess | Tracking timeout and receipt tests |
| DS2 reuse attestation is rejected by tracking | Allow the exact aggregate attestation, not arbitrary extra files | ClearML allowlist test |
| The notebook hides a long CPU operation | Stream aggregate progress every 25 scenes | Resource-release runner |
| Download is incomplete or contains altered bytes | Compare every file with the run's artifact register | All 1,007 Notebook 08E V5 entries verified |
| A model returns no usable region | Retain the terminal outcome and count it in reporting | Zero-proposal, empty-mask and insufficient-peer tests |
| Added executed notebooks change an older evidence count | Verify each evidence inventory separately | Inventory regression tests and clean-main transfer test |
| An archived clean notebook is treated as a second current source | Exclude the executed-evidence directory during source discovery | Clean-source archive regression test |
| Git line-ending conversion changes downloaded hashes | Preserve evidence bytes with scoped .gitattributes rules | Raw and filtered Git object hashes compared with core.autocrlf enabled |

## Recovery rules

1. Do not rerun model inference to repair tracking, figures or a failed download.
2. Preserve the original output directory. Put repaired reports and tracking receipts in a separate directory.
3. Reuse a shard only when the split, model settings, input hash, output hash and embedding hashes match.
4. Retry only bounded transient operations. Invalid data, credentials, permissions and hash mismatches need a specific fix, not an indefinite retry.
5. Write persistent checkpoints below /kaggle/working. A recovery ZIP must pass verification before loose state may be pruned.
6. On an interruption, retain the original error. A failed recovery export must not replace it.
7. Keep prediction inputs separate from evaluator references. Release protected resources only after their prerequisite lock exists.
8. A successful notebook run is not proof of perfect detection. Preserve missed objects, poor proposals and unsuccessful views in the evidence.

## Notebook 08E result

Kaggle Version 5 completed in 3 hours, 6 minutes and 31 seconds. The combined lock covers 1,000 development scenes and 20,000 views. All 1,007 registered output files passed their downloaded SHA-256 checks. The run produced 90 qualitative figures across the three cohorts.

ClearML was disabled for the screen and remainder cohorts in the Kaggle run. The confirmation tracking step also rejected the aggregate reuse attestation. These are tracking issues, not missing inference results. A separate post-run publication uses the original safe artifacts and records new receipts without changing the original Kaggle evidence.

Some qualitative examples contain missed objects and background proposals. Those examples remain in the review evidence. The next identity evaluation measures the saved predictions; it must not present them as ground truth or silently remove failures.

## Run order

- 08E.1: use locked development predictions and raw reference identities to create the private identity evaluator resource. CPU only.
- 08F: evaluate cross-view identity matching and freeze the application output contract using saved embeddings. No neural-model inference.
- 08F.1 and 09A: release calibration resources, then apply the frozen model route. Do not open internal-lock labels.
- 09B onward: follow the protected evaluation and official-test release gates already defined in the notebooks.

## Limits

Tests cover the known failure modes and the code-owned contracts. They cannot guarantee that Kaggle, a model host, the network or ClearML will never fail. Abrupt machine termination may occur before an exception handler runs. In that case, recovery depends on the most recent checkpoint files actually retained by Kaggle.

Use direct status and log checks while working. No recurring monitor is required.
