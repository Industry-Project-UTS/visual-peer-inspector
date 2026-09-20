# Current runtime and exposure preparation

## What is verified

No Kaggle notebook, model inference or protected dataset experiment ran during this preparation. The prepared source remains local and uncommitted. Runtime package metadata, public wheel downloads and neutral administrative membership checks are setup work, not accuracy evidence.

| Profile | Exact preparation | Still to verify on Kaggle |
| --- | --- | --- |
| CPU cached decisions | Existing 15 package pins; 15 Linux CPython 3.12 wheels, 86,017,627 bytes, SHA-256 and ZIP CRC verified | Installation, actual Python patch and development numerical compatibility |
| DINO FP32 CPU | Torch 2.13.0+cpu, Transformers 5.14.1, immutable DINO snapshot; 47-package Linux closure locked with hashes | Wheel preparation, installation, model loading and development-only feature compatibility |
| LocateAnything/SAM CUDA | Torch 2.10.0+cu126, Torchvision 0.25.0+cu126, Transformers 4.57.1 and exact worker dependencies; 70-package Linux closure locked with hashes | Wheel builds, installation, CUDA model loading, memory use and development-only output compatibility |

DINO and region workers use separate environments. The CPU decision profile does not contain either model stack. The candidate runtime profiles are in `requirements/profiles/`; they do not claim that the Kaggle image was observed. The current Kaggle Docker source uses Python 3.12 and freezes image dependencies, so the default image packages are not a reliable substitute for these exact profiles. The pinned Docker source and its latest release metadata do not identify a particular saved notebook's image. [Official Kaggle Docker source](https://github.com/Kaggle/docker-python/blob/1ed8d43599a7faa1b630baeba82c57def3a8e33b/Dockerfile.tmpl)

CUDA 12.6 wheels are prepared for the available T4 path. P100 retirement was announced for 15 September 2026; do not retain a P100-only launch assumption. [Kaggle accelerator announcement](https://www.kaggle.com/discussions/product-announcements/735239), [official PyTorch version/wheel matrix](https://pytorch.org/get-started/previous-versions/)

## Setup-only job

Use the clean `notebooks/input-preparation/current-kaggle-runtime-setup.ipynb`. Its first cell verifies source using the standard library before any scientific imports. Set `VPI_NEXT_CONFIG` to a concrete JSON configuration. Prepare one job per profile:

```json
{
  "profile": "dino",
  "action": "prepare",
  "build_wheelhouse": true,
  "wheelhouse": "/kaggle/working/setup-dino/wheelhouse",
  "target": "/tmp/vpi-runtime-setup-dino",
  "output_root": "/kaggle/working/setup-dino",
  "download_models": true,
  "models_only": "all",
  "model_asset_root": "/kaggle/working/setup-dino/model-assets"
}
```

Use `profile=cpu|dino|regions`. `action=probe` records metadata only. A CPU setup can use the already verified wheelhouse with `build_wheelhouse=false`. DINO and region wheel builders use exact resolved source locks, official indexes, hash verification and a 1,800-second subprocess limit. CUDA wheels can be prepared without allocating a GPU. Package installation targets a fresh `/tmp/vpi-runtime-*` directory, never the global notebook environment. Only the small dependency receipt is copied to retained output; installed package files are ephemeral. Keep wheels and observed profiles under `/kaggle/working` for later jobs. If a dependency fails, retain its receipt and retry setup in a fresh directory.

The optional model downloader downloads pinned artifacts without constructing or calling a model. Each selected model has a 1,800-second download budget; the wrapper allows another 120 seconds per model for verification (5,760 seconds total for all three). A timeout stops the complete child process group and retains a failure receipt. It emits immutable per-model manifests plus `model_assets_receipt.json`. Set `download_models` only after preparing a DINO or regions profile. Reuse its retained output in later jobs; do not upload the same large weights from a laptop unnecessarily.

The output contains `wheelhouse/`, `runtime-profile/runtime_profile.json`, `runtime-profile/setup_probe.json`, and `runtime-profile/runtime_profile_lock.json`. The observed profile can change only the Python 3.12 patch from the candidate profile. All package and device pins stay exact. This explicit setup record is created before the new scientific method freeze; it cannot bypass a later freeze or make a changed runtime resumable.

Before a downstream notebook's normal bootstrap, the generated launch first cell sets:

```python
os.environ['VPI_RUNTIME_PROFILE_ROOT'] = '/kaggle/input/SETUP_OUTPUT/runtime-profile'
os.environ['VPI_WHEELHOUSE'] = '/kaggle/input/SETUP_OUTPUT/wheelhouse'
os.environ['VPI_PACKAGE_TARGET'] = '/tmp/vpi-runtime-current-job'
os.environ['VPI_INSTALL_DEPENDENCIES'] = '1'
```

DINO additionally selects the source-relative `VPI_BASE_RUNTIME_PROFILE=requirements/profiles/linux-dino-fp32-cpu.json`. The region notebook chooses its registered GPU profile explicitly. Never pass an absolute path as the base-profile selector. Runtime mismatch receipts print safe expected/actual versions before scientific imports. Exact metadata agreement is necessary but does not replace a development compatibility check.

The source-only transport includes all transitive source files, profile locks, builders and clean preparation notebooks. Wheels, model assets, private reference packages, runtime outputs and credentials remain separate resources. Preserve the generated source manifest and per-job configuration alongside the setup receipts.

## Original-role exposure

The administrative source is the retained `restricted_source_map.jsonl` under the original 2,000-scene administrative vault. Preparation projects only `role`, `opaque_scene_id` and `source_base_asset_id`; it does not use anomaly answers or result scores.

| Original role | Scenes | Asset groups | Defensible access interpretation |
| --- | ---: | ---: | --- |
| Development | 1,000 | 273 | Documented reuse in completed development processing and evaluation |
| Calibration | 600 | 163 | Prior exploratory access remains unknown; potentially exposed |
| Internal lock | 400 | 109 | Prior exploratory access remains unknown; potentially exposed |

All memberships remain fixed. Renaming or reshuffling scenes does not remove previous access. The available handover identifies earlier exploratory work but does not resolve the exact reserved-asset access history. Accordingly, the new policy is **exposure-qualified fixed-method evaluation**, with `no_retuning=true`, `independent_confirmation=false` and `untouched_assets_claim=false`.

The new `vpi_next_evaluation.exposure` helper accepts this policy without changing the historical strict gate. It seals the actual neutral indexes, asset counts, evidence-source hashes and the unresolved-history statement. The calibration/internal protocol receipts carry that qualification. Accuracy can be measured and reported under this declared scope; it cannot be described as independent confirmation.

Local preparation configuration is `.local/exposure-preparation-config.json`; prepared private evidence is `.local/exposure-qualified-original2000/`. Recreate it with:

```bash
python scripts/prepare_exposure_qualified.py --config .local/exposure-preparation-config.json
```

Use a fresh output directory; preserve the existing audit. Its neutral index maps opaque scene IDs to `source_base_asset_id`. Native reference and evaluator producers must use the identical asset strings and canonical scene-index bytes for grouped uncertainty and exposure binding.

## Official roles remain separate

Official seen and unseen are later, separate scopes; they do not replace the original 2,000 scenes. `prepare_official_exposure(config)` accepts sealed neutral geometry roots for both roles. It verifies the official source membership, opaque scene membership, scene and asset counts, exact scene-index hashes and label-free geometry manifests before sealing a separate audit.

The resulting official access history is explicitly unknown/potentially exposed. An empty event list means missing history, not evidence of no access. The official claim stays exposure-qualified with no independent-confirmation claim, and current 09D completion remains a required prerequisite. The helper cannot create a valid audit from invented counts, arbitrary scene subsets or a hand-written success receipt.

## Remaining execution gates

1. Run setup-only jobs and retain the wheelhouse, observed profile and pinned model assets.
2. Validate the unchanged methods and required model outputs on development evidence only; record the actual runtime before creating the new method freeze.
3. Prepare and verify exact native RGB, regions, feature caches, neutral identity resources and separate evaluator packages for each fixed cohort. Predict and seal all selected-view decisions before evaluator access.
4. Execute the qualified original-role stages in order. Execute official roles only after current 09D and their separate audit are valid.
5. Save scientific seals before rendering or bounded aggregate-only ClearML publication. Report-only and publication-only retries reuse sealed science. Checkpoint survival still depends on actual retention/export of the required files.

This document describes prepared execution paths and verified setup evidence. It does not report new model performance or a successful Kaggle run.

## Separate publication runtime

ClearML 2.1.10 is prepared in `linux-tracking-cp312.json` with a 33-package exact Linux dependency lock. It does not change the scientific CPU profile. The runtime setup notebook accepts `profile=tracking`, with the same verified wheelhouse and observed-profile outputs and `download_models=false`.

After science and reports are sealed, set `VPI_TRACKING_PROFILE_ROOT`, `VPI_TRACKING_WHEELHOUSE` and `VPI_TRACKING_TARGET` for a fresh publisher target. Run `scripts/prepare_tracking_runtime.py --source-root CODE_ROOT` to perform the bounded offline dependency installation. This preparation contacts no ClearML service. Then run the existing `scripts/publish_next_stage.py` with the exact stage, sealed evaluation/report pair and receipt path.

The publisher verifies those seals first, checks the already prepared tracking target, and propagates its private dependency paths into the bounded child through `PYTHONPATH`. Dependency setup is outside the 180-second service-publication bound. Missing setup leaves a pending receipt and preserves scientific results; it cannot trigger inference or evaluation. Normal local ClearML configuration or configured Kaggle Secrets remain the credential sources. No credential value belongs in the setup config, source package or output log.
