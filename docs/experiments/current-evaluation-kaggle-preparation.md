# Current 09/10 Kaggle preparation

## Status and boundary

This source is prepared and tested locally with synthetic fixtures. No protected or official dataset experiment, Kaggle installation, upload, run or ClearML publication was performed during this preparation. Public runtime metadata and 15 Linux CPU wheels were downloaded and verified locally; neural dependency closures were resolved without installing them. Passing source and dependency checks does not establish actual scientific execution or satisfy the required retained input, exposure and protocol bindings.

The source-only package includes the complete Python dependency tree, clean current notebooks, their metadata and a runtime contract. It excludes dataset inputs, recovery files, notebook outputs, private review evidence, model weights and credentials. Current cache/input preparation is a separate notebook under `notebooks/input-preparation`; neural extraction requires its own pinned neural runtime. The current missing-DINO adapter uses CPU FP32; the prepared LocateAnything/SAM producer uses a separately pinned GPU runtime. The CPU 09/10 profile itself performs cached decisions only.

## 1. Review the target runtime before freezing

`requirements/current-evaluation-runtime.json` records the verified **local macOS CPU** profile: Python 3.12.14 and exact scientific/transitive package versions. These versions are not yet verified on Kaggle Linux. Do not start a protected stage just to discover whether they install.

Use the clean runtime-setup notebook to inspect the eventual Kaggle kernel and prepare one of the three isolated Linux profiles. The CPU wheel bytes are verified locally; the DINO and CUDA profiles have exact resolved dependency locks plus bounded wheel-build scripts. See [runtime and exposure preparation](current-runtime-and-exposure-preparation.md) for concrete paths and setup configuration. If a different profile is required, review and record the exact version change, run development-only numerical compatibility checks, then create the new method freeze with that profile. Do this before calibration/09D. Never silently repin a frozen run or mix profiles when resuming.

The bootstrap checks Python and installed package metadata **before** pandas, NumPy or application imports. A mismatch emits expected/actual Python and package versions and writes a safe receipt. It installs only when the launch configuration explicitly enables the verified offline wheelhouse into a fresh private target. Kaggle's notebook kernel/IPython is expected to exist already; live pip availability and wheel compatibility remain setup checks.

An optional offline installation requires all of:

- `VPI_INSTALL_DEPENDENCIES=1`.
- `VPI_WHEELHOUSE`: a reviewed wheel directory containing `wheelhouse_manifest.json` with schema `vpi-current-offline-wheels/v1`, the exact `runtime_contract` object, and `files` mapping every wheel filename to SHA-256.
- `VPI_PACKAGE_TARGET`: a fresh dependency-only directory under `/tmp/vpi-runtime-*`.

Installation uses `python -m pip`, no network/index, exact package pins, no dependency solving, a 300-second CPU or 900-second neural timeout, and a fresh target. Neural wheel manifests include the exact resolved dependency closure. It does not replace global packages; the private target shadows the selected pinned dependencies, including Torch only for a neural profile. Missing pip, a wrong wheel, incompatible Python, already-imported conflicting scientific modules or a partial install stops setup. Keep the failure receipt; restart before retrying with a fresh target. ClearML remains optional and is imported only in its post-science publisher; its Kaggle Secrets/environment configuration is never stored in the source package.

## 2. Build the source transport locally

From a fully materialized working-stage source tree, use the existing scientific Python environment:

```bash
python scripts/package_current_notebooks.py --source-root . --output-root /absolute/new/source-package --base-commit THE_RECORDED_BASE_COMMIT
```

This creates a verified `source/` tree, `current-evaluation-source.zip.bin`, and `package_receipt.json`. The manifest records exact file hashes and `source_state=working_directory_snapshot`; it makes no clean-Git or committed-source claim. Build again into a new directory after any edit. Do not modify a packaged source tree in place. The complete source snapshot and notebook hash appear in each invocation receipt, while the scientific method freeze binds all scoring/evaluation dependencies.

Kaggle can expose an expanded source tree or the archive. Prefer the expanded tree. Set `VPI_CODE_ROOT` when multiple copies are attached. Discovery supports flat mounts and `datasets`, `notebooks` and versioned `models` namespaces through fixed-depth bounded inventories; it rejects ambiguous roots and never scans raw-image trees recursively. For the archive fallback, explicitly set `VPI_SOURCE_ARCHIVE`, `VPI_SOURCE_ARCHIVE_SHA256` from the verified package receipt, and a fresh `VPI_SOURCE_DEST` under `/kaggle/working`. It checks archive size, CRC, paths, duplicate members and source hashes before imports.

## 3. Attach only the stage resources

Set `VPI_NEXT_CONFIG` to the exact reviewed JSON file. `VPI_CURRENT_STAGE_CONFIG` remains an alias. Automatic discovery is limited to fixed config subdirectories and rejects ambiguity. Model/prerequisite resource roots resolve a bounded number of nested/expanded levels. Evaluator discovery stays after prediction validation.

Use `/kaggle/working` for scientific output, reports, runtime receipts, verified wheels, model assets and recovery paths. Installed dependencies belong in fresh `/tmp/vpi-runtime-*` targets; this avoids retaining thousands of package files as Kaggle output. Prediction stages must not mount evaluator packages. Current 09 preserves the original calibration/internal-lock membership. The new exposure-qualified policy retains uncertain prior access explicitly, requires fixed methods, and makes no untouched or independent-confirmation claim. The historical strict policy is unchanged. Current 10 additionally requires completed current 09D and a separate official membership/exposure audit; seen and unseen stay separate. The existing historical P1 inputs are not current resources.

The typed CPU decision and evaluator paths work with compatible synthetic inputs. Prepared native producers and evaluator converters still need their actual source bindings, retained complete outputs and development compatibility checks before protected execution. See each current readiness document; do not satisfy a missing prerequisite with a hand-written success flag.

## 4. Checkpoints and actual retention

Scientific methods use atomic scene checkpoints and bind exact method, input, runtime, source and schedule evidence. `run_stage` writes safe elapsed-time/error-type receipts outside science. With `recovery_export_dir` configured, it attempts a versioned recovery export after completion or failure; the default export bound is 512 MiB and `recovery_max_bytes` must be explicitly raised for larger state after storage review.

The export includes only the explicitly named stage output tree with `run_binding.json`. It refuses mounted Kaggle inputs and symlinks. It uses the existing archive CRC, file hash, allowlist and binding verification. It does not export evaluator input packages. Recovery is private and may contain per-scene scientific outputs; it never goes to aggregate ClearML publication.

For an explicit snapshot after a completed scene checkpoint:

```python
from current_stage_recovery import export_state
receipt = export_state(output_root, new_archive_path, new_sidecar_path,
                       stage=STAGE, provenance=CODE_PROVENANCE)
```

Retain/download both archive and sidecar plus the receipt's `binding`. In a later session, restore before running science:

```python
from current_stage_recovery import restore_state
restored = restore_state(archive_path, sidecar_path, new_destination,
                         expected_binding=the_saved_binding)
```

Use `restored['state_directory']` as that invocation's output path. Normal method/input/runtime/scene-lock checks must still pass. A wrong binding, corruption or an existing restore destination stops restoration. Report-only repairs may use changed report renderer code in a **new report directory**; their separate renderer hashes are recorded and scientific source remains frozen.

A checkpoint survives interruption only if the required files were actually retained. This implementation does not guarantee persistence after abrupt session loss before export completes, nor does it automatically download files. During a long run, explicit exports after completed checkpoints are still required if Kaggle output retention is uncertain.

## 5. Reporting and tracking retries

Save and verify predictions before evaluator access. Seal scientific tables before plotting. Reports create coordinate-checked private galleries and nine aggregate figures from the sealed results. Missing media/undefined metrics remain explicit.

`10B` supports `action=report_only`. Current 09 reports can likewise be rerun from their sealed evaluation root. Changed renderer revisions use a fresh report directory; they do not change the scientific freeze or reopen scene labels. For publication-only retry, run `scripts/publish_next_stage.py` with the same receipt and exact sealed evaluation/report pair.

ClearML publication runs after sealing in a separate worker bounded to 180 seconds. Only seven allowed aggregate artifacts and nine allowed figures can be published. Task IDs are saved before upload and reused on retry. Failure leaves a pending receipt and preserves science. Keep tracking receipts outside scientific/report seals. Live delivery, credentials, service availability and eventual Kaggle recovery retention remain unverified external conditions.
