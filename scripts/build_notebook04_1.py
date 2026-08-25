#!/usr/bin/env python3
"""Build the clean Notebook 04.1 source from reviewed cells."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


CODE_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_RELATIVE = Path(
    "notebooks/detection-and-segmentation/toysad8k-ds1-locateanything/"
    "04.1-toysad8k-ds1-locateanything-screen.ipynb"
)


def markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(source).strip().splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).strip().splitlines(keepends=True),
    }


CELLS = [
    markdown(
        """
        # ToysAD-8K DS1 LocateAnything screen

        This notebook applies the frozen DS0 LocateAnything settings to the fixed
        100-scene DS1 screen. It runs one-image and 20-image gates before the
        resumable 2,000-image job. The model receives opaque RGB records only.

        Box accuracy is evaluated later in the protected evaluator. This notebook
        records operational reliability, proposal geometry, runtime, memory,
        retries and failures without opening labels or source identifiers.
        """
    ),
    markdown(
        """
        ## Acceptance contract

        The accepted run must use all 100 scenes and all 20 views, preserve the
        frozen cohort membership and model revision, pass exact coverage and box
        validity gates, reproduce the 20-image pilot, and finish aggregate-only
        ClearML finalisation after inference.
        """
    ),
    code(
        """
        from __future__ import annotations

        import importlib.metadata
        import json
        import os
        from pathlib import Path
        import platform
        import signal
        import subprocess
        import sys
        import time

        sys.dont_write_bytecode = True

        from IPython.display import HTML, Image as NotebookImage, display


        NOTEBOOK_RELATIVE = Path(
            "notebooks/detection-and-segmentation/toysad8k-ds1-locateanything/"
            "04.1-toysad8k-ds1-locateanything-screen.ipynb"
        )
        OUTPUT_ROOT = Path("/kaggle/working/toysad8k_ds1_locateanything_screen")
        EXECUTION_ENVIRONMENT = "kaggle" if Path("/kaggle/input").is_dir() else "local"
        RUN_MODE = "model_inference" if EXECUTION_ENVIRONMENT == "kaggle" else "contract_smoke"
        INFERENCE_STAGE = os.getenv(
            "VPI_INFERENCE_STAGE", "full" if EXECUTION_ENVIRONMENT == "kaggle" else "one_image"
        ).strip().lower()
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        """
    ),
    markdown("## Locate the pinned code and DS1 model input"),
    code(
        """
        def locate_code_root() -> Path:
            candidates = [Path.cwd(), *Path.cwd().parents]
            input_root = Path("/kaggle/input")
            if input_root.is_dir():
                candidates.extend(path for mounted in input_root.iterdir() for path in (mounted, mounted / "code"))
            configured = os.getenv("VPI_CODE_ROOT")
            if configured:
                candidates.insert(0, Path(configured).expanduser())
            for candidate in candidates:
                marker = candidate / "src/visual_peer_inspector/workflows/locateanything_ds1.py"
                if marker.is_file():
                    return candidate.resolve()
            raise FileNotFoundError("Could not locate the Visual Peer Inspector code snapshot")


        def locate_model_input_root() -> Path:
            configured = os.getenv("VPI_MODEL_INPUT_ROOT")
            candidates = [Path(configured).expanduser()] if configured else []
            input_root = Path("/kaggle/input")
            if input_root.is_dir():
                candidates.extend(path.parent for path in input_root.rglob("neutral_corpus_manifest.jsonl"))
            candidates.extend((Path.cwd(), *Path.cwd().parents))
            for candidate in candidates:
                if (candidate / "neutral_corpus_manifest.jsonl").is_file():
                    return candidate.resolve()
                if (candidate / "model-input/neutral_corpus_manifest.jsonl").is_file():
                    return (candidate / "model-input").resolve()
            raise FileNotFoundError("Could not locate the DS1 model-safe input package")


        CODE_ROOT = locate_code_root()
        MODEL_INPUT_ROOT = locate_model_input_root()
        sys.path.insert(0, str(CODE_ROOT / "src"))

        from visual_peer_inspector.data.cohort_contract import DS1_SCREEN_CONTRACT
        from visual_peer_inspector.data.model_safe_package import file_sha256
        from visual_peer_inspector.models.locateanything import LocateAnythingConfiguration

        print({
            "code_root": CODE_ROOT.name,
            "model_input_root": MODEL_INPUT_ROOT.name,
            "cohort": DS1_SCREEN_CONTRACT.cohort_id,
            "images": DS1_SCREEN_CONTRACT.image_count,
        })
        """
    ),
    markdown("## Install the pinned inference dependencies"),
    code(
        """
        if RUN_MODE == "model_inference":
            requirement = CODE_ROOT / "requirements/locateanything-lock.txt"
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "-r", str(requirement)],
                timeout=600,
            )

        configuration = LocateAnythingConfiguration()
        display({
            "model": configuration.model_id,
            "revision": configuration.model_revision,
            "prompt": configuration.prompt,
            "configuration_sha256": configuration.sha256,
            "run_mode": RUN_MODE,
            "inference_stage": INFERENCE_STAGE,
        })
        """
    ),
    markdown(
        """
        ## Restore a private checkpoint and run the disposable GPU watchdog

        A prior partial checkpoint may be attached as a private Kaggle dataset.
        The archive contains raw model responses and must never be public or
        committed to GitHub. Before the main worker is loaded, a separate process
        must load the model and finish one real image within the configured time.
        A timed-out process group is terminated so its CUDA context cannot be reused.
        Passing this gate proves only that this Kaggle runtime can process one image.
        It is not a general claim that LocateAnything supports T4.
        """
    ),
    code(
        """
        from visual_peer_inspector.models.locateanything import MODEL_PRODUCER_KIND
        from visual_peer_inspector.workflows.locateanything_ds1 import import_ds1_checkpoint


        input_manifest_sha256 = file_sha256(MODEL_INPUT_ROOT / "neutral_corpus_manifest.jsonl")
        configured_checkpoint = os.getenv("VPI_LOCATEANYTHING_CHECKPOINT_ARCHIVE")
        checkpoint_candidates = (
            [Path(configured_checkpoint).expanduser()]
            if configured_checkpoint
            else list(Path("/kaggle/input").rglob("ds1_locateanything_checkpoint.zip"))
            if Path("/kaggle/input").is_dir()
            else []
        )
        if len(checkpoint_candidates) > 1:
            raise RuntimeError("Attach only one private DS1 LocateAnything checkpoint archive")
        checkpoint_import = None
        if checkpoint_candidates:
            checkpoint_import = import_ds1_checkpoint(
                checkpoint_candidates[0],
                OUTPUT_ROOT,
                configuration_sha256=configuration.sha256,
                producer_kind=(
                    MODEL_PRODUCER_KIND if RUN_MODE == "model_inference" else "contract_smoke"
                ),
                input_manifest_sha256=input_manifest_sha256,
            )
            display({"checkpoint_imported": True, "files": sorted(checkpoint_import["files"])})

        watchdog_receipt = OUTPUT_ROOT / "disposable_preflight_receipt.json"
        watchdog_result = {"status": "not_required", "reason": "contract_smoke"}
        if RUN_MODE == "model_inference":
            watchdog_timeout = int(os.getenv("VPI_ONE_IMAGE_WATCHDOG_SECONDS", "900"))
            if not 120 <= watchdog_timeout <= 1800:
                raise ValueError("The one-image watchdog must be between 120 and 1,800 seconds")
            command = [
                sys.executable,
                str(CODE_ROOT / "scripts/run_locateanything_one_image_watchdog.py"),
                "--model-input-root",
                str(MODEL_INPUT_ROOT),
                "--receipt",
                str(watchdog_receipt),
            ]
            watchdog_environment = os.environ.copy()
            watchdog_environment["PYTHONPATH"] = str(CODE_ROOT / "src")
            process = subprocess.Popen(
                command,
                env=watchdog_environment,
                start_new_session=True,
            )
            watchdog_started = time.monotonic()
            while process.poll() is None:
                elapsed = time.monotonic() - watchdog_started
                if elapsed >= watchdog_timeout:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=20)
                    raise TimeoutError(
                        f"LocateAnything disposable one-image gate exceeded {watchdog_timeout} seconds"
                    )
                print(
                    json.dumps(
                        {"stage": "disposable_preflight", "elapsed_seconds": round(elapsed, 1)},
                        sort_keys=True,
                    ),
                    flush=True,
                )
                time.sleep(15)
            if not watchdog_receipt.is_file():
                raise RuntimeError("The disposable one-image process did not write a receipt")
            watchdog_result = json.loads(watchdog_receipt.read_text(encoding="utf-8"))
            display(watchdog_result)
            if process.returncode != 0 or watchdog_result.get("status") != "passed":
                raise RuntimeError("The disposable one-image model gate failed")
        """
    ),
    markdown(
        """
        ## Load the worker

        ClearML is not started here. Model loading and inference finish before
        tracking begins, so a tracking outage cannot freeze the GPU job.
        """
    ),
    code(
        """
        from visual_peer_inspector.models.locateanything import (
            LocateAnythingWorker,
            ProposalContractWorker,
        )

        load_checkpoints = []
        load_started = time.perf_counter()

        def record_load(stage: str, details: dict | None = None) -> None:
            load_checkpoints.append({
                "stage": stage,
                "elapsed_seconds": round(time.perf_counter() - load_started, 3),
                **dict(details or {}),
            })
            print(json.dumps(load_checkpoints[-1], sort_keys=True), flush=True)


        if RUN_MODE == "model_inference":
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("Notebook 04.1 requires a Kaggle GPU")
            torch.cuda.reset_peak_memory_stats()
            device = torch.cuda.get_device_properties(0)
            device_summary = {
                "cuda_available": True,
                "device_count": torch.cuda.device_count(),
                "device_name": torch.cuda.get_device_name(0),
                "compute_capability": f"{device.major}.{device.minor}",
                "total_memory_gib": round(device.total_memory / (1024 ** 3), 3),
            }
            worker = LocateAnythingWorker(configuration, progress_callback=record_load)
        else:
            device_summary = {"cuda_available": False, "device_count": 0}
            worker = ProposalContractWorker(configuration)

        model_load_seconds = round(time.perf_counter() - load_started, 3)
        display({"model_load_seconds": model_load_seconds, **device_summary})
        """
    ),
    markdown("## Run the preflight, repeatability gate and resumable proposal job"),
    code(
        """
        from visual_peer_inspector.verification import verify_code_snapshot_record
        from visual_peer_inspector.workflows.locateanything_ds1 import (
            run_ds1_locateanything_workflow,
        )

        snapshot_path = CODE_ROOT / "snapshot_record.json"
        if RUN_MODE == "model_inference" and INFERENCE_STAGE == "full" and not snapshot_path.is_file():
            raise RuntimeError("A canonical full run requires snapshot_record.json")
        snapshot = verify_code_snapshot_record(CODE_ROOT, snapshot_path) if snapshot_path.is_file() else {}
        notebook_path = CODE_ROOT / NOTEBOOK_RELATIVE
        if RUN_MODE == "model_inference" and INFERENCE_STAGE == "full":
            if snapshot.get("git_dirty") is not False or snapshot.get("precommit_candidate") is not False:
                raise RuntimeError("A canonical full run requires a clean committed code snapshot")
            if not notebook_path.is_file():
                raise RuntimeError("The code snapshot must include the exact Notebook 04.1 source")
        provenance = {
            "git_commit": snapshot.get("git_commit", "local-contract"),
            "git_dirty": snapshot.get("git_dirty", True),
            "precommit_candidate": snapshot.get("precommit_candidate", True),
            "snapshot_sha256": snapshot.get("snapshot_sha256", ""),
            "notebook_sha256": (
                __import__("hashlib").sha256(notebook_path.read_bytes()).hexdigest()
                if notebook_path.is_file() else ""
            ),
            "kaggle_notebook_slug": "esreekarreddy/toysad-8k-ds1-locateanything-screen",
        }
        platform_versions = {
            "python": platform.python_version(),
            "pillow": importlib.metadata.version("Pillow"),
            "numpy": importlib.metadata.version("numpy"),
            "transformers": (
                importlib.metadata.version("transformers")
                if RUN_MODE == "model_inference" else "not-loaded"
            ),
            "torch": (
                importlib.metadata.version("torch")
                if RUN_MODE == "model_inference" else "not-loaded"
            ),
            "torchvision": (
                importlib.metadata.version("torchvision")
                if RUN_MODE == "model_inference" else "not-loaded"
            ),
            "cuda_runtime": (
                str(torch.version.cuda) if RUN_MODE == "model_inference" else "not-loaded"
            ),
            "cudnn": (
                str(torch.backends.cudnn.version())
                if RUN_MODE == "model_inference" else "not-loaded"
            ),
            "kaggle_container_image": os.getenv("KAGGLE_CONTAINER_IMAGE", "unavailable"),
            "kaggle_kernel_run_type": os.getenv("KAGGLE_KERNEL_RUN_TYPE", "unavailable"),
        }
        load_evidence = {
            "load_seconds": model_load_seconds,
            "download_seconds": 0.0,
            "checkpoints": load_checkpoints,
            "disposable_preflight": watchdog_result,
            "device": device_summary,
            "peak_gpu_allocated_gib": (
                round(torch.cuda.max_memory_allocated() / (1024 ** 3), 4)
                if RUN_MODE == "model_inference" else 0.0
            ),
            "peak_gpu_reserved_gib": (
                round(torch.cuda.max_memory_reserved() / (1024 ** 3), 4)
                if RUN_MODE == "model_inference" else 0.0
            ),
        }
        run_record = run_ds1_locateanything_workflow(
            model_input_root=MODEL_INPUT_ROOT,
            output_root=OUTPUT_ROOT,
            worker=worker,
            configuration=configuration,
            run_mode=RUN_MODE,
            inference_stage=INFERENCE_STAGE,
            provenance=provenance,
            platform_versions=platform_versions,
            model_load_evidence=load_evidence,
        )
        display({
            "status": run_record["status"],
            "images": run_record["summary"]["job_count"],
            "successful": run_record["summary"]["successful_job_count"],
            "repeat_agreement": run_record["repeat_agreement"].get("box_agreement_rate"),
        })
        """
    ),
    markdown("## Review aggregate operational evidence"),
    code(
        """
        import pandas as pd

        safe_root = OUTPUT_ROOT / "safe"
        display(HTML("<h4>Completion gates</h4>"))
        display(pd.read_csv(safe_root / "completion_gates.csv"))
        display(HTML("<h4>Proposal summary by view</h4>"))
        display(pd.read_csv(safe_root / "proposal_summary_by_view.csv"))
        display(HTML("<h4>Failure summary</h4>"))
        display(pd.read_csv(safe_root / "failure_summary.csv"))

        for name in (
            "proposal_count_distribution.png",
            "runtime_distribution.png",
            "mean_proposals_by_view.png",
            "box_area_fraction_distribution.png",
            "box_aspect_ratio_distribution.png",
            "runtime_by_proposal_count.png",
        ):
            display(NotebookImage(filename=str(safe_root / name)))
        """
    ),
    markdown(
        """
        ## Private visual review

        The panel below is retained with the private run output. It is not sent
        to ClearML or committed to GitHub.
        """
    ),
    code(
        """
        example_path = OUTPUT_ROOT / "private/proposal_examples.png"
        if example_path.is_file():
            display(NotebookImage(filename=str(example_path)))
            outlier_path = OUTPUT_ROOT / "private/proposal_outliers.png"
            if outlier_path.is_file():
                display(NotebookImage(filename=str(outlier_path)))
            if RUN_MODE == "contract_smoke":
                print("These are synthetic contract-smoke boxes, not LocateAnything predictions.")
        else:
            print("Proposal examples are generated by the full run only.")
        """
    ),
    markdown(
        """
        ## Export restart state

        The archive below can be uploaded as a private Kaggle dataset and attached
        to a fresh version. It contains private raw responses. Keep it outside
        GitHub, ClearML and public Kaggle datasets.
        """
    ),
    code(
        """
        from visual_peer_inspector.workflows.locateanything_ds1 import export_ds1_checkpoint


        checkpoint_export = export_ds1_checkpoint(
            OUTPUT_ROOT,
            Path("/kaggle/working/ds1_locateanything_checkpoint.zip"),
            configuration_sha256=configuration.sha256,
            producer_kind=run_record["producer_kind"],
            input_manifest_sha256=run_record["cohort"]["input_manifest_sha256"],
        )
        display({
            "checkpoint_file_count": len(checkpoint_export["files"]),
            "checkpoint_archive_sha256": checkpoint_export["archive_sha256"],
        })
        """
    ),
    markdown(
        """
        ## Finalise aggregate ClearML evidence

        This isolated, wall-time-bounded child uses the official ClearML hosts.
        It retries only transient task-creation connection failures and writes
        sanitized retry counts and failure categories to the receipt.
        """
    ),
    code(
        """
        tracking_mode = os.getenv(
            "VPI_TRACKING_MODE", "online" if EXECUTION_ENVIRONMENT == "kaggle" else "disabled"
        ).strip().lower()
        tracking_environment = os.environ.copy()

        tracking_receipt = OUTPUT_ROOT / "tracking_receipt.json"
        command = [
            sys.executable,
            str(CODE_ROOT / "scripts/finalize_locateanything_ds1_clearml.py"),
            "--safe-root",
            str(safe_root),
            "--receipt",
            str(tracking_receipt),
            "--mode",
            tracking_mode,
            "--wall-time-seconds",
            "180",
            "--tag",
            "notebook-04.1",
            "--tag",
            "ds1",
        ]
        try:
            subprocess.run(command, check=True, env=tracking_environment, timeout=210)
            tracking_result = json.loads(tracking_receipt.read_text(encoding="utf-8"))
        except Exception as error:
            tracking_result = {
                "schema_version": 1,
                "stage": "notebook_04_1_clearml_finalisation",
                "mode": tracking_mode,
                "status": "failed",
                "task_id": None,
                "error_type": type(error).__name__,
            }
            tracking_receipt.write_text(
                json.dumps(tracking_result, indent=2, sort_keys=True) + "\\n",
                encoding="utf-8",
            )
        display(tracking_result)
        if tracking_result["status"] == "failed":
            print("ClearML finalisation failed after compute. Model outputs remain complete.")
        """
    ),
    markdown("## Completion"),
    code(
        """
        from visual_peer_inspector.workflows.locateanything_ds1 import write_artifact_manifest

        manifest = write_artifact_manifest(safe_root)
        final_summary = {
            "run_status": run_record["status"],
            "completion_gate_passed": run_record["completion_gate_passed"],
            "safe_artifact_count": manifest["artifact_count"],
            "tracking_status": tracking_result["status"],
            "clearml_task_id": tracking_result.get("task_id"),
            "private_results": str(OUTPUT_ROOT / "private/proposal_results.jsonl"),
        }
        display(final_summary)
        if INFERENCE_STAGE == "full" and not run_record["completion_gate_passed"]:
            raise RuntimeError("Notebook 04.1 did not pass the full DS1 operational contract")
        """
    ),
    markdown(
        """
        The next trusted step joins the frozen proposal file to protected reference
        geometry. The model-facing SAM package receives only opaque image keys and
        reference-box or automatic-box prompts.
        """
    ),
]


def main() -> None:
    notebook = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    target = CODE_ROOT / NOTEBOOK_RELATIVE
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
