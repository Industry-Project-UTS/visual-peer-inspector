#!/usr/bin/env python3
"""Build the clean Notebook 09B calibration and review-policy notebook."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_ROOT = ROOT / "notebooks" / "protected-evaluation"
NOTEBOOK_PATH = NOTEBOOK_ROOT / "09b-toysad8k-calibration-review-policy.ipynb"
METADATA_PATH = NOTEBOOK_ROOT / "kernel-metadata-09b.json"


def markdown(source: str):
    return nbformat.v4.new_markdown_cell(source.strip())


def code(source: str):
    return nbformat.v4.new_code_cell(source.strip())


def build_notebook() -> nbformat.NotebookNode:
    cells = [
        markdown(
            r'''
# ToysAD-8K Calibration Evaluation and Review-Policy Lock

**Notebook file:** `09b-toysad8k-calibration-review-policy.ipynb`

This notebook evaluates the frozen Visual Peer Inspector predictions for 600 calibration scenes. It verifies the complete Notebook 09A prediction lock before opening the private calibration correspondence. The LocateAnything proposal policy, SAM 2.1 masks, DINOv2-small features, P1 scoring, cross-view identity method, view contracts, and fusion rule remain unchanged.

Only the review and calibration policy is selected here. The system always reports relative unusualness among the uploaded peer objects because ToysAD-8K does not contain native all-normal scenes.
'''
        ),
        markdown(
            r'''
## Acceptance contract

- The immutable 09A bundle must contain 12,000 prediction rows, float16 global embeddings, a complete embedding index, and 1,800 multi-view decisions.
- The 09A run record must confirm that no evaluator was opened and no calibration policy was applied during inference.
- The private evaluator opens only after the prediction lock, method record, and every locked artifact verify.
- Base asset groups remain intact across cross-validation folds.
- Candidate review signals cover zero or incomplete proposals, duplicate or fragmented proposals, empty or low-quality masks, unmatched or ambiguous identity, small P1 margin, view disagreement, and stage failure.
- Reported policy states are `result_ready`, `review_required`, and `insufficient_evidence`.
- Raw P1 scores and margins are not probabilities. Error probabilities are reported only after grouped calibration.
- No arbitrary universal accuracy threshold is used. The selected policy records the stated risk and review costs and preserves the complete risk-coverage frontier.
- Wrong `result_ready` examples and all evaluator-informed examples remain in evaluator-only output.
- ClearML receives aggregate tables, aggregate figures, and the frozen policy only after local outputs are complete.
'''
        ),
        code(
            r'''
from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import pandas as pd
from IPython.display import Image as NotebookImage, display

sys.dont_write_bytecode = True

NOTEBOOK_FILE = "09b-toysad8k-calibration-review-policy.ipynb"
NOTEBOOK_CODE = "09b"
COHORT_ID = "toysad8k-calibration-600-v1"
EXPECTED_SCENES = 600
EXPECTED_IMAGE_ROWS = 12_000
EXPECTED_SCENE_CONTRACTS = 1_800
CLEARML_PROJECT = "Visual Peer Inspector/06 Protected Evaluation"
EXECUTION_ENVIRONMENT = "kaggle" if Path("/kaggle/input").is_dir() else "local"
OUTPUT_ROOT = Path(os.getenv(
    "VPI_OUTPUT_ROOT",
    "/kaggle/working/toysad8k_calibration_review_policy"
    if EXECUTION_ENVIRONMENT == "kaggle"
    else ".local/runs/toysad8k_calibration_review_policy",
)).expanduser().resolve()
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
'''
        ),
        code(
            r'''
def kaggle_roots() -> list[Path]:
    input_root = Path("/kaggle/input")
    if not input_root.is_dir():
        return []
    return sorted((path for path in input_root.iterdir() if path.is_dir()), key=lambda path: path.as_posix())


def find_code_root() -> Path:
    configured = os.getenv("VPI_CODE_ROOT")
    candidates = [Path(configured).expanduser()] if configured else []
    candidates.extend((Path.cwd(), *Path.cwd().parents, *kaggle_roots()))
    for candidate in candidates:
        if (candidate / "scripts/run_calibration_review_policy.py").is_file():
            return candidate.resolve()
        if (candidate / "code/scripts/run_calibration_review_policy.py").is_file():
            return (candidate / "code").resolve()
    raise FileNotFoundError("Could not locate the Visual Peer Inspector code snapshot")


def find_named_file(name: str, environment_name: str) -> Path:
    configured = os.getenv(environment_name)
    if configured and Path(configured).expanduser().is_file():
        return Path(configured).expanduser().resolve()
    matches = [path.resolve() for root in kaggle_roots() for path in root.rglob(name) if path.is_file()]
    if not matches:
        raise FileNotFoundError(f"Could not locate {name}")
    return sorted(matches)[0]


def find_09a_prediction_root() -> Path:
    configured = os.getenv("VPI_09A_MODEL_SAFE_ROOT")
    candidates = [Path(configured).expanduser()] if configured else []
    candidates.extend(path.parent for root in kaggle_roots() for path in root.rglob("prediction_lock.json"))
    matches = []
    for candidate in candidates:
        lock_path = candidate / "prediction_lock.json"
        if not lock_path.is_file():
            continue
        try:
            record = json.loads(lock_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if record.get("cohort_id") == COHORT_ID:
            matches.append(candidate.resolve())
    if len(matches) != 1:
        raise ValueError(f"Expected one 09A calibration prediction bundle, found {len(matches)}")
    return matches[0]


def find_model_input() -> Path:
    configured = os.getenv("VPI_MODEL_INPUT_ROOT")
    candidates = [Path(configured).expanduser()] if configured else []
    candidates.extend(path.parent for root in kaggle_roots() for path in root.rglob("package_record.json"))
    for candidate in candidates:
        record_path = candidate / "package_record.json"
        if not record_path.is_file() or not (candidate / "media").is_dir():
            continue
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if record.get("cohort_id") == COHORT_ID:
            return candidate.resolve()
    raise FileNotFoundError("Could not locate the 600-scene calibration RGB package")


def find_calibration_evaluator() -> Path:
    configured = os.getenv("VPI_CALIBRATION_EVALUATOR")
    candidates = [Path(configured).expanduser()] if configured else []
    candidates.extend(
        path
        for root in kaggle_roots()
        for path in root.rglob("evaluator_manifest.json")
    )
    matches = []
    for candidate in candidates:
        path = candidate if candidate.is_file() else candidate / "evaluator_manifest.json"
        if not path.is_file():
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if value.get("cohort_id") == COHORT_ID:
            matches.append(path.resolve())
    unique = sorted(set(matches))
    if len(unique) != 1:
        raise ValueError(f"Expected one raw calibration evaluator, found {len(unique)}")
    return unique[0]


def code_environment() -> dict[str, str]:
    environment = os.environ.copy()
    source_root = str(CODE_ROOT / "src")
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = source_root if not existing else source_root + os.pathsep + existing
    return environment


def read_secret(name: str) -> str | None:
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(name)
    except Exception:
        return None


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_tracking_environment() -> dict[str, str]:
    environment = code_environment()
    mode = os.getenv("VPI_TRACKING_MODE", "online" if EXECUTION_ENVIRONMENT == "kaggle" else "disabled").lower()
    environment["VPI_TRACKING_MODE"] = mode
    if mode == "online" and EXECUTION_ENVIRONMENT == "kaggle":
        for name in ("CLEARML_API_ACCESS_KEY", "CLEARML_API_SECRET_KEY"):
            value = read_secret(name)
            if value:
                environment[name] = value
        environment.setdefault("CLEARML_API_HOST", "https://api.clear.ml")
        environment.setdefault("CLEARML_WEB_HOST", "https://app.clear.ml")
        environment.setdefault("CLEARML_FILES_HOST", "https://files.clear.ml")
    return environment


def write_zip(source_root: Path, output_path: Path) -> dict[str, object]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output_path, "w", ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(source_root.rglob("*")):
            if path.is_file():
                member = ZipInfo(
                    path.relative_to(source_root).as_posix(),
                    date_time=(1980, 1, 1, 0, 0, 0),
                )
                member.compress_type = ZIP_DEFLATED
                member.external_attr = 0o100644 << 16
                with path.open("rb") as source, archive.open(member, "w") as destination:
                    shutil.copyfileobj(source, destination, length=1024 * 1024)
    return {
        "archive": output_path.as_posix(),
        "bytes": output_path.stat().st_size,
        "sha256": file_sha256(output_path),
    }
'''
        ),
        markdown(
            r'''
## Verify the prediction boundary

The evaluator path is not resolved in this section. The locked prediction manifest, embedding matrix, embedding index, multi-view decisions, method record, and 09A run record are checked first.
'''
        ),
        code(
            r'''
CODE_ROOT = find_code_root()
sys.path.insert(0, str(CODE_ROOT / "src"))
from visual_peer_inspector.notebook_provenance import verify_notebook_code_snapshot

CODE_PROVENANCE = verify_notebook_code_snapshot(
    CODE_ROOT,
    "notebooks/protected-evaluation/09b-toysad8k-calibration-review-policy.ipynb",
)

PREDICTION_ROOT = find_09a_prediction_root()
PREDICTION_LOCK = PREDICTION_ROOT / "prediction_lock.json"
RUN_RECORD = PREDICTION_ROOT.parent / "safe" / "run_record.json"
METHOD_RECORD = find_named_file("method_freeze.json", "VPI_METHOD_RECORD")
MODEL_INPUT_ROOT = find_model_input()

from visual_peer_inspector.workflows.calibration_review_policy import verify_calibration_boundary

prediction_boundary = verify_calibration_boundary(
    prediction_lock_path=PREDICTION_LOCK,
    run_record_path=RUN_RECORD,
    method_record_path=METHOD_RECORD,
)
display({
    "cohort": prediction_boundary.lock_record["cohort_id"],
    "scenes": prediction_boundary.lock_record["scene_count"],
    "images": prediction_boundary.lock_record["image_count"],
    "prediction lock": prediction_boundary.lock_sha256,
    "method record": prediction_boundary.method_record_sha256,
    "locked artifacts": prediction_boundary.lock_record["artifact_count"],
})
'''
        ),
        markdown(
            r'''
## Open the calibration evaluator

The private evaluator contains raw opaque labels and reference geometry. It is opened only after the verified boundary object above exists. Predicted candidates are then matched to reference objects, and acceptable anomalous identity tracks are derived without changing the locked predictions.
'''
        ),
        code(
            r'''
EVALUATOR_CORRESPONDENCE = find_calibration_evaluator()

from visual_peer_inspector.workflows.calibration_review_policy import open_calibration_evaluator

evaluator_rows, evaluator_access = open_calibration_evaluator(
    prediction_boundary,
    EVALUATOR_CORRESPONDENCE,
)
display({
    "status": evaluator_access["status"],
    "evaluator rows": evaluator_access["evaluator_row_count"],
    "asset groups": evaluator_access["asset_group_count"],
    "prediction lock verified first": evaluator_access["prediction_lock_sha256"] == prediction_boundary.lock_sha256,
})
'''
        ),
        markdown(
            r'''
## Fit the review policy with grouped cross-validation

Five folds keep every scene from the same base asset group together. Training folds choose only the margin, agreement, mask-quality, and review settings. Held-out folds measure automatic coverage, wrong-result rate, false-ready rate, review burden, and insufficient-evidence rate.

The stated selection loss assigns cost 1.00 to a wrong automatic result, 0.15 to a requested review, and 0.25 to insufficient evidence. The full candidate frontier is retained so the trade-off is visible rather than hidden behind one accuracy gate.
'''
        ),
        code(
            r'''
command = [
    sys.executable,
    str(CODE_ROOT / "scripts/run_calibration_review_policy.py"),
    "--prediction-lock", str(PREDICTION_LOCK),
    "--run-record", str(RUN_RECORD),
    "--method-record", str(METHOD_RECORD),
    "--evaluator-correspondence", str(EVALUATOR_CORRESPONDENCE),
    "--model-input-root", str(MODEL_INPUT_ROOT),
    "--output-root", str(OUTPUT_ROOT),
    "--fold-count", "5",
    "--review-cost", "0.15",
    "--insufficient-cost", "0.25",
    "--example-count", "16",
]
completed = subprocess.run(command, check=True, capture_output=True, text=True, env=code_environment())
evaluation_result = json.loads(completed.stdout.strip().splitlines()[-1])
RUN_ROOT = Path(evaluation_result["run_root"])
SAFE_ROOT = RUN_ROOT / "safe"
EVALUATOR_ONLY_ROOT = RUN_ROOT / "evaluator-only"
POLICY_PATH = RUN_ROOT / "downstream" / "review_policy_freeze.json"
display({
    "status": evaluation_result["status"],
    "scene contracts": evaluation_result["scene_contract_count"],
    "asset groups": evaluation_result["asset_group_count"],
    "exact restart used": evaluation_result["resumed"],
    "only review policy calibrated": evaluation_result["only_review_policy_calibrated"],
})
'''
        ),
        markdown(
            r'''
## Cross-validated calibration results

The grouped held-out folds provide the main estimate of how the review policy may behave on new asset groups. The full-set table is an apparent fit summary after threshold selection, so it is retained as a descriptive calibration check and is not presented as generalisation performance. The reported rates include confidence intervals and have separate denominators. In particular, `wrong_result_rate` describes every available fused result, while `false_ready_rate` describes only cases that the policy would release automatically.
'''
        ),
        code(
            r'''
apparent_fit = pd.read_csv(SAFE_ROOT / "apparent_fit_metrics.csv")
cross_validation = pd.read_csv(SAFE_ROOT / "cross_validation_folds.csv")
state_summary = pd.read_csv(SAFE_ROOT / "state_summary.csv")
display(cross_validation)
display({"descriptive full-set fit only": True})
display(apparent_fit)
display(state_summary)
display(NotebookImage(filename=str(SAFE_ROOT / "risk_coverage_frontier.png")))
display(NotebookImage(filename=str(SAFE_ROOT / "policy_state_by_view_count.png")))
'''
        ),
        markdown(
            r'''
## Reliability and calibrated error risk

The label-free review risk score combines operational and evidence warnings. It is not a probability. The reliability table and curve use error rates learned from grouped training folds and assessed on held-out asset groups.
'''
        ),
        code(
            r'''
reliability = pd.read_csv(SAFE_ROOT / "reliability_curve.csv")
display(reliability)
display(NotebookImage(filename=str(SAFE_ROOT / "held_out_reliability_curve.png")))
'''
        ),
        markdown(
            r'''
## Review reasons and subgroup checks

Reason-specific burden can sum to more than the overall review burden because one result may trigger several reasons. Subgroup rows report coverage and false-ready intervals for view count, category, anomaly type, and object-count band.
'''
        ),
        code(
            r'''
reason_burden = pd.read_csv(SAFE_ROOT / "reason_specific_review_burden.csv")
subgroup_metrics = pd.read_csv(SAFE_ROOT / "subgroup_metrics.csv")
display(reason_burden)
display(subgroup_metrics)
display(NotebookImage(filename=str(SAFE_ROOT / "reason_specific_review_burden.png")))
'''
        ),
        markdown(
            r'''
## Frozen review policy

The policy record carries the exact 09A prediction lock, the 08F method hash, the immutable pipeline identity, the selected thresholds, state definitions, and the calibration trade-off. Notebook 09C must verify this file before it runs internal-lock inference.
'''
        ),
        code(
            r'''
policy_record = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
POLICY_RESOURCE_ROOT = RUN_ROOT / "publish" / "review-policy"
display({
    "status": policy_record["status"],
    "minimum margin": policy_record["minimum_margin"],
    "minimum agreement": policy_record["minimum_agreement"],
    "minimum mask quality": policy_record["minimum_mask_quality"],
    "policy configuration": policy_record["policy_configuration_sha256"],
    "method record": policy_record["method_record_sha256"],
    "application claim": policy_record["application_claim"],
    "minimal policy resource": str(POLICY_RESOURCE_ROOT),
})
'''
        ),
        markdown(
            r'''
Publish only `publish/review-policy` as the private Kaggle dataset `vpi-toysad-calibration-review-policy`. Notebook 09C and all official inference stages must attach this minimal resource, not the complete Notebook 09B evaluator output.
'''
        ),
        markdown(
            r'''
## Evaluator-only qualitative audit

Wrong `result_ready` examples remain complete in the evaluator JSONL and CSV records. The rendered audit is a separate deterministic stratified subset capped at 16 panels. It represents wrong-ready cases, available policy states, terminal signals, review reasons, and multi-anomaly references before a stable-hash fill. Every private panel header lists all acceptable anomaly identities for that scene. These panels may contain evaluator correspondence and remain in the private evaluator-only archive.
'''
        ),
        code(
            r'''
example_inventory = json.loads((EVALUATOR_ONLY_ROOT / "example_inventory.json").read_text(encoding="utf-8"))
display({
    "total wrong result-ready records": example_inventory["total_wrong_result_ready_count"],
    "rendered wrong result-ready panels": example_inventory["rendered_wrong_result_ready_count"],
    "rendered panels": example_inventory["rendered_example_count"],
    "panel limit": example_inventory["example_count_limit"],
})
for item in example_inventory["items"][:6]:
    display({
        "policy state": item["policy_state"],
        "wrong result ready": item["wrong_result_ready"],
        "review reasons": item["review_reasons"],
        "selection strata": item["selection_strata"],
        "acceptable anomaly identities": item["acceptable_anomalous_identity_ids"],
    })
    display(NotebookImage(filename=item["path"]))
'''
        ),
        markdown(
            r'''
## Bounded ClearML finalisation

Tracking starts only after all scientific outputs and the review-policy lock exist. The subprocess has a fixed wall time. A tracking failure cannot change, delete, or invalidate the completed evaluation.
'''
        ),
        code(
            r'''
if EXECUTION_ENVIRONMENT == "kaggle":
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-r", str(CODE_ROOT / "requirements/confirmation-tracking-lock.txt")],
            check=True,
            timeout=180,
        )
    except Exception as error:
        print(f"ClearML dependency setup was unavailable: {type(error).__name__}")

tracking_environment = prepare_tracking_environment()
try:
    tracking_completed = subprocess.run(
        [
            sys.executable,
            str(CODE_ROOT / "scripts/run_calibration_review_policy.py"),
            "--clearml-finalize-only",
            "--run-root", str(RUN_ROOT),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
        env=tracking_environment,
    )
    tracking_receipt = json.loads(tracking_completed.stdout.strip().splitlines()[-1])
except subprocess.TimeoutExpired:
    tracking_receipt = {"schema_version": 1, "status": "timed_out"}
    (RUN_ROOT / "clearml_receipt.json").write_text(json.dumps(tracking_receipt, indent=2) + "\n", encoding="utf-8")
display(tracking_receipt)
'''
        ),
        markdown(
            r'''
## Build verified output archives

The repository archive contains aggregate tables, figures, and the run record. The downstream archive contains the frozen review policy for Notebook 09C. The evaluator archive contains correspondence-selected examples and evaluated scene records and must remain private. Credentials, RGB inputs, model caches, and the 09A embedding bundle are not duplicated.
'''
        ),
        code(
            r'''
BUNDLE_ROOT = Path("/kaggle/working/notebook-09b-bundles") if EXECUTION_ENVIRONMENT == "kaggle" else RUN_ROOT / "bundles"
bundles = [
    write_zip(SAFE_ROOT, BUNDLE_ROOT / "notebook-09b-repository-evidence.zip"),
    write_zip(RUN_ROOT / "downstream", BUNDLE_ROOT / "notebook-09b-downstream-policy.zip"),
    write_zip(EVALUATOR_ONLY_ROOT, BUNDLE_ROOT / "notebook-09b-private-evaluator-evidence.zip"),
]
(BUNDLE_ROOT / "bundle-checksums.json").write_text(
    json.dumps({"schema_version": 1, "bundles": bundles}, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
display(pd.DataFrame(bundles))
'''
        ),
        markdown(
            r'''
## Interpretation boundary

This notebook estimates calibration risk and automatic coverage for relative unusualness on the protected calibration role. It does not change a model or report internal-lock performance. The frozen policy is tested without further adjustment in Notebook 09C and its separate evaluator notebook. Human review remains a fallback for uncertain evidence, not a replacement for the automatic LocateAnything, SAM 2.1, DINOv2-small, P1, identity, and fusion route.
'''
        ),
    ]
    for index, cell in enumerate(cells):
        cell["id"] = sha256(
            f"{NOTEBOOK_PATH.name}|{index}|{cell.cell_type}|{cell.source}".encode()
        ).hexdigest()[:12]
        if cell.cell_type == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    notebook.metadata.language_info = {"name": "python", "version": "3.11"}
    return notebook


def main() -> int:
    NOTEBOOK_ROOT.mkdir(parents=True, exist_ok=True)
    nbformat.write(build_notebook(), NOTEBOOK_PATH)
    metadata = {
        "id": "esreekarreddy/toysad-8k-calibration-review-policy",
        "title": "ToysAD-8K Calibration Review Policy",
        "code_file": NOTEBOOK_PATH.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": True,
        "keywords": [],
        "dataset_sources": [
            "esreekarreddy/vpi-code-snapshot",
            "esreekarreddy/vpi-toysad-calibration600-model-input",
            "esreekarreddy/vpi-toysad-calibration600-evaluator",
            "esreekarreddy/vpi-toysad-development-method-freeze",
        ],
        "competition_sources": [],
        "kernel_sources": [
            "esreekarreddy/09a-toysad8k-protected-calibration-inference",
        ],
        "model_sources": [],
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
