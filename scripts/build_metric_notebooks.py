#!/usr/bin/env python3
"""Build clean Kaggle notebooks for the learned metric development stage."""

from __future__ import annotations

import argparse
from pathlib import Path
import textwrap

import nbformat as nbf


def code(value: str):
    return nbf.v4.new_code_cell(textwrap.dedent(value).strip())


def markdown(value: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(value).strip())


def common_preflight(stage: str) -> str:
    return f'''\
from pathlib import Path
from pathlib import PurePosixPath
import hashlib
import json
import os
import shutil
import subprocess
import sys
from zipfile import ZipFile

INPUT_ROOT = Path("/kaggle/input")
WORKING_ROOT = Path("/kaggle/working")
STAGE_NAME = "{stage}"
SOURCE_ARCHIVE_NAME = "learned-peer-view-source-v1.zip"
SOURCE_MANIFEST_NAME = "source-manifest.json"

def direct_matches(marker):
    matches = []
    for child in sorted(INPUT_ROOT.iterdir()):
        direct = child / marker
        if direct.exists():
            matches.append(direct.parent if direct.is_file() else direct)
        for nested in sorted(path for path in child.iterdir() if path.is_dir())[:20]:
            candidate = nested / marker
            if candidate.exists():
                matches.append(candidate.parent if candidate.is_file() else candidate)
    unique = sorted({{path.resolve() for path in matches}})
    if not unique:
        recursive = []
        for candidate in INPUT_ROOT.rglob(marker):
            recursive.append(candidate.parent if candidate.is_file() else candidate)
            if len(recursive) > 8:
                raise RuntimeError(f"Too many recursive matches for {{marker}}")
        unique = sorted({{path.resolve() for path in recursive}})
    return unique

def unpack_verified_source():
    manifests = sorted(INPUT_ROOT.rglob(SOURCE_MANIFEST_NAME))
    if len(manifests) != 1:
        raise RuntimeError(f"Expected one {{SOURCE_MANIFEST_NAME}} input, found {{manifests}}")
    manifest_path = manifests[0]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    archive = manifest_path.parent / manifest.get("archive", "")
    if not archive.is_file():
        raise RuntimeError(f"Missing source archive beside {{manifest_path}}")
    observed_hash = sha256_file(archive)
    if observed_hash != manifest.get("archive_sha256"):
        raise RuntimeError("The source archive hash does not match its manifest")
    extraction_root = Path("/kaggle/temp") / f"vpi-learned-source-{{observed_hash[:12]}}"
    marker = extraction_root / "learned-peer-view-source/scripts/run_metric_feature_preparation.py"
    if marker.is_file():
        return extraction_root
    temporary = extraction_root.with_name(extraction_root.name + ".partial")
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir(parents=True)
    with ZipFile(archive) as handle:
        members = handle.infolist()
        if not members or len(members) > 5000:
            raise RuntimeError("The source archive member count is outside the allowed range")
        if sum(member.file_size for member in members) > 100 * 1024 * 1024:
            raise RuntimeError("The source archive exceeds the extraction size limit")
        for member in members:
            relative = PurePosixPath(member.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError(f"Unsafe source archive member: {{member.filename}}")
            file_type = (member.external_attr >> 16) & 0o170000
            if file_type == 0o120000:
                raise RuntimeError(f"Source archive contains a symbolic link: {{member.filename}}")
        handle.extractall(temporary)
    if not (temporary / "learned-peer-view-source/scripts/run_metric_feature_preparation.py").is_file():
        raise RuntimeError("The verified source archive is missing the feature runner")
    shutil.rmtree(extraction_root, ignore_errors=True)
    temporary.replace(extraction_root)
    print(json.dumps({{"event": "source_extracted", "archive_sha256": observed_hash}}, sort_keys=True))
    return extraction_root

def find_bounded(marker):
    unique = direct_matches(marker)
    if not unique and marker == "scripts/run_metric_feature_preparation.py":
        unpack_verified_source()
        unique = direct_matches(marker)
    if len(unique) != 1:
        raise RuntimeError(f"Expected one {{marker}} input, found {{unique}}")
    return unique[0]

def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

runtime_versions = {{}}
for package in ("numpy", "pandas", "scipy", "matplotlib", "pillow"):
    try:
        runtime_versions[package] = __import__("importlib.metadata").metadata.version(package)
    except Exception:
        runtime_versions[package] = "unavailable"

SOURCE_ROOT = find_bounded("scripts/run_metric_feature_preparation.py").parent
SCORING_ROOT = find_bounded("prediction_lock.json")
if not (SCORING_ROOT / "scenes").is_dir():
    raise RuntimeError("The scoring input is missing sealed scene predictions")

sys.path.insert(0, str(SOURCE_ROOT / "src"))
print(json.dumps({{
    "event": "preflight_complete",
    "stage": STAGE_NAME,
    "source_root": str(SOURCE_ROOT),
    "scoring_root": str(SCORING_ROOT),
    "scoring_lock_sha256": sha256_file(SCORING_ROOT / "prediction_lock.json"),
    "runtime_versions": runtime_versions,
}}, sort_keys=True))
'''


def clearml_cell(shareable: str, receipt: str, task_name: str) -> str:
    return f'''\
# Tracking runs only after the scientific outputs and checksums exist.
tracking_env = os.environ.copy()
tracking_status = "disabled"
try:
    from kaggle_secrets import UserSecretsClient
    secrets = UserSecretsClient()
    for key in ("CLEARML_API_ACCESS_KEY", "CLEARML_API_SECRET_KEY"):
        tracking_env[key] = secrets.get_secret(key)
    tracking_env.setdefault("CLEARML_API_HOST", "https://api.clear.ml")
    tracking_env.setdefault("CLEARML_WEB_HOST", "https://app.clear.ml")
    tracking_env.setdefault("CLEARML_FILES_HOST", "https://files.clear.ml")
    tracking_site = WORKING_ROOT / "clearml-site"
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", "--target", str(tracking_site), "clearml==2.1.10"],
        check=True,
        timeout=120,
    )
    tracking_env["PYTHONPATH"] = os.pathsep.join([str(tracking_site), str(SOURCE_ROOT / "src")])
    tracking_env.setdefault("CLEARML_API_DEFAULT_REQ_TIMEOUT", "10")
    subprocess.run(
        [sys.executable, str(SOURCE_ROOT / "scripts/log_learned_metric_clearml.py"),
         "--shareable-root", str({shareable}), "--receipt", str({receipt}),
         "--task-name", "{task_name}"],
        check=True,
        timeout=120,
        env=tracking_env,
    )
    tracking_status = json.loads(Path({receipt}).read_text())["status"]
except Exception as error:
    receipt_path = Path({receipt})
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_receipt = receipt_path.with_suffix(receipt_path.suffix + ".partial")
    temporary_receipt.write_text(json.dumps({{"status": "failed", "error_type": type(error).__name__, "message": str(error)[:500]}}, sort_keys=True))
    temporary_receipt.replace(receipt_path)
    tracking_status = "failed"
finally:
    shutil.rmtree(WORKING_ROOT / "clearml-site", ignore_errors=True)
print(json.dumps({{"event": "tracking_finished", "status": tracking_status}}, sort_keys=True))
'''


def build_08l() -> nbf.NotebookNode:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kaggle": {"accelerator": "none", "dataSources": [], "isInternetEnabled": True},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    }
    notebook["cells"] = [
        markdown('''
        # Notebook 08L: Development learned peer and view feature preparation

        **File:** `08l-development-learned-peer-view-features.ipynb`

        This notebook reuses the sealed DINOv2-small outputs from Notebook 08G. It creates relative object features, fixed cross-view support measurements and automatic identity tracks for predeclared one, two, three and five-view schedules.

        No evaluator labels, categories or source object identifiers are available to this notebook. The complete feature package is locked before the metric evaluation begins in Notebook 08M.
        '''),
        markdown('''
        ## Experiment boundary

        - The established LocateAnything, SAM 2.1 and DINOv2-small pipeline is unchanged.
        - The same 100-scene development screen is reused. No scene split is changed.
        - The calibration and internal-lock roles remain unopened.
        - The primary web range is two to five views. Single-view schedules measure the fallback case.
        - Rotated schedules are sensitivity checks. They are not additional independent scenes.
        - Every scene checkpoint is bound to the upstream prediction lock and feature protocol.
        '''),
        code(common_preflight("notebook-08l")),
        markdown('''
        ## Prepare the sealed feature package

        The command verifies the upstream lock before processing any scene. A valid result and receipt pair is reused on restart. Missing or mismatched pairs are recomputed.
        '''),
        code('''
        OUTPUT_ROOT = WORKING_ROOT / "learned-peer-features-08l"
        run_env = os.environ.copy()
        run_env["PYTHONPATH"] = str(SOURCE_ROOT / "src")
        command = [
            sys.executable,
            str(SOURCE_ROOT / "scripts/run_metric_feature_preparation.py"),
            "--scoring-root", str(SCORING_ROOT),
            "--output", str(OUTPUT_ROOT),
        ]
        subprocess.run(command, check=True, timeout=1800, env=run_env)
        '''),
        markdown('''
        ## Verify the result

        The feature lock is checked again in a new process. This catches incomplete writes or changed checkpoint bytes before packaging.
        '''),
        code('''
        from visual_peer_inspector.workflows.learned_peer_features import feature_package_inventory

        inventory = feature_package_inventory(OUTPUT_ROOT)
        display(inventory)
        display(__import__("pandas").read_csv(OUTPUT_ROOT / "shareable/feature_schema.csv"))
        '''),
        markdown('''
        ## Create the downstream archive

        Notebook 08M needs the locked feature package, including the scene checkpoints. This archive is a derived stage input, not a new dataset split.
        '''),
        code('''
        import zipfile

        downstream = WORKING_ROOT / "learned-peer-features-08l-downstream.zip"
        temporary = downstream.with_suffix(".zip.partial")
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
            for path in sorted(OUTPUT_ROOT.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(OUTPUT_ROOT.parent))
            assert archive.testzip() is None
        temporary.replace(downstream)
        print(json.dumps({"archive": downstream.name, "bytes": downstream.stat().st_size, "sha256": sha256_file(downstream)}, sort_keys=True))

        PACKAGE_ROOT = WORKING_ROOT / "learned-peer-packages-08l"
        subprocess.run([
            sys.executable, str(SOURCE_ROOT / "scripts/package_metric_stage.py"),
            "--stage-root", str(OUTPUT_ROOT),
            "--output-root", str(PACKAGE_ROOT),
            "--prefix", "learned-peer-features-08l",
        ], check=True, timeout=300, env=run_env)
        display(json.loads((PACKAGE_ROOT / "learned-peer-features-08l-package-receipt.json").read_text()))
        '''),
        markdown('''
        ## Optional ClearML record

        ClearML receives only the aggregate preparation receipt and feature schema. Tracking is isolated and time-limited, so a network or credential failure cannot invalidate the scientific output.
        '''),
        code(clearml_cell("OUTPUT_ROOT / 'shareable'", "OUTPUT_ROOT / 'tracking_receipt.json'", "08L learned peer-view feature preparation")),
        markdown('''
        ## Completion

        The output is ready for Notebook 08M when the feature lock verifies and the downstream archive has a recorded SHA-256 value. The next notebook opens evaluator labels only after confirming those conditions.
        '''),
    ]
    return notebook


def build_08m() -> nbf.NotebookNode:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kaggle": {"accelerator": "none", "dataSources": [], "isInternetEnabled": True},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    }
    preflight = common_preflight("notebook-08m") + '''\
FEATURE_ROOT = find_bounded("shareable/preparation_receipt.json").parent
EVALUATOR_ROOT = find_bounded("artifact_lock.json")
MODEL_INPUT_ROOT = None
model_input_candidates = []
for index_path in INPUT_ROOT.rglob("index.json"):
    candidate = index_path.parent
    if not (candidate / "artifact_lock.json").is_file() or not (candidate / "scenes").is_dir() or not (candidate / "media").is_dir():
        continue
    try:
        index = json.loads(index_path.read_text())
    except Exception:
        continue
    if index.get("study") == "development-scoring-comparison-v1" and index.get("routes") == ["supplied", "automatic"] and len(index.get("scenes", [])) == 100:
        model_input_candidates.append(candidate.resolve())
if len(set(model_input_candidates)) != 1:
    raise RuntimeError(f"Expected one model-input package, found {{sorted(set(model_input_candidates))}}")
MODEL_INPUT_ROOT = sorted(set(model_input_candidates))[0]
print(json.dumps({"event": "evaluation_inputs_located", "feature_root": str(FEATURE_ROOT), "evaluator_root": str(EVALUATOR_ROOT), "model_input_root": str(MODEL_INPUT_ROOT)}, sort_keys=True))
'''
    # The generic artifact-lock finder is ambiguous for 08M. Replace it with
    # bounded schema checks in the notebook after locating all mounted roots.
    preflight = preflight.replace(
        'EVALUATOR_ROOT = find_bounded("artifact_lock.json")',
        '''evaluator_candidates = []
for lock_path in INPUT_ROOT.rglob("artifact_lock.json"):
    candidate = lock_path.parent
    index_path = candidate / "index.json"
    if not index_path.is_file() or not (candidate / "scenes").is_dir():
        continue
    try:
        index = json.loads(index_path.read_text())
    except Exception:
        continue
    expected_keys = {"attributes_source_sha256", "automatic_prediction_sha256", "correspondence_source_sha256", "model_lock_sha256", "role_ledger_sha256"}
    if expected_keys.issubset(index) and len(index.get("scenes", [])) == 100:
        evaluator_candidates.append(candidate.resolve())
if len(set(evaluator_candidates)) != 1:
    raise RuntimeError(f"Expected one evaluator package, found {sorted(set(evaluator_candidates))}")
EVALUATOR_ROOT = sorted(set(evaluator_candidates))[0]'''
    )
    notebook["cells"] = [
        markdown('''
        # Notebook 08M: Development learned peer and view ranker evaluation

        **File:** `08m-development-learned-peer-view-ranker.ipynb`

        This notebook verifies the locked 08L feature package before opening the development evaluator. It compares the current B0, P1, P2 and forward-patch scores with robust fusion, view voting, winner clustering, a scene-softmax linear ranker, a pairwise RankNet-style ranker and a hybrid objective.

        All learned metrics are out-of-fold by source asset. The protected calibration and internal-lock roles are not used.
        '''),
        markdown('''
        ## Scientific questions

        1. Does any fixed or learned combination improve the automatic three-view result?
        2. Is any gain stable across asset-group folds and view schedules?
        3. Does the method preserve tie-aware average precision, recall at two and reciprocal rank?
        4. Can low-confidence results be routed to review without inventing a universal accuracy gate?
        5. Do the predicted boxes, SAM masks and heatmaps support the numerical result?
        '''),
        markdown(r'''
        ## Candidate methods

        The fixed controls are B0, P1, P2 and the selected forward-patch score. The notebook also tests robust score fusion, view voting and three small linear ranking heads. The pairwise learned head follows the RankNet logistic ranking objective described by Burges and colleagues.

        The main new project candidate is **winner consensus**. Each view first votes for its strongest candidate using the forward, P1 and P2 ranks. DINO cosine distance links compatible winners across views. A cluster receives

        $$S(c)=\frac{v_c}{V}+\frac{0.5m_c+0.5q_c}{4V},$$

        where $v_c$ is the number of supporting views, $V$ is the uploaded view count, $m_c$ is the mean within-view winner margin, and $q_c$ is the mean DINO coherence. The second term is deliberately bounded, so it can break an equal-support tie but cannot overpower one additional view vote. Unsupported singleton winners remain visible and can be routed to review.

        Reference: [Learning to Rank using Gradient Descent](https://www.microsoft.com/en-us/research/publication/learning-to-rank-using-gradient-descent/).
        '''),
        code(preflight),
        markdown('''
        ## Run grouped out-of-fold evaluation

        Feature scaling and model weights are fitted only on the training asset groups in each fold. The automatic route is the deployable result. The supplied-mask route is retained only as a segmentation diagnostic.
        '''),
        code('''
        OUTPUT_ROOT = WORKING_ROOT / "learned-peer-evaluation-08m"
        run_env = os.environ.copy()
        run_env["PYTHONPATH"] = str(SOURCE_ROOT / "src")
        command = [
            sys.executable,
            str(SOURCE_ROOT / "scripts/run_metric_evaluation.py"),
            "--feature-root", str(FEATURE_ROOT),
            "--scoring-root", str(SCORING_ROOT),
            "--evaluator-root", str(EVALUATOR_ROOT),
            "--model-input-root", str(MODEL_INPUT_ROOT),
            "--output", str(OUTPUT_ROOT),
        ]
        subprocess.run(command, check=True, timeout=3600, env=run_env)
        '''),
        markdown('''
        ## Primary results

        The primary table is limited to the automatic route and the fixed three-view schedule. Equal-score candidates are averaged rather than ordered by candidate ID. It reports development evidence, not protected final accuracy.
        '''),
        code('''
        import pandas as pd
        from IPython.display import display, Image as DisplayImage

        summary = pd.read_csv(OUTPUT_ROOT / "shareable/method_summary.csv")
        primary = summary[(summary.route == "automatic") & (summary.view_set_id == "three-primary")]
        columns = [
            "method", "scenes", "asset_groups", "top1_tie_expected",
            "top1_tie_best", "top1_tie_worst", "ap_tie_expected",
            "recall_at_2_tie_expected", "reciprocal_rank_tie_expected",
            "mean_top_tie_count", "mean_margin",
        ]
        display(primary[columns].sort_values(
            ["top1_tie_expected", "reciprocal_rank_tie_expected"],
            ascending=False,
        ).reset_index(drop=True))
        display(json.loads((OUTPUT_ROOT / "shareable/development_decision.json").read_text()))
        '''),
        markdown('''
        ## Stability and paired evidence

        These tables keep the asset-group folds and view-schedule sensitivity visible. A schedule is a repeated measurement of the same scene, not a new sample.
        '''),
        code('''
        display(pd.read_csv(OUTPUT_ROOT / "shareable/fold_summary.csv").query("route == 'automatic' and view_set_id == 'three-primary'"))
        display(pd.read_csv(OUTPUT_ROOT / "shareable/view_schedule_sensitivity.csv").query("route == 'automatic'"))
        display(pd.read_csv(OUTPUT_ROOT / "shareable/paired_asset_intervals.csv"))
        '''),
        markdown('''
        ## Aggregate figures

        Figures cover the primary comparison, view-count behaviour, fold stability, risk-coverage and the strongest hybrid-model coefficients.
        '''),
        code('''
        for figure in sorted((OUTPUT_ROOT / "shareable/figures").glob("*.png")):
            print(figure.name)
            display(DisplayImage(filename=str(figure)))
        '''),
        markdown('''
        ## Qualitative evidence

        The private gallery is selected automatically from out-of-fold predictions. Each case shows the selected box, SAM overlay, masked object and forward-patch heatmap for the three uploaded views. It includes correct cases with the strongest cross-view support, uncertain cases and failures from distinct asset groups. Confidence remains an uncalibrated review diagnostic and is not used as an accuracy claim.
        '''),
        code('''
        gallery = OUTPUT_ROOT / "private/reference_gallery"
        display(json.loads((gallery / "gallery_manifest.json").read_text()))
        for figure in sorted(gallery.glob("example_*.png")):
            display(DisplayImage(filename=str(figure)))
        '''),
        markdown('''
        ## Package review and repository evidence

        Aggregate CSV, JSON and PNG files are separated from private scene-level metrics and image galleries. The later pull request can use the repository-evidence archive after review. The review archive stays outside GitHub.
        '''),
        code('''
        PACKAGE_ROOT = WORKING_ROOT / "learned-peer-packages-08m"
        subprocess.run([
            sys.executable, str(SOURCE_ROOT / "scripts/package_metric_stage.py"),
            "--stage-root", str(OUTPUT_ROOT),
            "--output-root", str(PACKAGE_ROOT),
            "--prefix", "learned-peer-view-08m",
        ], check=True, timeout=300, env=run_env)
        display(json.loads((PACKAGE_ROOT / "learned-peer-view-08m-package-receipt.json").read_text()))
        '''),
        markdown('''
        ## Optional ClearML record

        Only aggregate-safe tables and figures are sent to ClearML. This step runs after the scientific artifact lock and cannot trigger model fitting or change the selection.
        '''),
        code(clearml_cell("OUTPUT_ROOT / 'shareable'", "OUTPUT_ROOT / 'tracking_receipt.json'", "08M learned peer-view ranker evaluation")),
        markdown('''
        ## Interpretation boundary

        The one-standard-error heuristic keeps the simplest method close to the best grouped-fold result while protecting average precision and recall at two. It is a development selection aid, not a significance test. Winner clustering is also assessed as a selective review aid, but tied winners do not receive deterministic credit. The selected scorer is ready for frozen development confirmation only. Calibration and the internal lock remain later one-shot stages.
        '''),
    ]
    return notebook


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name, notebook in (
        ("08l-development-learned-peer-view-features.ipynb", build_08l()),
        ("08m-development-learned-peer-view-ranker.ipynb", build_08m()),
    ):
        path = args.output / name
        nbf.write(notebook, path)
        print(path)


if __name__ == "__main__":
    main()
