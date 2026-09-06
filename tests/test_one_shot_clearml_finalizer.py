from __future__ import annotations

import importlib.util
import json
import sys
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

from visual_peer_inspector.data.model_safe_package import (
    canonical_json_bytes,
    file_sha256,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_one_shot_evaluation.py"
SPEC = importlib.util.spec_from_file_location("run_one_shot_evaluation", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)

TABLES = {
    "clustered_confidence_intervals.csv",
    "decision_confusion.csv",
    "error_taxonomy.csv",
    "identity_failures.csv",
    "margin_agreement_reliability.csv",
    "pipeline_attrition.csv",
    "prediction_failures.csv",
    "runtime_summary.csv",
    "subgroup_and_category_metrics.csv",
}
FIGURES = {
    "coverage_and_review.png",
    "margin_reliability.png",
    "pipeline_attrition.png",
    "ranking_by_view.png",
}


class _Logger:
    def report_scalar(self, *args, **kwargs) -> None:
        return None

    def report_table(self, *args, **kwargs) -> None:
        return None

    def report_image(self, *args, **kwargs) -> None:
        return None


class _Task:
    id = "test-task"

    def get_logger(self) -> _Logger:
        return _Logger()

    def upload_artifact(self, *args, **kwargs) -> None:
        return None

    def add_tags(self, *args, **kwargs) -> None:
        return None

    def flush(self, *args, **kwargs) -> None:
        return None

    def close(self) -> None:
        return None


class _TaskFactory:
    @staticmethod
    def init(**kwargs) -> _Task:
        return _Task()


def _write_completed_result(tmp_path: Path) -> Path:
    root = tmp_path / "result"
    safe = root / "safe"
    figures = safe / "figures"
    figures.mkdir(parents=True)
    (safe / "summary_metrics.csv").write_text(
        "split_name,view_count,top1_accuracy,top2_accuracy,mean_reciprocal_rank,"
        "mean_average_precision,mean_recall_at_2,exact_set_accuracy,"
        "target_found_rate,mean_target_rank_when_found,coverage,selective_accuracy,"
        "selective_risk,review_rate\n"
        "seen-test,5,0.7,0.9,0.8,0.75,0.8,0.7,0.9,1.2,1.0,0.8,0.2,0.1\n",
        encoding="utf-8",
    )
    for name in TABLES:
        (safe / name).write_text("metric,value\ncount,1\n", encoding="utf-8")
    for name in FIGURES:
        (figures / name).write_bytes(b"aggregate figure")
    (safe / "run_record.json").write_text(
        json.dumps({"status": "completed", "stage": "10b"}),
        encoding="utf-8",
    )
    artifacts = [
        {
            "path": path.relative_to(safe).as_posix(),
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(safe.rglob("*"))
        if path.is_file() and path.name != "artifact_manifest.json"
    ]
    manifest = {
        "schema_version": "visual-peer-one-shot-evaluation/v1",
        "status": "complete",
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "artifact_inventory_sha256": sha256(canonical_json_bytes(artifacts)).hexdigest(),
    }
    (safe / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (root / "evaluation_complete.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "stage": "10b",
                "safe_manifest_sha256": file_sha256(safe / "artifact_manifest.json"),
                "safe_artifact_inventory_sha256": manifest[
                    "artifact_inventory_sha256"
                ],
            }
        ),
        encoding="utf-8",
    )
    return root


def test_completed_one_shot_outputs_pass_the_clearml_allowlist(
    tmp_path: Path, monkeypatch
) -> None:
    result = _write_completed_result(tmp_path)
    monkeypatch.setattr(runner, "_load_kaggle_secrets", lambda: None)
    monkeypatch.setattr(runner, "configure_official_clearml_hosts", lambda: None)
    monkeypatch.setitem(sys.modules, "clearml", SimpleNamespace(Task=_TaskFactory))
    receipt = runner.finalize_clearml(result)
    assert receipt["status"] == "completed"
    assert receipt["private_panels_uploaded"] is False


def test_chart_renderer_and_clearml_allowlist_use_the_same_filenames() -> None:
    renderer_source = (
        ROOT / "src/visual_peer_inspector/workflows/one_shot_evaluation.py"
    ).read_text(encoding="utf-8")
    finalizer_source = SCRIPT.read_text(encoding="utf-8")
    for name in FIGURES:
        assert f'"{name}"' in renderer_source
        assert f'"figures/{name}"' in finalizer_source
    for field in (
        "mean_average_precision",
        "mean_recall_at_2",
        "exact_set_accuracy",
    ):
        assert f'"{field}"' in renderer_source
        assert f'"{field}"' in finalizer_source
