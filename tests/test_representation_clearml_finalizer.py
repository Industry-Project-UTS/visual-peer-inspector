from __future__ import annotations

from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "finalize_representation_clearml.py"
SPEC = importlib.util.spec_from_file_location("representation_clearml", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ClearMLFinalizerTests(unittest.TestCase):
    def _build_safe_root(self, run_mode: str = "contract") -> Path:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)

        for name in MODULE.ALLOWED_TABLES:
            (root / name).write_text("measure,value\ncount,1\n", encoding="utf-8")
        for name in MODULE.ALLOWED_JSON - {"artifact_manifest.json"}:
            value = {"schema_version": 1, "status": "completed"}
            if name == "run_record.json":
                value.update(
                    {
                        "run_mode": run_mode,
                        "decision": "retain_dinov2_small_anchor_pending_ds1",
                    }
                )
            (root / name).write_text(
                json.dumps(value, sort_keys=True) + "\n", encoding="utf-8"
            )
        for relative in MODULE.ALLOWED_CHARTS:
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"safe-chart")

        expected = (
            MODULE.ALLOWED_TABLES
            | MODULE.ALLOWED_JSON
            | MODULE.ALLOWED_CHARTS
        ) - {"artifact_manifest.json"}
        rows = []
        for relative in sorted(expected):
            path = root / relative
            rows.append(
                {
                    "path": relative,
                    "sha256": sha256(path.read_bytes()).hexdigest(),
                    "size_bytes": path.stat().st_size,
                }
            )
        manifest = {
            "schema_version": 1,
            "artifact_count": len(rows),
            "artifacts": rows,
        }
        (root / "artifact_manifest.json").write_text(
            json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
        )
        return root

    def test_disabled_mode_can_validate_contract_evidence(self) -> None:
        root = self._build_safe_root(run_mode="contract")
        record = MODULE.validate_safe_root(root, require_model_evidence=False)
        self.assertEqual(record["run_mode"], "contract")

    def test_online_mode_rejects_contract_evidence(self) -> None:
        root = self._build_safe_root(run_mode="contract")
        with self.assertRaisesRegex(ValueError, "model-evidence"):
            MODULE.validate_safe_root(root, require_model_evidence=True)

    def test_online_mode_accepts_model_evidence(self) -> None:
        root = self._build_safe_root(run_mode="model_evidence")
        record = MODULE.validate_safe_root(root, require_model_evidence=True)
        self.assertEqual(record["status"], "completed")


if __name__ == "__main__":
    unittest.main()
