from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from visual_peer_inspector.data.cohort_contract import CohortContract
from visual_peer_inspector.data.model_safe_package import (
    PackageAudit,
    canonical_json_bytes,
    file_sha256,
    membership_sha256,
)
from visual_peer_inspector.data import sam_prompt_package_ds1 as PACKAGE
from visual_peer_inspector.data.toysad_selection import RELEASE_ID
from visual_peer_inspector.models.locateanything import (
    EXECUTION_POLICY_SHA256,
    LocateAnythingConfiguration,
    MODEL_PRODUCER_KIND,
)


class Ds1SamPromptPackageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.model_root = self.root / "model-input"
        self.geometry_path = self.root / "trusted/reference_manifest.jsonl"
        self.geometry_record_path = self.root / "trusted/package_record.json"
        self.proposal_path = self.root / "proposal/private/proposal_results.jsonl"
        self.safe_root = self.root / "proposal/safe"
        self.run_record_path = self.safe_root / "run_record.json"
        self.artifact_manifest_path = self.safe_root / "artifact_manifest.json"
        self.output_root = self.root / "prompts"
        scene_ids = ["S000001", "S000002"]
        self.contract = CohortContract(
            cohort_id="test-development-screen-2-v1",
            release_id=RELEASE_ID,
            role="development",
            pilot_stage="screen_2",
            membership_sha256=membership_sha256(scene_ids),
            scene_count=2,
            views_per_scene=20,
            image_count=40,
            asset_group_count=2,
        )
        self.configuration = LocateAnythingConfiguration()
        manifest_rows = []
        content_paths: list[Path] = []
        geometry_rows = []
        proposal_rows = []
        for scene_index, scene_id in enumerate(scene_ids, 1):
            view_ids = [f"V{index:02d}" for index in range(20)]
            manifest_rows.append(
                {
                    "schema_version": 1,
                    "release_id": self.contract.release_id,
                    "opaque_scene_id": scene_id,
                    "object_count": 3,
                    "object_ids": ["O00", "O01", "O02"],
                    "view_ids": view_ids,
                    "media_root": f"media/{scene_id}",
                }
            )
            for view_index, view_id in enumerate(view_ids):
                image_path = self.model_root / "media" / scene_id / "rgb" / f"{view_id}.png"
                image_path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (256, 256), (scene_index, view_index, 120)).save(image_path)
                content_paths.append(image_path.relative_to(self.model_root))
                geometry_rows.append(
                    {
                        "opaque_scene_id": scene_id,
                        "view_id": view_id,
                        "opaque_object_id": "O00",
                        "visible": True,
                        "box_xyxy": [8, 9, 80, 90],
                    }
                )
                proposal_rows.append(
                    {
                        "schema_version": 1,
                        "cohort_id": self.contract.cohort_id,
                        "input_manifest_sha256": "pending",
                        "opaque_scene_id": scene_id,
                        "view_id": view_id,
                        "attempt": 1,
                        "image_sha256": file_sha256(image_path),
                        "configuration_sha256": self.configuration.sha256,
                        "execution_policy_sha256": EXECUTION_POLICY_SHA256,
                        "effective_generation_mode": "hybrid",
                        "effective_max_new_tokens": 8192,
                        "producer_kind": MODEL_PRODUCER_KIND,
                        "model_id": self.configuration.model_id,
                        "model_revision": self.configuration.model_revision,
                        "prompt": self.configuration.prompt,
                        "proposal_count": 1,
                        "boxes_xyxy": [[10, 10, 82, 92]],
                        "raw_response": "<box><39><39><320><359></box>",
                        "generation_stats": {
                            "effective_generation_mode": "hybrid",
                            "effective_max_new_tokens": 8192,
                        },
                        "response_token_count": 1,
                        "possibly_truncated": False,
                        "truncation_check": "token_count_heuristic",
                        "runtime_seconds": 0.1,
                        "status": "ok",
                        "error_type": None,
                        "error_message": None,
                    }
                )
        manifest_path = self.model_root / "neutral_corpus_manifest.jsonl"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.write_jsonl(manifest_path, manifest_rows)
        manifest_sha256 = file_sha256(manifest_path)
        for row in proposal_rows:
            row["input_manifest_sha256"] = manifest_sha256
        content_paths.append(manifest_path.relative_to(self.model_root))
        checksum_path = self.model_root / "content_checksums.sha256"
        checksum_path.write_text(
            "".join(
                f"{file_sha256(self.model_root / path)}  {path.as_posix()}\n"
                for path in sorted(content_paths, key=lambda value: value.as_posix())
            ),
            encoding="utf-8",
        )
        package_audit = PackageAudit(
            cohort_id=self.contract.cohort_id,
            release_id=self.contract.release_id,
            scene_count=2,
            view_count=20,
            image_count=40,
            width=256,
            height=256,
            media_mode="RGB",
            total_media_bytes=sum((self.model_root / path).stat().st_size for path in content_paths if path.suffix == ".png"),
            manifest_sha256=manifest_sha256,
            membership_sha256=self.contract.membership_sha256,
            content_checksums_sha256=file_sha256(checksum_path),
            verified_image_count=40,
            duplicate_archive_member_count=0,
            duplicate_package_path_count=0,
            decoded_image_failure_count=0,
            all_checks_passed=True,
        )
        self.write_json(self.model_root / "package_record.json", package_audit.to_dict())
        self.geometry_path.parent.mkdir(parents=True, exist_ok=True)
        self.write_jsonl(self.geometry_path, geometry_rows)
        self.write_json(
            self.geometry_record_path,
            {
                "package_kind": "ds1-trusted-geometry",
                "cohort_id": self.contract.cohort_id,
                "release_id": self.contract.release_id,
                "membership_sha256": self.contract.membership_sha256,
                "reference_manifest_sha256": file_sha256(self.geometry_path),
                "all_checks_passed": True,
            },
        )
        self.proposal_path.parent.mkdir(parents=True, exist_ok=True)
        self.write_jsonl(self.proposal_path, proposal_rows)
        self.safe_root.mkdir(parents=True, exist_ok=True)
        self.write_json(
            self.run_record_path,
            {
                "stage": "ds1_locateanything_screen",
                "status": "completed",
                "completion_gate_passed": True,
                "run_mode": "model_inference",
                "producer_kind": MODEL_PRODUCER_KIND,
                "inference_stage": "full",
                "cohort": {
                    "cohort_id": self.contract.cohort_id,
                    "release_id": self.contract.release_id,
                    "role": self.contract.role,
                    "pilot_stage": self.contract.pilot_stage,
                    "membership_sha256": self.contract.membership_sha256,
                    "scene_count": 2,
                    "views_per_scene": 20,
                    "image_count": 40,
                    "asset_group_count": 2,
                },
                "configuration": asdict(self.configuration),
                "configuration_sha256": self.configuration.sha256,
                "output_identity": {
                    "result_filename": "proposal_results.jsonl",
                    "result_sha256": file_sha256(self.proposal_path),
                    "proposal_results_sha256": file_sha256(self.proposal_path),
                    "result_count": 40,
                },
                "provenance": {
                    "git_commit": "a" * 40,
                    "git_dirty": False,
                    "precommit_candidate": False,
                    "snapshot_sha256": "b" * 64,
                    "notebook_sha256": "c" * 64,
                },
                "summary": {
                    "job_count": 40,
                    "successful_job_count": 40,
                    "failed_job_count": 0,
                },
            },
        )
        artifact_row = {
            "path": "run_record.json",
            "sha256": file_sha256(self.run_record_path),
            "size_bytes": self.run_record_path.stat().st_size,
        }
        self.write_json(
            self.artifact_manifest_path,
            {
                "schema_version": 1,
                "artifact_count": 1,
                "artifacts": [artifact_row],
                "manifest_content_sha256": sha256(canonical_json_bytes(artifact_row) + b"\n").hexdigest(),
            },
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def write_json(path: Path, value: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @staticmethod
    def write_jsonl(path: Path, rows: list[dict]) -> None:
        path.write_text(
            "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
            encoding="utf-8",
        )

    def build(self) -> dict:
        with patch.object(PACKAGE, "DS1_SCREEN_CONTRACT", self.contract):
            return PACKAGE.build_ds1_sam_prompt_package(
                model_input_root=self.model_root,
                geometry_manifest_path=self.geometry_path,
                geometry_package_record_path=self.geometry_record_path,
                proposal_results_path=self.proposal_path,
                proposal_run_record_path=self.run_record_path,
                proposal_artifact_manifest_path=self.artifact_manifest_path,
                output_root=self.output_root,
            )

    def refresh_safe_artifact_manifest(self) -> None:
        artifact_row = {
            "path": "run_record.json",
            "sha256": file_sha256(self.run_record_path),
            "size_bytes": self.run_record_path.stat().st_size,
        }
        self.write_json(
            self.artifact_manifest_path,
            {
                "schema_version": 1,
                "artifact_count": 1,
                "artifacts": [artifact_row],
                "manifest_content_sha256": sha256(
                    canonical_json_bytes(artifact_row) + b"\n"
                ).hexdigest(),
            },
        )

    def commit_proposal_history(
        self,
        rows: list[dict],
        *,
        result_count: int = 40,
    ) -> None:
        self.write_jsonl(self.proposal_path, rows)
        record = json.loads(self.run_record_path.read_text(encoding="utf-8"))
        digest = file_sha256(self.proposal_path)
        record["output_identity"].update(
            {
                "result_sha256": digest,
                "proposal_results_sha256": digest,
                "result_count": result_count,
            }
        )
        self.write_json(self.run_record_path, record)
        self.refresh_safe_artifact_manifest()

    @staticmethod
    def retry_row(
        source: dict,
        attempt: int,
        *,
        status: str,
        error_type: str | None = None,
    ) -> dict:
        row = deepcopy(source)
        mode = "hybrid" if attempt <= 4 else "slow"
        row.update(
            {
                "attempt": attempt,
                "effective_generation_mode": mode,
                "effective_max_new_tokens": 8192,
                "generation_stats": {
                    "effective_generation_mode": mode,
                    "effective_max_new_tokens": 8192,
                },
                "status": status,
            }
        )
        if status == "ok":
            row.update(
                {
                    "proposal_count": 1,
                    "boxes_xyxy": [[10, 10, 82, 92]],
                    "possibly_truncated": False,
                    "error_type": None,
                    "error_message": None,
                }
            )
        else:
            row.update(
                {
                    "proposal_count": 0,
                    "boxes_xyxy": [],
                    "possibly_truncated": error_type == "PossibleTruncation",
                    "error_type": error_type,
                    "error_message": f"Sanitized {error_type} failure",
                }
            )
        return row

    def test_builds_two_routes_without_copying_evaluator_records(self) -> None:
        record = self.build()
        text = (self.output_root / "sam_prompt_manifest.jsonl").read_text(encoding="utf-8")
        self.assertEqual(record["image_count"], 40)
        self.assertEqual(record["reference_prompt_count"], 40)
        self.assertEqual(record["automatic_prompt_count"], 40)
        self.assertIn('"reference_box"', text)
        self.assertIn('"automatic_box"', text)
        self.assertNotIn("raw_response", text)
        self.assertNotIn("opaque_object_id", text)
        self.assertNotIn("mask", text.lower())
        with patch.object(PACKAGE, "DS1_SCREEN_CONTRACT", self.contract):
            PACKAGE.verify_ds1_sam_prompt_package(self.output_root)

    def test_rejects_results_from_another_configuration(self) -> None:
        rows = [json.loads(line) for line in self.proposal_path.read_text().splitlines()]
        rows[0]["configuration_sha256"] = "0" * 64
        self.commit_proposal_history(rows)
        with self.assertRaisesRegex(ValueError, "configuration"):
            self.build()

    def test_accepts_append_only_retry_history_and_uses_latest_success(self) -> None:
        rows = [json.loads(line) for line in self.proposal_path.read_text().splitlines()]
        source = rows.pop(0)
        history = [
            self.retry_row(
                source,
                attempt,
                status="error",
                error_type="PossibleTruncation",
            )
            for attempt in range(1, 5)
        ]
        history.append(self.retry_row(source, 5, status="ok"))
        self.commit_proposal_history(history + rows)

        record = self.build()

        self.assertEqual(record["image_count"], 40)
        self.assertEqual(record["automatic_prompt_count"], 40)
        upstream = json.loads(
            (self.output_root / "upstream_decision.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            upstream["proposal_results_sha256"], file_sha256(self.proposal_path)
        )

    def test_rejects_duplicate_retry_attempt(self) -> None:
        rows = [json.loads(line) for line in self.proposal_path.read_text().splitlines()]
        rows.insert(1, deepcopy(rows[0]))
        self.commit_proposal_history(rows)
        with self.assertRaisesRegex(ValueError, "duplicate attempt"):
            self.build()

    def test_rejects_skipped_retry_attempt(self) -> None:
        rows = [json.loads(line) for line in self.proposal_path.read_text().splitlines()]
        source = rows.pop(0)
        history = [
            self.retry_row(
                source,
                1,
                status="error",
                error_type="PossibleTruncation",
            ),
            self.retry_row(source, 3, status="ok"),
        ]
        self.commit_proposal_history(history + rows)
        with self.assertRaisesRegex(ValueError, "not sequential"):
            self.build()

    def test_rejects_retry_history_continuing_after_success(self) -> None:
        rows = [json.loads(line) for line in self.proposal_path.read_text().splitlines()]
        source = rows.pop(0)
        history = [self.retry_row(source, 1, status="ok")]
        history.append(
            self.retry_row(
                source,
                2,
                status="error",
                error_type="PossibleTruncation",
            )
        )
        self.commit_proposal_history(history + rows)
        with self.assertRaisesRegex(ValueError, "continues after success"):
            self.build()

    def test_rejects_invalid_slow_retry_transition(self) -> None:
        rows = [json.loads(line) for line in self.proposal_path.read_text().splitlines()]
        source = rows.pop(0)
        history = [
            self.retry_row(
                source,
                attempt,
                status="error",
                error_type="NoValidBoxes",
            )
            for attempt in range(1, 5)
        ]
        history.append(self.retry_row(source, 5, status="ok"))
        self.commit_proposal_history(history + rows)
        with self.assertRaisesRegex(ValueError, "directly preceding"):
            self.build()

    def test_rejects_history_whose_latest_attempt_failed(self) -> None:
        rows = [json.loads(line) for line in self.proposal_path.read_text().splitlines()]
        rows[0] = self.retry_row(
            rows[0],
            1,
            status="error",
            error_type="NoValidBoxes",
        )
        self.commit_proposal_history(rows)
        with self.assertRaisesRegex(ValueError, "not accepted model outputs"):
            self.build()

    def test_rejects_unverified_trusted_geometry(self) -> None:
        record = json.loads(self.geometry_record_path.read_text(encoding="utf-8"))
        record["all_checks_passed"] = False
        self.write_json(self.geometry_record_path, record)
        with self.assertRaisesRegex(ValueError, "not verified"):
            self.build()

    def test_rejects_dirty_proposal_provenance(self) -> None:
        record = json.loads(self.run_record_path.read_text(encoding="utf-8"))
        record["provenance"]["git_dirty"] = True
        self.write_json(self.run_record_path, record)
        self.refresh_safe_artifact_manifest()
        with self.assertRaisesRegex(ValueError, "clean committed"):
            self.build()

    def test_rejects_tampered_history_not_committed_by_run_record(self) -> None:
        rows = [json.loads(line) for line in self.proposal_path.read_text().splitlines()]
        rows[0]["boxes_xyxy"] = [[11, 11, 83, 93]]
        self.write_jsonl(self.proposal_path, rows)
        with self.assertRaisesRegex(ValueError, "another proposal file"):
            self.build()


if __name__ == "__main__":
    unittest.main()
