import json
import sqlite3
from contextlib import ExitStack
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock
import zipfile

from visual_peer_inspector.data.model_safe_package import (
    SMOKE20_COHORT_ID,
    canonical_json_bytes,
    file_sha256,
    membership_sha256,
)
from visual_peer_inspector.evaluation import sam_mask_package as package


class SamMaskEvaluatorPackageTests(unittest.TestCase):
    @staticmethod
    def write_jsonl(path: Path, rows) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(
            b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
        )

    @staticmethod
    def write_json(path: Path, value) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    @staticmethod
    def build_archive_index(archive_path: Path, index_path: Path) -> None:
        with zipfile.ZipFile(archive_path) as archive:
            infos = archive.infolist()
        with sqlite3.connect(index_path) as connection:
            connection.execute(
                "CREATE TABLE members ("
                "id INTEGER PRIMARY KEY, name TEXT NOT NULL, "
                "compressed_size INTEGER NOT NULL, "
                "uncompressed_size INTEGER NOT NULL, crc32 INTEGER NOT NULL, "
                "compression_method INTEGER NOT NULL, flags INTEGER NOT NULL, "
                "local_header_offset INTEGER NOT NULL, is_directory INTEGER NOT NULL, "
                "is_symlink INTEGER NOT NULL, unsafe_reason TEXT, top_level TEXT, "
                "extension TEXT)"
            )
            for index, info in enumerate(infos, 1):
                connection.execute(
                    "INSERT INTO members VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        index,
                        info.filename,
                        info.compress_size,
                        info.file_size,
                        info.CRC,
                        info.compress_type,
                        info.flag_bits,
                        info.header_offset,
                        int(info.is_dir()),
                        0,
                        None,
                        "toysad8k",
                        ".png",
                    ),
                )

    def create_inputs(self, root: Path):
        geometry_root = root / "geometry"
        crosswalk_root = root / "crosswalk"
        attributes_root = root / "attributes"
        archive_path = root / "toysAD8K.zip"
        index_path = root / "archive_index.sqlite3"
        scenes = ["S00", "S01"]
        views = ["V00", "V01"]
        object_id = "O00"
        geometry_rows = []
        source_by_key = {}
        with zipfile.ZipFile(
            archive_path, mode="w", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            for scene_id in scenes:
                for view_id in views:
                    data = f"supplied-mask-{scene_id}-{view_id}".encode("utf-8")
                    source_member = (
                        f"toysad8k/{scene_id}/segmentations/object.001/"
                        f"object.001_{view_id}.png"
                    )
                    archive.writestr(source_member, data)
                    relative = Path("masks") / scene_id / object_id / f"{view_id}.png"
                    mask_path = geometry_root / relative
                    mask_path.parent.mkdir(parents=True, exist_ok=True)
                    mask_path.write_bytes(data)
                    box = [1, 2, 7, 9]
                    geometry_rows.append(
                        {
                            "schema_version": 1,
                            "opaque_scene_id": scene_id,
                            "view_id": view_id,
                            "opaque_object_id": object_id,
                            "mask_path": relative.as_posix(),
                            "mask_sha256": sha256(data).hexdigest(),
                            "foreground_pixels": 42,
                            "visible": True,
                            "box_xyxy": box,
                        }
                    )
                    source_by_key[(scene_id, view_id, object_id)] = source_member
        self.build_archive_index(archive_path, index_path)

        self.write_jsonl(geometry_root / "reference_manifest.jsonl", geometry_rows)
        (geometry_root / "content_checksums.sha256").write_text(
            "geometry-checksum-placeholder\n", encoding="utf-8"
        )
        geometry_record = {
            "schema_version": 1,
            "package_kind": "ds0-oracle-geometry",
            "cohort_id": SMOKE20_COHORT_ID,
            "membership_sha256": membership_sha256(scenes),
            "scene_count": 2,
            "view_count": 2,
            "image_count": 4,
            "object_view_count": 4,
            "visible_object_view_count": 4,
            "empty_object_view_count": 0,
            "archive_index_sha256": file_sha256(index_path),
            "all_checks_passed": True,
        }
        self.write_json(geometry_root / "package_record.json", geometry_record)

        crosswalk_rows = []
        for row in geometry_rows:
            scene_id = row["opaque_scene_id"]
            view_id = row["view_id"]
            crosswalk_rows.extend(
                [
                    {
                        "schema_version": 1,
                        "opaque_scene_id": scene_id,
                        "view_id": view_id,
                        "route": "reference_box",
                        "prompt_key": "R000",
                        "prompt_box_xyxy": row["box_xyxy"],
                        "opaque_object_id": object_id,
                        "proposal_index": None,
                        "matched_opaque_object_id": object_id,
                        "matched_box_iou": 1.0,
                    },
                    {
                        "schema_version": 1,
                        "opaque_scene_id": scene_id,
                        "view_id": view_id,
                        "route": "automatic_box",
                        "prompt_key": "P000",
                        "prompt_box_xyxy": [0, 1, 8, 10],
                        "opaque_object_id": None,
                        "proposal_index": 0,
                        "matched_opaque_object_id": object_id,
                        "matched_box_iou": 0.9,
                    },
                ]
            )
        self.write_jsonl(
            crosswalk_root / "sam_evaluator_crosswalk.jsonl", crosswalk_rows
        )
        (crosswalk_root / "content_checksums.sha256").write_text(
            "crosswalk-checksum-placeholder\n", encoding="utf-8"
        )
        crosswalk_record = {
            "schema_version": 1,
            "package_kind": "ds0-sam-evaluator-crosswalk",
            "row_count": 8,
            "reference_prompt_count": 4,
            "automatic_prompt_count": 4,
        }
        self.write_json(crosswalk_root / "package_record.json", crosswalk_record)

        attribute_rows = [
            {
                "schema_version": 1,
                "opaque_scene_id": scene_id,
                "opaque_asset_group_id": f"A-{scene_id}",
                "category": f"category-{scene_id}",
                "object_count": 1,
                "anomaly_count": 1,
                "anomaly_families": ["test"],
                "labels_by_object_id": {object_id: 1},
            }
            for scene_id in scenes
        ]
        self.write_jsonl(
            attributes_root / "protected_scene_attributes.jsonl", attribute_rows
        )
        (attributes_root / "content_checksums.sha256").write_text(
            "attributes-checksum-placeholder\n", encoding="utf-8"
        )
        attributes_record = {
            "schema_version": 1,
            "package_kind": "ds0-evaluator-attributes",
            "scene_count": 2,
        }
        self.write_json(attributes_root / "package_record.json", attributes_record)
        return {
            "geometry_root": geometry_root,
            "crosswalk_root": crosswalk_root,
            "attributes_root": attributes_root,
            "archive_path": archive_path,
            "index_path": index_path,
            "geometry_record": geometry_record,
            "crosswalk_record": crosswalk_record,
            "attributes_record": attributes_record,
            "source_by_key": source_by_key,
            "scenes": scenes,
        }

    def patched_ds0(self, inputs, *, archive_sha256=None):
        stack = ExitStack()
        stack.enter_context(mock.patch.object(package, "EXPECTED_DS0_SCENES", 2))
        stack.enter_context(mock.patch.object(package, "EXPECTED_DS0_VIEWS_PER_SCENE", 2))
        stack.enter_context(mock.patch.object(package, "EXPECTED_DS0_IMAGES", 4))
        stack.enter_context(mock.patch.object(package, "EXPECTED_VISIBLE_MASKS", 4))
        stack.enter_context(mock.patch.object(package, "EXPECTED_REFERENCE_PROMPTS", 4))
        stack.enter_context(mock.patch.object(package, "EXPECTED_AUTOMATIC_PROMPTS", 4))
        stack.enter_context(mock.patch.object(package, "EXPECTED_SAM_CROSSWALK_ROWS", 8))
        stack.enter_context(
            mock.patch.object(
                package,
                "SMOKE20_MEMBERSHIP_SHA256",
                membership_sha256(inputs["scenes"]),
            )
        )
        stack.enter_context(
            mock.patch.object(
                package,
                "EXPECTED_ARCHIVE_SIZE_BYTES",
                inputs["archive_path"].stat().st_size,
            )
        )
        stack.enter_context(
            mock.patch.object(
                package,
                "EXPECTED_ARCHIVE_SHA256",
                archive_sha256 or file_sha256(inputs["archive_path"]),
            )
        )
        stack.enter_context(
            mock.patch.object(
                package,
                "verify_ds0_geometry_package",
                return_value=inputs["geometry_record"],
            )
        )
        stack.enter_context(
            mock.patch.object(
                package,
                "verify_sam_evaluator_crosswalk",
                return_value=inputs["crosswalk_record"],
            )
        )
        stack.enter_context(
            mock.patch.object(
                package,
                "verify_ds0_attributes_package",
                return_value=inputs["attributes_record"],
            )
        )
        return stack

    def test_builds_atomic_private_package_from_exact_archive_masks(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = self.create_inputs(root)
            output = root / "private-evaluator"
            with self.patched_ds0(inputs):
                record = package.build_ds0_mask_evaluator_package(
                    geometry_root=inputs["geometry_root"],
                    crosswalk_root=inputs["crosswalk_root"],
                    attributes_root=inputs["attributes_root"],
                    source_archive_path=inputs["archive_path"],
                    archive_index_path=inputs["index_path"],
                    output_root=output,
                )
                verified = package.verify_ds0_mask_evaluator_package(output)

            self.assertEqual(record, verified)
            self.assertFalse((root / "private-evaluator.incomplete").exists())
            self.assertEqual(record["supplied_mask_count"], 4)
            self.assertEqual(record["crosswalk_row_count"], 8)
            self.assertFalse(record["model_facing_allowed"])
            self.assertEqual(
                (output / "oracle_geometry/reference_manifest.jsonl").read_bytes(),
                (inputs["geometry_root"] / "reference_manifest.jsonl").read_bytes(),
            )
            self.assertEqual(
                (output / "sam_crosswalk/sam_evaluator_crosswalk.jsonl").read_bytes(),
                (
                    inputs["crosswalk_root"] / "sam_evaluator_crosswalk.jsonl"
                ).read_bytes(),
            )
            self.assertEqual(
                (
                    output
                    / "scene_attributes/protected_scene_attributes.jsonl"
                ).read_bytes(),
                (
                    inputs["attributes_root"] / "protected_scene_attributes.jsonl"
                ).read_bytes(),
            )
            archive_rows = [
                json.loads(line)
                for line in (output / "archive_mask_manifest.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(
                {row["source_member"] for row in archive_rows},
                set(inputs["source_by_key"].values()),
            )

    def test_model_facing_use_is_rejected_before_private_data_is_read(self):
        with self.assertRaisesRegex(ValueError, "evaluator-only"):
            package.verify_ds0_mask_evaluator_package(
                Path("not-needed"), intended_use="model_facing"
            )

    def test_wrong_archive_sha_fails_without_publishing_output(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = self.create_inputs(root)
            output = root / "private-evaluator"
            with self.patched_ds0(inputs, archive_sha256="0" * 64):
                with self.assertRaisesRegex(ValueError, "archive SHA-256"):
                    package.build_ds0_mask_evaluator_package(
                        geometry_root=inputs["geometry_root"],
                        crosswalk_root=inputs["crosswalk_root"],
                        attributes_root=inputs["attributes_root"],
                        source_archive_path=inputs["archive_path"],
                        archive_index_path=inputs["index_path"],
                        output_root=output,
                    )
            self.assertFalse(output.exists())
            self.assertFalse((root / "private-evaluator.incomplete").exists())

    def test_wrong_frozen_mask_sha_cleans_the_atomic_staging_directory(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = self.create_inputs(root)
            manifest_path = inputs["geometry_root"] / "reference_manifest.jsonl"
            rows = [
                json.loads(line)
                for line in manifest_path.read_text(encoding="utf-8").splitlines()
            ]
            rows[0]["mask_sha256"] = "0" * 64
            self.write_jsonl(manifest_path, rows)
            output = root / "private-evaluator"
            with self.patched_ds0(inputs):
                with self.assertRaisesRegex(ValueError, "Frozen oracle mask"):
                    package.build_ds0_mask_evaluator_package(
                        geometry_root=inputs["geometry_root"],
                        crosswalk_root=inputs["crosswalk_root"],
                        attributes_root=inputs["attributes_root"],
                        source_archive_path=inputs["archive_path"],
                        archive_index_path=inputs["index_path"],
                        output_root=output,
                    )
            self.assertFalse(output.exists())
            self.assertFalse((root / "private-evaluator.incomplete").exists())

    def test_unlisted_file_invalidates_the_package(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = self.create_inputs(root)
            output = root / "private-evaluator"
            with self.patched_ds0(inputs):
                package.build_ds0_mask_evaluator_package(
                    geometry_root=inputs["geometry_root"],
                    crosswalk_root=inputs["crosswalk_root"],
                    attributes_root=inputs["attributes_root"],
                    source_archive_path=inputs["archive_path"],
                    archive_index_path=inputs["index_path"],
                    output_root=output,
                )
                (output / "unexpected.txt").write_text("private", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "file allowlist"):
                    package.verify_ds0_mask_evaluator_package(output)


if __name__ == "__main__":
    unittest.main()
