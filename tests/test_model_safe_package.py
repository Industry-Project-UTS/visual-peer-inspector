from __future__ import annotations

from hashlib import sha256
import binascii
import json
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))

from visual_peer_inspector.data.model_safe_package import (
    PackageAudit,
    SMOKE20_COHORT_ID,
    SMOKE20_MEMBERSHIP_SHA256,
    choose_smoke20_rows,
    file_sha256,
    verify_checksum_manifest,
    verify_model_input_package,
    _read_extracted_member,
)
from visual_peer_inspector.data.toysad_selection import RELEASE_ID


FROZEN_SCENE_IDS = (
    "S000021",
    "S000110",
    "S000300",
    "S000318",
    "S000547",
    "S000821",
    "S000845",
    "S001012",
    "S001107",
    "S001261",
    "S001331",
    "S001388",
    "S001505",
    "S001625",
    "S001676",
    "S001700",
    "S001726",
    "S001783",
    "S001843",
    "S001866",
)


class ModelSafePackageTests(unittest.TestCase):
    def test_extracted_source_member_checks_path_size_and_crc(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            member = root / "toysad8k" / "scene" / "RGB" / "0000.png"
            member.parent.mkdir(parents=True)
            member.write_bytes(b"verified-rgb-bytes")
            expected_crc = binascii.crc32(member.read_bytes()) & 0xFFFFFFFF

            data = _read_extracted_member(
                root,
                "toysad8k/scene/RGB/0000.png",
                expected_crc32=expected_crc,
                expected_size=member.stat().st_size,
            )

            self.assertEqual(data, b"verified-rgb-bytes")
            with self.assertRaisesRegex(ValueError, "CRC or size"):
                _read_extracted_member(
                    root,
                    "toysad8k/scene/RGB/0000.png",
                    expected_crc32=expected_crc + 1,
                    expected_size=member.stat().st_size,
                )
            with self.assertRaisesRegex(ValueError, "unsafe path"):
                _read_extracted_member(
                    root,
                    "../outside.png",
                    expected_crc32=0,
                    expected_size=0,
                )

    def test_checksum_manifest_rejects_changed_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            data_path = root / "accepted.json"
            data_path.write_text("{}\n", encoding="utf-8")
            (root / "checksums.sha256").write_text(
                f"{file_sha256(data_path)}  accepted.json\n",
                encoding="utf-8",
            )
            verify_checksum_manifest(root, manifest_name="checksums.sha256")
            data_path.write_text('{"changed": true}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                verify_checksum_manifest(root, manifest_name="checksums.sha256")

    def test_greedy_smoke_selection_covers_required_dimensions(self) -> None:
        prefixes = ("bump", "crack", "defr", "disc", "frac", "miss", "rota", "tran")
        strata = ((3, 1), (4, 1), (4, 2), (5, 1), (5, 2))
        rows = []
        for index in range(100):
            objects, anomalies = strata[index % len(strata)]
            rows.append(
                {
                    "role": "development",
                    "pilot_stage": "screen_100",
                    "opaque_scene_id": f"S{index + 1:06d}",
                    "opaque_asset_group_id": f"A{index + 1:06d}",
                    "category": f"category-{index % 20:02d}",
                    "object_count": objects,
                    "anomaly_count": anomalies,
                    "raw_anomaly_prefixes": [prefixes[index % len(prefixes)]],
                }
            )

        selected, audit = choose_smoke20_rows(rows, enforce_frozen_hash=False)

        self.assertEqual(len(selected), 20)
        self.assertEqual(audit["asset_group_count"], 20)
        self.assertEqual(audit["category_count"], 20)
        self.assertEqual(audit["prefix_coverage_count"], 8)
        self.assertEqual(audit["stratum_coverage_count"], 5)

    def test_verified_model_input_package_contains_only_opaque_rgb(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest_rows = []
            content_paths = []
            for scene_index, scene_id in enumerate(FROZEN_SCENE_IDS):
                view_ids = [f"V{index:02d}" for index in range(20)]
                manifest_rows.append(
                    {
                        "schema_version": 1,
                        "release_id": RELEASE_ID,
                        "opaque_scene_id": scene_id,
                        "object_count": 3 + scene_index % 3,
                        "object_ids": [
                            f"O{index:02d}" for index in range(3 + scene_index % 3)
                        ],
                        "view_ids": view_ids,
                        "media_root": f"media/{scene_id}",
                    }
                )
                for view_index, view_id in enumerate(view_ids):
                    path = root / "media" / scene_id / "rgb" / f"{view_id}.png"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    Image.new(
                        "RGB", (256, 256), (scene_index, view_index, 100)
                    ).save(path)
                    content_paths.append(path.relative_to(root))

            manifest_path = root / "neutral_corpus_manifest.jsonl"
            manifest_path.write_text(
                "".join(
                    json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                    for row in manifest_rows
                ),
                encoding="utf-8",
            )
            content_paths.append(manifest_path.relative_to(root))
            checksum_path = root / "content_checksums.sha256"
            checksum_path.write_text(
                "".join(
                    f"{file_sha256(root / path)}  {path.as_posix()}\n"
                    for path in sorted(content_paths, key=lambda value: value.as_posix())
                ),
                encoding="utf-8",
            )
            record = PackageAudit(
                cohort_id=SMOKE20_COHORT_ID,
                release_id=RELEASE_ID,
                scene_count=20,
                view_count=20,
                image_count=400,
                width=256,
                height=256,
                media_mode="RGB",
                total_media_bytes=sum(
                    (root / path).stat().st_size
                    for path in content_paths
                    if path.suffix == ".png"
                ),
                manifest_sha256=file_sha256(manifest_path),
                membership_sha256=SMOKE20_MEMBERSHIP_SHA256,
                content_checksums_sha256=file_sha256(checksum_path),
                verified_image_count=400,
                duplicate_archive_member_count=0,
                duplicate_package_path_count=0,
                decoded_image_failure_count=0,
                all_checks_passed=True,
            )
            (root / "package_record.json").write_text(
                json.dumps(record.to_dict(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            verified = verify_model_input_package(root)

            self.assertTrue(verified.all_checks_passed)
            self.assertEqual(verified.image_count, 400)
            self.assertEqual(
                sha256("\n".join(FROZEN_SCENE_IDS).encode()).hexdigest(),
                SMOKE20_MEMBERSHIP_SHA256,
            )

            record_path = root / "package_record.json"
            original_record = json.loads(record_path.read_text(encoding="utf-8"))
            altered_record = dict(original_record)
            altered_record["image_count"] = 1
            record_path.write_text(
                json.dumps(altered_record, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "image_count"):
                verify_model_input_package(root)

            record_path.write_text(
                json.dumps(original_record, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            (root / "labels.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "allowlist"):
                verify_model_input_package(root)


if __name__ == "__main__":
    unittest.main()
