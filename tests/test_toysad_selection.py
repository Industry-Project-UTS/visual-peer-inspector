from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))

from visual_peer_inspector.data.toysad_selection import (
    RELEASE_ID,
    anomaly_prefixes,
    assemble_eligible_scene_pool,
    deterministic_object_permutation,
    validate_model_facing_media_record,
    verify_release_checksums,
)


class ToysSelectionHelperTests(unittest.TestCase):
    def test_anomaly_prefixes_are_unique_and_sorted(self) -> None:
        scene = {"anomaly_variants": ["miss-001", "crack-003", "miss-009"]}

        self.assertEqual(anomaly_prefixes(scene), ("crack", "miss"))

    def test_object_permutation_is_stable_and_complete(self) -> None:
        first = deterministic_object_permutation("airplane_000-001", 5)
        second = deterministic_object_permutation("airplane_000-001", 5)

        self.assertEqual(first, second)
        self.assertEqual(sorted(first), list(range(5)))

    def test_eligibility_excludes_seen_asset_and_unavailable_scene(self) -> None:
        scenes = [
            self.scene("apple_000-001", "apple_000", "train", True),
            self.scene("apple_000-002", "apple_000", "seen-test", True),
            self.scene("boat_000-001", "boat_000", "train", False),
            self.scene("car_000-001", "car_000", "train", True),
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            split = Path(temporary_directory) / "split.json"
            split.write_text(
                '{"train":["apple_000-001","boat_000-001","car_000-001"],'
                '"seen-test":["apple_000-002"],"unseen-test":[]}',
                encoding="utf-8",
            )

            eligible = assemble_eligible_scene_pool(
                {"scenes": scenes}, split, enforce_expected_counts=False
            )

        self.assertEqual([scene["scene_id"] for scene in eligible], ["car_000-001"])

    def test_checksum_verifier_detects_valid_release_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            payload = root / "payload.txt"
            payload.write_text(RELEASE_ID, encoding="utf-8")
            from hashlib import sha256

            digest = sha256(RELEASE_ID.encode("utf-8")).hexdigest()
            (root / "checksums.sha256").write_text(
                f"{digest}  payload.txt\n", encoding="utf-8"
            )

            result = verify_release_checksums(root)

        self.assertEqual(result, {"payload.txt": digest})

    def test_checksum_verifier_rejects_unlisted_extra_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            payload = root / "payload.txt"
            payload.write_text(RELEASE_ID, encoding="utf-8")
            (root / "extra.txt").write_text("unexpected", encoding="utf-8")
            from hashlib import sha256

            digest = sha256(RELEASE_ID.encode("utf-8")).hexdigest()
            (root / "checksums.sha256").write_text(
                f"{digest}  payload.txt\n", encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "unlisted"):
                verify_release_checksums(root, {"payload.txt"})

    def test_model_facing_record_rejects_asset_group_and_labels(self) -> None:
        row = self.media_record()
        row["opaque_asset_group_id"] = "A000001"

        with self.assertRaisesRegex(ValueError, "not exact"):
            validate_model_facing_media_record(row)

        row = self.media_record()
        row["labels_by_object_id"] = {"O00": 1, "O01": 0, "O02": 0}
        with self.assertRaisesRegex(ValueError, "not exact"):
            validate_model_facing_media_record(row)

    def test_model_facing_record_accepts_only_media_fields(self) -> None:
        validate_model_facing_media_record(self.media_record())

    @staticmethod
    def scene(
        scene_id: str,
        asset_id: str,
        split: str,
        availability_ok: bool,
    ) -> dict[str, object]:
        return {
            "scene_id": scene_id,
            "base_asset_id": asset_id,
            "split": split,
            "availability_ok": availability_ok,
            "category": scene_id.split("_", 1)[0],
            "object_count": 3,
            "anomaly_count": 1,
            "anomaly_variants": ["miss-001"],
        }

    @staticmethod
    def media_record() -> dict[str, object]:
        return {
            "schema_version": 1,
            "release_id": RELEASE_ID,
            "opaque_scene_id": "S000001",
            "object_count": 3,
            "object_ids": ["O00", "O01", "O02"],
            "view_ids": [f"V{index:02d}" for index in range(20)],
            "media_root": "media/S000001",
        }


if __name__ == "__main__":
    unittest.main()
