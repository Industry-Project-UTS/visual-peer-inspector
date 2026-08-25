from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))

from visual_peer_inspector.data.cohort_contract import (
    CohortContract,
    DS1_SCREEN_CONTRACT,
)
from visual_peer_inspector.data.model_safe_package import (
    PackageAudit,
    file_sha256,
    membership_sha256,
    select_contract_rows,
    verify_model_input_package,
)
from visual_peer_inspector.data.toysad_selection import RELEASE_ID


class CohortContractTests(unittest.TestCase):
    def test_ds1_screen_contract_is_frozen(self) -> None:
        self.assertEqual(
            DS1_SCREEN_CONTRACT.cohort_id,
            "toysad8k-development-screen-100-v1",
        )
        self.assertEqual(DS1_SCREEN_CONTRACT.release_id, "toysad8k-train-2000-v2")
        self.assertEqual(DS1_SCREEN_CONTRACT.role, "development")
        self.assertEqual(DS1_SCREEN_CONTRACT.pilot_stage, "screen_100")
        self.assertEqual(DS1_SCREEN_CONTRACT.scene_count, 100)
        self.assertEqual(DS1_SCREEN_CONTRACT.views_per_scene, 20)
        self.assertEqual(DS1_SCREEN_CONTRACT.image_count, 2_000)
        self.assertEqual(DS1_SCREEN_CONTRACT.asset_group_count, 27)
        self.assertEqual(
            DS1_SCREEN_CONTRACT.membership_sha256,
            "308e2e104448612dc60b5392a5258d97f6d93f52ca553aa68fa1b447500f4f2f",
        )

    def test_contract_rejects_inconsistent_image_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "scenes times views"):
            CohortContract(
                cohort_id="test-cohort",
                release_id="test-release",
                role="development",
                pilot_stage="screen",
                membership_sha256="0" * 64,
                scene_count=3,
                views_per_scene=2,
                image_count=5,
                asset_group_count=2,
            )

    def test_contract_selection_checks_membership_and_assets(self) -> None:
        scene_ids = ["S000001", "S000002", "S000003"]
        contract = CohortContract(
            cohort_id="test-cohort",
            release_id=RELEASE_ID,
            role="development",
            pilot_stage="screen",
            membership_sha256=membership_sha256(scene_ids),
            scene_count=3,
            views_per_scene=20,
            image_count=60,
            asset_group_count=2,
        )
        rows = [
            {
                "role": "development",
                "pilot_stage": "screen",
                "opaque_scene_id": scene_id,
                "opaque_asset_group_id": f"A{index // 2:02d}",
                "category": f"category-{index}",
                "object_count": 3,
                "anomaly_count": 1,
                "raw_anomaly_prefixes": ["bump"],
            }
            for index, scene_id in enumerate(reversed(scene_ids))
        ]

        selected, audit = select_contract_rows(rows, contract)

        self.assertEqual(
            [row["opaque_scene_id"] for row in selected],
            scene_ids,
        )
        self.assertEqual(audit["scene_count"], 3)
        self.assertEqual(audit["asset_group_count"], 2)
        self.assertTrue(audit["all_checks_passed"])

    def test_generic_package_verifier_uses_supplied_contract(self) -> None:
        scene_ids = ["S000001", "S000002", "S000003"]
        contract = CohortContract(
            cohort_id="test-cohort",
            release_id=RELEASE_ID,
            role="development",
            pilot_stage="screen",
            membership_sha256=membership_sha256(scene_ids),
            scene_count=3,
            views_per_scene=20,
            image_count=60,
            asset_group_count=2,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest_rows = []
            content_paths = []
            for scene_index, scene_id in enumerate(scene_ids):
                view_ids = [f"V{index:02d}" for index in range(20)]
                manifest_rows.append(
                    {
                        "schema_version": 1,
                        "release_id": contract.release_id,
                        "opaque_scene_id": scene_id,
                        "object_count": 3,
                        "object_ids": ["O00", "O01", "O02"],
                        "view_ids": view_ids,
                        "media_root": f"media/{scene_id}",
                    }
                )
                for view_index, view_id in enumerate(view_ids):
                    path = root / "media" / scene_id / "rgb" / f"{view_id}.png"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    Image.new(
                        "RGB",
                        (256, 256),
                        (scene_index, view_index, 100),
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
                    for path in sorted(
                        content_paths,
                        key=lambda value: value.as_posix(),
                    )
                ),
                encoding="utf-8",
            )
            record = PackageAudit(
                cohort_id=contract.cohort_id,
                release_id=contract.release_id,
                scene_count=3,
                view_count=20,
                image_count=60,
                width=256,
                height=256,
                media_mode="RGB",
                total_media_bytes=sum(
                    (root / path).stat().st_size
                    for path in content_paths
                    if path.suffix == ".png"
                ),
                manifest_sha256=file_sha256(manifest_path),
                membership_sha256=contract.membership_sha256,
                content_checksums_sha256=file_sha256(checksum_path),
                verified_image_count=60,
                duplicate_archive_member_count=0,
                duplicate_package_path_count=0,
                decoded_image_failure_count=0,
                all_checks_passed=True,
            )
            (root / "package_record.json").write_text(
                json.dumps(record.to_dict(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            verified = verify_model_input_package(root, contract=contract)

            self.assertEqual(verified.cohort_id, contract.cohort_id)
            self.assertEqual(verified.image_count, 60)


if __name__ == "__main__":
    unittest.main()
