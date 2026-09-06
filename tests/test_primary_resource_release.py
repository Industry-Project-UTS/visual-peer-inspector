from __future__ import annotations

import json
import sqlite3
import zipfile
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import nbformat
from PIL import Image

from visual_peer_inspector.data.cohort_contract import CohortContract
from visual_peer_inspector.data.model_safe_package import file_sha256, membership_sha256
from visual_peer_inspector.workflows import primary_resource_release as release


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _fixture(tmp_path: Path) -> tuple[dict[str, Path], CohortContract]:
    scene_ids = ["S001", "S002"]
    contract = CohortContract(
        cohort_id="fixture-calibration",
        release_id="toysad8k-train-2000-v2",
        role="calibration",
        pilot_stage=None,
        membership_sha256=membership_sha256(scene_ids),
        scene_count=2,
        views_per_scene=20,
        image_count=40,
        asset_group_count=2,
    )
    selection = tmp_path / "selection"
    administration = tmp_path / "administration"
    labels = tmp_path / "labels"
    source = tmp_path / "source"
    for root in (selection, administration, labels, source):
        root.mkdir()
        (root / "checksums.sha256").write_text("fixture\n", encoding="utf-8")
    roles = []
    neutral = []
    trusted = []
    required = []
    label_rows = []
    image_path = tmp_path / "rgb.png"
    Image.new("RGB", (256, 256), (60, 90, 120)).save(image_path)
    rgb = image_path.read_bytes()
    members: dict[str, bytes] = {}
    for scene_index, scene_id in enumerate(scene_ids):
        roles.append(
            {
                "opaque_scene_id": scene_id,
                "opaque_asset_group_id": f"A{scene_index}",
                "role": "calibration",
                "pilot_stage": None,
            }
        )
        neutral.append(
            {
                "schema_version": 1,
                "release_id": contract.release_id,
                "opaque_scene_id": scene_id,
                "object_count": 3,
                "object_ids": ["O00", "O01", "O02"],
                "view_ids": [f"V{index:02d}" for index in range(20)],
                "media_root": f"media/{scene_id}",
            }
        )
        trusted.append(
            {
                "opaque_scene_id": scene_id,
                "opaque_asset_group_id": f"A{scene_index}",
                "role": "calibration",
                "source_scene_id": f"source-{scene_index}",
            }
        )
        label_rows.append(
            {
                "opaque_scene_id": scene_id,
                "labels_by_object_id": {"O00": 1, "O01": 0, "O02": 0},
            }
        )
        for view_index in range(20):
            view_id = f"V{view_index:02d}"
            rgb_name = f"archive/{scene_id}/rgb/{view_index}.png"
            members[rgb_name] = rgb
            required.append(
                {
                    "opaque_scene_id": scene_id,
                    "component": "rgb",
                    "opaque_object_id": None,
                    "view_id": view_id,
                    "source_member": rgb_name,
                }
            )
            for object_index in range(3):
                mask_path = tmp_path / f"mask-{scene_id}-{view_index}-{object_index}.png"
                mask = Image.new("L", (256, 256), 0)
                for x in range(20 + object_index * 60, 60 + object_index * 60):
                    for y in range(30, 80):
                        mask.putpixel((x, y), 255)
                mask.save(mask_path)
                mask_name = f"archive/{scene_id}/mask/{view_index}-{object_index}.png"
                members[mask_name] = mask_path.read_bytes()
                required.append(
                    {
                        "opaque_scene_id": scene_id,
                        "component": "object_mask",
                        "opaque_object_id": f"O{object_index:02d}",
                        "view_id": view_id,
                        "source_member": mask_name,
                    }
                )
    _write_jsonl(selection / "experiment_roles.jsonl", roles)
    _write_jsonl(selection / "neutral_corpus_manifest.jsonl", neutral)
    _write_jsonl(administration / "restricted_source_map.jsonl", trusted)
    _write_jsonl(administration / "required_source_members.jsonl", required)
    _write_jsonl(labels / "calibration_labels.jsonl", label_rows)
    (source / "toysad8k_scene_master.json").write_text(
        json.dumps(
            {
                "scenes": [
                    {
                        "scene_id": f"source-{index}",
                        "category": f"category-{index}",
                        "anomaly_variants": ["colour-red"],
                    }
                    for index in range(2)
                ]
            }
        ),
        encoding="utf-8",
    )
    archive_path = source / "toysAD8K.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    with zipfile.ZipFile(archive_path) as archive, sqlite3.connect(
        source / "archive_index.sqlite3"
    ) as database:
        database.execute(
            "CREATE TABLE members(name TEXT, compressed_size INTEGER, uncompressed_size INTEGER, "
            "crc32 INTEGER, compression_method INTEGER, flags INTEGER, local_header_offset INTEGER, "
            "is_directory INTEGER, is_symlink INTEGER, unsafe_reason TEXT)"
        )
        for info in archive.infolist():
            database.execute(
                "INSERT INTO members VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    info.filename,
                    info.compress_size,
                    info.file_size,
                    info.CRC,
                    info.compress_type,
                    info.flag_bits,
                    info.header_offset,
                    0,
                    0,
                    None,
                ),
            )
    (source / "package-checksums.sha256").write_text(
        "".join(
            f"{file_sha256(source / name)}  {name}\n"
            for name in (
                "archive_index.sqlite3",
                "toysAD8K.zip",
                "toysad8k_scene_master.json",
            )
        ),
        encoding="utf-8",
    )
    prerequisite = tmp_path / "method_freeze.json"
    prerequisite.write_text(json.dumps({"status": "frozen"}), encoding="utf-8")
    (tmp_path / "application_output_contract.json").write_text("{}", encoding="utf-8")
    (tmp_path / "artifact_manifest.json").write_text("{}", encoding="utf-8")
    return {
        "selection": selection,
        "administration": administration,
        "labels": labels,
        "source": source,
        "prerequisite": prerequisite,
    }, contract


def test_protected_resource_release_is_bounded_separate_and_equivalent(
    tmp_path: Path, monkeypatch
) -> None:
    paths, contract = _fixture(tmp_path)
    monkeypatch.setitem(release.ROLE_CONTRACTS, "calibration", contract)
    monkeypatch.setattr(
        release,
        "load_method_freeze",
        lambda *_args, **_kwargs: {
            "sha256": sha256(b"method").hexdigest(),
            "identity_configuration_sha256": sha256(b"identity").hexdigest(),
            "frozen_model_identity_sha256": sha256(b"pipeline").hexdigest(),
        },
    )
    bounded = tmp_path / "bounded"
    first = release.build_protected_role_resources(
        role="calibration",
        selection_root=paths["selection"],
        administrative_root=paths["administration"],
        label_root=paths["labels"],
        source_root=paths["source"],
        prerequisite_lock=paths["prerequisite"],
        output_root=bounded,
        maximum_new_scenes=1,
    )
    assert first["status"] == "checkpointed"
    assert not (bounded / "calibration-model-input/package_record.json").exists()
    assert not (bounded / "calibration-evaluator-only/evaluator_manifest.json").exists()
    second = release.build_protected_role_resources(
        role="calibration",
        selection_root=paths["selection"],
        administrative_root=paths["administration"],
        label_root=paths["labels"],
        source_root=paths["source"],
        prerequisite_lock=paths["prerequisite"],
        output_root=bounded,
        maximum_new_scenes=1,
    )
    assert second["status"] == "complete"
    model_root = bounded / "calibration-model-input"
    evaluator_root = bounded / "calibration-evaluator-only"
    assert model_root.resolve() != evaluator_root.resolve()
    assert "anomaly" not in (model_root / "neutral_corpus_manifest.jsonl").read_text().lower()
    private_labels = (evaluator_root / "labels_private.jsonl").read_text()
    private_crosswalk = (evaluator_root / "crosswalk_private.jsonl").read_text()
    assert '"anomaly_type":"colour"' in private_labels
    assert '"category":"category-0"' in private_crosswalk
    uninterrupted = tmp_path / "uninterrupted"
    complete = release.build_protected_role_resources(
        role="calibration",
        selection_root=paths["selection"],
        administrative_root=paths["administration"],
        label_root=paths["labels"],
        source_root=paths["source"],
        prerequisite_lock=paths["prerequisite"],
        output_root=uninterrupted,
    )
    assert complete["status"] == "complete"
    for relative in (
        "calibration-model-input/neutral_corpus_manifest.jsonl",
        "calibration-model-input/content_checksums.sha256",
        "calibration-model-input/package_record.json",
        "calibration-evaluator-only/crosswalk_private.jsonl",
        "calibration-evaluator-only/labels_private.jsonl",
        "calibration-evaluator-only/geometry_private.jsonl",
        "calibration-evaluator-only/evaluator_manifest.json",
    ):
        assert file_sha256(bounded / relative) == file_sha256(uninterrupted / relative)


def test_protected_resource_release_rejects_a_bare_prerequisite_file(
    tmp_path: Path, monkeypatch
) -> None:
    paths, contract = _fixture(tmp_path)
    monkeypatch.setitem(release.ROLE_CONTRACTS, "calibration", contract)
    try:
        release.build_protected_role_resources(
            role="calibration",
            selection_root=paths["selection"],
            administrative_root=paths["administration"],
            label_root=paths["labels"],
            source_root=paths["source"],
            prerequisite_lock=paths["prerequisite"],
            output_root=tmp_path / "rejected",
        )
    except (FileNotFoundError, ValueError):
        pass
    else:
        raise AssertionError("A bare prerequisite file must not authorize release")


def test_internal_release_binds_exact_policy_to_exact_method(
    tmp_path: Path, monkeypatch
) -> None:
    paths, calibration_contract = _fixture(tmp_path)
    roles = [
        {**row, "role": "internal_lock"}
        for row in (
            json.loads(line)
            for line in (paths["selection"] / "experiment_roles.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
    ]
    _write_jsonl(paths["selection"] / "experiment_roles.jsonl", roles)
    trusted = [
        {**row, "role": "internal_lock"}
        for row in (
            json.loads(line)
            for line in (paths["administration"] / "restricted_source_map.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
    ]
    _write_jsonl(paths["administration"] / "restricted_source_map.jsonl", trusted)
    (paths["labels"] / "internal_lock_labels.jsonl").write_bytes(
        (paths["labels"] / "calibration_labels.jsonl").read_bytes()
    )
    contract = CohortContract(
        cohort_id="fixture-internal-lock",
        release_id=calibration_contract.release_id,
        role="internal_lock",
        pilot_stage=None,
        membership_sha256=calibration_contract.membership_sha256,
        scene_count=calibration_contract.scene_count,
        views_per_scene=calibration_contract.views_per_scene,
        image_count=calibration_contract.image_count,
        asset_group_count=calibration_contract.asset_group_count,
    )
    monkeypatch.setitem(release.ROLE_CONTRACTS, "internal-lock", contract)
    method_digest = sha256(b"method-resource").hexdigest()
    calls: dict[str, object] = {}

    def load_method(path, *, application_contract_path, require_resource):
        calls["method"] = (path, application_contract_path, require_resource)
        return {
            "sha256": method_digest,
            "identity_configuration_sha256": sha256(b"identity").hexdigest(),
            "frozen_model_identity_sha256": sha256(b"pipeline").hexdigest(),
        }

    def load_policy(path, *, method_sha256, require_resource):
        calls["policy"] = (path, method_sha256, require_resource)
        return {
            "sha256": sha256(b"policy-resource").hexdigest(),
            "policy_configuration_sha256": sha256(b"policy-config").hexdigest(),
        }

    monkeypatch.setattr(release, "load_method_freeze", load_method)
    monkeypatch.setattr(release, "load_policy_freeze", load_policy)
    method_lock = tmp_path / "method" / "method_freeze.json"
    method_lock.parent.mkdir()
    method_lock.write_text("{}", encoding="utf-8")
    (method_lock.parent / "application_output_contract.json").write_text(
        "{}", encoding="utf-8"
    )
    (method_lock.parent / "artifact_manifest.json").write_text("{}", encoding="utf-8")
    policy_lock = tmp_path / "policy" / "review_policy_freeze.json"
    policy_lock.parent.mkdir()
    policy_lock.write_text("{}", encoding="utf-8")
    (policy_lock.parent / "artifact_manifest.json").write_text("{}", encoding="utf-8")

    result = release.build_protected_role_resources(
        role="internal-lock",
        selection_root=paths["selection"],
        administrative_root=paths["administration"],
        label_root=paths["labels"],
        source_root=paths["source"],
        prerequisite_lock=policy_lock,
        method_lock=method_lock,
        output_root=tmp_path / "internal-release",
    )

    assert result["status"] == "complete"
    assert calls["method"] == (
        method_lock.resolve(),
        method_lock.parent.resolve() / "application_output_contract.json",
        True,
    )
    assert calls["policy"] == (policy_lock.resolve(), method_digest, True)


def test_development_identity_evaluator_is_prediction_bound_and_bounded(
    tmp_path: Path, monkeypatch
) -> None:
    paths, _contract = _fixture(tmp_path)
    scene_ids = ["S001", "S002"]
    development_contract = CohortContract(
        cohort_id="fixture-development",
        release_id="toysad8k-train-2000-v2",
        role="development",
        pilot_stage="all_development",
        membership_sha256=membership_sha256(scene_ids),
        scene_count=2,
        views_per_scene=2,
        image_count=4,
        asset_group_count=2,
    )
    trusted = [
        {
            "opaque_scene_id": scene_id,
            "opaque_asset_group_id": f"A{index}",
            "role": "development",
            "source_scene_id": f"source-{index}",
        }
        for index, scene_id in enumerate(scene_ids)
    ]
    _write_jsonl(paths["administration"] / "restricted_source_map.jsonl", trusted)
    views = {}
    for scene_id in scene_ids:
        views[scene_id] = tuple(
            SimpleNamespace(
                view_id=f"V{view_index:02d}",
                objects=tuple(
                    SimpleNamespace(
                        object_id=f"V{view_index:02d}-P{object_index:02d}",
                        box_xyxy=(20 + object_index * 60, 30, 60 + object_index * 60, 80),
                    )
                    for object_index in range(3)
                ),
            )
            for view_index in range(2)
        )
    verified = SimpleNamespace(
        views_by_scene=views,
        input_lock_sha256s=(sha256(b"first").hexdigest(), sha256(b"second").hexdigest()),
        frozen_identity_sha256=sha256(b"pipeline").hexdigest(),
    )
    monkeypatch.setattr(release, "DEVELOPMENT_FULL_CONTRACT", development_contract)
    monkeypatch.setattr(release, "load_08e_predictions", lambda _roots: verified)
    output = tmp_path / "identity"
    first = release.build_development_identity_evaluator(
        prediction_roots=[tmp_path],
        administrative_root=paths["administration"],
        source_root=paths["source"],
        output_root=output,
        maximum_new_scenes=1,
    )
    assert first["status"] == "checkpointed"
    assert not (output / "development-identity-evaluator/evaluator_manifest.json").exists()
    second = release.build_development_identity_evaluator(
        prediction_roots=[tmp_path],
        administrative_root=paths["administration"],
        source_root=paths["source"],
        output_root=output,
        maximum_new_scenes=1,
    )
    assert second["status"] == "complete"
    rows = [
        json.loads(line)
        for line in (
            output / "development-identity-evaluator/evaluator_correspondence.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 12
    assert {row["reference_identity_id"] for row in rows} == {"O00", "O01", "O02"}


def test_primary_resource_notebooks_are_clean_cpu_release_stages() -> None:
    root = Path(__file__).resolve().parents[1]
    configurations = (
        (
            "development-freeze",
            "08e1-toysad8k-development-identity-evaluator-release.ipynb",
            "08e1",
            ("vpi-toysad-development-identity-evaluator",),
        ),
        (
            "protected-evaluation",
            "08f1-toysad8k-calibration-resource-release.ipynb",
            "08f1",
            (
                "vpi-toysad-calibration600-model-input",
                "vpi-toysad-calibration600-evaluator",
            ),
        ),
        (
            "protected-evaluation",
            "09b1-toysad8k-internal-lock-resource-release.ipynb",
            "09b1",
            (
                "vpi-toysad-internal-lock400-model-input",
                "vpi-toysad-internal-lock400-evaluator",
            ),
        ),
    )
    for folder, name, code, publication_names in configurations:
        notebook = nbformat.read(root / "notebooks" / folder / name, as_version=4)
        source = "\n".join(cell.source for cell in notebook.cells)
        assert "recovery" in source.lower()
        assert "ClearML" in source
        assert "exact full coverage" in source
        assert all(publication in source for publication in publication_names)
        assert all(not cell.get("outputs") for cell in notebook.cells if cell.cell_type == "code")
        assert [cell.id for cell in notebook.cells] == [
            sha256(f"{name}:{index}".encode()).hexdigest()[:8]
            for index in range(len(notebook.cells))
        ]
        metadata = json.loads(
            (root / "notebooks" / folder / f"kernel-metadata-{code}.json").read_text()
        )
        assert metadata["enable_gpu"] is False
        assert metadata["enable_tpu"] is False
        assert metadata["is_private"] is True
