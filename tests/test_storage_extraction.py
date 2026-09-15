"""Checkpoint and extraction contracts with synthetic images and a fake encoder.

These tests exercise the real mask/crop/projection code, not model inference.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from visual_peer_inspector.models.sam2 import encode_binary_mask
from visual_peer_inspector.models.visual_encoder import EncodedBatch
from visual_peer_inspector.representation.crops import prepare_masked_crop, project_crop_heatmap
from visual_peer_inspector.scoring.forward_patch import patch_policy
from vpi_object_bank import extraction, storage


class ContractEncoder:
    """Return identifiable finite tensors while recording the actual input crops."""

    def __init__(self, *, mode="valid"):
        self.calls = []
        self.mode = mode

    def encode(self, images):
        self.calls.append([np.array(image, copy=True) for image in images])
        count = len(images)
        patches = tuple(
            np.repeat(np.linspace(0.1, 1.0, 256, dtype=np.float32)[:, None], 384, axis=1)
            for _ in images
        )
        globals_ = np.ones((count, 384), dtype=np.float32)
        if self.mode == "short_batch":
            patches = patches[:-1]
        elif self.mode == "wrong_patch_shape":
            patches = tuple(matrix[:, :383] for matrix in patches)
        elif self.mode == "wrong_global_shape":
            globals_ = globals_[:, :383]
        elif self.mode == "nonfinite":
            patches[0][0, 0] = np.nan
        return EncodedBatch(globals_, patches, 16, 384)


@pytest.fixture
def package(tmp_path):
    model = tmp_path / "model"
    prepared = tmp_path / "prepared"
    output = tmp_path / "features"
    model.mkdir()
    prepared.mkdir()
    image = np.zeros((40, 60, 3), dtype=np.uint8)
    image[..., 0] = np.arange(60, dtype=np.uint8)
    image[..., 1] = np.arange(40, dtype=np.uint8)[:, None]
    image[..., 2] = 120
    Image.fromarray(image).save(model / "rgb.png")
    masks = []
    for ys, xs in [(slice(0, 15), slice(0, 12)), (slice(12, 29), slice(23, 36)), (slice(24, 40), slice(45, 60))]:
        mask = np.zeros((40, 60), dtype=bool)
        mask[ys, xs] = True
        masks.append(mask)
    objects = []
    for index, mask in enumerate(masks):
        encoded = encode_binary_mask(mask)
        ys, xs = np.nonzero(mask)
        objects.append({
            "candidate_id": f"C{index:02d}",
            "mask": encoded,
            "mask_sha256": storage.digest(encoded),
            "box": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
        })
    rows = [{
        "scene": "S0001", "route": route, "view": view,
        "rgb": "rgb.png", "rgb_sha256": storage.digest_file(model / "rgb.png"),
        "objects": objects,
    } for route in ["supplied", "automatic"] for view in ["V00", "V01"]]
    storage.atomic_json(model / "scenes" / "S0001.json", rows)
    storage.seal(model, [model / "rgb.png", model / "scenes" / "S0001.json"])
    manifest = {
        "model_input_lock_sha256": storage.digest_file(model / "artifact_lock.json"),
        "scenes": ["S0001"], "view_sets": {"two": ["V00", "V01"]},
    }
    storage.atomic_json(prepared / "input_manifest.json", manifest)
    storage.seal(prepared, [prepared / "input_manifest.json"])
    return SimpleNamespace(model=model, prepared=prepared, output=output, image=image,
                           masks=masks, rows=rows, source=Path(extraction.__file__).resolve().parents[2])


def run(package, encoder, **kwargs):
    return extraction.extract_features(
        package.model, package.prepared, package.output, package.source,
        device="cpu", encoder_factory=lambda: encoder, producer="contract-encoder", **kwargs,
    )


def unit(package, route="supplied", view="V00"):
    return package.output / "views" / "S0001" / f"{route}-{view}"


def identity_map(views=("V00", "V01")):
    return {(view, f"C{index:02d}"): f"O{index:02d}" for view in views for index in range(3)}


def file_snapshot(path):
    return {str(item.relative_to(path)): (item.read_bytes(), item.stat().st_mtime_ns)
            for item in path.rglob("*") if item.is_file()}


def reseal_unit(path, metadata_update=None, binding_hash=None):
    lock = storage.read_json(path / "artifact_lock.json")
    if metadata_update is not None:
        metadata = storage.read_json(path / "observations.json")
        metadata.update(metadata_update)
        storage.atomic_json(path / "observations.json", metadata)
    if binding_hash is not None:
        lock["metadata"]["binding_sha256"] = binding_hash
    storage.seal(path, [p for p in path.rglob("*") if p.is_file() and p.name != "artifact_lock.json"], metadata=lock["metadata"])


def test_atomic_interruption_preserves_completed_unit_and_recovers_partial(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    with storage.atomic_unit(first, "binding") as temporary:
        (temporary / "result.bin").write_bytes(b"completed evidence")
    original = file_snapshot(first)
    with pytest.raises(KeyboardInterrupt):
        with storage.atomic_unit(second, "binding") as temporary:
            (temporary / "result.bin").write_bytes(b"interrupted evidence")
            raise KeyboardInterrupt()
    assert not storage.verified_unit(second, "binding")
    assert (tmp_path / "second.partial" / "result.bin").read_bytes() == b"interrupted evidence"
    with storage.atomic_unit(second, "binding") as temporary:
        (temporary / "result.bin").write_bytes(b"new complete evidence")
    assert storage.verified_unit(first, "binding")
    assert storage.verified_unit(second, "binding")
    assert file_snapshot(first) == original
    archived = list(tmp_path.glob("second.interrupted-*"))
    assert len(archived) == 1
    assert (archived[0] / "result.bin").read_bytes() == b"interrupted evidence"


def test_atomic_unit_never_overwrites_completed_checkpoint(tmp_path):
    target = tmp_path / "unit"
    with storage.atomic_unit(target, "binding") as temporary:
        (temporary / "file").write_bytes(b"original")
    before = file_snapshot(target)
    with pytest.raises(FileExistsError):
        with storage.atomic_unit(target, "binding"):
            pytest.fail("Completed checkpoint was opened for writing")
    assert file_snapshot(target) == before


@pytest.mark.parametrize("damage", ["changed", "missing", "extra", "binding"])
def test_verified_checkpoint_fails_closed_on_corruption_or_foreign_binding(tmp_path, damage):
    target = tmp_path / "unit"
    with storage.atomic_unit(target, "binding") as temporary:
        (temporary / "file").write_bytes(b"original")
    if damage == "changed":
        (target / "file").write_bytes(b"changed")
    elif damage == "missing":
        (target / "file").unlink()
    elif damage == "extra":
        (target / "unexpected").write_bytes(b"extra")
    with pytest.raises((ValueError, FileNotFoundError)):
        storage.verified_unit(target, "other" if damage == "binding" else "binding")


def test_changed_run_binding_or_unbound_nonempty_root_is_rejected(tmp_path):
    root = tmp_path / "run"
    expected = {"source": "v1", "configuration": {"views": ["V00"]}}
    first = storage.bind(root, expected)
    assert storage.bind(root, expected) == first
    with pytest.raises(ValueError, match="different"):
        storage.bind(root, {"source": "v1", "configuration": {"views": ["V01"]}})
    unbound = tmp_path / "unbound"
    unbound.mkdir()
    (unbound / "private.bin").write_bytes(b"keep")
    with pytest.raises(ValueError, match="Nonempty"):
        storage.bind(unbound, expected)


def test_terminal_failure_is_saved_without_exception_credential_text(tmp_path):
    with pytest.raises(RuntimeError):
        with storage.terminal_status(tmp_path, "test-stage"):
            raise RuntimeError("synthetic https://test-user:test-password@example.invalid/private")
    status = storage.read_json(tmp_path / "run_status.json")
    assert status["status"] == "failed"
    assert status["error_type"] == "RuntimeError"
    assert len(list((tmp_path / "failures").glob("*.json"))) == 1
    assert "test-password" not in "".join(path.read_text() for path in tmp_path.rglob("*.json"))


def test_extraction_resume_reuses_completed_unit_without_reencoding(package):
    encoder = ContractEncoder()
    first = run(package, encoder, max_units=1)
    assert first["completed_units"] == 1
    assert first["extracted_units"] == 1
    assert not first["sealed"]
    completed = file_snapshot(unit(package))
    assert len(encoder.calls) == 1
    second = run(package, encoder)
    assert second["completed_units"] == 4
    assert second["extracted_units"] == 3
    assert second["sealed"]
    assert len(encoder.calls) == 4
    assert file_snapshot(unit(package)) == completed
    storage.verify_seal(package.output, name="feature_lock.json")


def test_completed_extraction_resume_never_loads_encoder(package):
    run(package, ContractEncoder())
    original = file_snapshot(unit(package))

    def forbidden_factory():
        pytest.fail("Verified completed run attempted to load encoder")

    result = extraction.extract_features(package.model, package.prepared, package.output, package.source,
                                         device="cpu", encoder_factory=forbidden_factory, producer="contract-encoder")
    assert result["extracted_units"] == 0
    assert result["completed_units"] == 4
    assert file_snapshot(unit(package)) == original


def test_corrupt_feature_resume_stops_before_encoder_and_preserves_damaged_bytes(package):
    run(package, ContractEncoder(), max_units=1)
    damaged = unit(package) / "features.npz"
    damaged.write_bytes(b"corrupt checkpoint must be preserved")
    encoder = ContractEncoder()
    with pytest.raises(ValueError, match="Checksum"):
        run(package, encoder)
    assert encoder.calls == []
    assert damaged.read_bytes() == b"corrupt checkpoint must be preserved"


def test_extraction_changed_runtime_binding_stops_before_encoder(package):
    run(package, ContractEncoder(), max_units=1)
    encoder = ContractEncoder()
    with pytest.raises(ValueError, match="different"):
        extraction.extract_features(package.model, package.prepared, package.output, package.source,
                                    device="mps", encoder_factory=lambda: encoder, producer="contract-encoder")
    assert encoder.calls == []


def test_partial_extraction_status_does_not_claim_complete_scientific_stage(package):
    result = run(package, ContractEncoder(), max_units=1)
    status = storage.read_json(package.output / "run_status.json")
    assert not result["sealed"]
    assert status["status"] in {"partial", "pilot_complete", "pilot_completed"}


def test_missing_required_model_input_binding_is_rejected(package):
    manifest = storage.read_json(package.prepared / "input_manifest.json")
    manifest.pop("model_input_lock_sha256")
    storage.atomic_json(package.prepared / "input_manifest.json", manifest)
    storage.seal(package.prepared, [package.prepared / "input_manifest.json"])
    encoder = ContractEncoder()
    with pytest.raises(ValueError, match="(?i)(binding|hash|lock|manifest|registry|packages)"):
        run(package, encoder)
    assert encoder.calls == []


@pytest.mark.parametrize("mode", ["short_batch", "wrong_patch_shape", "wrong_global_shape", "nonfinite"])
def test_malformed_encoder_results_cannot_seal_an_incomplete_unit(package, mode):
    with pytest.raises(ValueError):
        run(package, ContractEncoder(mode=mode), max_units=1)
    assert not unit(package).exists()
    assert not (package.output / "feature_lock.json").exists()
    assert storage.read_json(package.output / "run_status.json")["status"] == "failed"


def test_no_objects_seals_empty_units_without_encoder_initialization(package):
    for row in package.rows:
        row["objects"] = []
    storage.atomic_json(package.model / "scenes" / "S0001.json", package.rows)
    storage.seal(package.model, [package.model / "rgb.png", package.model / "scenes" / "S0001.json"])
    manifest = storage.read_json(package.prepared / "input_manifest.json")
    manifest["model_input_lock_sha256"] = storage.digest_file(package.model / "artifact_lock.json")
    storage.atomic_json(package.prepared / "input_manifest.json", manifest)
    storage.seal(package.prepared, [package.prepared / "input_manifest.json"])

    def forbidden_factory():
        pytest.fail("Empty scene attempted neural-model initialization")

    result = extraction.extract_features(package.model, package.prepared, package.output, package.source,
                                         device="cpu", encoder_factory=forbidden_factory, producer="contract-encoder")
    assert result["completed_units"] == 4
    rows, geometry = extraction.load_observations(package.output, "S0001", "supplied", ["V00"], {})
    assert rows == []
    assert geometry == {}


def test_loader_uses_only_selected_view_even_when_cache_has_extra_views(package):
    run(package, ContractEncoder())
    rows, geometry = extraction.load_observations(package.output, "S0001", "supplied", ["V00"], identity_map(["V00"]))
    assert len(rows) == 3
    assert {row["view_id"] for row in rows} == {"V00"}
    assert set(geometry) == set(identity_map(["V00"]))


@pytest.mark.parametrize("damage", ["extra_file", "foreign_binding", "wrong_scene", "wrong_view", "wrong_route"])
def test_loader_rejects_foreign_or_inconsistent_sealed_units(package, damage):
    run(package, ContractEncoder())
    target = unit(package)
    if damage == "extra_file":
        (target / "unlocked.npy").write_bytes(b"unexpected")
    elif damage == "foreign_binding":
        reseal_unit(target, binding_hash="foreign-run")
    else:
        field, replacement = {"wrong_scene": ("scene", "S0002"), "wrong_view": ("view", "V19"), "wrong_route": ("route", "automatic")}[damage]
        reseal_unit(target, metadata_update={field: replacement})
    with pytest.raises(ValueError):
        extraction.load_observations(package.output, "S0001", "supplied", ["V00"], identity_map(["V00"]))


def test_loader_rejects_duplicate_selected_views(package):
    run(package, ContractEncoder())
    with pytest.raises(ValueError):
        extraction.load_observations(package.output, "S0001", "supplied", ["V00", "V00"], identity_map(["V00"]))


@pytest.mark.parametrize("scene,views", [("S9999", ["V00"]), ("S0001", ["V00", "V19"])])
def test_loader_rejects_scene_or_view_outside_the_feature_binding(package, scene, views):
    run(package, ContractEncoder())
    with pytest.raises(ValueError, match="outside feature binding"):
        extraction.load_observations(package.output, scene, "supplied", views, identity_map(views))


def test_real_crop_masks_and_projected_heatmaps_preserve_source_geometry(package):
    encoder = ContractEncoder()
    run(package, encoder, max_units=1)
    metadata = storage.read_json(unit(package) / "observations.json")
    loaded, geometry = extraction.load_observations(package.output, "S0001", "supplied", ["V00"], identity_map(["V00"]))
    assert metadata["image_shape"] == [40, 60, 3]
    for index, source_mask in enumerate(package.masks):
        crop = prepare_masked_crop(package.image, source_mask)
        valid, fallback = patch_policy(crop, 16, 0.70, fallback="all")
        stored = geometry[("V00", f"C{index:02d}")]
        assert stored["source_box_xyxy"] == list(crop.source_box_xyxy)
        assert stored["padded_box_xyxy"] == list(crop.padded_box_xyxy)
        assert stored["crop_sha256"] == crop.sha256
        assert stored["foreground_patch_count"] == int(valid.sum())
        assert stored["fallback_all"] == fallback
        np.testing.assert_array_equal(encoder.calls[0][index], crop.rgb)
        assert loaded[index]["patches"].shape == (int(valid.sum()), 384)
        heatmap = np.tile(np.linspace(0.0, 1.0, 16), (16, 1))
        projected = project_crop_heatmap(heatmap, crop, source_mask)
        assert projected.shape == source_mask.shape
        assert np.isnan(projected[~source_mask]).all()
        assert np.isfinite(projected[source_mask]).all()
        ys, xs = np.nonzero(source_mask)
        middle_y = int((ys.min() + ys.max()) // 2)
        assert projected[middle_y, int(xs.min()) + 2] < projected[middle_y, int(xs.max()) - 2]
    assert metadata["objects"][0]["padded_box_xyxy"][0] < 0
    assert metadata["objects"][0]["padded_box_xyxy"][1] < 0
