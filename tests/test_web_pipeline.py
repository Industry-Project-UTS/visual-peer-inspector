"""CPU integration tests. No model weights, network, GPU or research data."""

from __future__ import annotations

import importlib.util
import json
import socket
import sys
import threading
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from visual_peer_inspector.models.visual_encoder import EncodedBatch
from visual_peer_inspector.web.backend import SubprocessBackend
from visual_peer_inspector.web.core import (
    Box,
    InspectionError,
    InspectionRequest,
    Session,
    load_views,
    validate_result,
)
from visual_peer_inspector.web.deployment import (
    WorkerSettings,
    load_review_policy,
    pipeline_config,
    pipeline_fingerprint,
)
from visual_peer_inspector.web.pipeline import ImageTransform, UploadPipeline
from visual_peer_inspector.web.presentation import discard_exports, export_report
from visual_peer_inspector.web.transport import (
    read_request,
    read_result,
    write_request,
    write_result,
)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def deny(*args, **kwargs):
        raise AssertionError("Model-free tests must never contact a network.")

    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setenv("VPI_ALLOW_MODEL_DOWNLOADS", "0")


@pytest.fixture
def request_images(tmp_path):
    paths = []
    for index, colour in enumerate(("#aaaaaa", "#bbbbbb")):
        path = tmp_path / f"{index}.png"
        Image.new("RGB", (256, 256), colour).save(path)
        paths.append(str(path))
    return InspectionRequest(load_views(paths))


class PixelModels:
    """Supplies stage outputs; the frozen decisions and upload code still execute."""

    boxes = ((20, 50, 75, 190), (92, 50, 147, 190), (164, 50, 219, 190))

    def __init__(self, *, count=3, empty_mask=False, zero_view=None):
        self.count = count
        self.empty_mask = empty_mask
        self.zero_view = zero_view
        self.proposal_calls = 0
        self.released = 0

    def propose(self, image, view_id):
        self.proposal_calls += 1
        return ([] if view_id == self.zero_view else self.boxes[: self.count]), []

    def segment(self, image, boxes):
        masks = np.zeros((len(boxes), 256, 256), dtype=bool)
        for index, (left, top, right, bottom) in enumerate(boxes):
            if not (self.empty_mask and index == 0):
                masks[index, top:bottom, left:right] = True
        return masks, np.full(len(boxes), 0.95)

    def encode(self, crops):
        vectors = np.array([(1, 0, 0), (0.98, 0.2, 0), (0, 0, 1)], dtype=np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        selected = [vectors[i % len(vectors)] for i in range(len(crops))]
        patches = tuple(np.tile(vector, (256, 1)) for vector in selected)
        return EncodedBatch(np.asarray(selected), patches, 16, 3)

    def release(self):
        self.released += 1


def test_real_cpu_pipeline_and_report_roundtrip(request_images, tmp_path):
    models = PixelModels()
    stages = []
    result = UploadPipeline(models).inspect(request_images, stages.append)
    validate_result(result, request_images)
    assert len(result.observations) == 6
    # This fixture's batch-local encoder changes its winner between views.
    assert len(result.ranking) == 2
    assert all(r.score_method == 'winner_cluster_consensus' for r in result.ranking)
    assert result.ranking[0].relative_score == result.ranking[1].relative_score
    assert result.selected_identity_id is None
    assert all(len(r.view_ids) == 1 for r in result.ranking)
    assert result.review_state == "review_required"
    assert result.policy_version == "calibration-selective-v1"
    assert stages == [
        "Finding objects",
        "Creating masks",
        "Extracting features",
        "Comparing peers",
        "Combining views",
    ]
    assert models.released >= 4
    assert all(
        o.mask.size == (256, 256) and o.heatmap.size == (256, 256)
        for o in result.observations
    )
    write_request(tmp_path, request_images)
    restored = read_request(tmp_path)
    assert [v.digest for v in restored.views] == [
        v.digest for v in request_images.views
    ]
    write_result(tmp_path, result)
    output = read_result(tmp_path, request_images)
    assert output.ranking == result.ranking
    assert output.provenance["pipeline_sha256"] == pipeline_fingerprint()
    session = Session(request_images.views, output)
    try:
        html, report = export_report(session, "Contract verification")
        assert Path(html).is_file()
        exported = json.loads(Path(report).read_text())
        assert exported["provenance"] == result.provenance
        assert exported["provenance"]["review_policy"]["sha256"] == result.provenance["review_policy"]["sha256"]
        assert exported["mode"] == "automatic" and exported["correspondence"] == "Predicted matches"
    finally:
        discard_exports(session)


def test_assisted_boxes_keep_user_identities(request_images):
    corrections = tuple(
        Box(v.view_id, f"object_{i}", f"my_identity_{i}", box)
        for v in request_images.views
        for i, box in enumerate(PixelModels.boxes)
    )
    models = PixelModels()
    result = UploadPipeline(models).inspect(
        replace(request_images, corrections=corrections), lambda _: None
    )
    assert models.proposal_calls == 0
    assert result.mode == "assisted"
    assert {r.identity_id for r in result.ranking} == {
        f"my_identity_{i}" for i in range(3)
    }


@pytest.mark.parametrize("view_count", [2, 3, 5])
def test_frozen_policy_and_assisted_reranking_across_supported_view_counts(tmp_path, view_count):
    paths = []
    for index in range(view_count):
        path = tmp_path / f"view_{index}.png"
        Image.new("RGB", (256, 256), (80 + index * 15, 100, 120)).save(path)
        paths.append(str(path))
    request = InspectionRequest(load_views(paths))
    models = PixelModels()
    pipeline = UploadPipeline(models, cache_root=tmp_path / "cache")
    automatic = pipeline.inspect(request, lambda _: None)
    assert automatic.review_state in {"review_required", "insufficient_evidence"}
    assert automatic.provenance["review_signals"]["view_count"] == view_count
    corrections = tuple(replace(o.box, identity_id=o.box.object_id) for o in automatic.observations)
    assisted = pipeline.inspect(replace(request, mode="assisted", corrections=corrections,
                                        cache_token=automatic.cache_token), lambda _: None)
    assert models.proposal_calls == view_count
    assert assisted.provenance["cache"] == {"reused_candidates": 3 * view_count, "encoded_candidates": 0}
    assert assisted.policy_version == "calibration-selective-v1"
    assert assisted.provenance["review_signals"]["view_count"] == view_count
    assert assisted.provenance["correspondence_source"] == "user_confirmed"
    validate_result(assisted, replace(request, mode="assisted", corrections=corrections))


@pytest.mark.parametrize("count", [0, 1, 2])
def test_too_few_peers_never_invents_scores(request_images, count):
    result = UploadPipeline(PixelModels(count=count)).inspect(
        request_images, lambda _: None
    )
    assert result.review_state == "insufficient_evidence"
    assert not result.ranking
    assert all(o.p1_score is None and o.heatmap is None for o in result.observations)


def test_empty_mask_is_preserved_for_correction(request_images):
    result = UploadPipeline(PixelModels(empty_mask=True)).inspect(
        request_images, lambda _: None
    )
    assert len(result.observations) == 6
    assert sum(o.mask.getbbox() is None for o in result.observations) == 2
    assert result.review_state == "insufficient_evidence"


def test_one_missing_view_cannot_claim_multiview_support(request_images):
    result = UploadPipeline(
        PixelModels(zero_view=request_images.views[0].view_id)
    ).inspect(request_images, lambda _: None)
    assert result.review_state == "review_required"
    assert all(len(r.view_ids) == 1 for r in result.ranking)


@pytest.mark.parametrize("size", [(1000, 500), (500, 1000), (640, 480), (64, 1024)])
def test_letterbox_roundtrip_and_padding(size):
    image, transform = ImageTransform.prepare(Image.new("RGB", size, "red"))
    assert image.size == (256, 256)
    box = (0, 0, *size)
    assert transform.to_original_box(transform.to_model_box(box)) == box
    mask = transform.trim_mask(np.ones((256, 256), dtype=bool))
    restored = transform.restore(
        Image.fromarray(mask.astype(np.uint8) * 255), nearest=True
    )
    assert restored.size == size and restored.getextrema() == (255, 255)
    if transform.offset[0] > 0:
        assert transform.to_original_box((0, 0, transform.offset[0], 256)) is None


def test_result_mapping_on_nonsquare_images(request_images):
    views = tuple(
        replace(v, image=v.image.resize((512, 320))) for v in request_images.views
    )
    request = replace(request_images, views=views)
    result = UploadPipeline(PixelModels()).inspect(request, lambda _: None)
    validate_result(result, request)
    assert all(o.mask.size == (512, 320) for o in result.observations)
    assert all(
        o.heatmap is None or o.heatmap.size == (512, 320) for o in result.observations
    )


def test_transport_rejects_path_escape(request_images, tmp_path):
    write_request(tmp_path, request_images)
    path = tmp_path / "request.json"
    record = json.loads(path.read_text())
    record["views"][0]["image"] = "../outside.png"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Unsafe"):
        read_request(tmp_path)


def test_review_policy_requires_exact_sealed_artifact(tmp_path):
    from visual_peer_inspector.web.frozen_policy import POLICY_PATH, POLICY_SHA256

    path = tmp_path / "policy.json"
    path.write_bytes(POLICY_PATH.read_bytes())
    loaded, version = load_review_policy(str(path))
    assert loaded.source_sha256 == POLICY_SHA256 and version == "calibration-selective-v1"
    value = json.loads(path.read_text())
    value["version"] = "altered"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="integrity hash"):
        load_review_policy(str(path))


def test_subprocess_complete_and_failure(request_images, monkeypatch):
    backend = SubprocessBackend(
        WorkerSettings(sys.executable, timeout_seconds=10),
        worker_module="tests.web_worker_stub",
    )
    result = backend.inspect(request_images, lambda _: None)
    assert result.review_state == "insufficient_evidence"
    monkeypatch.setenv("VPI_TEST_WORKER_MODE", "fail")
    with pytest.raises(InspectionError, match="CUDA GPU"):
        backend.inspect(request_images, lambda _: None)


def test_timeout_kills_worker_and_removes_job(request_images, tmp_path, monkeypatch):
    import tempfile

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setenv("VPI_TEST_WORKER_MODE", "hang")
    backend = SubprocessBackend(
        WorkerSettings(sys.executable, timeout_seconds=1),
        worker_module="tests.web_worker_stub",
    )
    with pytest.raises(InspectionError, match="timed out"):
        backend.inspect(request_images, lambda _: None)
    assert not list(tmp_path.glob("vpi-job-*"))
    assert not backend._active


def test_cancellation_is_scoped_and_stops_worker(request_images, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    monkeypatch.setenv("VPI_TEST_WORKER_MODE", "hang")
    backend = SubprocessBackend(
        WorkerSettings(sys.executable, timeout_seconds=10),
        worker_module="tests.web_worker_stub",
    )
    started = threading.Event()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            backend.inspect, request_images, lambda _: started.set()
        )
        assert started.wait(5)
        assert not backend.cancel("another-session")
        assert backend.cancel(request_images.request_id)
        with pytest.raises(InspectionError, match="cancelled"):
            future.result(timeout=5)
    assert not backend._active


def test_package_contains_only_code_and_pinned_runtime(tmp_path):
    path = Path(__file__).resolve().parents[1] / "apps/web/package_space.py"
    spec = importlib.util.spec_from_file_location("package_space", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path / "space"
    summary = module.package(output)
    assert not summary["uploaded"] and not summary["model_weights_included"]
    assert (output / "Dockerfile").is_file()
    assert (output / "src/visual_peer_inspector/web/pipeline_config.json").is_file()
    assert (output / "src/visual_peer_inspector/web/frozen_review_policy.json").is_file()
    assert not (output / "tests").exists()
    assert not (output / "reports").exists()
    assert not list(output.rglob("*.pt")) and not list(output.rglob("*.safetensors"))
    assert "VPI_ALLOW_MODEL_DOWNLOADS=0" in (output / "Dockerfile").read_text()
    with pytest.raises(ValueError, match="empty"):
        module.package(output)


def test_checked_in_model_pins_match_research():
    from visual_peer_inspector.models.locateanything import (
        DEFAULT_LOCATEANYTHING_REVISION,
    )
    from visual_peer_inspector.models.sam2 import (
        SAM2_CODE_REVISION,
        SAM2_MODEL_REVISION,
    )

    config = pipeline_config()
    assert config["locateanything"]["revision"] == DEFAULT_LOCATEANYTHING_REVISION
    assert config["sam2"]["revision"] == SAM2_MODEL_REVISION
    assert config["sam2"]["code_revision"] == SAM2_CODE_REVISION
    assert config["patch_validity_minimum_fraction"] == 0.5


def test_stage_failure_releases_model_resources(request_images):
    class BrokenModels(PixelModels):
        def segment(self, image, boxes):
            raise RuntimeError("stage failure")

    models = BrokenModels()
    with pytest.raises(RuntimeError, match="stage failure"):
        UploadPipeline(models).inspect(request_images, lambda _: None)
    assert models.released >= 2


def test_runtime_constructor_does_not_import_gpu_libraries(monkeypatch):
    from visual_peer_inspector.web.model_runtime import GPUModels

    before = {name for name in ("torch", "transformers", "sam2") if name in sys.modules}
    models = GPUModels(allow_downloads=False)
    assert models.resource is None
    assert {
        name for name in ("torch", "transformers", "sam2") if name in sys.modules
    } == before


def test_offline_snapshot_never_enables_downloads(monkeypatch):
    import huggingface_hub
    from huggingface_hub.errors import LocalEntryNotFoundError

    from visual_peer_inspector.web.model_runtime import GPUModels, RuntimeUnavailable

    calls = []

    def missing(**kwargs):
        calls.append(kwargs)
        raise LocalEntryNotFoundError("test-only missing cache")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", missing)
    with pytest.raises(RuntimeUnavailable, match="models_not_cached"):
        GPUModels(allow_downloads=False)._snapshot(pipeline_config()["encoder"])
    assert calls[0]["local_files_only"] is True
    assert calls[0]["revision"] == pipeline_config()["encoder"]["revision"]


def test_cuda_guard_runs_before_checkpoint_access(monkeypatch):
    from visual_peer_inspector.web.model_runtime import GPUModels, RuntimeUnavailable

    models = GPUModels()

    def unavailable():
        raise RuntimeUnavailable("cuda_unavailable")

    def forbidden(*args):
        raise AssertionError("A CPU-only request must not reach checkpoint access")

    monkeypatch.setattr(models, "_torch", unavailable)
    monkeypatch.setattr(models, "_snapshot", forbidden)
    with pytest.raises(RuntimeUnavailable, match="cuda_unavailable"):
        models._load("locate")


def test_locate_adapter_reuses_parser_policy_and_rejects_truncation(monkeypatch):
    from types import SimpleNamespace

    from visual_peer_inspector.web.model_runtime import GPUModels, RuntimeUnavailable

    models = GPUModels()
    worker = SimpleNamespace(
        tokenizer=SimpleNamespace(encode=lambda *a, **kw: []),
        predict=lambda *a, **kw: {
            "answer": "<box><100><100><300><500></box>",
            "stats": {"truncated": False},
        },
    )
    monkeypatch.setattr(models, "_load", lambda _: worker)
    boxes, notes = models.propose(Image.new("RGB", (256, 256)), "view")
    assert boxes == [[26, 26, 77, 128]] and not notes
    worker.predict = lambda *a, **kw: {"answer": "", "stats": {"truncated": True}}
    with pytest.raises(RuntimeUnavailable, match="proposal_truncated"):
        models.propose(Image.new("RGB", (256, 256)), "view")


def test_subprocess_cpu_pipeline_runs_across_serialized_boundary(
    request_images, monkeypatch
):
    monkeypatch.setenv("VPI_TEST_WORKER_MODE", "pipeline")
    backend = SubprocessBackend(
        WorkerSettings(sys.executable, timeout_seconds=15),
        worker_module="tests.web_worker_stub",
    )
    result = backend.inspect(request_images, lambda _: None)
    assert result.provenance["test_only"] is True
    assert result.ranking[0].score_method == 'winner_cluster_consensus'
    assert result.selected_identity_id is None
    assert result.provenance["review_signals"]["ranked_count"] > 0
