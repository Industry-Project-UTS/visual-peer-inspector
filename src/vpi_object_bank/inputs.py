"""Prepare an explicitly assisted identity input without opening anomaly labels.

The trusted geometry and correspondence sources contain opaque identities, not
anomaly answers. The exported contract whitelists only selected observations.
Images, masks and cached scores remain in their separately sealed input roots.
"""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np

from visual_peer_inspector.workflows.comparison_io import (
    atomic_json, canonical, digest_file, read_json, seal, verify_seal,
)
from visual_peer_inspector.workflows.scoring_comparison import (
    METHODS, PROTOCOL, preflight as preflight_model,
)

SCHEMA = "vpi-object-bank-neutral-input/v1"
_SUPPLIED_FIELDS = {
    "box_xyxy", "foreground_pixels", "mask_path", "mask_sha256",
    "opaque_object_id", "opaque_scene_id", "schema_version", "view_id", "visible",
}
_AUTOMATIC_FIELDS = {
    "object_id", "opaque_asset_group_id", "opaque_scene_id",
    "reference_identity_id", "schema_version", "view_id",
}
_SCORE_FIELDS = {
    "feature_cache_hit", "objects", "patch_fallbacks", "route", "scene",
    "seconds", "status", "view",
}
_OBJECT_FIELDS = {"candidate_id", "embedding", "geometry", "heatmaps", "scores"}


def _opaque(value: Any, pattern: str, kind: str) -> str:
    if not isinstance(value, str) or re.fullmatch(pattern, value) is None:
        raise ValueError(f"Invalid opaque {kind}")
    return value


def _candidate_id(scene: str, view: str, route: str, original: str) -> str:
    return "C" + sha256(f"{scene}/{view}/{route}/{original}".encode()).hexdigest()[:16]


def _neutral_id(scene: str, original: str) -> str:
    # Preserve identity across mask routes without exposing source ordering.
    return "N" + sha256(f"{SCHEMA}/{scene}/{original}".encode()).hexdigest()[:24]


def _source_correspondence(path: Path, route: str) -> tuple[dict, dict]:
    expected = _SUPPLIED_FIELDS if route == "supplied" else _AUTOMATIC_FIELDS
    mapping: dict[tuple[str, str, str], str | None] = {}
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or set(row) != expected:
                raise ValueError(f"Unexpected {route} source schema on line {line_number}")
            source_version = 1 if route == "supplied" else "visual-peer-identity-evaluator/v1"
            if row["schema_version"] != source_version:
                raise ValueError("Unsupported neutral source schema version")
            scene = _opaque(row["opaque_scene_id"], r"S\d{6}", "scene")
            view = _opaque(row["view_id"], r"V\d{2}", "view")
            original = row["opaque_object_id"] if route == "supplied" else row["object_id"]
            original = _opaque(original, r"[A-Za-z0-9_-]+", "candidate source")
            reference = original if route == "supplied" else row["reference_identity_id"]
            if reference is not None:
                reference = _opaque(reference, r"O\d{2,}", "reference identity")
            candidate = _candidate_id(scene, view, route, original)
            key = (scene, view, candidate)
            if key in mapping:
                raise ValueError("Duplicate or conflicting source correspondence")
            mapping[key] = reference

    source_digest = digest_file(path)
    receipt_name, hash_key = (
        ("package_record.json", "reference_manifest_sha256")
        if route == "supplied"
        else ("evaluator_manifest.json", "correspondence_sha256")
    )
    receipt_path = path.parent / receipt_name
    receipt_digest = None
    if receipt_path.is_file():
        receipt = read_json(receipt_path)
        if receipt.get(hash_key) != source_digest:
            raise ValueError("Neutral source differs from its provenance receipt")
        receipt_digest = digest_file(receipt_path)
    return mapping, {"sha256": source_digest, "receipt_sha256": receipt_digest}


def _view_sets(value: Mapping[str, Sequence[str]], available: set[str]) -> dict[str, list[str]]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("At least one named selected-view schedule is required")
    result = {}
    for name, views in sorted(value.items()):
        if not isinstance(name, str) or re.fullmatch(r"[A-Za-z0-9_-]+", name) is None:
            raise ValueError("Invalid view-set name")
        if isinstance(views, (str, bytes)) or not isinstance(views, Sequence):
            raise ValueError("A view schedule must be a sequence of view IDs")
        chosen = list(views)
        if len(chosen) not in {1, 2, 3, 5} or len(set(chosen)) != len(chosen):
            raise ValueError("View schedules require 1, 2, 3 or 5 distinct views")
        for view in chosen:
            _opaque(view, r"V\d{2}", "view")
        if not set(chosen) <= available:
            raise ValueError("Selected view is absent from the sealed input")
        result[name] = chosen
    return result


def _score_rows(path: Path, scene: str, model_rows: list[dict]) -> dict:
    records = read_json(path)
    if not isinstance(records, list):
        raise ValueError("Scoring scene must be a list of route/view records")
    model = {(r["route"], r["view"]): r for r in model_rows}
    rows = {}
    for row in records:
        if not isinstance(row, dict) or set(row) != _SCORE_FIELDS or row["scene"] != scene:
            raise ValueError("Unexpected scoring record schema")
        key = (row["route"], row["view"])
        if key not in model or key in rows:
            raise ValueError("Missing, unexpected or duplicated scoring route/view")
        if row["status"] not in {"scored", "insufficient_peers"}:
            raise ValueError("Unexpected scoring status")
        if not isinstance(row["objects"], list):
            raise ValueError("Scoring objects must be a list")
        seen = set()
        for obj in row["objects"]:
            if not isinstance(obj, dict) or set(obj) != _OBJECT_FIELDS:
                raise ValueError("Unexpected scoring object schema")
            candidate = _opaque(obj["candidate_id"], r"C[0-9a-f]{16}", "candidate")
            if candidate in seen:
                raise ValueError("Duplicate scoring candidate")
            seen.add(candidate)
            embedding = np.asarray(obj["embedding"], dtype=float)
            geometry = np.asarray(obj["geometry"], dtype=float)
            if embedding.shape != (384,) or not np.isfinite(embedding).all() or np.linalg.norm(embedding) <= 1e-12:
                raise ValueError("Candidate requires a finite nonzero 384-dimensional embedding")
            if geometry.shape != (7,) or not np.isfinite(geometry).all():
                raise ValueError("Candidate geometry schema differs")
            if not isinstance(obj["scores"], dict) or set(obj["scores"]) != set(METHODS):
                raise ValueError("Cached score methods differ from the frozen protocol")
            if any(v is not None and (isinstance(v, bool) or not isinstance(v, (float, int)) or not np.isfinite(v)) for v in obj["scores"].values()):
                raise ValueError("Cached scores must be finite or explicitly missing")
            if not isinstance(obj["heatmaps"], dict) or not set(obj["heatmaps"]) <= set(METHODS):
                raise ValueError("Unexpected heatmap schema")
        if seen != {o["candidate_id"] for o in model[key]["objects"]}:
            raise ValueError("Cached candidates differ from the sealed model input")
        rows[key] = row
    if set(rows) != set(model):
        raise ValueError("Scoring scene is missing a model route/view")
    return rows


def prepare_inputs(
    model_root: Path,
    scoring_root: Path,
    neutral_supplied_jsonl: Path,
    neutral_automatic_jsonl: Path,
    output_root: Path,
    *,
    view_sets: Mapping[str, Sequence[str]],
    scenes: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Seal selected neutral correspondence; never read evaluator scene labels.

    Explicit null correspondences are retained. Missing source rows, ambiguous
    same-view identities, incompatible cached inputs and unexpected fields fail
    before output creation. Existing output packages are never overwritten.
    """
    model_root, scoring_root, output_root = map(Path, (model_root, scoring_root, output_root))
    if output_root.exists():
        raise FileExistsError("Use a new neutral-input package directory")
    preflight_model(model_root)
    prediction_lock = verify_seal(scoring_root, name="prediction_lock.json")
    verify_seal(scoring_root, name="feature_lock.json")
    binding = read_json(scoring_root / "run_binding.json")
    model_digest = digest_file(model_root / "artifact_lock.json")
    if binding.get("input_lock_sha256") != model_digest:
        raise ValueError("Scoring predictions belong to a different model input")
    if binding.get("protocol") != json.loads(json.dumps(PROTOCOL)):
        raise ValueError("Cached scoring protocol differs from the frozen implementation")
    if binding.get("producer") != "dinov2-small":
        raise ValueError("Scientific preparation requires real frozen DINO evidence")
    index = read_json(model_root / "index.json")
    expected = prediction_lock.get("metadata", {}).get("scenes")
    if not isinstance(expected, list) or len(expected) != len(set(expected)) or not set(expected) <= set(index["scenes"]):
        raise ValueError("Scoring scene membership is invalid")
    selected = sorted(expected if scenes is None else scenes)
    if not selected or len(selected) != len(set(selected)) or not set(selected) <= set(expected):
        raise ValueError("Requested scenes must be a unique nonempty subset of sealed scenes")
    for scene in selected:
        _opaque(scene, r"S\d{6}", "scene")
    schedules = _view_sets(view_sets, set(index["views"]))
    views = sorted({v for values in schedules.values() for v in values})
    supplied, supplied_receipt = _source_correspondence(Path(neutral_supplied_jsonl), "supplied")
    automatic, automatic_receipt = _source_correspondence(Path(neutral_automatic_jsonl), "automatic")

    payloads = []
    counts = Counter()
    for scene in selected:
        model_rows = read_json(model_root / "scenes" / f"{scene}.json")
        rows = _score_rows(scoring_root / "scenes" / f"{scene}.json", scene, model_rows)
        observations = []
        for route, source in (("supplied", supplied), ("automatic", automatic)):
            for view in views:
                identities = set()
                for obj in sorted(rows[route, view]["objects"], key=lambda o: o["candidate_id"]):
                    candidate = obj["candidate_id"]
                    key = (scene, view, candidate)
                    if key not in source:
                        raise ValueError("Selected candidate is missing a trusted correspondence row")
                    reference = source[key]
                    neutral = None if reference is None else _neutral_id(scene, reference)
                    if neutral is not None and neutral in identities:
                        raise ValueError("Conflicting candidates share an identity in the same view")
                    identities.add(neutral)
                    observations.append({"route": route, "view": view, "candidate_id": candidate, "neutral_id": neutral})
                    counts[f"{route}_observations"] += 1
                    counts[f"{route}_unknown_observations"] += neutral is None
        payloads.append({"schema_version": SCHEMA, "scene": scene, "selected_views": views, "observations": observations})

    manifest = {
        "schema_version": SCHEMA,
        "scene_count": len(selected), "scenes": selected,
        "view_sets": schedules, "selected_views": views,
        "counts": dict(sorted(counts.items())),
        "model_input_lock_sha256": model_digest,
        "scoring_prediction_lock_sha256": digest_file(scoring_root / "prediction_lock.json"),
        "scoring_feature_lock_sha256": digest_file(scoring_root / "feature_lock.json"),
        "neutral_sources": {"supplied": supplied_receipt, "automatic": automatic_receipt},
        "implementation_sha256": digest_file(Path(__file__)),
        "reference_correspondence_supplied": True,
        "anomaly_labels_opened": False,
        "raw_source_identities_exported": False,
        "scope": "existing development scenes; selected-view assisted inputs",
    }
    # Validate everything first. Atomic files plus a final seal distinguish a
    # complete package from a preparation interrupted during writing.
    for payload in payloads:
        atomic_json(output_root / "scenes" / f"{payload['scene']}.json", payload)
    atomic_json(output_root / "input_manifest.json", manifest)
    seal(output_root, [output_root / "input_manifest.json", *sorted((output_root / "scenes").glob("*.json"))], metadata={"kind": SCHEMA, "anomaly_labels_opened": False, "reference_correspondence_supplied": True})
    return manifest | {"artifact_lock_sha256": digest_file(output_root / "artifact_lock.json")}
