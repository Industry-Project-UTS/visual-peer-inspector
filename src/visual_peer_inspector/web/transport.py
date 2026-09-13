"""Private, temporary JSON/PNG exchange between the two Python environments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image

from .core import (
    MAX_PIXELS,
    MAX_VIEWS,
    MIN_VIEWS,
    Box,
    InspectionRequest,
    InspectionResult,
    Observation,
    RankedObject,
    View,
    validate_boxes,
    validate_result,
)


def _image(root: Path, name: str, expected_size=None):
    if not isinstance(name, str) or Path(name).name != name:
        raise ValueError("Unsafe image filename in worker exchange.")
    path = (root / name).resolve(strict=True)
    if path.parent != root.resolve() or (root / name).is_symlink():
        raise ValueError("Worker image escapes its private job directory.")
    with Image.open(path) as source:
        if source.format != "PNG" or source.width * source.height > MAX_PIXELS:
            raise ValueError("Invalid worker image format or dimensions.")
        if expected_size is not None and source.size != expected_size:
            raise ValueError(
                "Worker overlay dimensions differ from the uploaded image."
            )
        source.load()
        return source.copy()


def _json(path):
    if path.stat().st_size > 5_000_000:
        raise ValueError("Worker metadata is too large.")
    return json.loads(path.read_text(encoding="utf-8"))


def write_request(root: Path, request: InspectionRequest):
    views = []
    for index, view in enumerate(request.views):
        name = f"input-{index}.png"
        view.image.save(root / name, format="PNG")
        views.append({"view_id": view.view_id, "digest": view.digest, "image": name})
    record = {"views": views, "request_id": request.request_id, "corrections": None}
    if request.corrections is not None:
        record["corrections"] = [vars(box) for box in request.corrections]
    (root / "request.json").write_text(
        json.dumps(record, allow_nan=False), encoding="utf-8"
    )


def read_request(root: Path):
    value = _json(root / "request.json")
    if not MIN_VIEWS <= len(value["views"]) <= MAX_VIEWS:
        raise ValueError("Invalid number of views in worker exchange.")
    views = []
    for row in value["views"]:
        image = _image(root, row["image"]).convert("RGB")
        digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
        if digest != row["digest"] or row["view_id"] != f"v_{digest[:16]}":
            raise ValueError("Worker input pixels differ from their content identity.")
        views.append(View(row["view_id"], image, digest))
    corrections = (
        None
        if value["corrections"] is None
        else tuple(
            Box(
                row["view_id"], row["object_id"], row["identity_id"], tuple(row["xyxy"])
            )
            for row in value["corrections"]
        )
    )
    if corrections is not None:
        validate_boxes(corrections, views)
    return InspectionRequest(tuple(views), corrections, request_id=value["request_id"])


def write_result(root: Path, result: InspectionResult):
    observations = []
    for index, observation in enumerate(result.observations):
        row = {"box": vars(observation.box), "p1_score": observation.p1_score}
        for layer in ("mask", "heatmap"):
            overlay = getattr(observation, layer)
            name = None if overlay is None else f"output-{index}-{layer}.png"
            if overlay is not None:
                overlay.save(root / name, format="PNG")
            row[layer] = name
        observations.append(row)
    record = {
        "ranking": [vars(item) for item in result.ranking],
        "observations": observations,
        "review_state": result.review_state,
        "reasons": result.reasons,
        "method_version": result.method_version,
        "policy_version": result.policy_version,
        "mode": result.mode,
        "provenance": result.provenance,
    }
    (root / "result.json").write_text(
        json.dumps(record, allow_nan=False), encoding="utf-8"
    )


def read_result(root: Path, request: InspectionRequest):
    value = _json(root / "result.json")
    dimensions = {v.view_id: v.image.size for v in request.views}
    if len(value["observations"]) > 8 * len(request.views):
        raise ValueError("Worker returned more than eight objects per uploaded view.")
    observations = []
    for row in value["observations"]:
        box = row["box"]
        size = dimensions[box["view_id"]]
        layers = [
            None if row[layer] is None else _image(root, row[layer], size)
            for layer in ("mask", "heatmap")
        ]
        observations.append(
            Observation(
                Box(
                    box["view_id"],
                    box["object_id"],
                    box["identity_id"],
                    tuple(box["xyxy"]),
                ),
                row["p1_score"],
                *layers,
            )
        )
    result = InspectionResult(
        tuple(
            RankedObject(r["identity_id"], r["relative_score"], tuple(r["view_ids"]))
            for r in value["ranking"]
        ),
        tuple(observations),
        value["review_state"],
        tuple(value["reasons"]),
        value["method_version"],
        value["policy_version"],
        value["mode"],
        value["provenance"],
    )
    validate_result(result, request)
    return result
