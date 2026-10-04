"""Upload validation and the contract between the web app and inference."""

from __future__ import annotations

import hashlib
import importlib
import math
import re
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from PIL import Image, ImageOps, UnidentifiedImageError

MIN_VIEWS, MAX_VIEWS = 2, 5
MAX_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 12_000_000
TOKEN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
REVIEW_STATES = {"result_ready", "review_required", "insufficient_evidence"}
STAGES = (
    "Finding objects",
    "Creating masks",
    "Extracting features",
    "Comparing peers",
    "Combining views",
)


class InspectionError(ValueError):
    """An actionable error that can be shown to the user."""


class InspectionCancelled(InspectionError):
    """An intentional user cancellation, rather than a failed model run."""


@dataclass(frozen=True)
class View:
    view_id: str
    image: Image.Image
    digest: str


@dataclass(frozen=True)
class Box:
    view_id: str
    object_id: str
    identity_id: str
    xyxy: tuple[int, int, int, int]


@dataclass(frozen=True)
class InspectionRequest:
    views: tuple[View, ...]
    # None means automatic; an explicit sequence is an assisted rerun.
    corrections: tuple[Box, ...] | None = None
    request_id: str = field(default_factory=lambda: uuid4().hex)
    session_id: str = field(default_factory=lambda: uuid4().hex)
    cache_token: str | None = None
    mode: str | None = None

    @property
    def decision_mode(self):
        return self.mode or ("assisted" if self.corrections is not None else "automatic")


@dataclass(frozen=True)
class RankedObject:
    identity_id: str
    relative_score: float
    view_ids: tuple[str, ...]
    contributors: tuple[dict, ...] = ()
    score_method: str = ""


@dataclass(frozen=True)
class Observation:
    box: Box
    p1_score: float | None
    mask: Image.Image | None = None
    heatmap: Image.Image | None = None
    score: float | None = None
    score_method: str = ""
    component_scores: dict = field(default_factory=dict)
    normalized_rank: float | None = None
    identity_status: str = ""
    identity_source: str = ""


@dataclass(frozen=True)
class InspectionResult:
    ranking: tuple[RankedObject, ...]
    observations: tuple[Observation, ...]
    review_state: str
    reasons: tuple[str, ...] = ()
    method_version: str = ""
    policy_version: str = ""
    mode: str = "automatic"
    provenance: dict = field(default_factory=dict)
    selected_identity_id: str | None = None
    session_id: str = ""
    cache_token: str | None = None
    score_metadata: dict = field(default_factory=dict)


class InspectionBackend(Protocol):
    def inspect(
        self, request: InspectionRequest, progress: Callable[[str], None]
    ) -> InspectionResult:
        """Consume RGB images and return actual inference outputs, without labels."""
        ...


def load_backend(spec: str | None) -> InspectionBackend | None:
    """An operator-configured module:factory; never taken from a web request."""
    if not spec:
        return None
    module_name, separator, factory_name = spec.partition(":")
    if not separator or not module_name or not factory_name:
        raise ValueError("VPI_BACKEND must be module:factory")
    backend = getattr(importlib.import_module(module_name), factory_name)()
    if not callable(getattr(backend, "inspect", None)):
        raise TypeError("The backend factory must return an object with inspect().")
    return backend


def load_views(paths: Sequence[str] | None) -> tuple[View, ...]:
    """Decode real uploads, strip metadata, and assign content-stable opaque IDs."""
    if not paths:
        return ()
    if len(paths) > MAX_VIEWS:
        raise InspectionError(
            "Choose up to five images. Remove the extra views to continue."
        )
    views: list[View] = []
    digests: set[str] = set()
    for index, value in enumerate(paths, 1):
        path = Path(value)
        try:
            if path.stat().st_size > MAX_BYTES:
                raise InspectionError(
                    f"Image {index} is over 10 MB. Choose a smaller file."
                )
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(path) as source:
                    if source.format not in {"JPEG", "PNG", "WEBP"}:
                        raise InspectionError(
                            f"Image {index}: use a JPEG, PNG or WebP photo."
                        )
                    if getattr(source, "n_frames", 1) != 1:
                        raise InspectionError(
                            f"Image {index} is animated. Choose a still photo."
                        )
                    if source.width * source.height > MAX_PIXELS:
                        raise InspectionError(
                            f"Image {index} is over 12 megapixels. Resize it first."
                        )
                    if min(source.size) < 64:
                        raise InspectionError(
                            f"Image {index} is too small. Use at least 64 × 64 pixels."
                        )
                    source.load()
                    rgb = ImageOps.exif_transpose(source).convert("RGB")
                    # A fresh image prevents source EXIF/text metadata entering inference.
                    image = Image.frombytes("RGB", rgb.size, rgb.tobytes())
        except InspectionError:
            raise
        except (
            OSError,
            UnidentifiedImageError,
            Image.DecompressionBombError,
            Image.DecompressionBombWarning,
            ValueError,
        ) as error:
            raise InspectionError(
                f"Image {index} could not be opened. Choose a valid photo."
            ) from error
        digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
        if digest in digests:
            raise InspectionError(
                "Two uploads contain the same image. Choose different camera views."
            )
        digests.add(digest)
        views.append(View(f"v_{digest[:16]}", image, digest))
    return tuple(views)


def validate_boxes(boxes: Sequence[Box], views: Sequence[View]) -> None:
    dimensions = {view.view_id: view.image.size for view in views}
    keys = set()
    identities = set()
    for box in boxes:
        if box.view_id not in dimensions:
            raise InspectionError("A box refers to a view that is no longer uploaded.")
        if not TOKEN.fullmatch(box.object_id) or (box.identity_id and not TOKEN.fullmatch(box.identity_id)):
            raise InspectionError(
                "Object and identity IDs must use letters, numbers, underscores or hyphens."
            )
        if (box.view_id, box.object_id) in keys:
            raise InspectionError("Each object ID must appear only once in a view.")
        keys.add((box.view_id, box.object_id))
        if box.identity_id and (box.view_id, box.identity_id) in identities:
            raise InspectionError(
                "One identity cannot belong to two objects in the same view."
            )
        if box.identity_id:
            identities.add((box.view_id, box.identity_id))
        if len(box.xyxy) != 4 or any(type(v) is not int for v in box.xyxy):
            raise InspectionError("Box coordinates must be four whole numbers.")
        x1, y1, x2, y2 = box.xyxy
        width, height = dimensions[box.view_id]
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise InspectionError(
                "Boxes must have positive width and height and stay inside their image."
            )


def parse_corrections(
    rows: Sequence[Sequence],
    views: Sequence[View],
    *,
    require_multiple_views: bool = True,
) -> tuple[Box, ...]:
    # The review table uses the same upload-order labels as the image galleries.
    view_labels = {f"View {i}": view.view_id for i, view in enumerate(views, 1)}
    boxes = []
    for row in rows:
        if len(row) != 7:
            raise InspectionError(
                "Each box needs a view, object ID, identity ID and four coordinates."
            )
        try:
            values = [float(value) for value in row[3:]]
            if not all(math.isfinite(value) and value.is_integer() for value in values):
                raise ValueError
            boxes.append(
                Box(
                    view_labels.get(str(row[0]), str(row[0])),
                    str(row[1]), str(row[2]), tuple(int(v) for v in values)
                )
            )
        except (TypeError, ValueError) as error:
            raise InspectionError("Box coordinates must be whole numbers.") from error
    validate_boxes(boxes, views)
    if require_multiple_views and len({box.view_id for box in boxes}) < MIN_VIEWS:
        raise InspectionError("Keep at least one box in each of two different views.")
    return tuple(boxes)


def validate_result(result: InspectionResult, request: InspectionRequest) -> None:
    """Reject broken or misleading backend output before rendering or exporting."""
    if not isinstance(result, InspectionResult):
        raise InspectionError("The inspection service returned an unsupported result.")
    if result.review_state not in REVIEW_STATES:
        raise InspectionError(
            "The inspection service returned an unknown review state."
        )
    expected_mode = request.decision_mode
    if result.mode != expected_mode:
        raise InspectionError(
            "The inspection result has an incorrect automatic/assisted label."
        )
    if not result.method_version or not result.policy_version:
        raise InspectionError(
            "The inspection result is missing its method or review policy version."
        )
    if result.review_state != "result_ready" and not result.reasons:
        raise InspectionError(
            "The inspection service must explain why review is needed."
        )
    if result.review_state == "result_ready" and not result.ranking:
        raise InspectionError("The inspection service returned no ranked objects.")
    if result.review_state == "result_ready" and result.selected_identity_id is None:
        raise InspectionError("A ready result must select one physical object.")
    validate_boxes([o.box for o in result.observations], request.views)
    sizes = {v.view_id: v.image.size for v in request.views}
    evidence: dict[str, set[str]] = {}
    for observation in result.observations:
        if observation.p1_score is not None and not math.isfinite(observation.p1_score):
            raise InspectionError(
                "The inspection service returned an invalid object score."
            )
        score = observation.score if observation.score_method else observation.p1_score
        if score is not None and not math.isfinite(score):
            raise InspectionError("The inspection service returned an invalid score.")
        if (observation.normalized_rank is not None if observation.score_method else score is not None):
            evidence.setdefault(observation.box.identity_id, set()).add(
                observation.box.view_id
            )
        for overlay in (observation.mask, observation.heatmap):
            if overlay is not None and overlay.size != sizes[observation.box.view_id]:
                raise InspectionError(
                    "An evidence overlay does not match its image dimensions."
                )
    ids: set[str] = set()
    previous_score = math.inf
    for ranked in result.ranking:
        if ranked.identity_id in ids or not TOKEN.fullmatch(ranked.identity_id):
            raise InspectionError(
                "The inspection service returned duplicate or invalid identities."
            )
        ids.add(ranked.identity_id)
        if (
            not math.isfinite(ranked.relative_score)
            or ranked.relative_score > previous_score
        ):
            raise InspectionError(
                "The inspection service returned invalid or unsorted scores."
            )
        previous_score = ranked.relative_score
        if not ranked.view_ids or len(set(ranked.view_ids)) != len(ranked.view_ids):
            raise InspectionError("An object is missing valid supporting views.")
        if set(ranked.view_ids) != evidence.get(ranked.identity_id, set()):
            raise InspectionError(
                "Object view support does not match the displayed evidence."
            )
    if result.selected_identity_id is not None:
        if not result.ranking or result.selected_identity_id != result.ranking[0].identity_id:
            raise InspectionError("The selected object is not the leading ranked object.")
        if len(result.ranking) > 1 and math.isclose(result.ranking[0].relative_score, result.ranking[1].relative_score, abs_tol=1e-12, rel_tol=0):
            raise InspectionError("Tied rankings cannot select one object.")


@dataclass
class Session:
    views: tuple[View, ...] = ()
    result: InspectionResult | None = None
    export_directory: str | None = None
    pending_corner: tuple[int, int] | None = None
    corrections: list[list] = field(default_factory=list)
    active_request_id: str | None = None
    session_id: str = field(default_factory=lambda: uuid4().hex)
    cache_token: str | None = None
    mode: str = "automatic"
    # An InspectionCase from case.py; the operator record never feeds inference.
    case: object | None = None
