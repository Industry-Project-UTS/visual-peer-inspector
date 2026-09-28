"""Inspection cases: context, operator decisions, timeline and portable case files.

The operator's record is kept apart from the model result. Nothing here runs a
model, reads or creates an inference cache, or changes a ranking or review state.
"""

from __future__ import annotations

import hashlib
import html
import io
import json
import math
import re
import secrets
import shutil
import tempfile
import warnings
import zipfile
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from .core import (
    MAX_PIXELS,
    MAX_VIEWS,
    MIN_VIEWS,
    TOKEN,
    Box,
    InspectionError,
    InspectionRequest,
    InspectionResult,
    Observation,
    RankedObject,
    Session,
    View,
    validate_boxes,
    validate_result,
)
from .presentation import (
    LABELS,
    ORIGINS,
    export_metadata,
    export_report,
    identity_aliases,
    method_label,
    result_record,
    selected_identity,
)

CASE_SCHEMA = "visual-peer-inspection-case/v1"
CASE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{2,39}")
TIMESTAMP = "%Y-%m-%dT%H:%M:%SZ"
GROUP = "*"  # Dropdown value for "whole peer group"; never a valid identity token.
ACTIONS = {
    "inspect_further": "Inspect further",
    "hold": "Hold",
    "correct_rework": "Correct / rework",
    "no_issue_found": "No issue found",
}
STATUSES = {
    None: ("open", "Open · awaiting operator decision"),
    "inspect_further": ("open", "Open · further inspection"),
    "hold": ("on_hold", "On hold"),
    "correct_rework": ("rework", "Rework required"),
    "no_issue_found": ("closed", "Closed · no issue found"),
}
LIMITS = {"item": 120, "batch": 80, "operator": 80, "finding": 300, "reason": 300, "responsible": 80, "event": 300}
MAX_CASE_BYTES = 100 * 1024 * 1024
MAX_UNPACKED_BYTES = 400 * 1024 * 1024
MAX_JSON_BYTES = 5 * 1024 * 1024
MAX_EVENTS = 200
MAX_DECISIONS = 50
MAX_OBSERVATIONS_PER_VIEW = 8
DATA_ORIGINS = {*ORIGINS, "backend_output"}
SEPARATION_NOTE = (
    "The operator decision is a human record. It is stored separately from the model ranking, does not "
    "change scores, the selected object or the review state, and does not approve objects it does not name."
)
FOLLOW_UP_NOTE = (
    "This inspection is a new capture of the full peer group. Scores are relative to the objects in each "
    "capture and are not compared between cases; a lower score does not show that a repair worked. Check "
    "the item physically and record the outcome as the operator decision."
)
READ_ONLY_NOTE = (
    "Opened read-only from a saved case file. The models were not rerun and no inference cache was created. "
    "The file was checked for structure and image fingerprints; it is not signed."
)
MEMBER = re.compile(r"case\.json|views/view-[1-5]\.(?:png|webp)|evidence/obs-\d{3}-(?:mask\.png|heatmap\.(?:png|webp))")
MACHINE_KEYS = frozenset({
    "mode", "method_label", "correspondence", "selected_identity_id", "selected_locations", "missing_links",
    "review_signals", "overlay_aliases", "score_definitions", "review_state", "reasons", "method_version",
    "policy_version", "provenance", "score_metadata", "views", "ranking", "observations", "claim_limits",
})
OBSERVATION_KEYS = frozenset({
    "view_id", "object_id", "identity_id", "box_xyxy", "score", "score_method", "p1_score", "component_scores",
    "normalized_rank", "identity_status", "identity_source", "mask_available", "heatmap_available",
})
DECISION_KEYS = frozenset({
    "action", "action_label", "finding", "reason", "object_scope", "responsible", "operator", "recorded_at",
    "inspection_number", "current",
})
CASE_KEYS = frozenset({
    "case_id", "item", "batch", "operator", "created_at", "status", "inspection_number", "operator_decision",
    "decision_history", "corrections", "follow_up_of", "timeline", "separation_note",
})
SUMMARY_KEYS = frozenset({
    "case_id", "created_at", "saved_at", "item", "batch", "operator", "case_file_sha256", "status", "machine",
    "decisions",
})
SUMMARY_MACHINE_KEYS = frozenset({
    "mode", "method_label", "review_state", "selected_identity_id", "tied", "view_count", "data_origin",
})


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime(TIMESTAMP)


def display_time(value: str) -> str:
    return datetime.strptime(value, TIMESTAMP).strftime("%d %b %Y, %H:%M UTC")


def clean_text(value, limit: int, label: str, *, required: bool = False) -> str:
    """Single-line printable text; HTML is always escaped when it is shown."""
    if value is not None and not isinstance(value, str):
        raise InspectionError(f"The {label} must be text.")
    text = "".join(ch for ch in " ".join((value or "").split()) if ch.isprintable())
    if required and not text:
        raise InspectionError(f"Enter the {label}.")
    if len(text) > limit:
        raise InspectionError(f"Keep the {label} to {limit} characters or fewer.")
    return text


@dataclass(frozen=True)
class OperatorDecision:
    action: str
    finding: str
    reason: str
    object_scope: str  # "" is the whole peer group
    responsible: str
    operator: str
    recorded_at: str
    inspection_number: int


@dataclass
class InspectionCase:
    case_id: str
    created_at: str
    item: str = ""
    batch: str = ""
    operator: str = ""
    follow_up_of: dict | None = None
    decisions: list = field(default_factory=list)
    events: list = field(default_factory=list)
    corrections: dict | None = None
    inspection_number: int = 0
    # Boxes proposed by the most recent uncorrected run; used to describe corrections.
    model_boxes: dict = field(default_factory=dict)


@dataclass(frozen=True)
class OpenedCase:
    case: InspectionCase
    session: Session
    sha256: str
    saved_at: str
    data_origin: str


def new_case(*, item="", batch="", operator="", follow_up_of=None, now=None) -> InspectionCase:
    now = now or utc_now()
    case_id = f"VPI-{now[:10].replace('-', '')}-{secrets.token_hex(2).upper()}"
    case = InspectionCase(case_id, now, item, batch, operator, follow_up_of)
    log(case, "Case created", now)
    if follow_up_of:
        log(case, f"Follow-up of case {follow_up_of['case_id']}; the full peer group must be captured again", now)
    return case


def ensure_case(session: Session) -> InspectionCase:
    if session.case is None:
        session.case = new_case()
    return session.case


def log(case: InspectionCase, event: str, now=None) -> None:
    case.events.append({"time": now or utc_now(), "event": event[:LIMITS["event"]]})
    del case.events[:-MAX_EVENTS]


def update_details(case: InspectionCase, case_id, item, batch, operator) -> None:
    case_id = (case_id or "").strip() if isinstance(case_id, str) else ""
    if not CASE_ID.fullmatch(case_id):
        raise InspectionError("Case IDs use 3–40 letters, numbers, hyphens or underscores.")
    item = clean_text(item, LIMITS["item"], "item description")
    batch = clean_text(batch, LIMITS["batch"], "batch or reference")
    operator = clean_text(operator, LIMITS["operator"], "operator name")
    if case_id != case.case_id:
        log(case, f"Case ID changed from {case.case_id} to {case_id}")
    case.case_id, case.item, case.batch, case.operator = case_id, item, batch, operator


def _tied(result) -> bool:
    return len(result.ranking) > 1 and math.isclose(
        result.ranking[0].relative_score, result.ranking[1].relative_score, abs_tol=1e-12, rel_tol=0)


def _legend(identity: str, result) -> str:
    aliases = identity_aliases(o.box.identity_id for o in result.observations) if result else {}
    alias = aliases.get(identity, identity)
    return f"{alias} · {identity}" if alias != identity else identity


def _machine_outcome(result) -> str:
    selected = selected_identity(result)
    if selected:
        return f"highest combined score: {_legend(selected, result)}"
    if _tied(result) or (result.score_metadata or {}).get("selection_status") == "tied":
        return "leading scores tied, no unique selection"
    return "no object selected"


def correction_record(boxes, model_boxes: dict, now: str) -> dict:
    submitted = {(box.view_id, box.object_id): tuple(box.xyxy) for box in boxes}

    def keys(items):
        return [{"view_id": view, "object_id": item} for view, item in sorted(items)]

    return {
        "confirmed_at": now,
        "boxes": [{"view_id": b.view_id, "object_id": b.object_id, "identity_id": b.identity_id,
                   "box_xyxy": list(b.xyxy)} for b in boxes],
        "added": keys(k for k in submitted if k not in model_boxes),
        "edited": keys(k for k in submitted if k in model_boxes and tuple(model_boxes[k]) != submitted[k]),
        "removed": keys(k for k in model_boxes if k not in submitted),
        "blank_identities": sum(not box.identity_id for box in boxes),
    }


def record_inspection(case: InspectionCase, result: InspectionResult, view_count: int,
                      corrections=None, now=None) -> None:
    now = now or utc_now()
    case.inspection_number += 1
    if corrections is None:
        case.model_boxes = {(o.box.view_id, o.box.object_id): tuple(o.box.xyxy) for o in result.observations}
    else:
        case.corrections = correction_record(corrections, case.model_boxes, now)
        record = case.corrections
        log(case, (f"Object matches confirmed: {len(record['boxes'])} boxes · {len(record['added'])} added · "
                   f"{len(record['edited'])} edited · {len(record['removed'])} removed · "
                   f"{record['blank_identities']} left blank"), now)
    log(case, (f"{result.mode.title()} inspection {case.inspection_number} · {view_count} views · "
               f"model: {LABELS[result.review_state]} · {_machine_outcome(result)}"), now)


def record_decision(case: InspectionCase, result, *, action, finding, reason, object_scope,
                    responsible="", now=None) -> OperatorDecision:
    """Store the operator's finding; the model result is read but never changed."""
    if result is None:
        raise InspectionError("Run an inspection before recording a decision.")
    if not case.operator:
        raise InspectionError("Enter the operator name in the case details before recording a decision.")
    if action not in ACTIONS:
        raise InspectionError("Choose an action.")
    if object_scope is None:
        raise InspectionError("Choose the object your finding refers to, or the whole peer group.")
    object_scope = "" if object_scope == GROUP else object_scope
    identities = {o.box.identity_id for o in result.observations if o.box.identity_id}
    if object_scope and object_scope not in identities:
        raise InspectionError("Choose an object from this inspection, or the whole peer group.")
    if len(case.decisions) >= MAX_DECISIONS:
        raise InspectionError("This case has reached the maximum number of recorded decisions.")
    decision = OperatorDecision(
        action,
        clean_text(finding, LIMITS["finding"], "finding", required=True),
        clean_text(reason, LIMITS["reason"], "reason", required=True),
        object_scope,
        clean_text(responsible, LIMITS["responsible"], "responsible person"),
        case.operator,
        now or utc_now(),
        case.inspection_number,
    )
    case.decisions.append(decision)
    target = _legend(object_scope, result) if object_scope else "whole peer group"
    log(case, f"Operator decision: {ACTIONS[action]} · {target} · recorded by {case.operator}", decision.recorded_at)
    return decision


def current_decision(case: InspectionCase | None) -> OperatorDecision | None:
    """Only a decision made on the latest inspection result is current."""
    if case and case.decisions and case.decisions[-1].inspection_number == case.inspection_number:
        return case.decisions[-1]
    return None


def case_status(case: InspectionCase | None) -> tuple[str, str]:
    decision = current_decision(case)
    return STATUSES[decision.action if decision else None]


def decision_record(decision: OperatorDecision, current: bool = False) -> dict:
    return {
        "action": decision.action, "action_label": ACTIONS[decision.action], "finding": decision.finding,
        "reason": decision.reason, "object_scope": decision.object_scope, "responsible": decision.responsible,
        "operator": decision.operator, "recorded_at": decision.recorded_at,
        "inspection_number": decision.inspection_number, "current": current,
    }


def case_record(case: InspectionCase) -> dict:
    current = current_decision(case)
    code, label = case_status(case)
    return {
        "case_id": case.case_id, "item": case.item, "batch": case.batch, "operator": case.operator,
        "created_at": case.created_at,
        "status": {"code": code, "label": label},
        "inspection_number": case.inspection_number,
        "operator_decision": decision_record(current, True) if current else None,
        "decision_history": [decision_record(d, d is current) for d in case.decisions],
        "corrections": case.corrections,
        "follow_up_of": case.follow_up_of,
        "timeline": [dict(event) for event in case.events],
        "separation_note": SEPARATION_NOTE,
    }


def data_origin(result) -> str:
    source = result.provenance.get("input_source")
    return source if source in ORIGINS else "backend_output"


def follow_up_summary(opened: OpenedCase) -> dict:
    """What a later case keeps about an earlier one: records, not scores to compare."""
    case, result = opened.case, opened.session.result
    current = current_decision(case)
    code, label = case_status(case)
    return {
        "case_id": case.case_id, "created_at": case.created_at, "saved_at": opened.saved_at,
        "item": case.item, "batch": case.batch, "operator": case.operator,
        "case_file_sha256": opened.sha256,
        "status": {"code": code, "label": label},
        "machine": {"mode": result.mode, "method_label": method_label(result),
                    "review_state": result.review_state, "selected_identity_id": selected_identity(result),
                    "tied": _tied(result), "view_count": len(opened.session.views),
                    "data_origin": opened.data_origin},
        "decisions": [decision_record(d, d is current) for d in case.decisions],
    }


def start_follow_up(opened: OpenedCase, operator: str = "") -> InspectionCase:
    return new_case(item=opened.case.item, batch=opened.case.batch, operator=operator,
                    follow_up_of=follow_up_summary(opened))


def decision_choices(result) -> list[tuple[str, str]]:
    if result is None:
        return []
    ranked = [item.identity_id for item in result.ranking]
    others = sorted({o.box.identity_id for o in result.observations if o.box.identity_id} - set(ranked))
    return [("Whole peer group", GROUP)] + [(_legend(identity, result), identity) for identity in ranked + others]


# Portable case files ---------------------------------------------------------


def _encode_rgb(image: Image.Image) -> tuple[str, bytes]:
    """Lossless WebP when it decodes to identical pixels; PNG otherwise."""
    image = image.convert("RGB")
    if max(image.size) <= 16383:
        buffer = io.BytesIO()
        try:
            image.save(buffer, format="WEBP", lossless=True, quality=50)
            with Image.open(io.BytesIO(buffer.getvalue())) as check:
                if check.convert("RGB").tobytes() == image.tobytes():
                    return "webp", buffer.getvalue()
        except (OSError, ValueError, KeyError):
            pass
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "png", buffer.getvalue()


def _support_region(observation: Observation, size) -> tuple[int, int, int, int]:
    """The only pixels a heatmap overlay can change: its mask and its (inclusive) box."""
    x1, y1, x2, y2 = observation.box.xyxy
    x2, y2 = x2 + 1, y2 + 1
    bounds = observation.mask.convert("L").getbbox() if observation.mask is not None else None
    if bounds:
        x1, y1, x2, y2 = min(x1, bounds[0]), min(y1, bounds[1]), max(x2, bounds[2]), max(y2, bounds[3])
    return max(0, x1), max(0, y1), min(size[0], x2), min(size[1], y2)


def write_case_bundle(session: Session, path: Path, saved_at: str) -> None:
    result, case = session.result, session.case
    images = {view.view_id: view.image for view in session.views}
    members, view_files, evidence = {}, [], []
    for index, view in enumerate(session.views, 1):
        kind, data = _encode_rgb(view.image)
        name = f"views/view-{index}.{kind}"
        members[name] = data
        view_files.append({"view_id": view.view_id, "file": name})
    for index, observation in enumerate(result.observations, 1):
        entry = {"mask": None, "heatmap": None, "heatmap_region": None}
        if observation.mask is not None:
            buffer = io.BytesIO()
            observation.mask.convert("L").save(buffer, format="PNG")
            entry["mask"] = f"evidence/obs-{index:03d}-mask.png"
            members[entry["mask"]] = buffer.getvalue()
        if observation.heatmap is not None:
            region = _support_region(observation, images[observation.box.view_id].size)
            kind, data = _encode_rgb(observation.heatmap.crop(region))
            entry["heatmap"] = f"evidence/obs-{index:03d}-heatmap.{kind}"
            entry["heatmap_region"] = list(region)
            members[entry["heatmap"]] = data
        evidence.append(entry)
    document = {
        "schema_version": CASE_SCHEMA,
        "saved_at": saved_at,
        "application": "Visual Peer Inspector web app",
        "data_origin": data_origin(result),
        "case": case_record(case),
        "machine_result": result_record(session),
        "files": {"views": view_files, "observations": evidence},
        "storage_note": ("Views are stored losslessly. Heatmap pixels are stored inside each object's mask and "
                         "box only; the rest of a heatmap is the unchanged view."),
    }
    payload = json.dumps(document, indent=2, allow_nan=False).encode("utf-8")
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("case.json", payload, compress_type=zipfile.ZIP_DEFLATED)
        for name, data in members.items():
            archive.writestr(name, data, compress_type=zipfile.ZIP_STORED)


def save_case(session: Session, title: str = "") -> tuple[str, str, str]:
    """Write the HTML report, JSON result and portable case file for the current result."""
    if session.result is None:
        raise InspectionError("Run an inspection before saving a case.")
    case = ensure_case(session)
    saved_at = utc_now()
    previous_events = list(case.events)
    log(case, "Case saved with images, model evidence, match corrections and operator record", saved_at)
    try:
        # Check the portable file before replacing any earlier report downloads.
        with tempfile.TemporaryDirectory(prefix="vpi-case-") as staging:
            staged_bundle = Path(staging) / f"{case.case_id}-case.zip"
            write_case_bundle(session, staged_bundle, saved_at)
            if staged_bundle.stat().st_size > MAX_CASE_BYTES:
                raise InspectionError(
                    "The case file is too large to reopen in this app. Use fewer or lower-resolution views."
                )
            report, data = export_report(session, title)
            bundle = Path(session.export_directory) / staged_bundle.name
            shutil.move(str(staged_bundle), str(bundle))
            return report, data, str(bundle)
    except Exception:
        # A failed export must not appear as a completed save in the timeline.
        case.events[:] = previous_events
        raise


def _invalid(label: str) -> InspectionError:
    return InspectionError(f"The {label} in this case file is incomplete or has unexpected fields.")


def _mapping(value, keys, label: str) -> dict:
    if not isinstance(value, dict) or set(value) != set(keys):
        raise _invalid(label)
    return value


def _plain(value, label: str, depth: int = 0):
    """JSON data with bounded nesting and finite numbers only."""
    if depth > 8:
        raise _invalid(label)
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        if len(value) > 10_000:
            raise _invalid(label)
        return value
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            raise _invalid(label)
        return value
    if isinstance(value, list):
        return [_plain(item, label, depth + 1) for item in value]
    if isinstance(value, dict):
        return {key: _plain(item, label, depth + 1) for key, item in value.items()}
    raise _invalid(label)


def _public(value, label: str):
    """Reject anything the exporter would have removed (cache handles, tensors, local paths)."""
    value = _plain(value, label)
    if export_metadata(value) != value:
        raise InspectionError(f"The {label} in this case file contains private or unsupported fields.")
    return value


def _string(value, label: str, limit: int = 500) -> str:
    if not isinstance(value, str) or len(value) > limit:
        raise _invalid(label)
    return value


def _integer(value, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise _invalid(label)
    return value


def _number(value, label: str, optional: bool = True):
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise _invalid(label)
    return value


def _timestamp(value, label: str) -> str:
    try:
        if not isinstance(value, str) or datetime.strptime(value, TIMESTAMP).strftime(TIMESTAMP) != value:
            raise ValueError
    except ValueError as error:
        raise InspectionError(f"A {label} timestamp in this case file is invalid.") from error
    return value


def _members(archive: zipfile.ZipFile) -> dict:
    infos = archive.infolist()
    if len(infos) > 1 + MAX_VIEWS * (1 + 2 * MAX_OBSERVATIONS_PER_VIEW):
        raise InspectionError("The case file contains too many files.")
    members = {}
    for info in infos:
        if info.is_dir() or not MEMBER.fullmatch(info.filename) or info.filename in members:
            raise InspectionError("The case file contains unexpected content.")
        if info.flag_bits & 0x1:
            raise InspectionError("Encrypted case files are not supported.")
        members[info.filename] = info
    if sum(info.file_size for info in infos) > MAX_UNPACKED_BYTES:
        raise InspectionError("The case file expands beyond the supported size.")
    if "case.json" not in members:
        raise InspectionError("The case file has no case record.")
    return members


def _read(archive: zipfile.ZipFile, info: zipfile.ZipInfo, limit: int) -> bytes:
    if info.file_size > limit:
        raise InspectionError("A file inside the case file is too large.")
    with archive.open(info) as source:
        data = source.read(limit + 1)
    if len(data) > limit:
        raise InspectionError("A file inside the case file is too large.")
    return data


class _Images:
    """Reads each referenced image exactly once and reports anything left over."""

    def __init__(self, archive, members):
        self.archive, self.members, self.used = archive, members, {"case.json"}

    def load(self, name: str, mode: str, size) -> Image.Image:
        if name not in self.members or name in self.used:
            raise InspectionError("An image referenced by this case file is missing or repeated.")
        self.used.add(name)
        data = _read(self.archive, self.members[name], 64 * 1024 * 1024)
        expected = "WEBP" if name.endswith(".webp") else "PNG"
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(data)) as source:
                    if (source.format != expected or getattr(source, "n_frames", 1) != 1
                            or source.size != tuple(size) or source.width * source.height > MAX_PIXELS):
                        raise InspectionError("An image in this case file has an unexpected format or size.")
                    source.load()
                    return source.convert(mode)
        except InspectionError:
            raise
        except (OSError, SyntaxError, ValueError, Image.DecompressionBombError,
                Image.DecompressionBombWarning) as error:
            raise InspectionError("An image in this case file could not be opened.") from error

    def check_unused(self) -> None:
        if set(self.members) != self.used:
            raise InspectionError("The case file contains files that its record does not reference.")


def _views(images: _Images, rows, files) -> tuple[View, ...]:
    if (not isinstance(rows, list) or not isinstance(files, list)
            or not MIN_VIEWS <= len(rows) <= MAX_VIEWS or len(files) != len(rows)):
        raise InspectionError("A case file must contain 2–5 views.")
    views = []
    for index, (row, entry) in enumerate(zip(rows, files), 1):
        _mapping(row, {"view_id", "sha256", "width", "height"}, "view record")
        _mapping(entry, {"view_id", "file"}, "view file record")
        size = (_integer(row["width"], "view record", 1), _integer(row["height"], "view record", 1))
        if (entry["file"] not in (f"views/view-{index}.webp", f"views/view-{index}.png")
                or entry["view_id"] != row["view_id"] or size[0] * size[1] > MAX_PIXELS):
            raise _invalid("view file record")
        image = images.load(entry["file"], "RGB", size)
        digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
        if digest != row["sha256"] or row["view_id"] != f"v_{digest[:16]}":
            raise InspectionError(f"View {index} does not match its recorded fingerprint.")
        views.append(View(row["view_id"], image, digest))
    if len({view.view_id for view in views}) != len(views):
        raise InspectionError("The case file repeats a view.")
    return tuple(views)


def _observation(images: _Images, index: int, row, entry, views) -> Observation:
    _mapping(row, OBSERVATION_KEYS, "observation")
    _mapping(entry, {"mask", "heatmap", "heatmap_region"}, "evidence file record")
    if row["view_id"] not in views:
        raise InspectionError("An observation refers to a view that is not in the case file.")
    image = views[row["view_id"]]
    mask = heatmap = None
    if entry["mask"] is not None:
        if entry["mask"] != f"evidence/obs-{index:03d}-mask.png":
            raise _invalid("evidence file record")
        mask = images.load(entry["mask"], "L", image.size)
    if entry["heatmap"] is not None:
        region = entry["heatmap_region"]
        if (entry["heatmap"] not in (f"evidence/obs-{index:03d}-heatmap.webp", f"evidence/obs-{index:03d}-heatmap.png")
                or not isinstance(region, list) or len(region) != 4 or any(type(v) is not int for v in region)
                or not (0 <= region[0] < region[2] <= image.width and 0 <= region[1] < region[3] <= image.height)):
            raise _invalid("evidence file record")
        crop = images.load(entry["heatmap"], "RGB", (region[2] - region[0], region[3] - region[1]))
        heatmap = image.copy()
        heatmap.paste(crop, tuple(region[:2]))
    elif entry["heatmap_region"] is not None:
        raise _invalid("evidence file record")
    if row["mask_available"] is not (mask is not None) or row["heatmap_available"] is not (heatmap is not None):
        raise InspectionError("Evidence availability in this case file does not match its stored images.")
    box = row["box_xyxy"]
    if not isinstance(box, list) or len(box) != 4 or any(type(v) is not int for v in box):
        raise _invalid("observation")
    components = _public(row["component_scores"], "component scores")
    if not isinstance(components, dict):
        raise _invalid("component scores")
    return Observation(
        Box(_string(row["view_id"], "observation", 64), _string(row["object_id"], "observation", 64),
            _string(row["identity_id"], "observation", 64), tuple(box)),
        _number(row["p1_score"], "observation"), mask, heatmap,
        score=_number(row["score"], "observation"),
        score_method=_string(row["score_method"], "observation", 120),
        component_scores=components,
        normalized_rank=_number(row["normalized_rank"], "observation"),
        identity_status=_string(row["identity_status"], "observation", 64),
        identity_source=_string(row["identity_source"], "observation", 64),
    )


def _result(images: _Images, machine, files, views) -> InspectionResult:
    _mapping(machine, MACHINE_KEYS, "model result")
    if machine["mode"] not in ("automatic", "assisted"):
        raise _invalid("model result")
    rows, ranking_rows = machine["observations"], machine["ranking"]
    if (not isinstance(rows, list) or not isinstance(files, list) or len(files) != len(rows)
            or len(rows) > MAX_OBSERVATIONS_PER_VIEW * len(views) or not isinstance(ranking_rows, list)
            or len(ranking_rows) > len(rows)):
        raise _invalid("model result")
    images_by_view = {view.view_id: view.image for view in views}
    observations = tuple(_observation(images, index, row, entry, images_by_view)
                         for index, (row, entry) in enumerate(zip(rows, files), 1))
    ranking = []
    for row in ranking_rows:
        _mapping(row, {"identity_id", "relative_score", "view_ids", "score_method", "contributors"}, "ranked object")
        contributors = _public(row["contributors"], "fusion contributors")
        if (not isinstance(row["view_ids"], list) or not all(isinstance(v, str) for v in row["view_ids"])
                or not isinstance(contributors, list) or not all(isinstance(c, dict) for c in contributors)):
            raise _invalid("ranked object")
        ranking.append(RankedObject(_string(row["identity_id"], "ranked object", 64),
                                    _number(row["relative_score"], "ranked object", optional=False),
                                    tuple(row["view_ids"]), tuple(contributors),
                                    _string(row["score_method"], "ranked object", 120)))
    reasons = machine["reasons"]
    if not isinstance(reasons, list) or len(reasons) > 50 or not all(isinstance(r, str) and len(r) <= 500 for r in reasons):
        raise _invalid("model result")
    provenance = _public(machine["provenance"], "model provenance")
    metadata = _public(machine["score_metadata"], "score metadata")
    selected = machine["selected_identity_id"]
    if not isinstance(provenance, dict) or not isinstance(metadata, dict) or not (selected is None or isinstance(selected, str)):
        raise _invalid("model result")
    result = InspectionResult(
        tuple(ranking), observations, machine["review_state"], tuple(reasons),
        _string(machine["method_version"], "model result"), _string(machine["policy_version"], "model result"),
        machine["mode"], provenance, selected, "", None, metadata,
    )
    try:
        validate_result(result, InspectionRequest(views, mode=result.mode))
    except InspectionError as error:
        raise InspectionError(f"The saved model result is inconsistent: {error}") from error
    _plain(machine["claim_limits"], "model result")
    _plain(machine["score_definitions"], "model result")
    rebuilt = result_record(Session(views, result))
    if any(rebuilt[key] != machine[key] for key in MACHINE_KEYS - {"claim_limits", "score_definitions"}):
        raise InspectionError("The saved model summary does not match its stored evidence.")
    return result


def _decision(row, label: str = "decision") -> OperatorDecision:
    _mapping(row, DECISION_KEYS, label)
    scope = row["object_scope"]
    if row["action"] not in ACTIONS:
        raise InspectionError("A recorded decision in this case file uses an unknown action.")
    if not isinstance(scope, str) or (scope and not TOKEN.fullmatch(scope)) or not isinstance(row["current"], bool):
        raise _invalid(label)
    return OperatorDecision(
        row["action"],
        clean_text(row["finding"], LIMITS["finding"], "finding", required=True),
        clean_text(row["reason"], LIMITS["reason"], "reason", required=True),
        scope,
        clean_text(row["responsible"], LIMITS["responsible"], "responsible person"),
        clean_text(row["operator"], LIMITS["operator"], "operator name", required=True),
        _timestamp(row["recorded_at"], label),
        _integer(row["inspection_number"], label, 1),
    )


def _status(decisions: list[dict]) -> dict:
    current = [row for row in decisions if row["current"]]
    if len(current) > 1:
        raise _invalid("decision history")
    code, label = STATUSES[current[0]["action"] if current else None]
    return {"code": code, "label": label}


def _summary(value) -> dict:
    """An earlier case referenced by a follow-up; checked for shape and self-consistency."""
    _mapping(value, SUMMARY_KEYS, "follow-up reference")
    machine = _mapping(value["machine"], SUMMARY_MACHINE_KEYS, "follow-up reference")
    if (not CASE_ID.fullmatch(value["case_id"] if isinstance(value["case_id"], str) else "")
            or not isinstance(value["case_file_sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", value["case_file_sha256"])
            or machine["mode"] not in ("automatic", "assisted") or machine["review_state"] not in LABELS
            or machine["data_origin"] not in DATA_ORIGINS or not isinstance(machine["tied"], bool)
            or not (machine["selected_identity_id"] is None or TOKEN.fullmatch(str(machine["selected_identity_id"])))
            or not isinstance(value["decisions"], list) or len(value["decisions"]) > MAX_DECISIONS):
        raise _invalid("follow-up reference")
    _string(machine["method_label"], "follow-up reference", 120)
    if _integer(machine["view_count"], "follow-up reference", MIN_VIEWS) > MAX_VIEWS:
        raise _invalid("follow-up reference")
    _timestamp(value["created_at"], "follow-up")
    _timestamp(value["saved_at"], "follow-up")
    for key, limit in (("item", LIMITS["item"]), ("batch", LIMITS["batch"]), ("operator", LIMITS["operator"])):
        if clean_text(value[key], limit, "follow-up reference") != value[key]:
            raise _invalid("follow-up reference")
    for row in value["decisions"]:
        if decision_record(_decision(row, "follow-up decision"), row["current"]) != row:
            raise _invalid("follow-up decision")
    if value["status"] != _status(value["decisions"]):
        raise InspectionError("The follow-up reference status does not match its decisions.")
    return value


def _corrections(value, views) -> dict | None:
    if value is None:
        return None
    _mapping(value, {"confirmed_at", "boxes", "added", "edited", "removed", "blank_identities"}, "match corrections")
    _timestamp(value["confirmed_at"], "match correction")
    rows = value["boxes"]
    if not isinstance(rows, list) or len(rows) > MAX_OBSERVATIONS_PER_VIEW * len(views):
        raise _invalid("match corrections")
    boxes = []
    for row in rows:
        _mapping(row, {"view_id", "object_id", "identity_id", "box_xyxy"}, "match corrections")
        box = row["box_xyxy"]
        if (not isinstance(box, list) or len(box) != 4 or any(type(v) is not int for v in box)
                or not all(isinstance(row[k], str) for k in ("view_id", "object_id", "identity_id"))):
            raise _invalid("match corrections")
        boxes.append(Box(row["view_id"], row["object_id"], row["identity_id"], tuple(box)))
    validate_boxes(boxes, views)
    view_ids = {view.view_id for view in views}
    for key in ("added", "edited", "removed"):
        if not isinstance(value[key], list):
            raise _invalid("match corrections")
        for item in value[key]:
            _mapping(item, {"view_id", "object_id"}, "match corrections")
            if item["view_id"] not in view_ids or not TOKEN.fullmatch(str(item["object_id"])):
                raise _invalid("match corrections")
    if value["blank_identities"] != sum(not box.identity_id for box in boxes):
        raise _invalid("match corrections")
    return value


def _case(value, result, views) -> InspectionCase:
    _mapping(value, CASE_KEYS, "case record")
    if not isinstance(value["case_id"], str) or not CASE_ID.fullmatch(value["case_id"]):
        raise _invalid("case record")
    case = InspectionCase(
        value["case_id"], _timestamp(value["created_at"], "case"),
        clean_text(value["item"], LIMITS["item"], "item description"),
        clean_text(value["batch"], LIMITS["batch"], "batch or reference"),
        clean_text(value["operator"], LIMITS["operator"], "operator name"),
        None if value["follow_up_of"] is None else _summary(value["follow_up_of"]),
        inspection_number=_integer(value["inspection_number"], "case record", 1),
    )
    decisions, timeline = value["decision_history"], value["timeline"]
    if not isinstance(decisions, list) or len(decisions) > MAX_DECISIONS:
        raise _invalid("decision history")
    case.decisions = [_decision(row) for row in decisions]
    if any(d.inspection_number > case.inspection_number for d in case.decisions):
        raise _invalid("decision history")
    if not isinstance(timeline, list) or len(timeline) > MAX_EVENTS:
        raise _invalid("case timeline")
    for event in timeline:
        _mapping(event, {"time", "event"}, "case timeline")
        _timestamp(event["time"], "timeline")
        if clean_text(event["event"], LIMITS["event"], "timeline entry", required=True) != event["event"]:
            raise _invalid("case timeline")
    case.events = [dict(event) for event in timeline]
    case.corrections = _corrections(value["corrections"], views)
    current = current_decision(case)
    identities = {o.box.identity_id for o in result.observations if o.box.identity_id}
    if current is not None and current.object_scope and current.object_scope not in identities:
        raise InspectionError("The current decision refers to an object that is not in this result.")
    _string(value["separation_note"], "case record")
    rebuilt = case_record(case)
    if any(rebuilt[key] != value[key] for key in CASE_KEYS - {"separation_note"}):
        raise InspectionError("The case record is inconsistent with its decisions or details.")
    return case


def _reject_constant(name):
    raise ValueError(f"Non-finite number {name}")


def open_case(path) -> OpenedCase:
    """Validate a saved case file and rebuild it for read-only review, without any model."""
    path = Path(path)
    try:
        if path.stat().st_size > MAX_CASE_BYTES:
            raise InspectionError("The case file is larger than 100 MiB.")
        raw = path.read_bytes()
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            members = _members(archive)
            record = _read(archive, members["case.json"], MAX_JSON_BYTES)
            try:
                document = json.loads(record.decode("utf-8"), parse_constant=_reject_constant)
            except (UnicodeDecodeError, ValueError, RecursionError) as error:
                raise InspectionError("The case record is not valid JSON.") from error
            _mapping(document, {"schema_version", "saved_at", "application", "data_origin", "case",
                                "machine_result", "files", "storage_note"}, "case file")
            if document["schema_version"] != CASE_SCHEMA:
                raise InspectionError("This case file uses an unsupported format version.")
            saved_at = _timestamp(document["saved_at"], "save")
            _string(document["application"], "case file", 120)
            _string(document["storage_note"], "case file")
            files = _mapping(document["files"], {"views", "observations"}, "file list")
            machine = _mapping(document["machine_result"], MACHINE_KEYS, "model result")
            images = _Images(archive, members)
            views = _views(images, machine["views"], files["views"])
            result = _result(images, machine, files["observations"], views)
            images.check_unused()
        if document["data_origin"] != data_origin(result):
            raise InspectionError("The recorded data origin does not match the model provenance.")
        case = _case(document["case"], result, views)
    except InspectionError:
        raise
    except (OSError, EOFError, zipfile.BadZipFile, zipfile.LargeZipFile, NotImplementedError,
            RuntimeError, zlib.error, ValueError, TypeError, KeyError) as error:
        raise InspectionError("Choose a case file saved by this app (.zip).") from error
    return OpenedCase(case, Session(views, result, case=case), hashlib.sha256(raw).hexdigest(),
                      saved_at, document["data_origin"])


# HTML ------------------------------------------------------------------------


def _escape(value) -> str:
    return html.escape(str(value))


def _facts(rows) -> str:
    return '<ul class="case-facts">' + "".join(
        f'<li><b>{_escape(label)}</b>{_escape(value) if value else "—"}</li>' for label, value in rows) + '</ul>'


def case_banner_html(case: InspectionCase | None, message: str = "") -> str:
    if case is None:
        return ""
    code, label = case_status(case)
    follow = ""
    if case.follow_up_of:
        earlier = case.follow_up_of
        follow = (f'<p class="follow-up-link">Follow-up of case <b>{_escape(earlier["case_id"])}</b> '
                  f'({_escape(earlier["status"]["label"])}). Capture the full peer group again.</p>')
    notice = f'<p class="case-error" role="alert">{_escape(message)}</p>' if message else ""
    return (f'<div class="case-banner"><span class="case-status {code}">{_escape(label)}</span>'
            f'<span>Created {_escape(display_time(case.created_at))}</span></div>{follow}{notice}')


def case_details_html(case: InspectionCase | None, heading: str = "h2") -> str:
    if case is None:
        return ""
    code, label = case_status(case)
    follow = ""
    if case.follow_up_of:
        follow = (f'<p class="follow-up-link">Follow-up of case <b>{_escape(case.follow_up_of["case_id"])}</b>; '
                  'both records are shown below.</p>')
    return (f'<section class="case-card"><div class="case-title"><{heading}>Case {_escape(case.case_id)}</{heading}>'
            f'<span class="case-status {code}">{_escape(label)}</span></div>'
            + _facts([("Item or product", case.item), ("Batch or reference", case.batch), ("Operator", case.operator),
                      ("Created", display_time(case.created_at)), ("Inspections run", str(case.inspection_number))])
            + follow + '</section>')


def decision_card_html(case: InspectionCase | None, result=None) -> str:
    current = current_decision(case)
    if current is None:
        body = ('<h3>No decision recorded for this inspection</h3><p>The model result is not an approval. '
                'Check the items physically and record what you find.</p>')
    else:
        target = _legend(current.object_scope, result) if current.object_scope else "Whole peer group"
        scope_note = ('<p class="scope-note">This decision covers only the object named here. Other objects '
                      'have no recorded outcome.</p>') if current.object_scope else ''
        body = (f'<span class="decision-pill {current.action}">{_escape(ACTIONS[current.action])}</span>'
                f'<h3>{_escape(current.finding)}</h3>'
                + _facts([("Applies to", target), ("Reason", current.reason),
                          ("Responsible", current.responsible or "Not assigned"), ("Recorded by", current.operator),
                          ("Recorded", display_time(current.recorded_at))]) + scope_note)
    earlier = len(case.decisions) - (1 if current else 0) if case else 0
    history = (f'<p class="history-note">{earlier} earlier decision{"s" if earlier != 1 else ""} '
               'kept in the case history.</p>') if earlier else ''
    return f'<div class="human-card"><p class="card-eyebrow">Operator decision · human record</p>{body}{history}</div>'


def _record_side(title: str, machine: str, human: str) -> str:
    return f'<div class="follow-up-side"><h4>{_escape(title)}</h4><p><b>Model</b> {_escape(machine)}</p><p><b>Operator</b> {_escape(human)}</p></div>'


def _decision_text(label, target, finding, operator, recorded_at) -> str:
    return f'{label} · {target} · “{finding}” · {operator}, {display_time(recorded_at)}'


def follow_up_html(case: InspectionCase | None, result=None) -> str:
    earlier = case.follow_up_of if case else None
    if not earlier:
        return ""
    machine = earlier["machine"]
    if machine["selected_identity_id"]:
        outcome = f'highest combined score: {machine["selected_identity_id"]}'
    else:
        outcome = "leading scores tied" if machine["tied"] else "no object selected"
    earlier_decision = next((row for row in earlier["decisions"] if row["current"]), None)
    left = _record_side(
        f'Earlier case {earlier["case_id"]} · saved {display_time(earlier["saved_at"])}',
        f'{machine["mode"].title()} · {LABELS[machine["review_state"]]} · {outcome}',
        _decision_text(earlier_decision["action_label"], earlier_decision["object_scope"] or "whole peer group",
                       earlier_decision["finding"], earlier_decision["operator"], earlier_decision["recorded_at"])
        if earlier_decision else "No decision was recorded.")
    current = current_decision(case)
    this_machine = (f'{result.mode.title()} · {LABELS[result.review_state]} · {_machine_outcome(result)}'
                    if result is not None else "Not inspected yet.")
    this_decision = (_decision_text(ACTIONS[current.action],
                                    _legend(current.object_scope, result) if current.object_scope else "whole peer group",
                                    current.finding, current.operator, current.recorded_at)
                     if current else "No decision recorded yet.")
    right = _record_side(f"This case {case.case_id}", this_machine, this_decision)
    return (f'<section class="follow-up"><h3>Follow-up of case {_escape(earlier["case_id"])}</h3>'
            f'<p class="follow-up-note">{_escape(FOLLOW_UP_NOTE)}</p><div class="follow-up-grid">{left}{right}</div></section>')


def timeline_html(case: InspectionCase | None) -> str:
    if case is None or not case.events:
        return ""
    items = "".join(f'<li><time datetime="{_escape(e["time"])}">{_escape(display_time(e["time"]))}</time>'
                    f'{_escape(e["event"])}</li>' for e in case.events)
    return f'<section class="case-timeline"><h3>Case timeline</h3><ol class="timeline">{items}</ol></section>'


def _table(headers, rows) -> str:
    head = "".join(f"<th>{_escape(value)}</th>" for value in headers)
    body = "".join("<tr>" + "".join(f"<td>{_escape(value)}</td>" for value in row) + "</tr>" for row in rows)
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def decision_history_html(case: InspectionCase | None) -> str:
    if case is None or not case.decisions:
        return ""
    current = current_decision(case)
    rows = [[d.inspection_number, ACTIONS[d.action], d.object_scope or "Whole peer group", d.finding, d.reason,
             d.responsible or "—", d.operator, display_time(d.recorded_at), "current" if d is current else "superseded"]
            for d in case.decisions]
    return ('<h2>Operator decision history</h2>' + _table(
        ["Inspection", "Action", "Applies to", "Finding", "Reason", "Responsible", "Recorded by", "Recorded", "Status"], rows))


def corrections_html(session: Session) -> str:
    record = session.case.corrections if session.case else None
    if not record:
        return ""
    names = {view.view_id: f"View {index}" for index, view in enumerate(session.views, 1)}
    rows = [[names.get(row["view_id"], row["view_id"]), row["object_id"], row["identity_id"] or "left blank",
             ", ".join(str(v) for v in row["box_xyxy"])] for row in record["boxes"]]
    summary = (f'{len(record["added"])} boxes added, {len(record["edited"])} edited and {len(record["removed"])} removed '
               f'relative to the model proposals; {record["blank_identities"]} identities left blank. Confirming '
               'matches does not choose the unusual object; the scorer still ranks the objects.')
    return (f'<h2>Object match corrections</h2><p>Confirmed {_escape(display_time(record["confirmed_at"]))}. '
            f'{_escape(summary)}</p>' + _table(["View", "Object", "Confirmed identity", "Box (left, top, right, bottom)"], rows))


def report_sections(session: Session) -> tuple[str, str, str]:
    """Case HTML placed before, beside and after the machine evidence in a report."""
    case, result = session.case, session.result
    decision = (f'<h2>Operator decision</h2>{decision_card_html(case, result)}'
                f'<p class="separation-note">{_escape(SEPARATION_NOTE)}</p>{follow_up_html(case, result)}')
    history = decision_history_html(case) + corrections_html(session) + timeline_html(case)
    return case_details_html(case), decision, history


def opened_case_html(opened: OpenedCase) -> str:
    origin = ORIGINS.get(opened.data_origin, "Output of the configured inspection backend.")
    return (f'<div class="read-only-banner"><b>Read-only</b> {_escape(READ_ONLY_NOTE)}'
            f'<br>Saved {_escape(display_time(opened.saved_at))} · file fingerprint {_escape(opened.sha256[:12])} · '
            f'{_escape(origin)}</div>' + case_details_html(opened.case, "h3"))
