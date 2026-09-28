"""Physical-object evidence and self-contained inspection reports."""

from __future__ import annotations

import base64
import hashlib
import html
import io
import json
import math
import re
import shutil
import tempfile
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw

from .core import Box, Session

LABELS = {
    "result_ready": "Result ready",
    "review_required": "Review required",
    "insufficient_evidence": "Insufficient evidence",
}
SOURCES = {
    "predicted": "Predicted matches",
    "user_confirmed": "User-confirmed matches",
    "dataset_reference": "Dataset-supplied matches",
    "unknown": "Unconfirmed identity",
}
SCORE_HEADERS = [
    "View", "Candidate", "Overlay / full object ID", "Match status", "Observation score",
    "Within-view rank", "Fusion contribution", "Component scores",
]
RANK_HEADERS = ["Rank", "Overlay / full object ID", "Combined score", "Scored views", "Fusion contributors"]
PALETTE = ("#4f46a5", "#2563a6", "#0f766e", "#975925", "#8c4678")
PRIVATE_KEYS = frozenset({
    "session_id", "cache_token", "cache_root", "cache_path", "cache_directory",
    "api_key", "api_secret_key", "api_access_key", "password", "token",
    "patches", "patch_features", "global_features", "embedding", "embeddings",
    "features", "tensors", "pixel_values", "mask_rle",
})
CLAIM_LIMITS = (
    "Model scores describe relative unusualness within this group of objects; they are not defect probabilities.",
    "Masks cover whole objects; they are not validated defect boundaries. Heatmaps show coarse feature mismatch.",
    "An object that is not linked in a view is not proof that it is absent.",
    "Dataset-supplied matches and human-confirmed matches are different sources of assistance.",
    "Result ready and Review required are model review-routing states. They do not approve, hold or release "
    "any item; only the operator decision records an action.",
)
ORIGINS = {
    "recorded_development": "Recorded development example · inference was not run on these uploads.",
    "synthetic_test": "Synthetic test data · generated pixels and stand-in model stages, not a model inference result.",
}
UNLINKED = re.compile(r"(?:unlinked|unknown)_\d+")


def discard_exports(session: Session | None) -> None:
    if session and session.export_directory:
        shutil.rmtree(session.export_directory, ignore_errors=True)
        session.export_directory = None


def selected_identity(result) -> str | None:
    """A sorted ranking alone never authorizes a unique object highlight."""
    if result is None:
        return None
    identity = getattr(result, "selected_identity_id", None)
    metadata = getattr(result, "score_metadata", {}) or {}
    if metadata.get("selection_status") in {"tied", "insufficient"}:
        return None
    if not result.ranking or identity != result.ranking[0].identity_id:
        return None
    if len(result.ranking) > 1 and math.isclose(
        result.ranking[0].relative_score, result.ranking[1].relative_score,
        rel_tol=0.0, abs_tol=1e-12,
    ):
        return None
    return identity


def method_label(result) -> str:
    metadata = getattr(result, "score_metadata", {}) or {}
    if metadata.get("method_label"):
        label = str(metadata["method_label"])
        prefix = result.mode + " "
        return label[len(prefix):].capitalize() if label.lower().startswith(prefix) else label
    if "web-p1" in result.method_version:
        return "P1 with mean rank fusion"
    return "Cross-view object banks · strongest two views" if result.mode == "assisted" else "Winner consensus"


def correspondence_label(result) -> str:
    sources = {getattr(o, "identity_source", "") for o in result.observations}
    explicit = (getattr(result, "score_metadata", {}) or {}).get("identity_source")
    explicit = explicit or result.provenance.get("identity_source")
    if explicit:
        sources.add(explicit)
    if "dataset_reference" in sources:
        return SOURCES["dataset_reference"]
    if "user_confirmed" in sources:
        return SOURCES["user_confirmed"]
    return SOURCES["predicted"] if result.mode == "automatic" else SOURCES["unknown"]


def _colour(identity):
    return PALETTE[int(hashlib.sha256(identity.encode()).hexdigest()[:8], 16) % len(PALETTE)]


def identity_aliases(identities) -> dict[str, str]:
    """Compact display names remain stable across views and candidate ordering."""
    aliases = {}
    for identity in sorted(set(identities)):
        winner = re.fullmatch(r"winner-consensus-0*(\d+)", identity)
        unlinked = re.fullmatch(r"(?:unlinked|unknown)_0*(\d+)", identity)
        if winner:
            alias = f"W{int(winner.group(1))}"
        elif unlinked:
            alias = f"?{int(unlinked.group(1))}"
        elif not identity:
            alias = "?"
        elif len(identity) <= 12:
            alias = identity
        else:
            alias = identity[:6] + "~" + hashlib.sha256(identity.encode()).hexdigest()[:4]
        aliases[identity] = alias
    # A user can choose an ID such as W1; preserve it and disambiguate any alias.
    for alias in sorted(set(aliases.values())):
        collisions = [identity for identity, value in aliases.items() if value == alias]
        if len(collisions) > 1:
            for index, identity in enumerate((item for item in collisions if item != alias), 1):
                aliases[identity] = f"{alias}~{index}"
    return aliases


def _session_aliases(session):
    return identity_aliases(o.box.identity_id for o in session.result.observations) if session.result else {}


def _legend_name(identity, aliases):
    alias = aliases.get(identity, identity)
    return f"{alias} · {identity}" if alias != identity else identity


def _label_layout(draw, text, xy, size):
    """Return a clipped text label and rectangle wholly inside the image."""
    width, height = size
    padding = 3
    while text and draw.textbbox((0, 0), text)[2] + padding * 2 > width:
        text = text[:-1]
    bounds = draw.textbbox((0, 0), text)
    label_width = min(width, bounds[2] + padding * 2)
    label_height = min(height, bounds[3] - bounds[1] + padding * 2)
    x = max(0, min(int(xy[0]), width - label_width))
    y = max(0, min(int(xy[1]) - label_height, height - label_height))
    return text, (x, y, x + label_width - 1, y + label_height - 1), (x + padding, y + padding - bounds[1])


def draw_boxes(image: Image.Image, boxes: list[Box], winner: str | None = None,
               aliases: dict[str, str] | None = None) -> Image.Image:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    width = max(2, round(min(image.size) / 250))
    aliases = aliases if aliases is not None else identity_aliases(box.identity_id for box in boxes)
    for box in boxes:
        chosen = box.identity_id == winner and winner is not None
        colour = "#0f766e" if chosen else _colour(box.identity_id)
        draw.rectangle(box.xyxy, outline=colour, width=width * (2 if chosen else 1))
        label = aliases.get(box.identity_id, "?")
        if not box.identity_id:
            label += " " + box.object_id[:9]
        label, bounds, origin = _label_layout(draw, label, box.xyxy[:2], image.size)
        draw.rectangle(bounds, fill=colour)
        draw.text(origin, label, fill="white")
    return canvas


def evidence_gallery(session: Session, identity: str | None, layer: str):
    if not session.result:
        return []
    images = []
    winner = selected_identity(session.result)
    aliases = _session_aliases(session)
    for index, view in enumerate(session.views, 1):
        observations = [o for o in session.result.observations if o.box.view_id == view.view_id
                        and (identity in (None, "All objects") or o.box.identity_id == identity)]
        canvas = view.image.copy()
        note = ""
        if not observations:
            note = " · Not linked in this view. No matching observation."
        elif winner and identity in (None, "All objects") and not any(o.box.identity_id == winner for o in observations):
            note = " · Selected object not linked in this view"
        if layer == "Objects on black":
            combined = Image.new("L", canvas.size)
            for observation in observations:
                if observation.mask is not None:
                    combined = ImageChops.lighter(combined, observation.mask.convert("L"))
            canvas = Image.composite(canvas, Image.new("RGB", canvas.size), combined)
            if observations and not any(o.mask is not None for o in observations):
                note += " · Object masks unavailable"
        elif layer in {"Masks", "Heatmaps"}:
            applied = False
            for observation in observations:
                overlay = observation.mask if layer == "Masks" else observation.heatmap
                if overlay is None:
                    continue
                if layer == "Masks":
                    colour = Image.new("RGB", canvas.size, _colour(observation.box.identity_id))
                    alpha = overlay.convert("L").point(lambda value: int(value * 0.45))
                    canvas = Image.composite(colour, canvas, alpha)
                else:
                    support = observation.mask.convert("L") if observation.mask is not None else Image.new("L", canvas.size)
                    if observation.mask is None:
                        ImageDraw.Draw(support).rectangle(observation.box.xyxy, fill=255)
                    alpha = support.point(lambda value: int(value * 0.45))
                    canvas = Image.composite(overlay.convert("RGB"), canvas, alpha)
                applied = True
            if observations and not applied:
                note += " · This evidence layer is unavailable"
        if layer != "Original":
            canvas = draw_boxes(canvas, [o.box for o in observations], winner, aliases)
        images.append((canvas, f"View {index}{note}"))
    return images


def _score(observation):
    score = getattr(observation, "score", None)
    if score is None and not getattr(observation, "score_method", ""):
        score = observation.p1_score
    return score


def _number(value):
    return None if value is None else round(float(value), 6)


def _view_names(session):
    return {view.view_id: f"View {i}" for i, view in enumerate(session.views, 1)}


def observation_rows(session: Session):
    if session.result is None:
        return []
    names = _view_names(session)
    aliases = _session_aliases(session)
    contributors = {}
    for ranked in session.result.ranking:
        for item in getattr(ranked, "contributors", ()):
            key = (item.get("view_id"), item.get("candidate_id", item.get("object_id")))
            contributors[key] = bool(item.get("used_in_fusion", True))
    rows = []
    for observation in session.result.observations:
        box = observation.box
        status = getattr(observation, "identity_status", "") or "not recorded"
        source = SOURCES.get(getattr(observation, "identity_source", ""), "")
        component_scores = getattr(observation, "component_scores", {}) or {}
        components = "; ".join(f"{name}: {'unavailable' if value is None else format(float(value), '.5g')}"
                               for name, value in sorted(component_scores.items()))
        key = (box.view_id, box.object_id)
        contribution = "used" if contributors.get(key) else "available, not selected" if key in contributors else "not contributing"
        rows.append([names.get(box.view_id, box.view_id), box.object_id, _legend_name(box.identity_id, aliases),
                     status.replace("_", " ") + (" · " + source if source else ""),
                     _number(_score(observation)), _number(getattr(observation, "normalized_rank", None)),
                     contribution, components])
    return rows


def _positions(result):
    """Competition ranks: tied scores share a position."""
    positions, position, previous = {}, 0, None
    for index, ranked in enumerate(result.ranking, 1):
        if previous is None or not math.isclose(previous, ranked.relative_score, abs_tol=1e-12, rel_tol=0):
            position = index
        previous = ranked.relative_score
        positions[ranked.identity_id] = position
    return positions


def ranking_rows(session: Session):
    if session.result is None:
        return []
    rows = []
    names = _view_names(session)
    aliases = _session_aliases(session)
    positions = _positions(session.result)
    for ranked in session.result.ranking:
        contributions = [names.get(item.get("view_id"), item.get("view_id", ""))
                         for item in getattr(ranked, "contributors", ()) if item.get("used_in_fusion", True)]
        rows.append([positions[ranked.identity_id], _legend_name(ranked.identity_id, aliases), _number(ranked.relative_score),
                     ", ".join(names.get(view, view) for view in ranked.view_ids),
                     ", ".join(contributions) if contributions else "not recorded"])
    return rows


def _unlinked(observation, ranked):
    identity = observation.box.identity_id
    if identity in ranked:
        return False
    return (not identity or getattr(observation, "identity_status", "") in {"unlinked", "unknown"}
            or UNLINKED.fullmatch(identity) is not None)


def object_board(session: Session) -> dict:
    """One entry per physical object with a slot for every uploaded view.

    Observations that were never linked across views stay separate; they are
    not merged into an object or treated as missing objects.
    """
    result = session.result
    if result is None:
        return {"objects": [], "unlinked": []}
    names = _view_names(session)
    aliases = _session_aliases(session)
    selected = selected_identity(result)
    positions = _positions(result)
    ranked = {item.identity_id: item for item in result.ranking}
    top = result.ranking[0].relative_score if result.ranking else None
    tied = {item.identity_id for item in result.ranking
            if math.isclose(item.relative_score, top, abs_tol=1e-12, rel_tol=0)} if result.ranking else set()
    if len(tied) < 2:
        tied = set()
    used = {(item.get("view_id"), item.get("candidate_id", item.get("object_id"))): bool(item.get("used_in_fusion", True))
            for entry in result.ranking for item in getattr(entry, "contributors", ())}
    linked, unlinked = {}, []
    for observation in result.observations:
        box = observation.box
        cell = {"view": names.get(box.view_id, box.view_id), "view_id": box.view_id, "linked": True,
                "object_id": box.object_id, "box_xyxy": list(box.xyxy), "score": _number(_score(observation)),
                "normalized_rank": _number(getattr(observation, "normalized_rank", None)),
                "used_in_fusion": used.get((box.view_id, box.object_id)),
                "mask_available": observation.mask is not None}
        if _unlinked(observation, ranked):
            unlinked.append({**cell, "identity_id": box.identity_id, "alias": aliases.get(box.identity_id, "?"),
                             "identity_status": getattr(observation, "identity_status", "") or "not recorded"})
        else:
            linked.setdefault(box.identity_id, {})[box.view_id] = cell
    order = [item.identity_id for item in result.ranking] + sorted(set(linked) - set(ranked))
    shared = list(positions.values())
    objects = []
    for identity in order:
        cells = linked.get(identity, {})
        entry = ranked.get(identity)
        rank = positions.get(identity)
        objects.append({
            "identity_id": identity, "alias": aliases.get(identity, identity),
            "rank": rank, "rank_shared": rank is not None and shared.count(rank) > 1,
            "score": _number(entry.relative_score) if entry else None,
            "selected": identity == selected, "tied_for_top": identity in tied,
            "views_linked": len(cells), "view_count": len(session.views),
            "cells": [cells.get(view.view_id) or {"view": names[view.view_id], "view_id": view.view_id, "linked": False}
                      for view in session.views],
        })
    return {"objects": objects, "unlinked": unlinked}


def _thumbnail(image, xyxy, colour, size=220):
    x1, y1, x2, y2 = xyxy
    margin_x, margin_y = max(4, (x2 - x1) // 8), max(4, (y2 - y1) // 8)
    region = (max(0, x1 - margin_x), max(0, y1 - margin_y),
              min(image.width, x2 + margin_x), min(image.height, y2 + margin_y))
    crop = image.crop(region)
    scale = min(1.0, size / max(crop.size))
    crop = crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))))
    left, top = round((x1 - region[0]) * scale), round((y1 - region[1]) * scale)
    right = max(left, round((x2 - region[0]) * scale) - 1)
    bottom = max(top, round((y2 - region[1]) * scale) - 1)
    ImageDraw.Draw(crop).rectangle((left, top, right, bottom), outline=colour, width=2)
    buffer = io.BytesIO()
    crop.convert("RGB").save(buffer, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _cell_caption(cell):
    parts = []
    if cell["score"] is not None:
        parts.append(f"score {cell['score']:.3f}")
    if cell["normalized_rank"] is not None:
        parts.append(f"view rank {cell['normalized_rank']:.2f}")
    if cell["used_in_fusion"] is not None:
        parts.append("used in fusion" if cell["used_in_fusion"] else "not used in fusion")
    if cell["score"] is None:
        parts.append("not scored")
    return " · ".join(parts)


def object_board_html(session: Session) -> str:
    if session.result is None:
        return ""
    board = object_board(session)
    images = {view.view_id: view.image for view in session.views}
    rows = []
    for item in board["objects"]:
        colour = "#0f766e" if item["selected"] else _colour(item["identity_id"])
        chips = []
        if item["selected"]:
            chips.append('<span class="chip chip-selected">Highest combined score</span>')
        if item["tied_for_top"]:
            chips.append('<span class="chip chip-tied">Tied for highest score</span>')
        if item["score"] is None:
            chips.append('<span class="chip">Not scored</span>')
        if item["views_linked"] < item["view_count"]:
            chips.append(f'<span class="chip chip-missing">Linked in {item["views_linked"]} of {item["view_count"]} views</span>')
        if item["score"] is not None:
            rank = ("=" if item["rank_shared"] else "") + str(item["rank"])
            meta = f"Rank {rank} · combined score {item['score']:.3f}"
        else:
            meta = "No combined score"
        cells = []
        for cell in item["cells"]:
            view = html.escape(cell["view"])
            if cell["linked"]:
                source = _thumbnail(images[cell["view_id"]], cell["box_xyxy"], colour)
                cells.append(f'<figure class="board-cell"><img alt="{html.escape(item["alias"])} in {view}" src="{source}">'
                             f'<figcaption><b>{view}</b> {html.escape(_cell_caption(cell))}</figcaption></figure>')
            else:
                cells.append(f'<figure class="board-cell missing"><div class="missing-slot">Not linked</div>'
                             f'<figcaption><b>{view}</b> No matching observation. This is not proof that the object is absent.</figcaption></figure>')
        full_id = (f'<span class="board-id">{html.escape(item["identity_id"])}</span>'
                   if item["alias"] != item["identity_id"] else '')
        rows.append(f'<div class="board-row{" selected" if item["selected"] else ""}"><div class="board-head">'
                    f'<span class="alias-chip" style="background:{colour}">{html.escape(item["alias"])}</span>'
                    f'{full_id}{"".join(chips)}<span class="board-meta">{html.escape(meta)}</span></div>'
                    f'<div class="board-cells">{"".join(cells)}</div></div>')
    if not rows:
        rows.append('<p class="board-empty">No physical object is linked across views yet. '
                    'In Assisted mode, confirm the object matches in Review.</p>')
    loose = []
    for cell in board["unlinked"]:
        blank = cell["identity_status"] == "unknown" or not cell["identity_id"]
        note = "identity left blank · not scored" if blank else "not linked to other views"
        source = _thumbnail(images[cell["view_id"]], cell["box_xyxy"], _colour(cell["identity_id"]), size=160)
        loose.append(f'<figure class="board-cell"><img alt="Unlinked observation in {html.escape(cell["view"])}" src="{source}">'
                     f'<figcaption><b>{html.escape(cell["view"])}</b> {html.escape(cell["alias"])} · {note}'
                     f'{" · " + html.escape(_cell_caption(cell)) if cell["score"] is not None else ""}</figcaption></figure>')
    unlinked = (f'<details class="board-unlinked"><summary>Observations without a cross-view link ({len(loose)})</summary>'
                f'<div class="board-cells">{"".join(loose)}</div></details>') if loose else ""
    return ('<section class="object-board"><h3>Objects across views</h3>'
            '<p class="board-intro">Each row is one physical object matched across your views. Scores rank relative '
            'unusualness within this group; they do not decide whether anything is defective.</p>'
            + "".join(rows) + unlinked + '</section>')


def summary_html(session: Session) -> str:
    result = session.result
    if result is None:
        return '<p class="empty-result">Run an inspection to see results.</p>'
    state = LABELS[result.review_state]
    reasons = "".join(f"<li>{html.escape(reason)}</li>" for reason in result.reasons)
    selected = selected_identity(result)
    metadata = getattr(result, "score_metadata", {}) or {}
    signals = result.provenance.get("review_signals", {})
    confidence = signals.get("confidence")
    confidence_text = (f'<span><b>{confidence:.3f}</b>Ranking confidence index</span>'
                       if isinstance(confidence, (int, float)) and math.isfinite(confidence) else '')
    if selected:
        winner = next(item for item in result.ranking if item.identity_id == selected)
        used = sum(item.get("used_in_fusion", True) for item in getattr(winner, "contributors", ()))
        detail = (f'<h2>{html.escape(selected)}</h2><p>Highest combined unusualness score</p>'
                  f'<div class="result-facts"><span><b>{winner.relative_score:.3f}</b>Relative score</span>'
                  f'<span><b>{len(winner.view_ids)} / {len(session.views)}</b>View support</span>'
                  + confidence_text
                  + (f'<span><b>{used}</b>Fusion contributors</span>' if used else '') + '</div>')
    elif metadata.get("selection_status") == "tied" or (len(result.ranking) > 1 and math.isclose(
        result.ranking[0].relative_score, result.ranking[1].relative_score, abs_tol=1e-12, rel_tol=0)):
        detail = '<h2>No unique selection</h2><p>The highest combined scores are tied. Review the object matches and available views.</p>'
    else:
        detail = '<h2>More evidence needed</h2><p>No unique physical object is selected. Review the matches and available scores below.</p>'
    origin = ORIGINS.get(result.provenance.get("input_source"))
    recording = f'<p class="evidence-origin">{html.escape(origin)}</p>' if origin else ''
    if selected:
        next_step = "Check this object physically, compare it with its peers, then record your finding in Decision."
    elif result.ranking:
        next_step = "No single object stands out. Check the leading objects physically and record your finding in Decision."
    else:
        next_step = "Correct the object matches in Review, or check the group physically and record your finding in Decision."
    source = correspondence_label(result)
    if confidence_text and not selected:
        detail += f'<p class="confidence-index">Ranking confidence index: {confidence:.3f}</p>'
    missing = [f"View {i}" for i, view in enumerate(session.views, 1)
               if selected and not any(o.box.view_id == view.view_id and o.box.identity_id == selected for o in result.observations)]
    missing_note = (f'<p class="missing-link">Selected object not linked in {html.escape(", ".join(missing))}; this does not establish absence.</p>'
                    if missing else '')
    return (f'<div class="result-card"><p class="result-eyebrow">Model result · machine ranking</p>'
            f'<span class="state-pill {result.review_state}">{state}</span>'
            f'<span class="mode-label">{html.escape(result.mode.title())} · {html.escape(method_label(result))}</span>'
            f'{recording}{detail}<p class="next-step">{html.escape(next_step)}</p>'
            f'<p class="identity-source">{html.escape(source)}</p>{missing_note}'
            '<p class="score-explanation">The ranking confidence index measures score separation for review routing; it is not a correctness probability. '
            f'{html.escape(state)} is a review-routing state, not an approval of any item.</p>'
            f'<ul>{reasons}</ul></div>')


def export_metadata(value):
    """Keep readable result metadata; omit server handles and neural arrays."""
    if isinstance(value, dict):
        return {str(key): export_metadata(item) for key, item in value.items()
                if str(key).lower() not in PRIVATE_KEYS and not str(key).lower().endswith(('_path', '_directory', '_root'))}
    if isinstance(value, (list, tuple)):
        return [export_metadata(item) for item in value]
    if isinstance(value, str):
        return '[local path omitted]' if value.startswith(('/Users/', '/home/', '/kaggle/', 'file://')) else value
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return None


def _table(headers, rows):
    cells = ''.join('<th>' + html.escape(str(value)) + '</th>' for value in headers)
    body = ''.join('<tr>' + ''.join('<td>' + html.escape('unavailable' if value is None else str(value)) + '</td>' for value in row) + '</tr>' for row in rows)
    return f'<div class="table-wrap"><table><thead><tr>{cells}</tr></thead><tbody>{body}</tbody></table></div>'


def result_record(session: Session) -> dict:
    """Public machine evidence shared by the JSON report and the saved case file."""
    result = session.result
    selected = selected_identity(result)
    selected_locations = [
        {"view_id": o.box.view_id, "object_id": o.box.object_id, "box_xyxy": list(o.box.xyxy)}
        for o in result.observations if selected and o.box.identity_id == selected
    ]
    matched_views = {row["view_id"] for row in selected_locations}
    missing_links = [
        {"view_id": view.view_id, "explanation": "Selected object not linked in this view; absence is not established."}
        for view in session.views if selected and view.view_id not in matched_views
    ]
    return {
        "mode": result.mode, "method_label": method_label(result),
        "correspondence": correspondence_label(result),
        "selected_identity_id": selected,
        "selected_locations": selected_locations,
        "missing_links": missing_links,
        "review_signals": export_metadata(result.provenance.get("review_signals", {})),
        "overlay_aliases": _session_aliases(session),
        "score_definitions": {
            "automatic_observation_score": "Mean within-view normalized rank of the available forward, P1 and P2 component scores.",
            "assisted_observation_score": "Object-bank foreground patch mismatch against other confirmed physical objects.",
            "normalized_rank": "Within-view relative rank; higher values are more unusual.",
            "combined_score": "The versioned method's fusion score; it is not a confidence probability.",
        },
        "review_state": result.review_state, "reasons": list(result.reasons),
        "method_version": result.method_version, "policy_version": result.policy_version,
        "provenance": export_metadata(result.provenance),
        "score_metadata": export_metadata(getattr(result, "score_metadata", {})),
        "views": [{"view_id": v.view_id, "sha256": v.digest, "width": v.image.width,
                   "height": v.image.height} for v in session.views],
        "ranking": [{"identity_id": r.identity_id, "relative_score": r.relative_score,
                     "view_ids": list(r.view_ids), "score_method": getattr(r, "score_method", ""),
                     "contributors": export_metadata(getattr(r, "contributors", ()))} for r in result.ranking],
        "observations": [{"view_id": o.box.view_id, "object_id": o.box.object_id,
                          "identity_id": o.box.identity_id, "box_xyxy": list(o.box.xyxy),
                          "score": _score(o), "score_method": getattr(o, "score_method", ""),
                          "p1_score": o.p1_score,
                          "component_scores": export_metadata(getattr(o, "component_scores", {})),
                          "normalized_rank": getattr(o, "normalized_rank", None),
                          "identity_status": getattr(o, "identity_status", ""),
                          "identity_source": getattr(o, "identity_source", ""),
                          "mask_available": o.mask is not None, "heatmap_available": o.heatmap is not None}
                         for o in result.observations],
        "claim_limits": list(CLAIM_LIMITS),
    }


REPORT_CSS = (
    'body{font:15px system-ui;color:#202124;max-width:1120px;margin:40px auto;padding:0 24px}h1{font-size:32px}h2{margin-top:30px}'
    '.result-card{border:1px solid #c9deda;border-left:4px solid #0f766e;padding:20px;border-radius:10px}.result-eyebrow,.card-eyebrow{margin:0 0 8px;font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:#525c65}'
    '.state-pill{padding:4px 8px;background:#eef5f4;border-radius:5px}.mode-label{margin-left:10px}.result-facts{display:flex;gap:32px;flex-wrap:wrap}.result-facts span{display:grid;gap:6px}'
    '.table-wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;margin:20px 0;font-size:13px}td,th{text-align:left;padding:10px;border-bottom:1px solid #ddd;vertical-align:top}th{background:#f3f5fb}'
    '.evidence-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px}figure{margin:0;break-inside:avoid}img{max-width:100%}figcaption,p{color:#525c65}'
    'pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f7f8;padding:16px;font-size:12px}.evidence-origin{font-weight:600;color:#725021}'
    '.human-card{border:1px solid #d8d4ef;border-left:4px solid #4f46a5;padding:20px;border-radius:10px}.decision-pill,.case-status{padding:4px 8px;border-radius:5px;background:#efedf9;color:#3d3780}'
    '.case-facts{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:8px 24px;padding:0;list-style:none}.case-facts b{display:block;font-size:12px;color:#525c65;font-weight:500}'
    '.follow-up{border:1px dashed #b9b4df;padding:16px 20px;border-radius:10px}.follow-up-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px}'
    '.timeline{padding-left:18px}.timeline li{margin:4px 0}.timeline time{color:#525c65;font-size:12px;margin-right:8px}'
    '.object-board h3{margin:24px 0 4px}.board-row{border-top:1px solid #e3e3e6;padding:12px 0}.board-head{display:flex;flex-wrap:wrap;align-items:center;gap:6px 10px}'
    '.alias-chip{color:#fff;padding:2px 8px;border-radius:4px;font-weight:600}.board-id{font-size:13px;color:#525c65}.board-meta{margin-left:auto;font-size:13px}'
    '.chip{font-size:12px;padding:2px 7px;border-radius:4px;background:#f1f1f3}.chip-selected{background:#e9f4f1;color:#115e59}.chip-tied,.chip-missing{background:#fbf1e1;color:#835c1c}'
    '.board-cells{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:12px;margin-top:10px}.board-cell figcaption{font-size:12px}'
    '.missing-slot{display:grid;place-items:center;min-height:120px;border:1px dashed #c9c9cf;border-radius:6px;color:#835c1c;background:#fafafa}'
    '@media print{body{margin:0}.table-wrap{overflow:visible}.evidence-grid{grid-template-columns:1fr 1fr}details{display:block}}'
)


def export_report(session: Session, title: str) -> tuple[str, str]:
    result = session.result
    if result is None:
        raise ValueError("Run an inspection before exporting a report.")
    discard_exports(session)
    session.export_directory = tempfile.mkdtemp(prefix="vpi-report-")
    directory = Path(session.export_directory)
    case = getattr(session, "case", None)
    record = {
        "schema_version": "visual-peer-inspection-report/v3",
        "title": title[:120].strip() or (f"Inspection case {case.case_id}" if case else "Untitled inspection"),
        **result_record(session),
    }
    details = decision = history = ""
    if case is not None:
        from .case import case_record, report_sections  # case.py builds on this module
        record["inspection_case"] = case_record(case)
        details, decision, history = report_sections(session)
    names = ((f"{case.case_id}-report.html", f"{case.case_id}-result.json") if case is not None
             else ("inspection.html", "inspection.json"))
    json_path = directory / names[1]
    json_path.write_text(json.dumps(record, indent=2, allow_nan=False), encoding="utf-8")
    sections = []
    for layer in ("Original", "Boxes", "Masks", "Objects on black", "Heatmaps"):
        figures = []
        for image, caption in evidence_gallery(session, "All objects", layer):
            image = image.copy()
            image.thumbnail((1200, 1200))
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
            figures.append(f'<figure><img alt="{html.escape(caption)}" src="data:image/png;base64,{encoded}"><figcaption>{html.escape(caption)}</figcaption></figure>')
        sections.append(f'<section><h2>{layer}</h2><div class="evidence-grid">' + ''.join(figures) + '</div></section>')
    metadata = {key: record[key] for key in ('method_version', 'policy_version', 'score_definitions', 'overlay_aliases', 'score_metadata', 'provenance')}
    limits = ''.join(f'<li>{html.escape(item)}</li>' for item in record['claim_limits'])
    document = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(record["title"])}</title><style>{REPORT_CSS}</style></head><body>'
                f'<p>VISUAL PEER INSPECTOR · INSPECTION REPORT</p><h1>{html.escape(record["title"])}</h1>{details}'
                f'<h2>Model result</h2>{summary_html(session)}{decision}{object_board_html(session)}'
                '<h2>Combined object ranking</h2>' + _table(RANK_HEADERS, ranking_rows(session))
                + '<h2>Scores and evidence in each view</h2>' + _table(SCORE_HEADERS, observation_rows(session))
                + f'<h2>How to read this report</h2><ul class="claim-limits">{limits}</ul>' + ''.join(sections) + history
                + '<h2>Method and run record</h2><pre>' + html.escape(json.dumps(metadata, indent=2, allow_nan=False)) + '</pre></body></html>')
    report_path = directory / names[0]
    report_path.write_text(document, encoding="utf-8")
    return str(report_path), str(json_path)
