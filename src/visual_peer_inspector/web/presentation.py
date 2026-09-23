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


def ranking_rows(session: Session):
    if session.result is None:
        return []
    rows, position, previous = [], 0, None
    names = _view_names(session)
    aliases = _session_aliases(session)
    for index, ranked in enumerate(session.result.ranking, 1):
        if previous is None or not math.isclose(previous, ranked.relative_score, abs_tol=1e-12, rel_tol=0):
            position = index
        previous = ranked.relative_score
        contributions = [names.get(item.get("view_id"), item.get("view_id", ""))
                         for item in getattr(ranked, "contributors", ()) if item.get("used_in_fusion", True)]
        rows.append([position, _legend_name(ranked.identity_id, aliases), _number(ranked.relative_score),
                     ", ".join(names.get(view, view) for view in ranked.view_ids),
                     ", ".join(contributions) if contributions else "not recorded"])
    return rows


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
    recorded = result.provenance.get("input_source") == "recorded_development"
    recording = '<p class="evidence-origin">Recorded development example · inference was not run on these uploads.</p>' if recorded else ''
    source = correspondence_label(result)
    if confidence_text and not selected:
        detail += f'<p class="confidence-index">Ranking confidence index: {confidence:.3f}</p>'
    missing = [f"View {i}" for i, view in enumerate(session.views, 1)
               if selected and not any(o.box.view_id == view.view_id and o.box.identity_id == selected for o in result.observations)]
    missing_note = (f'<p class="missing-link">Selected object not linked in {html.escape(", ".join(missing))}; this does not establish absence.</p>'
                    if missing else '')
    return (f'<div class="result-card"><span class="state-pill {result.review_state}">{state}</span>'
            f'<span class="mode-label">{html.escape(result.mode.title())} · {html.escape(method_label(result))}</span>'
            f'{recording}{detail}<p class="identity-source">{html.escape(source)}</p>{missing_note}'
            '<p class="score-explanation">The ranking confidence index measures score separation for review routing; it is not a correctness probability.</p>'
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


def export_report(session: Session, title: str) -> tuple[str, str]:
    result = session.result
    if result is None:
        raise ValueError("Run an inspection before exporting a report.")
    discard_exports(session)
    session.export_directory = tempfile.mkdtemp(prefix="vpi-report-")
    directory = Path(session.export_directory)
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
    record = {
        "schema_version": "visual-peer-inspection-report/v2",
        "title": title[:120].strip() or "Untitled inspection",
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
        "claim_limits": ["Scores describe relative unusualness, not defect probabilities.",
                         "Whole-object masks are not defect masks. Heatmaps show coarse feature mismatch.",
                         "An unlinked observation does not establish that an object is absent.",
                         "Dataset-supplied matches and human-confirmed matches are different sources of assistance."],
    }
    json_path = directory / "inspection.json"
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
    css = 'body{font:15px system-ui;color:#202124;max-width:1120px;margin:40px auto;padding:0 24px}h1{font-size:32px}h2{margin-top:30px}.result-card{border:1px solid #c9deda;border-left:4px solid #0f766e;padding:20px;border-radius:10px}.state-pill{padding:4px 8px;background:#eef5f4;border-radius:5px}.mode-label{margin-left:10px}.result-facts{display:flex;gap:32px;flex-wrap:wrap}.result-facts span{display:grid;gap:6px}.table-wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;margin:20px 0;font-size:13px}td,th{text-align:left;padding:10px;border-bottom:1px solid #ddd;vertical-align:top}th{background:#f3f5fb}.evidence-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px}figure{margin:0;break-inside:avoid}img{max-width:100%}figcaption,p{color:#525c65}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f7f8;padding:16px;font-size:12px}.evidence-origin{font-weight:600;color:#725021}@media print{body{margin:0}.table-wrap{overflow:visible}.evidence-grid{grid-template-columns:1fr 1fr}}'
    metadata = {key: record[key] for key in ('method_version', 'policy_version', 'score_definitions', 'overlay_aliases', 'score_metadata', 'provenance')}
    document = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(record["title"])}</title><style>{css}</style></head><body>'
                f'<p>VISUAL PEER INSPECTOR · INSPECTION REPORT</p><h1>{html.escape(record["title"])}</h1>{summary_html(session)}'
                '<h2>Combined object ranking</h2>' + _table(RANK_HEADERS, ranking_rows(session))
                + '<h2>Scores and evidence in each view</h2>' + _table(SCORE_HEADERS, observation_rows(session))
                + '<p>' + html.escape(' '.join(record['claim_limits'])) + '</p>' + ''.join(sections)
                + '<h2>Method and run record</h2><pre>' + html.escape(json.dumps(metadata, indent=2, allow_nan=False)) + '</pre></body></html>')
    report_path = directory / "inspection.html"
    report_path.write_text(document, encoding="utf-8")
    return str(report_path), str(json_path)
