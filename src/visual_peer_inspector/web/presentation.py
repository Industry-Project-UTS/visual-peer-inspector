"""Evidence rendering and downloadable, self-contained inspection reports."""

from __future__ import annotations

import base64
import html
import io
import json
import shutil
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

from .core import Box, Session

LABELS = {
    "result_ready": "Result ready",
    "review_recommended": "Review recommended",
    "insufficient_evidence": "Insufficient evidence",
}


def discard_exports(session: Session | None) -> None:
    if session and session.export_directory:
        shutil.rmtree(session.export_directory, ignore_errors=True)
        session.export_directory = None


def draw_boxes(
    image: Image.Image, boxes: list[Box], winner: str | None = None
) -> Image.Image:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    width = max(2, round(min(image.size) / 250))
    for box in boxes:
        colour = "#e88738" if box.identity_id == winner else "#8ba1e8"
        draw.rectangle(box.xyxy, outline=colour, width=width)
        x, y = box.xyxy[:2]
        label = f"{box.object_id} / {box.identity_id}"
        bounds = draw.textbbox((x + 4, y + 4), label)
        draw.rectangle((x, y, bounds[2] + 4, bounds[3] + 4), fill=colour)
        draw.text((x + 4, y + 4), label, fill="#202124")
    return canvas


def evidence_gallery(session: Session, identity: str | None, layer: str):
    if not session.result:
        return []
    images = []
    for index, view in enumerate(session.views, 1):
        observations = [
            o
            for o in session.result.observations
            if o.box.view_id == view.view_id
            and (identity == "All objects" or o.box.identity_id == identity)
        ]
        canvas = view.image.copy()
        note = ""
        if not observations:
            note = " · No matching observation"
        if layer in {"Masks", "Heatmaps"}:
            applied = False
            for observation in observations:
                overlay = observation.mask if layer == "Masks" else observation.heatmap
                if overlay is None:
                    continue
                if layer == "Masks":
                    colour = Image.new("RGB", canvas.size, "#8ba1e8")
                    alpha = overlay.convert("L").point(lambda value: int(value * 0.45))
                    canvas = Image.composite(colour, canvas, alpha)
                else:
                    canvas = Image.blend(canvas, overlay.convert("RGB"), 0.45)
                applied = True
            if observations and not applied:
                note = " · This evidence layer is unavailable"
        winner = (
            session.result.ranking[0].identity_id if session.result.ranking else None
        )
        if layer != "Original":
            canvas = draw_boxes(canvas, [o.box for o in observations], winner)
        images.append((canvas, f"View {index}{note}"))
    return images


def summary_html(session: Session) -> str:
    result = session.result
    if result is None:
        return '<p class="empty-result">Run an inspection to see results.</p>'
    state = LABELS[result.review_state]
    reasons = "".join(f"<li>{html.escape(reason)}</li>" for reason in result.reasons)
    if result.ranking:
        winner = result.ranking[0]
        detail = f'<h2>{html.escape(winner.identity_id)}</h2><p>Most unusual relative to its peers</p><div class="result-facts"><span><b>{winner.relative_score:.3f}</b>Relative score</span><span><b>{len(winner.view_ids)} / {len(session.views)}</b>Supporting views</span></div>'
    else:
        detail = "<h2>No reliable ranking</h2><p>Review the available evidence or try clearer views.</p>"
    return f'<div class="result-card"><span class="state-pill {result.review_state}">{state}</span><span class="mode-label">{html.escape(result.mode.title())}</span>{detail}<ul>{reasons}</ul></div>'


def export_report(session: Session, title: str) -> tuple[str, str]:
    result = session.result
    if result is None:
        raise ValueError("Run an inspection before exporting a report.")
    discard_exports(session)
    session.export_directory = tempfile.mkdtemp(prefix="vpi-report-")
    directory = Path(session.export_directory)
    record = {
        "schema_version": "visual-peer-inspection-report/v1",
        "title": title[:120].strip() or "Untitled inspection",
        "mode": result.mode,
        "review_state": result.review_state,
        "reasons": list(result.reasons),
        "method_version": result.method_version,
        "policy_version": result.policy_version,
        "provenance": result.provenance,
        "views": [
            {
                "view_id": v.view_id,
                "sha256": v.digest,
                "width": v.image.width,
                "height": v.image.height,
            }
            for v in session.views
        ],
        "ranking": [
            {
                "identity_id": r.identity_id,
                "relative_score": r.relative_score,
                "view_ids": list(r.view_ids),
            }
            for r in result.ranking
        ],
        "observations": [
            {
                "view_id": o.box.view_id,
                "object_id": o.box.object_id,
                "identity_id": o.box.identity_id,
                "box_xyxy": list(o.box.xyxy),
                "p1_score": o.p1_score,
                "mask_available": o.mask is not None,
                "heatmap_available": o.heatmap is not None,
            }
            for o in result.observations
        ],
        "claim_limits": [
            "Scores describe relative unusualness, not defect probabilities.",
            "Whole-object masks are not defect masks.",
            "Heatmaps show coarse feature mismatch.",
        ],
    }
    json_path = directory / "inspection.json"
    json_path.write_text(
        json.dumps(record, indent=2, allow_nan=False), encoding="utf-8"
    )
    figures = []
    for layer in ("Boxes", "Masks", "Heatmaps"):
        for image, caption in evidence_gallery(session, "All objects", layer):
            image.thumbnail((1200, 1200))
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
            figures.append(
                f'<figure><img alt="{html.escape(caption)}" src="data:image/png;base64,{encoded}"><figcaption>{html.escape(caption)} · {layer}</figcaption></figure>'
            )
    rows = "".join(
        f"<tr><td>{html.escape(r.identity_id)}</td><td>{r.relative_score:.4f}</td><td>{len(r.view_ids)} / {len(session.views)}</td></tr>"
        for r in result.ranking
    )
    document = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(record["title"])}</title><style>body{{font:16px system-ui;color:#202124;max-width:1000px;margin:48px auto;padding:0 24px}}h1{{font-size:36px}}.result-card{{border:1px solid #dedee2;padding:24px;border-radius:12px}}.result-facts{{display:flex;gap:40px}}.result-facts span{{display:grid}}table{{border-collapse:collapse;width:100%;margin:24px 0}}td,th{{text-align:left;padding:12px;border-bottom:1px solid #ddd}}figure{{margin:24px 0;break-inside:avoid}}img{{max-width:100%}}figcaption,p{{color:#626268}}@media print{{body{{margin:0}}}}</style><p>VISUAL PEER INSPECTOR · INSPECTION REPORT</p><h1>{html.escape(record["title"])}</h1>{summary_html(session)}<table><thead><tr><th>Object identity</th><th>Relative score</th><th>Supporting views</th></tr></thead><tbody>{rows}</tbody></table><p>{" ".join(record["claim_limits"])}</p>{"".join(figures)}<p>Method: {html.escape(result.method_version)} · Review policy: {html.escape(result.policy_version)}</p></html>"""
    report_path = directory / "inspection.html"
    report_path.write_text(document, encoding="utf-8")
    return str(report_path), str(json_path)
