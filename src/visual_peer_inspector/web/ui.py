"""Gradio UI that works independently of GPU inference and research notebooks."""

from __future__ import annotations

import logging
from pathlib import Path
from uuid import uuid4

import gradio as gr

from .core import (
    MIN_VIEWS,
    STAGES,
    Box,
    InspectionBackend,
    InspectionCancelled,
    InspectionError,
    InspectionRequest,
    Session,
    load_views,
    parse_corrections,
    validate_result,
)
from .help_panel import app_header
from .presentation import (
    RANK_HEADERS,
    SCORE_HEADERS,
    discard_exports,
    draw_boxes,
    evidence_gallery,
    export_report,
    identity_aliases,
    observation_rows,
    ranking_rows,
    summary_html,
)

CSS = Path(__file__).with_name("styles.css").read_text(encoding="utf-8")
LOG = logging.getLogger(__name__)
HEADERS = ["View ID", "Object ID", "Identity ID", "Left", "Top", "Right", "Bottom"]


def build_app(backend: InspectionBackend | None = None) -> gr.Blocks:
    connected = backend is not None

    def dispose_session(current):
        discard_exports(current)
        clear = getattr(backend, "clear_session", None)
        if current and callable(clear):
            clear(current.session_id, current.cache_token)

    theme = gr.themes.Base(
        primary_hue="teal",
        secondary_hue="indigo",
        neutral_hue="gray",
        font=["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
    ).set(
        body_background_fill="#ffffff",
        body_background_fill_dark="#ffffff",
        block_background_fill="#ffffff",
        block_background_fill_dark="#ffffff",
        body_text_color="#202124",
        body_text_color_dark="#202124",
        button_primary_background_fill="#0f766e",
        button_primary_background_fill_dark="#0f766e",
        button_primary_text_color="#ffffff",
        button_primary_text_color_dark="#ffffff",
        block_border_color="#e3e3e6",
        block_border_color_dark="#e3e3e6",
        input_background_fill="#ffffff",
        input_background_fill_dark="#ffffff",
        input_border_color="#dedee2",
        input_border_color_dark="#dedee2",
        input_radius="8px",
        block_radius="10px",
    )
    with gr.Blocks(
        title="Visual Peer Inspector", analytics_enabled=False, delete_cache=(60, 1800)
    ) as app:
        session = gr.State(
            Session(), time_to_live=1800, delete_callback=dispose_session
        )
        app_header(connected)
        with gr.Tabs(selected="scene", elem_id="inspection-tabs") as tabs:
            with gr.Tab("Images", id="scene"):
                gr.HTML(
                    '<div class="section-intro"><h2>Add your images</h2><p>Upload 2–5 photos of the same scene from different angles.</p></div>'
                )
                files = gr.File(
                    label="Images",
                    show_label=False,
                    file_count="multiple",
                    file_types=[".jpg", ".jpeg", ".png", ".webp"],
                    type="filepath",
                    height=190,
                    elem_id="upload-zone",
                )
                gr.Markdown(
                    "JPG, PNG or WebP · Max 10 MB and 12 MP per image",
                    elem_classes="quiet",
                )
                upload_status = gr.Markdown("", visible=False, elem_id="upload-status")
                previews = gr.Gallery(
                    label="Your views",
                    columns=3,
                    height=240,
                    object_fit="contain",
                    interactive=False,
                    buttons=["fullscreen"],
                    visible=False,
                )
                mode = gr.Radio(
                    choices=[("Automatic", "automatic"), ("Assisted", "assisted")],
                    value="automatic", label="Inspection mode", elem_id="inspection-mode",
                )
                gr.Markdown(
                    "Automatic combines predicted object evidence. Assisted lets you check which objects match before combining their scores.",
                    elem_classes="quiet",
                )
                with gr.Row(elem_id="image-actions"):
                    inspect = gr.Button(
                        "Inspect scene",
                        variant="primary",
                        interactive=False,
                        elem_id="inspect-button",
                        scale=0,
                        min_width=155,
                    )
                    reset = gr.Button(
                        "Clear images",
                        size="sm",
                        elem_id="reset-button",
                        scale=0,
                        min_width=115,
                    )
                    cancel_run = gr.Button(
                        "Cancel inspection",
                        visible=False,
                        elem_id="cancel-inspection",
                        scale=0,
                        min_width=135,
                    )
                readiness = gr.Markdown(
                    "" if connected else "Inspection is not connected yet.",
                    elem_id="readiness",
                )
            with gr.Tab("Results", id="results"):
                overview = gr.HTML(summary_html(Session()))
                with gr.Column(
                    visible=False, elem_id="result-controls"
                ) as result_controls:
                    with gr.Row():
                        identity = gr.Dropdown(
                            label="Object",
                            choices=["All objects"],
                            value="All objects",
                            interactive=False,
                        )
                        layer = gr.Radio(
                            ["Boxes", "Masks", "Objects on black", "Heatmaps", "Original"],
                            value="Boxes",
                            label="Show",
                            interactive=False,
                        )
                    evidence = gr.Gallery(
                        label="Evidence by view",
                        columns=2,
                        height=420,
                        object_fit="contain",
                        interactive=False,
                        buttons=["fullscreen"],
                    )
                    rankings = gr.Dataframe(
                        headers=RANK_HEADERS,
                        datatype=["number", "str", "number", "str", "str"],
                        value=[],
                        interactive=False,
                        label="Object ranking",
                        type="array",
                    )
                    scores = gr.Dataframe(
                        headers=SCORE_HEADERS,
                        datatype=["str", "str", "str", "str", "number", "number", "str", "str"],
                        value=[], interactive=False, label="Scores in each view", type="array",
                    )
                    gr.Markdown(
                        "Scores describe relative unusualness, not probability. Heatmaps are coarse feature evidence. “Not linked” means there is no supported match in that view.",
                        elem_classes="quiet",
                    )
                    gr.Markdown(
                        "**Observation scores:** Automatic averages the normalized ranks of its component scores. Assisted measures patch mismatch against other confirmed objects. Within-view ranks are used to combine views. Compact labels on images map to the full object IDs in the tables.",
                        elem_classes="quiet",
                    )
            with gr.Tab("Review", id="review"):
                review_notice = gr.Markdown(
                    "Run an inspection to review and correct objects.",
                    elem_classes="empty-notice",
                )
                with gr.Column(
                    visible=False, elem_id="review-controls"
                ) as review_controls:
                    gr.Markdown(
                        "Check the proposed IDs across all views. Use the same ID for the same physical object; leave an ID blank when unsure. You only confirm matches—the scorer selects the unusual object."
                    )
                    review_evidence = gr.Gallery(
                        label="Proposed object identities · check before confirming",
                        columns=3, height=300, object_fit="contain", interactive=False,
                        buttons=["fullscreen"],
                    )
                    review_view = gr.Dropdown(
                        label="View", choices=[], interactive=False
                    )
                    review_image = gr.Image(
                        label="Click two corners to add a box",
                        type="pil",
                        interactive=False,
                        height=340,
                        buttons=["fullscreen"],
                        alt_text="Selected view for box correction",
                    )
                    with gr.Row():
                        new_identity = gr.Textbox(
                            label="Identity for new box",
                            value="object_1",
                            interactive=False,
                        )
                        cancel_corner = gr.Button(
                            "Cancel drawing", size="sm", interactive=False
                        )
                    drawing_status = gr.Markdown("")
                    corrections = gr.Dataframe(
                        headers=HEADERS,
                        datatype=[
                            "str",
                            "str",
                            "str",
                            "number",
                            "number",
                            "number",
                            "number",
                        ],
                        type="array",
                        column_count=7,
                        row_count=0,
                        value=[],
                        interactive=False,
                        label="Boxes and identities",
                    )
                    gr.Markdown(
                        "Edit IDs in the table. IDs must be unique within each view. Box edits use original image pixels and refresh only the affected object features.",
                        elem_classes="quiet",
                    )
                    confirmed = gr.Checkbox(
                        label="I checked the object IDs across all views. Blank IDs remain unconfirmed.",
                        value=False, interactive=False,
                    )
                    rerun = gr.Button(
                        "Confirm identities and rerank", interactive=False, variant="primary"
                    )
            with gr.Tab("Report", id="report"):
                report_note = gr.Markdown(
                    "Run an inspection to create a report.", elem_classes="empty-notice"
                )
                with gr.Column(
                    visible=False, elem_id="report-controls"
                ) as report_controls:
                    title = gr.Textbox(
                        label="Report name (optional)",
                        placeholder="Untitled inspection",
                        max_length=120,
                    )
                    export = gr.Button(
                        "Prepare report", interactive=False, variant="primary"
                    )
                    gr.Markdown(
                        "Includes images, scores and review notes. Open the HTML report to print or save as PDF.",
                        elem_classes="quiet",
                    )
                    report_file = gr.File(
                        label="HTML report", interactive=False, visible=False
                    )
                    json_file = gr.File(
                        label="JSON result", interactive=False, visible=False
                    )
        gr.HTML(
            '<footer class="app-footer">Temporary uploads are cleared after 30 minutes.</footer>'
        )

        result_outputs = [
            result_controls,
            review_controls,
            report_controls,
            review_notice,
            overview,
            identity,
            layer,
            evidence,
            rankings,
            scores,
            review_evidence,
            review_view,
            review_image,
            new_identity,
            cancel_corner,
            drawing_status,
            corrections,
            confirmed,
            rerun,
            export,
            report_note,
            report_file,
            json_file,
        ]

        def empty_results():
            return {
                result_controls: gr.update(visible=False),
                review_controls: gr.update(visible=False),
                report_controls: gr.update(visible=False),
                review_notice: gr.update(visible=True),
                overview: summary_html(Session()),
                identity: gr.update(
                    choices=["All objects"], value="All objects", interactive=False
                ),
                layer: gr.update(value="Boxes", interactive=False),
                evidence: [],
                rankings: [],
                scores: [],
                review_evidence: [],
                review_view: gr.update(choices=[], value=None, interactive=False),
                review_image: None,
                new_identity: gr.update(value="object_1", interactive=False),
                cancel_corner: gr.update(interactive=False),
                drawing_status: "Run an inspection to start reviewing objects.",
                corrections: gr.update(value=[], interactive=False),
                confirmed: gr.update(value=False, interactive=False),
                rerun: gr.update(interactive=False),
                export: gr.update(interactive=False),
                report_note: "Run an inspection to create a report.",
                report_file: gr.update(value=None, visible=False),
                json_file: gr.update(value=None, visible=False),
            }

        all_outputs = [
            session,
            previews,
            upload_status,
            readiness,
            inspect,
            *result_outputs,
        ]

        def prepare_uploads(paths, current):
            selected_mode = current.mode
            dispose_session(current)
            current = Session(mode=selected_mode)
            message = ""
            try:
                current.views = load_views(paths)
                if current.views:
                    count = len(current.views)
                    message = f"**{count} / 5 images added.** " + (
                        "Add one more view to continue." if count < MIN_VIEWS else ""
                    )
            except InspectionError as error:
                message = f"**Check your images.** {error}"
            ready = len(current.views) >= MIN_VIEWS
            return {
                **empty_results(),
                session: current,
                previews: gr.update(
                    value=[
                        (v.image, f"View {i} · {v.image.width} × {v.image.height}")
                        for i, v in enumerate(current.views, 1)
                    ],
                    visible=bool(current.views),
                ),
                upload_status: gr.update(value=message, visible=bool(message)),
                inspect: gr.update(interactive=ready and connected),
                readiness: "" if connected else "Inspection is not connected yet.",
            }

        def start_over(current):
            return {
                **prepare_uploads(None, current),
                files: None,
                title: "",
                tabs: gr.update(selected="scene"),
            }

        def change_mode(current, value):
            current.mode = value
            return current, gr.update(value="Find objects for review" if value == "assisted" else "Inspect scene")

        def review_gallery(current, rows):
            try:
                boxes = parse_corrections(rows, current.views, require_multiple_views=False) if rows else ()
            except InspectionError:
                return [(view.image, f"View {i} · Correct the table to preview identities") for i, view in enumerate(current.views, 1)]
            return [
                (draw_boxes(view.image, [box for box in boxes if box.view_id == view.view_id],
                            aliases=identity_aliases(box.identity_id for box in boxes)),
                 f"View {i} · Proposed identities" )
                for i, view in enumerate(current.views, 1)
            ]

        progress_tracker = gr.Progress()

        def do_inspect(current, rows=None, progress=progress_tracker):
            if backend is None:
                raise gr.Error(
                    "Inspection is not connected yet. You can still add and preview your images."
                )
            if len(current.views) < MIN_VIEWS:
                raise gr.Error("Add at least two different views before inspecting.")
            try:
                corrected = (
                    parse_corrections(rows, current.views) if rows is not None else None
                )
            except InspectionError as error:
                raise gr.Error(str(error)) from error
            discard_exports(current)
            current.result = None
            current.pending_corner = None
            request = InspectionRequest(
                current.views, corrected, session_id=current.session_id,
                cache_token=current.cache_token,
                mode="assisted" if corrected is not None else current.mode,
            )
            current.active_request_id = request.request_id
            yield {
                **empty_results(),
                session: current,
                readiness: "Inspection in progress…",
                inspect: gr.update(interactive=False),
                files: gr.update(interactive=False),
                reset: gr.update(interactive=False),
                mode: gr.update(interactive=False),
                cancel_run: gr.update(
                    value="Cancel inspection",
                    visible=callable(getattr(backend, "cancel", None)),
                    interactive=True,
                ),
            }
            try:

                def report_progress(stage):
                    if stage not in STAGES:
                        raise ValueError("The backend reported an unknown stage.")
                    progress((STAGES.index(stage), len(STAGES)), desc=stage)

                result = backend.inspect(request, report_progress)
                validate_result(result, request)
                current.result = result
                current.cache_token = result.cache_token or current.cache_token
                suggestions = {
                    (row["view_id"], row["candidate_id"]): row.get("identity_id", "")
                    for row in result.score_metadata.get("suggested_identities", [])
                }
                current.corrections = [
                    [o.box.view_id, o.box.object_id,
                     suggestions.get((o.box.view_id, o.box.object_id), "" if o.identity_status == "unknown" else o.box.identity_id),
                     *o.box.xyxy]
                    for o in result.observations
                ]
                choices = ["All objects", *dict.fromkeys(
                    [r.identity_id for r in result.ranking] + [o.box.identity_id for o in result.observations]
                )]
                selected = "All objects"
                first_view = current.views[0]
                yield {
                    session: current,
                    result_controls: gr.update(visible=True),
                    review_controls: gr.update(visible=True),
                    report_controls: gr.update(visible=True),
                    review_notice: gr.update(value="Check the proposed object identities across your views.", visible=False),
                    overview: summary_html(current),
                    readiness: "",
                    identity: gr.update(
                        choices=choices, value=selected, interactive=True
                    ),
                    layer: gr.update(interactive=True),
                    evidence: evidence_gallery(current, selected, "Boxes"),
                    rankings: ranking_rows(current),
                    scores: observation_rows(current),
                    review_evidence: review_gallery(current, current.corrections),
                    review_view: gr.update(
                        choices=[
                            (f"View {i}", v.view_id)
                            for i, v in enumerate(current.views, 1)
                        ],
                        value=first_view.view_id,
                        interactive=True,
                    ),
                    review_image: review_preview(current, first_view.view_id, current.corrections)[1],
                    corrections: gr.update(value=current.corrections, interactive=True),
                    new_identity: gr.update(interactive=True),
                    cancel_corner: gr.update(interactive=True),
                    drawing_status: "Click two opposite corners on the image to add a box, or edit the table below.",
                    confirmed: gr.update(value=False, interactive=True),
                    rerun: gr.update(interactive=False),
                    export: gr.update(interactive=True),
                    report_note: f"Ready to export this {result.mode} inspection.",
                    tabs: gr.update(selected="review" if request.decision_mode == "assisted" and corrected is None else "results"),
                }
            except Exception as error:  # noqa: BLE001 - contain failures at the backend boundary
                # Do not expose model errors, file paths or uploaded data in UI/logs.
                if not isinstance(error, InspectionCancelled):
                    LOG.warning("Inspection failed (%s)", type(error).__name__)
                else:
                    current.cache_token = None
                message = (
                    str(error)
                    if isinstance(error, InspectionError)
                    else "The inspection could not finish. Your images are still here; try again."
                )
                yield {
                    **empty_results(),
                    session: current,
                    readiness: message
                    if isinstance(error, InspectionCancelled)
                    else f"**Inspection failed.** {message}",
                    tabs: gr.update(selected="scene"),
                }
            finally:
                current.active_request_id = None
                yield {
                    session: current,
                    inspect: gr.update(interactive=True),
                    files: gr.update(interactive=True),
                    reset: gr.update(interactive=True),
                    mode: gr.update(interactive=True),
                    cancel_run: gr.update(visible=False),
                }

        def cancel_inspection(current):
            cancel = getattr(backend, "cancel", None)
            if (
                current.active_request_id
                and callable(cancel)
                and cancel(current.active_request_id)
            ):
                return gr.update(value="Cancelling…", interactive=False)
            return gr.update(value="Cancel inspection", interactive=True)

        def confirm_and_rerun(current, rows, checked):
            if not checked:
                raise gr.Error("Check the object identities across all views, then confirm them before reranking.")
            yield from do_inspect(current, rows)

        def edit_identities(current, view_id, rows):
            current, preview, notice = review_preview(current, view_id, rows)
            return current, preview, notice, review_gallery(current, rows), False, gr.update(interactive=False)

        cancel_run.click(
            cancel_inspection,
            [session],
            [cancel_run],
            queue=False,
            api_visibility="private",
        )

        def review_preview(current, view_id, rows):
            current.pending_corner = None
            view = next((v for v in current.views if v.view_id == view_id), None)
            if view is None:
                return current, None, "Choose a view."
            try:
                boxes = (
                    parse_corrections(rows, current.views, require_multiple_views=False)
                    if rows
                    else ()
                )
            except InspectionError as error:
                return current, view.image, str(error)
            return (
                current,
                draw_boxes(view.image, [b for b in boxes if b.view_id == view_id],
                           aliases=identity_aliases(b.identity_id for b in boxes)),
                "Click the first corner of a new box.",
            )

        def draw_corner(current, view_id, identity_id, rows, event: gr.SelectData):
            if current.result is None:
                raise gr.Error("Run an inspection before correcting objects.")
            view = next(v for v in current.views if v.view_id == view_id)
            point = tuple(int(value) for value in event.index)
            if current.pending_corner is None:
                current.pending_corner = point
                return (
                    current,
                    gr.skip(),
                    gr.skip(),
                    "First corner set. Click the opposite corner.",
                )
            previous = current.pending_corner
            current.pending_corner = None
            xyxy = (
                min(previous[0], point[0]),
                min(previous[1], point[1]),
                max(previous[0], point[0]),
                max(previous[1], point[1]),
            )
            object_id = f"manual_{uuid4().hex[:8]}"
            box = Box(view_id, object_id, identity_id.strip(), xyxy)
            updated = [list(row) for row in rows] + [
                [view_id, object_id, box.identity_id, *xyxy]
            ]
            try:
                boxes = parse_corrections(
                    updated, current.views, require_multiple_views=False
                )
            except InspectionError as error:
                return current, gr.skip(), gr.skip(), str(error)
            return (
                current,
                updated,
                draw_boxes(view.image, [b for b in boxes if b.view_id == view_id],
                           aliases=identity_aliases(b.identity_id for b in boxes)),
                "Box added. Edit its identity in the table if needed.",
            )

        def cancel_drawing(current):
            current.pending_corner = None
            return current, "Drawing cancelled. Click a first corner to start again."

        def download(current, name):
            if current.result is None:
                raise gr.Error("Run an inspection before preparing a report.")
            report, data = export_report(current, name)
            return (
                current,
                gr.update(value=report, visible=True),
                gr.update(value=data, visible=True),
                "Report ready. Downloads include the current result and review notes.",
            )

        # One shared queue also prevents reset/upload callbacks racing inference.
        event_options = {
            "concurrency_limit": 1,
            "concurrency_id": "inspection",
            "api_visibility": "private",
        }
        files.change(prepare_uploads, [files, session], all_outputs, **event_options)
        mode.change(change_mode, [session, mode], [session, inspect], **event_options)
        reset.click(
            start_over, [session], [*all_outputs, files, title, tabs], **event_options
        )
        run_outputs = [*all_outputs, files, reset, tabs, cancel_run, mode]
        inspect.click(do_inspect, [session], run_outputs, **event_options)
        rerun.click(confirm_and_rerun, [session, corrections, confirmed], run_outputs, **event_options)
        confirmed.change(lambda checked: gr.update(interactive=bool(checked)), [confirmed], [rerun], **event_options)
        corrections.change(
            edit_identities, [session, review_view, corrections],
            [session, review_image, drawing_status, review_evidence, confirmed, rerun], **event_options,
        )
        gr.on(
            [identity.change, layer.change],
            evidence_gallery,
            [session, identity, layer],
            evidence,
            **event_options,
        )
        review_view.change(
            review_preview,
            [session, review_view, corrections],
            [session, review_image, drawing_status],
            **event_options,
        )
        review_image.select(
            draw_corner,
            [session, review_view, new_identity, corrections],
            [session, corrections, review_image, drawing_status],
            **event_options,
        )
        cancel_corner.click(
            cancel_drawing, [session], [session, drawing_status], **event_options
        )
        export.click(
            download,
            [session, title],
            [session, report_file, json_file, report_note],
            **event_options,
        )
    app.queue(max_size=8, default_concurrency_limit=1)
    # Gradio 6 accepts theme and CSS at launch, not Blocks construction.
    app.vpi_theme = theme
    return app
