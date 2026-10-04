"""Gradio UI that works independently of GPU inference and research notebooks."""

from __future__ import annotations

import logging
import os
import threading
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
from .case import (
    ACTIONS,
    SEPARATION_NOTE,
    case_banner_html,
    case_details_html,
    case_status,
    decision_card_html,
    decision_choices,
    ensure_case,
    follow_up_html,
    log,
    new_case,
    open_case,
    opened_case_html,
    record_decision,
    record_inspection,
    save_case,
    start_follow_up,
    timeline_html,
    update_details,
)
from .help_panel import app_header
from .presentation import (
    RANK_HEADERS,
    SCORE_HEADERS,
    discard_exports,
    draw_boxes,
    evidence_gallery,
    evidence_note,
    identity_aliases,
    object_board_html,
    observation_rows,
    ranking_rows,
    summary_html,
)

CSS = Path(__file__).with_name("styles.css").read_text(encoding="utf-8")
LOG = logging.getLogger(__name__)
HEADERS = ["View", "Object ID", "Identity ID", "Left", "Top", "Right", "Bottom"]
DECISION_HINT = '<p class="empty-notice">Run an inspection, check the items physically, then record your decision here.</p>'


def build_app(backend: InspectionBackend | None = None) -> gr.Blocks:
    connected = backend is not None
    browser_jobs = {}
    browser_jobs_lock = threading.Lock()

    def browser_disconnected(request: gr.Request):
        # Refresh/close must release only this browser session's GPU work.
        with browser_jobs_lock:
            job = browser_jobs.pop(request.session_hash, None)
        if job:
            request_id, disconnected = job
            disconnected.set()
            cancel = getattr(backend, "cancel", None)
            if callable(cancel):
                cancel(request_id)

    def dispose_session(current):
        discard_exports(current)
        clear = getattr(backend, "clear_session", None)
        if current and callable(clear):
            clear(current.session_id, current.cache_token)

    theme = gr.themes.Base(
        primary_hue="orange",
        secondary_hue="red",
        neutral_hue="stone",
        font=["Avenir Next", "IBM Plex Sans", "ui-sans-serif", "system-ui", "sans-serif"],
    ).set(
        body_background_fill="#fbf9f7",
        body_background_fill_dark="#fbf9f7",
        block_background_fill="#fffdfb",
        block_background_fill_dark="#fffdfb",
        body_text_color="#2a1110",
        body_text_color_dark="#2a1110",
        button_primary_background_fill="#680904",
        button_primary_background_fill_dark="#680904",
        button_primary_text_color="#ffffff",
        button_primary_text_color_dark="#ffffff",
        block_border_color="#efd2ca",
        block_border_color_dark="#efd2ca",
        input_background_fill="#fffdfb",
        input_background_fill_dark="#fffdfb",
        input_border_color="#e9c9c0",
        input_border_color_dark="#e9c9c0",
        input_radius="10px",
        block_radius="14px",
    )
    with gr.Blocks(
        title="Visual Peer Inspector", analytics_enabled=False, delete_cache=(60, 1800)
    ) as app:
        session = gr.State(
            Session(), time_to_live=1800, delete_callback=dispose_session
        )
        # A reopened case is kept apart from the live session and never reaches a backend.
        opened = gr.State(None, time_to_live=1800)
        app_header(connected)
        with gr.Tabs(selected="scene", elem_id="inspection-tabs") as tabs:
            # Keep inactive controls mounted so cross-tab updates retain their values.
            with gr.Tab("Images", id="scene", render_children=True):
                gr.HTML(
                    '<div class="section-intro"><h2>New inspection</h2><p>Upload 2–5 clear images of the same group of objects from different angles.</p></div>'
                )
                with gr.Row(elem_id="case-details"):
                    item = gr.Textbox(
                        label="What are you inspecting?", placeholder="e.g. a group of 500 ml bottles",
                        max_length=120, max_lines=1, elem_id="case-item", scale=2, min_width=200,
                    )
                    operator = gr.Textbox(label="Your name (for the final decision)", max_length=80, max_lines=1, elem_id="case-operator", min_width=160)
                with gr.Accordion("Reference details (optional)", open=False, elem_id="case-extra"):
                    case_id = gr.Textbox(label="Case ID", max_length=40, max_lines=1, elem_id="case-id", min_width=160)
                    batch = gr.Textbox(
                        label="Batch or reference (optional)", max_length=80, max_lines=1, elem_id="case-batch", min_width=160
                    )
                with gr.Row(elem_id="case-actions"):
                    case_banner = gr.HTML("", elem_id="case-banner")
                    new_case_button = gr.Button(
                        "Start new case", size="sm", elem_id="new-case-button", scale=0, min_width=130
                    )
                files = gr.File(
                    label="Images",
                    show_label=False,
                    file_count="multiple",
                    file_types=[".jpg", ".jpeg", ".png", ".webp"],
                    type="filepath",
                    height="auto",
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
                    height="auto",
                    object_fit="contain",
                    interactive=False,
                    buttons=["fullscreen"],
                    visible=False,
                    elem_id="view-previews",
                )
                mode = gr.Radio(
                    choices=[("Automatic", "automatic"), ("Assisted", "assisted")],
                    value="automatic", label="Inspection mode", elem_id="inspection-mode",
                )
                gr.Markdown(
                    "Automatic uses predicted matches. Assisted asks you to confirm which boxes show the same physical object before ranking.",
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
            with gr.Tab("Results", id="results", render_children=True):
                overview = gr.HTML(summary_html(Session()))
                with gr.Column(
                    visible=False, elem_id="result-controls"
                ) as result_controls:
                    board = gr.HTML("", elem_id="object-board")
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
                        height="auto",
                        object_fit="contain",
                        interactive=False,
                        buttons=["fullscreen"],
                        elem_id="result-evidence",
                    )
                    layer_note = gr.Markdown(
                        "",
                        elem_classes="quiet",
                    )
                    with gr.Accordion("Score tables and components", open=False, elem_id="score-details"):
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
                            "**Observation scores:** Automatic averages the normalized ranks of its component scores. Assisted measures patch mismatch against other confirmed objects. Within-view ranks are used to combine views. Compact labels on images map to the full object IDs in the tables.",
                            elem_classes="quiet",
                        )
            with gr.Tab("Review", id="review", render_children=True):
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
                        columns=3, height="auto", object_fit="contain", interactive=False,
                        buttons=["fullscreen"],
                        elem_id="review-evidence",
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
            with gr.Tab("Decision", id="decision", render_children=True):
                gr.Markdown("**Optional human record.** Record what you found after checking the objects. This does not change the model ranking; you can save a report without it.", elem_classes="quiet")
                decision_case = gr.HTML("", elem_id="decision-case")
                decision_notice = gr.HTML(
                    DECISION_HINT
                )
                with gr.Column(visible=False, elem_id="decision-controls") as decision_controls:
                    with gr.Row(elem_id="decision-records", equal_height=False):
                        machine_card = gr.HTML("", elem_id="machine-card", min_width=280)
                        human_card = gr.HTML("", elem_id="human-card", min_width=280)
                    gr.HTML(
                        '<div class="section-intro compact"><h2>Record your finding</h2><p>'
                        + SEPARATION_NOTE + '</p></div>'
                    )
                    decision_operator = gr.Textbox(
                        label="Your name (required to record a decision)",
                        max_length=80, max_lines=1, elem_id="decision-operator",
                    )
                    decision_object = gr.Dropdown(
                        label="Object your finding refers to", choices=[], value=None, elem_id="decision-object"
                    )
                    decision_action = gr.Radio(
                        choices=[(label, code) for code, label in ACTIONS.items()], value=None,
                        label="Action", elem_id="decision-action",
                    )
                    finding = gr.Textbox(
                        label="Finding", placeholder="What did you observe on the item?", max_length=300, lines=2, max_lines=4,
                        elem_id="decision-finding",
                    )
                    reason = gr.Textbox(
                        label="Reason", placeholder="Why this action?", max_length=300, lines=2, max_lines=4, elem_id="decision-reason"
                    )
                    responsible = gr.Textbox(
                        label="Responsible person (optional)", max_length=80, max_lines=1, elem_id="decision-responsible"
                    )
                    record = gr.Button("Record decision", variant="primary", elem_id="record-decision")
                    decision_status = gr.Markdown("", elem_id="decision-status")
                follow_up_panel = gr.HTML("", elem_id="follow-up-panel")
                timeline = gr.HTML("", elem_id="case-timeline")
            with gr.Tab("Report", id="report", render_children=True):
                report_note = gr.Markdown(
                    "Run an inspection to create a report.", elem_classes="empty-notice"
                )
                with gr.Column(
                    visible=False, elem_id="report-controls"
                ) as report_controls:
                    title = gr.Textbox(
                        label="Report title (optional)",
                        placeholder="Defaults to the case ID",
                        max_length=120,
                    )
                    export = gr.Button(
                        "Save case and report", interactive=False, variant="primary"
                    )
                    gr.Markdown(
                        "Download the ZIP to reopen this case later. The HTML report is for reading or printing; JSON contains the result without images.",
                        elem_classes="quiet",
                    )
                    case_file = gr.File(
                        label="Inspection case file", interactive=False, visible=False
                    )
                    report_file = gr.File(
                        label="HTML report", interactive=False, visible=False
                    )
                    json_file = gr.File(
                        label="JSON result", interactive=False, visible=False
                    )
            with gr.Tab("Saved cases", id="cases", render_children=True):
                gr.HTML(
                    '<div class="section-intro"><h2>Open a saved case</h2><p>Upload a case ZIP you downloaded earlier. It opens for review without running the models again.</p></div>'
                )
                case_upload = gr.File(
                    label="Saved case file (.zip)", file_count="single", file_types=[".zip"],
                    type="filepath", elem_id="case-upload",
                )
                opened_status = gr.Markdown("", elem_id="opened-status")
                with gr.Column(visible=False, elem_id="opened-case") as opened_view:
                    opened_header = gr.HTML("")
                    with gr.Row(equal_height=False, elem_id="opened-records"):
                        opened_machine = gr.HTML("", min_width=280)
                        opened_human = gr.HTML("", min_width=280)
                    opened_follow_up = gr.HTML("")
                    opened_board = gr.HTML("")
                    with gr.Accordion("Evidence layers and score tables", open=False):
                        opened_layer = gr.Radio(
                            ["Boxes", "Masks", "Objects on black", "Heatmaps", "Original"],
                            value="Boxes", label="Show",
                        )
                        opened_evidence = gr.Gallery(
                            label="Saved evidence by view", columns=2, height="auto", object_fit="contain",
                            interactive=False, buttons=["fullscreen"], elem_id="opened-evidence",
                        )
                        opened_layer_note = gr.Markdown("", elem_classes="quiet")
                        opened_rankings = gr.Dataframe(
                            headers=RANK_HEADERS, datatype=["number", "str", "number", "str", "str"],
                            value=[], interactive=False, label="Object ranking", type="array",
                        )
                        opened_scores = gr.Dataframe(
                            headers=SCORE_HEADERS,
                            datatype=["str", "str", "str", "str", "number", "number", "str", "str"],
                            value=[], interactive=False, label="Scores in each view", type="array",
                        )
                    opened_timeline = gr.HTML("")
                    follow_up_button = gr.Button(
                        "Start a follow-up inspection", variant="primary", elem_id="follow-up-button"
                    )
                    gr.Markdown(
                        "Starts a new case linked to this one and clears any unsaved inspection. Photograph the full group of items again; scores from the two cases are not compared.",
                        elem_classes="quiet",
                    )
        readiness = gr.Markdown(
            "" if connected else "Inspection is not connected yet.", elem_id="readiness"
        )
        cancel_run = gr.Button(
            "Cancel inspection", visible=False, elem_id="cancel-inspection"
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
            layer_note,
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
            case_file,
            board,
            decision_notice,
            decision_controls,
            machine_card,
            decision_object,
            decision_status,
            decision_operator,
            case_banner,
            decision_case,
            human_card,
            follow_up_panel,
            timeline,
        ]
        case_fields = [case_id, item, batch, operator]
        case_outputs = [case_banner, decision_case, human_card, follow_up_panel, timeline, decision_operator]

        def case_view(current, message=""):
            case = ensure_case(current)
            return {
                case_banner: case_banner_html(case, message),
                decision_case: case_details_html(case),
                human_card: decision_card_html(case, current.result),
                follow_up_panel: follow_up_html(case, current.result),
                timeline: timeline_html(case),
                decision_operator: case.operator,
            }

        def empty_results():
            return {
                result_controls: gr.update(visible=False),
                review_controls: gr.update(visible=False),
                report_controls: gr.update(visible=False),
                review_notice: gr.update(visible=True),
                board: "",
                decision_notice: DECISION_HINT,
                decision_controls: gr.update(visible=False),
                machine_card: "",
                decision_object: gr.update(choices=[], value=None),
                decision_status: "",
                case_file: gr.update(value=None, visible=False),
                overview: summary_html(Session()),
                identity: gr.update(
                    choices=["All objects"], value="All objects", interactive=False
                ),
                layer: gr.update(value="Boxes", interactive=False),
                evidence: [],
                layer_note: "",
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

        def prepare_uploads(paths, current, *details):
            selected_mode = current.mode
            if details:
                apply_details(current, details)
            case = ensure_case(current)
            dispose_session(current)
            # New photos start a new capture within the same case.
            current = Session(mode=selected_mode, case=case)
            case.model_boxes = {}
            message = ""
            try:
                current.views = load_views(paths)
                if current.views:
                    count = len(current.views)
                    message = f"**{count} / 5 images added.** " + (
                        "Add one more view to continue." if count < MIN_VIEWS else ""
                    )
                    log(case, f"Photos loaded: {count} view{'s' if count != 1 else ''}")
            except InspectionError as error:
                message = f"**Check your images.** {error}"
            ready = len(current.views) >= MIN_VIEWS
            return {
                **empty_results(),
                **case_view(current),
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
            if current.views:
                log(ensure_case(current), "Photos cleared")
            return {
                **prepare_uploads(None, current),
                files: None,
                title: "",
                tabs: gr.update(selected="scene"),
            }

        def fresh_case(current, case):
            output = start_over(current)
            fresh = output[session]
            fresh.case = case
            return {
                **output,
                **case_view(fresh),
                case_id: case.case_id,
                item: case.item,
                batch: case.batch,
                operator: case.operator,
            }

        def start_new_case(current, *details):
            # Keep the operator name; everything else belongs to the previous case.
            if details:
                apply_details(current, details)
            return fresh_case(current, new_case(operator=current.case.operator if current.case else ""))

        def initialise(current):
            case = ensure_case(current)
            return {
                session: current,
                case_id: case.case_id,
                item: case.item,
                batch: case.batch,
                operator: case.operator,
                **case_view(current),
            }

        def apply_details(current, values):
            """Store the case fields as shown; an invalid case ID keeps the previous one."""
            case = ensure_case(current)
            try:
                update_details(case, *values)
            except InspectionError as error:
                try:
                    update_details(case, case.case_id, *values[1:])
                except InspectionError:
                    pass
                return str(error)
            return ""

        def edit_case(current, *values):
            message = apply_details(current, values)
            return {session: current, case_id: current.case.case_id, **case_view(current, message)}

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

        def prepare_inspection(current, rows=None, checked=None):
            if backend is None:
                raise gr.Error("Inspection is not connected yet.")
            if len(current.views) < MIN_VIEWS:
                raise gr.Error("Add at least two different views before inspecting.")
            if rows is not None:
                if not checked:
                    raise gr.Error("Check the object identities across all views, then confirm them before reranking.")
                try:
                    parse_corrections(rows, current.views)
                except InspectionError as error:
                    raise gr.Error(str(error)) from error
            # Show cancellation before the model callback returns its complete result.
            return {
                **case_view(current), session: current,
                overview: summary_html(Session()), board: "", machine_card: "",
                # Hiding a mounted parent resets child values in Gradio's client.
                # Keep correction inputs mounted while clearing prior evidence.
                evidence: [], rankings: [], scores: [],
                case_file: gr.update(value=None, visible=False),
                report_file: gr.update(value=None, visible=False),
                json_file: gr.update(value=None, visible=False),
                export: gr.update(interactive=False),
                report_note: "Inspection in progress. Save a fresh report after reranking.",
                readiness: "Inspection in progress…",
                inspect: gr.update(interactive=False), files: gr.update(interactive=False),
                reset: gr.update(interactive=False), mode: gr.update(interactive=False),
                cancel_run: gr.update(value="Cancel inspection", interactive=True,
                                      visible=callable(getattr(backend, "cancel", None))),
            }

        def do_inspect(current, rows=None, progress=progress_tracker, browser_request: gr.Request = None):
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
            disconnected = threading.Event()
            browser_id = getattr(browser_request, "session_hash", None)
            if browser_id:
                with browser_jobs_lock:
                    browser_jobs[browser_id] = (request.request_id, disconnected)
            completion_update = {}
            try:

                def report_progress(stage):
                    if disconnected.is_set():
                        raise InspectionCancelled("Inspection cancelled because its browser session closed.")
                    if stage not in STAGES:
                        raise ValueError("The backend reported an unknown stage.")
                    detail = getattr(stage, 'detail', '')
                    elapsed = int(getattr(stage, 'elapsed_seconds', 0))
                    label = ' · '.join(part for part in (str(stage), detail, f'{elapsed // 60}:{elapsed % 60:02d} elapsed') if part)
                    # Gradio hides the description when progress has no total.
                    # This counter measures completed stages, never remaining time.
                    progress((STAGES.index(stage), len(STAGES)), desc=label, unit="stages")

                result = backend.inspect(request, report_progress)
                validate_result(result, request)
                current.result = result
                current.cache_token = result.cache_token or current.cache_token
                record_inspection(ensure_case(current), result, len(current.views), corrected)
                suggestions = {
                    (row["view_id"], row["candidate_id"]): row.get("identity_id", "")
                    for row in result.score_metadata.get("suggested_identities", [])
                }
                view_labels = {view.view_id: f"View {i}" for i, view in enumerate(current.views, 1)}
                view_order = {view.view_id: i for i, view in enumerate(current.views)}
                current.corrections = [
                    [view_labels[o.box.view_id], o.box.object_id,
                     suggestions.get((o.box.view_id, o.box.object_id), "" if o.identity_status == "unknown" else o.box.identity_id),
                     *o.box.xyxy]
                    for o in sorted(result.observations, key=lambda o: (view_order[o.box.view_id], o.box.object_id))
                ]
                choices = ["All objects", *dict.fromkeys(
                    [r.identity_id for r in result.ranking] + [o.box.identity_id for o in result.observations]
                )]
                selected = "All objects"
                first_view = current.views[0]
                completion_update = {
                    session: current,
                    result_controls: gr.update(visible=True),
                    review_controls: gr.update(visible=True),
                    report_controls: gr.update(visible=True),
                    review_notice: gr.update(value="Check the proposed object identities across your views.", visible=False),
                    overview: summary_html(current),
                    readiness: ("**Objects found.** Open **Review** to confirm the physical object IDs, then rerank."
                                if request.decision_mode == "assisted" and corrected is None
                                else f"**Inspection complete in {result.provenance.get('timing', {}).get('request_seconds', 0):.0f}s.** Open **Results** to see the combined ranking."),
                    identity: gr.update(
                        choices=choices, value=selected, interactive=True
                    ),
                    layer: gr.update(value="Boxes", interactive=True),
                    evidence: evidence_gallery(current, selected, "Boxes"),
                    layer_note: evidence_note(current),
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
                    report_note: f"Ready to save this {result.mode} inspection with its case record.",
                    board: object_board_html(current),
                    decision_notice: "",
                    decision_controls: gr.update(visible=True),
                    machine_card: summary_html(current),
                    decision_object: gr.update(choices=decision_choices(result), value=None),
                    decision_status: "",
                    **case_view(current),
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
                completion_update = {
                    **empty_results(),
                    **case_view(current),
                    session: current,
                    readiness: message
                    if isinstance(error, InspectionCancelled)
                    else f"**Inspection failed.** {message}",
                }
            finally:
                current.active_request_id = None
                if browser_id:
                    with browser_jobs_lock:
                        job = browser_jobs.get(browser_id)
                        if job and job[0] == request.request_id:
                            browser_jobs.pop(browser_id, None)
            # The preparation event already exposed progress and cancellation.
            # Return all result/visibility updates once, without stream deltas.
            return {
                **completion_update,
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

        def confirm_and_rerun(current, rows, checked, progress=progress_tracker, browser_request: gr.Request = None):
            if not checked:
                raise gr.Error("Check the object identities across all views, then confirm them before reranking.")
            return do_inspect(current, rows, progress, browser_request)

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
            view_label = next(f"View {i}" for i, v in enumerate(current.views, 1) if v.view_id == view_id)
            updated = [list(row) for row in rows] + [
                [view_label, object_id, box.identity_id, *xyxy]
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

        saved_outputs = [report_file, json_file, case_file]

        def record_finding(current, case_id_value, item_value, batch_value, operator_value,
                           scope, action, finding_text, reason_text, responsible_text,
                           decision_operator_value=None):
            if current.result is None:
                return {decision_status: "**Decision not recorded:** Run an inspection first."}
            case = ensure_case(current)
            try:
                if decision_operator_value is not None:
                    operator_value = decision_operator_value
                update_details(case, case_id_value, item_value, batch_value, operator_value)
                record_decision(case, current.result, action=action, finding=finding_text, reason=reason_text,
                                object_scope=scope, responsible=responsible_text)
            except InspectionError as error:
                # Expected form validation must not replace the result cards with
                # Gradio's generic Error overlays or discard entered findings.
                return {decision_status: f"**Decision not recorded:** {error}"}
            # Earlier downloads no longer contain the latest human record.
            discard_exports(current)
            return {
                session: current,
                operator: case.operator,
                **case_view(current),
                **{component: gr.update(value=None, visible=False) for component in saved_outputs},
                report_note: "Save the case again to include your latest decision.",
                decision_object: gr.update(value=None),
                decision_action: None,
                finding: "",
                reason: "",
                responsible: "",
                decision_status: "Decision recorded. It is stored separately from the model result.",
            }

        def save_case_files(current, name, *details):
            if current.result is None:
                raise gr.Error("Run an inspection before preparing a report.")
            case = ensure_case(current)
            try:
                update_details(case, *details)
                report, data, bundle = save_case(current, name)
            except InspectionError as error:
                raise gr.Error(str(error)) from error
            return {
                session: current,
                case_file: gr.update(value=bundle, visible=True),
                report_file: gr.update(value=report, visible=True),
                report_note: f"Case {case.case_id} saved · {case_status(case)[1]}. The files keep the model result and the operator record separately.",
                json_file: gr.update(value=data, visible=True),
                **case_view(current),
            }

        opened_outputs = [
            opened, opened_view, opened_status, opened_header, opened_machine, opened_human,
            opened_follow_up, opened_board, opened_layer, opened_evidence, opened_rankings,
            opened_scores, opened_timeline, opened_layer_note,
        ]

        def open_saved_case(path):
            if not path:
                return {opened: None, opened_view: gr.update(visible=False), opened_status: ""}
            try:
                earlier = open_case(path)
            except InspectionError as error:
                return {
                    opened: None,
                    opened_view: gr.update(visible=False),
                    opened_status: f"**This case file could not be opened.** {error}",
                }
            view = earlier.session
            return {
                opened: earlier,
                opened_view: gr.update(visible=True),
                opened_status: f"Opened case **{earlier.case.case_id}** read-only.",
                opened_header: opened_case_html(earlier),
                opened_machine: summary_html(view),
                opened_human: decision_card_html(earlier.case, view.result),
                opened_follow_up: follow_up_html(earlier.case, view.result),
                opened_board: object_board_html(view),
                opened_layer: "Boxes",
                opened_evidence: evidence_gallery(view, "All objects", "Boxes"),
                opened_layer_note: evidence_note(view),
                opened_rankings: ranking_rows(view),
                opened_scores: observation_rows(view),
                opened_timeline: timeline_html(earlier.case),
            }

        def opened_gallery(earlier, value):
            return evidence_gallery(earlier.session, "All objects", value) if earlier else []

        def begin_follow_up(current, earlier, *details):
            if earlier is None:
                raise gr.Error("Open a saved case first.")
            if details:
                apply_details(current, details)
            operator_name = current.case.operator if current.case else ""
            output = fresh_case(current, start_follow_up(earlier, operator_name))
            # Stay on this tab: switching tabs from here can leave later result updates unapplied.
            output.pop(tabs)
            case = output[session].case
            output[opened_status] = (f"Follow-up case **{case.case_id}** started. Open **Images** and photograph "
                                     "the full group of items again.")
            return output

        # One shared queue also prevents reset/upload callbacks racing inference.
        event_options = {
            "concurrency_limit": 1,
            "concurrency_id": "inspection",
            "api_visibility": "private",
            "show_progress": "hidden",
        }
        files.change(prepare_uploads, [files, session, *case_fields], all_outputs, **event_options)
        mode.change(change_mode, [session, mode], [session, inspect], **event_options)
        reset.click(
            start_over, [session], [*all_outputs, files, title, tabs], **event_options
        )
        run_outputs = [*all_outputs, files, reset, cancel_run, mode]
        # Keep the cancel button and uploaded images usable while the worker runs.
        run_options = {**event_options, "show_progress": "full", "show_progress_on": readiness}
        # Store the case fields as shown before inspecting, so a missed blur event cannot lose them.
        inspect.click(
            edit_case, [session, *case_fields], [session, case_id, *case_outputs], **event_options
        ).success(prepare_inspection, [session], run_outputs, **event_options).success(
            do_inspect, [session], run_outputs, **run_options
        )
        rerun.click(
            prepare_inspection, [session, corrections, confirmed], run_outputs, **event_options
        ).success(confirm_and_rerun, [session, corrections, confirmed], run_outputs, **run_options)
        confirmed.change(lambda checked: gr.update(interactive=bool(checked)), [confirmed], [rerun], **event_options)
        # Only user edits trigger these callbacks. Programmatic completion updates
        # otherwise race with old dropdown values and erase the Review preview.
        corrections.input(
            edit_identities, [session, review_view, corrections],
            [session, review_image, drawing_status, review_evidence, confirmed, rerun], **event_options,
        )
        gr.on(
            [identity.input, layer.input],
            evidence_gallery,
            [session, identity, layer],
            evidence,
            **event_options,
        )
        review_view.input(
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
        # A new browser session never waits behind someone else's model job.
        app.load(initialise, [session], [session, *case_fields, *case_outputs],
                 queue=False, show_progress="hidden", api_visibility="private")
        app.unload(browser_disconnected)
        gr.on(
            [field.blur for field in case_fields] + [field.submit for field in case_fields],
            edit_case, [session, *case_fields], [session, case_id, *case_outputs], **event_options,
        )
        new_case_button.click(
            start_new_case, [session, *case_fields], [*all_outputs, files, title, tabs, *case_fields],
            **event_options,
        )
        record.click(
            record_finding,
            [session, *case_fields, decision_object, decision_action, finding, reason, responsible, decision_operator],
            [session, *case_outputs, *saved_outputs, report_note, decision_object, decision_action,
             finding, reason, responsible, decision_status, operator],
            **event_options,
        )
        export.click(
            save_case_files,
            [session, title, *case_fields],
            [session, *saved_outputs, report_note, *case_outputs],
            **event_options,
        )
        # Reading a saved case never touches the live session or a backend.
        saved_options = {"concurrency_limit": 1, "concurrency_id": "saved-cases", "api_visibility": "private"}
        case_upload.change(open_saved_case, [case_upload], opened_outputs, **saved_options)
        opened_layer.input(opened_gallery, [opened, opened_layer], opened_evidence, **saved_options)
        follow_up_button.click(
            begin_follow_up, [session, opened, *case_fields],
            [*all_outputs, files, title, opened_status, *case_fields],
            **event_options,
        )
    app.queue(max_size=8, default_concurrency_limit=1)
    # Gradio 6 accepts theme and CSS at launch, not Blocks construction.
    app.vpi_theme = theme
    return app
