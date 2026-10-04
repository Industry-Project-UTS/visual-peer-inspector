"""Inspection cases, operator decisions and saved case files.

Every image here is generated in a temporary directory and every model output is
synthetic test data (provenance input_source "synthetic_test"). Nothing here runs
LocateAnything, SAM or DINO, and no result is evidence of inspection accuracy.
"""

from __future__ import annotations

import io
import json
import math
import re
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

gr = pytest.importorskip("gradio")

from visual_peer_inspector.models.visual_encoder import EncodedBatch  # noqa: E402
from visual_peer_inspector.web import case as case_module  # noqa: E402
from visual_peer_inspector.web.case import (  # noqa: E402
    FOLLOW_UP_NOTE,
    GROUP,
    STATUSES,
    case_record,
    case_status,
    current_decision,
    decision_card_html,
    follow_up_html,
    new_case,
    open_case,
    opened_case_html,
    record_decision,
    record_inspection,
    save_case,
    start_follow_up,
    update_details,
)
from visual_peer_inspector.web.core import (  # noqa: E402
    Box,
    InspectionError,
    InspectionRequest,
    InspectionResult,
    Observation,
    RankedObject,
    Session,
    load_views,
    validate_result,
)
from visual_peer_inspector.web.pipeline import UploadPipeline  # noqa: E402
from visual_peer_inspector.web.presentation import (  # noqa: E402
    CLAIM_LIMITS,
    discard_exports,
    evidence_gallery,
    object_board,
    object_board_html,
    result_record,
    summary_html,
)

BOXES = {"item_a": (10, 40, 50, 110), "item_b": (70, 40, 110, 110), "item_c": (130, 40, 170, 110)}
PRIVATE = ("private-cache-handle", "private-session", "/Users/private", "embedding")


def write_views(root: Path, count=3, *, odd=True, size=(180, 130)):
    """Generated tabletop scenes: identical items, optionally one with a wrong cap."""
    paths = []
    for index in range(count):
        image = Image.new("RGB", size, (200 - index * 9, 204, 198 + index * 7))
        draw = ImageDraw.Draw(image)
        for number, (x1, y1, x2, y2) in enumerate(BOXES.values()):
            draw.rectangle((x1, y1 + 12, x2, y2), fill=(40, 90, 160))
            draw.rectangle((x1 + 10, y1, x2 - 10, y1 + 11), fill=(40, 170, 220) if odd and number == 0 else (220, 60, 40))
        path = root / f"synthetic-view-{index}.png"
        image.save(path)
        paths.append(str(path))
    return paths


def synthetic_result(views, *, scores=(0.9, 0.4, 0.1), missing=None, state="review_required", mode="assisted"):
    """A hand-built assisted result labelled as synthetic test data."""
    observations, ranking = [], []
    for identity, score in zip(BOXES, scores):
        linked = []
        for index, view in enumerate(views):
            if missing == (identity, index):
                continue
            mask = Image.new("L", view.image.size)
            ImageDraw.Draw(mask).rectangle(BOXES[identity], fill=255)
            heatmap = view.image.copy()
            ImageDraw.Draw(heatmap).rectangle(BOXES[identity][:2] + (BOXES[identity][2], BOXES[identity][1] + 20), fill="#ec7c36")
            observations.append(Observation(
                Box(view.view_id, f"obj_{identity[-1]}", identity, BOXES[identity]), None, mask, heatmap,
                score=score, score_method="object_bank", component_scores={"object_bank": score},
                normalized_rank=score, identity_status="confirmed", identity_source="user_confirmed"))
            linked.append(view.view_id)
        contributors = tuple({"view_id": view_id, "candidate_id": f"obj_{identity[-1]}", "normalized_rank": score,
                              "used_in_fusion": True} for view_id in linked)
        ranking.append(RankedObject(identity, score, tuple(linked), contributors, "object_bank_top2"))
    ranking.sort(key=lambda item: (-item.relative_score, item.identity_id))
    tied = math.isclose(ranking[0].relative_score, ranking[1].relative_score)
    result = InspectionResult(
        tuple(ranking), tuple(observations), state,
        () if state == "result_ready" else ("Synthetic test result; review required.",),
        "synthetic-test-method", "synthetic-test-policy", mode,
        {"input_source": "synthetic_test", "cache_token": "private-cache-handle",
         "cache_root": "/Users/private/cache", "nested": {"embedding": [1, 2], "runtime_seconds": 0.5}},
        None if tied else ranking[0].identity_id, "private-session", "private-cache-handle",
        {"selection_status": "tied" if tied else "unique", "method_label": "Assisted patch bank"},
    )
    validate_result(result, InspectionRequest(views, mode=mode))
    return result


@pytest.fixture
def views(tmp_path):
    return load_views(write_views(tmp_path))


@pytest.fixture
def cleanup():
    sessions = []
    yield sessions.append
    for session in sessions:
        discard_exports(session)


def inspected(views, result, cleanup, **details):
    session = Session(views, result, case=new_case(**({"operator": "Operator A"} | details)))
    record_inspection(session.case, result, len(views))
    cleanup(session)
    return session


def test_new_case_has_context_and_validates_details():
    case = new_case(now="2026-09-27T01:02:03Z")
    assert re.fullmatch(r"VPI-20260927-[0-9A-F]{4}", case.case_id)
    assert case.created_at == "2026-09-27T01:02:03Z"
    assert case.events == [{"time": "2026-09-27T01:02:03Z", "event": "Case created"}]
    update_details(case, "BATCH-7_bottles", "  500 ml\tbottle ", "", "Operator A")
    assert (case.case_id, case.item, case.batch, case.operator) == ("BATCH-7_bottles", "500 ml bottle", "", "Operator A")
    assert "Case ID changed" in case.events[-1]["event"]
    for bad in ("", "ab", "has space", "../x", "x" * 41, "<b>"):
        with pytest.raises(InspectionError, match="Case IDs"):
            update_details(case, bad, "item", "", "Operator A")
    with pytest.raises(InspectionError, match="120 characters"):
        update_details(case, "CASE-2", "x" * 121, "", "Operator A")
    assert case.case_id == "BATCH-7_bottles"
    assert case_status(case) == STATUSES[None]


def test_decision_is_a_separate_human_record_that_never_changes_the_model_result(views, cleanup):
    result = synthetic_result(views)
    session = inspected(views, result, cleanup)
    before = (result.ranking, result.selected_identity_id, result.review_state, summary_html(session),
              object_board(session))
    decision = record_decision(session.case, result, action="hold", finding="Cap is the wrong colour",
                               reason="Mixed components", object_scope="item_a", responsible="Line lead")
    assert session.result is result
    assert (result.ranking, result.selected_identity_id, result.review_state, summary_html(session),
            object_board(session)) == before
    assert current_decision(session.case) == decision and decision.operator == "Operator A"
    assert case_status(session.case) == ("on_hold", "On hold")
    card = decision_card_html(session.case, result)
    assert "Operator decision · human record" in card and "covers only the object named here" in card
    machine = summary_html(session)
    assert "Model result · machine ranking" in machine and "Hold" not in machine
    assert "not an approval" in machine


@pytest.mark.parametrize("change, message", [
    ({"action": "approve"}, "Choose an action"),
    ({"action": None}, "Choose an action"),
    ({"object_scope": None}, "Choose the object"),
    ({"object_scope": "item_z"}, "from this inspection"),
    ({"finding": "  "}, "Enter the finding"),
    ({"reason": ""}, "Enter the reason"),
    ({"responsible": "x" * 81}, "80 characters"),
])
def test_decision_validation(views, cleanup, change, message):
    session = inspected(views, synthetic_result(views), cleanup)
    values = {"action": "hold", "finding": "Loose cap", "reason": "Leak risk", "object_scope": "item_a"} | change
    with pytest.raises(InspectionError, match=message):
        record_decision(session.case, session.result, **values)
    assert session.case.decisions == []


def test_decision_requires_an_operator_and_a_result(views, cleanup):
    session = inspected(views, synthetic_result(views), cleanup, operator="")
    with pytest.raises(InspectionError, match="operator name"):
        record_decision(session.case, session.result, action="hold", finding="f", reason="r", object_scope=GROUP)
    with pytest.raises(InspectionError, match="Run an inspection"):
        record_decision(session.case, None, action="hold", finding="f", reason="r", object_scope=GROUP)


def test_a_new_inspection_result_needs_a_new_decision(views, cleanup):
    session = inspected(views, synthetic_result(views), cleanup)
    record_decision(session.case, session.result, action="inspect_further", finding="Scuff?", reason="Unclear",
                    object_scope="item_a")
    assert case_status(session.case) == ("open", "Open · further inspection")
    record_inspection(session.case, session.result, 3, corrections=tuple(o.box for o in session.result.observations))
    assert current_decision(session.case) is None
    assert case_status(session.case) == STATUSES[None]
    assert "1 earlier decision kept" in decision_card_html(session.case, session.result)
    assert session.case.corrections["blank_identities"] == 0


def test_model_ranking_never_approves_the_group(views, cleanup):
    session = inspected(views, synthetic_result(views, state="result_ready"), cleanup)
    report, data, _ = save_case(session)
    record = json.loads(Path(data).read_text())
    assert record["review_state"] == "result_ready" and record["selected_identity_id"] == "item_a"
    assert record["inspection_case"]["operator_decision"] is None
    assert record["inspection_case"]["status"]["label"] == "Open · awaiting operator decision"
    document = Path(report).read_text()
    assert "No decision recorded for this inspection" in document
    assert "review-routing state, not an approval" in document
    assert "approved" not in document.lower()


def test_all_normal_group_is_a_human_decision_beside_the_model_ranking(views, cleanup):
    session = inspected(views, synthetic_result(views, state="result_ready"), cleanup)
    record_decision(session.case, session.result, action="no_issue_found", finding="All caps match on inspection",
                    reason="Colour difference was lighting", object_scope=GROUP)
    report, data, _ = save_case(session)
    record = json.loads(Path(data).read_text())
    decision = record["inspection_case"]["operator_decision"]
    assert record["selected_identity_id"] == "item_a" and record["review_state"] == "result_ready"
    assert (decision["action"], decision["object_scope"]) == ("no_issue_found", "")
    assert record["inspection_case"]["status"] == {"code": "closed", "label": "Closed · no issue found"}
    document = Path(report).read_text()
    assert "Highest combined unusualness score" in document and "Whole peer group" in document
    assert document.index("<h2>Model result</h2>") < document.index("<h2>Operator decision</h2>")


def test_board_keeps_missing_links_ties_and_unlinked_observations_explicit(views, cleanup):
    result = synthetic_result(views, scores=(0.7, 0.7, 0.2), missing=("item_a", 1))
    extra = Observation(Box(views[2].view_id, "obj_x", "unlinked_7", (5, 5, 30, 30)), None, None, None,
                        score=None, score_method="object_bank", identity_status="unlinked", identity_source="predicted")
    result = replace(result, observations=result.observations + (extra,))
    validate_result(result, InspectionRequest(views, mode="assisted"))
    session = Session(views, result)
    board = object_board(session)
    first, second, third = board["objects"]
    assert (first["identity_id"], second["identity_id"]) == ("item_a", "item_b")
    assert first["tied_for_top"] and second["tied_for_top"] and not third["tied_for_top"]
    assert not any(item["selected"] for item in board["objects"])
    assert (first["rank"], first["rank_shared"], third["rank"]) == (1, True, 3)
    assert [cell["linked"] for cell in first["cells"]] == [True, False, True]
    assert first["views_linked"] == 2 and board["unlinked"][0]["identity_id"] == "unlinked_7"
    page = object_board_html(session)
    assert page.count("Tied for highest score") == 2 and "Highest combined score" not in page
    assert "This is not proof that the object is absent" in page and "Linked in 2 of 3 views" in page
    assert "Observations without a cross-view link (1)" in page
    assert "No unique selection" in summary_html(session)


def test_tied_result_is_saved_open_for_human_review(views, cleanup):
    session = inspected(views, synthetic_result(views, scores=(0.5, 0.5, 0.1)), cleanup)
    _, data, bundle = save_case(session)
    record = json.loads(Path(data).read_text())
    assert record["selected_identity_id"] is None and record["review_state"] == "review_required"
    assert record["inspection_case"]["status"]["code"] == "open"
    opened = open_case(bundle)
    assert opened.session.result.selected_identity_id is None
    assert case_status(opened.case) == STATUSES[None]
    assert "No unique selection" in summary_html(opened.session)


def test_save_and_reopen_is_faithful_read_only_and_private(views, cleanup):
    result = synthetic_result(views, missing=("item_c", 2))
    session = inspected(views, result, cleanup, item="Dish soap bottle", batch="B-12")
    record_inspection(session.case, result, 3, corrections=tuple(o.box for o in result.observations))
    record_decision(session.case, result, action="correct_rework", finding="Wrong cap fitted",
                    reason="Cap colour differs from peers", object_scope="item_a", responsible="Rework bench")
    report, data, bundle = save_case(session, "Morning batch")
    names = zipfile.ZipFile(bundle).namelist()
    assert names[0] == "case.json" and {n.split("/")[0] for n in names} == {"case.json", "views", "evidence"}
    raw = zipfile.ZipFile(bundle).read("case.json").decode()
    for text in (raw, Path(data).read_text(), Path(report).read_text()):
        assert not any(private in text for private in PRIVATE)

    opened = open_case(bundle)
    reopened = opened.session
    assert reopened.cache_token is None and reopened.result.cache_token is None and reopened.result.session_id == ""
    assert [v.image.tobytes() for v in reopened.views] == [v.image.tobytes() for v in views]
    assert result_record(reopened) == result_record(session)
    assert case_record(opened.case) == case_record(session.case)
    assert opened.case.corrections == session.case.corrections
    for layer in ("Original", "Boxes", "Masks", "Objects on black", "Heatmaps"):
        assert ([image.tobytes() for image, _ in evidence_gallery(reopened, "All objects", layer)]
                == [image.tobytes() for image, _ in evidence_gallery(session, "All objects", layer)])
    banner = opened_case_html(opened)
    assert "Read-only" in banner and "not rerun" in banner and "Synthetic test data" in banner
    assert opened.data_origin == "synthetic_test"


def rewrite(bundle, target, change):
    """Copy a saved case with one change applied to its record or members."""
    with zipfile.ZipFile(bundle) as source:
        members = {name: source.read(name) for name in source.namelist()}
    document = json.loads(members["case.json"])
    change(members, document)
    if document is not None and "case.json" in members:
        members["case.json"] = json.dumps(document).encode()
    with zipfile.ZipFile(target, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return target


def _tamper_view(members, document):
    image = Image.open(io.BytesIO(members["views/view-1.webp"])).convert("RGB")
    image.putpixel((0, 0), (1, 2, 3))
    buffer = io.BytesIO()
    image.save(buffer, format="WEBP", lossless=True)
    members["views/view-1.webp"] = buffer.getvalue()


def _small_mask(members, document):
    buffer = io.BytesIO()
    Image.new("L", (5, 5)).save(buffer, format="PNG")
    members["evidence/obs-001-mask.png"] = buffer.getvalue()


MALFORMED = {
    "traversal": (lambda m, d: m.update({"../evil.png": b"x"}), "unexpected content"),
    "unreferenced file": (lambda m, d: m.update({"evidence/obs-099-mask.png": m["evidence/obs-001-mask.png"]}), "does not reference"),
    "no record": (lambda m, d: m.pop("case.json"), "no case record"),
    "bad json": (lambda m, d: d.clear() or d.update({"x": float("nan")}), "not valid JSON"),
    "schema": (lambda m, d: d.update({"schema_version": "visual-peer-inspection-case/v9"}), "unsupported format"),
    "extra field": (lambda m, d: d.update({"cache_token": "abc"}), "unexpected fields"),
    "pixels": (_tamper_view, "fingerprint"),
    "mask size": (_small_mask, "unexpected format or size"),
    "action": (lambda m, d: d["case"]["decision_history"][0].update({"action": "approve_batch"}), "unknown action"),
    "status": (lambda m, d: d["case"].update({"status": {"code": "closed", "label": "Closed · no issue found"}}), "inconsistent"),
    "cache handle": (lambda m, d: d["machine_result"]["provenance"].update({"cache_token": "secret"}), "private"),
    "local path": (lambda m, d: d["machine_result"]["provenance"].update({"note": "/Users/someone/photo.png"}), "private"),
    "summary": (lambda m, d: d["machine_result"].update({"selected_identity_id": "item_b"}), "inconsistent|does not match"),
    "score order": (lambda m, d: d["machine_result"]["ranking"][0].update({"relative_score": 0.01}), "inconsistent"),
    "box": (lambda m, d: d["machine_result"]["observations"][0].update({"box_xyxy": [0, 0, 999, 10]}), "inconsistent"),
    "timeline": (lambda m, d: d["case"]["timeline"][0].update({"event": "Case\ncreated"}), "timeline"),
    "heatmap region": (lambda m, d: d["files"]["observations"][0].update({"heatmap_region": [0, 0, 9999, 10]}), "evidence file record"),
    "availability": (lambda m, d: d["machine_result"]["observations"][0].update({"mask_available": False}), "availability"),
    "decision scope": (lambda m, d: d["case"]["decision_history"][0].update({"object_scope": "<script>"}), "decision"),
}


@pytest.fixture
def saved(views, cleanup):
    session = inspected(views, synthetic_result(views), cleanup)
    record_decision(session.case, session.result, action="hold", finding="Wrong cap", reason="Mismatch",
                    object_scope="item_a")
    return save_case(session)[2]


@pytest.mark.parametrize("name", sorted(MALFORMED))
def test_malformed_case_files_are_rejected(saved, tmp_path, name):
    change, message = MALFORMED[name]
    with pytest.raises(InspectionError, match=message):
        open_case(rewrite(saved, tmp_path / f"{name}.zip", change))


def test_non_zip_and_oversized_files_are_rejected(saved, tmp_path, monkeypatch):
    path = tmp_path / "notes.zip"
    path.write_text("not a case file")
    with pytest.raises(InspectionError, match="saved by this app"):
        open_case(path)
    monkeypatch.setattr(case_module, "MAX_JSON_BYTES", 100)
    with pytest.raises(InspectionError, match="too large"):
        open_case(saved)
    monkeypatch.setattr(case_module, "MAX_CASE_BYTES", 10)
    with pytest.raises(InspectionError, match="larger than"):
        open_case(saved)


def test_oversized_export_keeps_previous_downloads_and_does_not_log_a_save(views, cleanup, monkeypatch):
    session = inspected(views, synthetic_result(views), cleanup)
    report, data, bundle = save_case(session)
    previous_directory = session.export_directory
    session.case.events.extend({"time": session.case.created_at, "event": f"Earlier action {index}"}
                               for index in range(case_module.MAX_EVENTS - len(session.case.events)))
    previous_events = list(session.case.events)
    monkeypatch.setattr(case_module, "MAX_CASE_BYTES", Path(bundle).stat().st_size - 1)

    with pytest.raises(InspectionError, match="too large to reopen"):
        save_case(session)

    assert session.export_directory == previous_directory
    assert all(Path(path).is_file() for path in (report, data, bundle))
    assert session.case.events == previous_events


def test_follow_up_links_both_records_without_comparing_scores(views, tmp_path, cleanup):
    earlier = inspected(views, synthetic_result(views, scores=(0.2, 0.9, 0.1)), cleanup, item="Bottles", batch="B-12")
    record_decision(earlier.case, earlier.result, action="correct_rework", finding="Cap cross-threaded",
                    reason="Leak risk", object_scope="item_b")
    opened = open_case(save_case(earlier)[2])

    later_case = start_follow_up(opened, "Operator B")
    reference = later_case.follow_up_of
    assert (later_case.item, later_case.batch, later_case.operator) == ("Bottles", "B-12", "Operator B")
    assert later_case.case_id != earlier.case.case_id and later_case.inspection_number == 0
    assert reference["case_id"] == earlier.case.case_id and reference["case_file_sha256"] == opened.sha256
    assert reference["decisions"][0]["action"] == "correct_rework"

    def numbers(value):
        if isinstance(value, dict):
            return [n for item in value.values() for n in numbers(item)]
        if isinstance(value, list):
            return [n for item in value for n in numbers(item)]
        return [value] if isinstance(value, float) else []
    assert numbers(reference) == []

    recapture = tmp_path / "recapture"
    recapture.mkdir()
    later_views = load_views(write_views(recapture, odd=False, size=(184, 132)))
    later = Session(later_views, synthetic_result(later_views, scores=(0.3, 0.2, 0.1)), case=later_case)
    cleanup(later)
    record_inspection(later.case, later.result, 3)
    record_decision(later.case, later.result, action="no_issue_found", finding="Cap refitted and tight",
                    reason="Checked by hand", object_scope="item_b")
    report, data, bundle = save_case(later)
    document = Path(report).read_text()
    for text in (earlier.case.case_id, later.case.case_id, "Correct / rework", "No issue found",
                 "a lower score does not show that a repair worked"):
        assert text in document
    assert "improv" not in document.lower() and "repair succeeded" not in document.lower()
    assert json.loads(Path(data).read_text())["inspection_case"]["follow_up_of"] == reference
    reopened = open_case(bundle)
    panel = follow_up_html(reopened.case, reopened.session.result)
    assert earlier.case.case_id in panel and later.case.case_id in panel and FOLLOW_UP_NOTE in panel


def test_report_json_and_case_file_agree_and_state_claim_limits(views, cleanup):
    session = inspected(views, synthetic_result(views, missing=("item_b", 0)), cleanup)
    record_decision(session.case, session.result, action="inspect_further", finding="Possible scratch",
                    reason="Needs better light", object_scope="item_b")
    report, data, bundle = save_case(session)
    record = json.loads(Path(data).read_text())
    saved = json.loads(zipfile.ZipFile(bundle).read("case.json"))
    assert {key: record[key] for key in saved["machine_result"]} == saved["machine_result"]
    assert record["inspection_case"] == saved["case"]
    assert saved["case"]["operator_decision"]["finding"] == "Possible scratch"
    assert saved["case"]["timeline"][-1]["event"].startswith("Case saved")
    document = Path(report).read_text()
    assert "Possible scratch" in document and "Needs better light" in document
    for statement in CLAIM_LIMITS:
        assert statement in document
    assert "relative unusualness" in document and "not validated defect boundaries" in document
    assert "not proof that it is absent" in document


def test_origin_labels_distinguish_synthetic_data_from_backend_output(views, cleanup):
    synthetic = inspected(views, synthetic_result(views), cleanup)
    assert "Synthetic test data" in summary_html(synthetic)
    plain = inspected(views, replace(synthetic.result, provenance={}), cleanup)
    assert "Synthetic test data" not in summary_html(plain)
    assert open_case(save_case(plain)[2]).data_origin == "backend_output"


# Real upload pipeline and frozen decision code, with stand-in model stages ------


class SyntheticStages:
    """Test-only stand-ins for detection, masks and features; not a model."""

    def __init__(self):
        self.calls = {"propose": 0, "segment": 0, "encode": 0}

    def propose(self, image, view_id):
        self.calls["propose"] += 1
        pixels = np.asarray(image, dtype=np.int16)
        foreground = np.abs(pixels - pixels[1, 1]).sum(axis=2) > 90
        columns = np.flatnonzero(foreground.any(axis=0))
        boxes = []
        for run in np.split(columns, np.flatnonzero(np.diff(columns) > 1) + 1):
            rows = np.flatnonzero(foreground[:, run[0]:run[-1] + 1].any(axis=1))
            boxes.append((int(run[0]), int(rows[0]), int(run[-1]) + 1, int(rows[-1]) + 1))
        return boxes, []

    def segment(self, image, boxes):
        self.calls["segment"] += len(boxes)
        masks = np.zeros((len(boxes), 256, 256), bool)
        for mask, (x1, y1, x2, y2) in zip(masks, boxes):
            mask[y1:y2, x1:x2] = True
        return masks, np.full(len(boxes), 0.95)

    def encode(self, crops):
        self.calls["encode"] += len(crops)
        vectors, patches = [], []
        for crop in crops:
            grid = np.asarray(Image.fromarray(np.asarray(crop)).resize((16, 16), Image.Resampling.BOX), np.float32)
            patch = np.tile(np.concatenate([grid.reshape(256, 3) / 255, np.ones((256, 1))], axis=1), (1, 96))
            patch /= np.linalg.norm(patch, axis=1, keepdims=True)
            vector = patch.mean(axis=0)
            vectors.append(vector / np.linalg.norm(vector))
            patches.append(patch.astype(np.float32))
        return EncodedBatch(np.asarray(vectors, np.float32), tuple(patches), 16, 384)

    def release(self):
        pass


class SyntheticBackend:
    def __init__(self, root):
        self.stages = SyntheticStages()
        self.pipeline = UploadPipeline(self.stages, cache_root=root)
        self.requests = []

    def inspect(self, request, progress):
        self.requests.append(request)
        result = self.pipeline.inspect(request, progress)
        return replace(result, provenance={**result.provenance, "input_source": "synthetic_test"})

    def clear_session(self, session_id, cache_token=None):
        self.pipeline.clear_session(session_id, cache_token)


def scene(root, *, odd=True):
    """256 × 256 generated views: four bottles, one optionally with a different cap."""
    paths = []
    for index, shift in enumerate((0, 6, -5)):
        image = Image.new("RGB", (256, 256), (205 + index * 4, 208, 202))
        draw = ImageDraw.Draw(image)
        for number, left in enumerate((20, 80, 140, 200)):
            x = left + shift
            draw.rectangle((x, 80, x + 36, 220), fill=(40, 90, 160))
            draw.rectangle((x + 8, 60, x + 28, 79), fill=(40, 200, 90) if odd and number == 2 else (220, 60, 40))
        path = root / f"scene-{index}.png"
        image.save(path)
        paths.append(str(path))
    return paths


def component(outputs, elem_id):
    return next(value for key, value in outputs.items() if getattr(key, "elem_id", None) == elem_id)


def live(outputs):
    return next(value for key, value in outputs.items() if isinstance(value, Session))


def test_ui_workflow_from_upload_through_assisted_review_decision_save_and_reopen(tmp_path, cleanup):
    from visual_peer_inspector.web.ui import build_app

    backend = SyntheticBackend(tmp_path / "cache")
    app = build_app(backend)
    callbacks = {f.fn.__name__: f.fn for f in app.fns.values() if f.fn}
    state = live(callbacks["initialise"](Session()))
    first_case = state.case.case_id
    state = live(callbacks["prepare_uploads"](scene(tmp_path), state))
    assert state.case.case_id == first_case and len(state.views) == 3
    assert state.case.events[-1]["event"] == "Photos loaded: 3 views"
    callbacks["edit_case"](state, first_case, "Dish soap bottle", "B-12", "")
    callbacks["change_mode"](state, "assisted")

    callbacks["do_inspect"](state)
    assert state.result.mode == "assisted" and not state.result.ranking
    assert state.result.review_state == "insufficient_evidence"
    assert state.case.inspection_number == 1
    failed = callbacks["record_finding"](state, first_case, "Dish soap bottle", "B-12", "", GROUP, "hold", "f", "r", "")
    assert "operator name" in component(failed, "decision-status")
    assert len(failed) == 1 and not state.case.decisions

    updates = callbacks["confirm_and_rerun"](state, state.corrections, True)
    result = state.result
    assert backend.requests[-1].corrections is not None and result.mode == "assisted"
    assert result.provenance["input_source"] == "synthetic_test"
    assert result.ranking and state.case.inspection_number == 2 and state.case.corrections
    assert any("Synthetic test data" in str(value) for value in updates.values())
    calls = dict(backend.stages.calls)
    requests = len(backend.requests)

    top = result.ranking[0].identity_id
    output = callbacks["record_finding"](state, first_case, "Dish soap bottle", "B-12", "", top,
                                         "correct_rework", "Cap is a different part", "Mixed parts bin", "Bench 2", "Operator A")
    assert state.case.operator == "Operator A"
    assert state.result is result and result.ranking[0].identity_id == top
    assert case_status(state.case) == ("rework", "Rework required")
    assert "Correct / rework" in component(output, "human-card")

    saved = callbacks["save_case_files"](state, "", first_case, "Dish soap bottle", "B-12", "Operator A")
    cleanup(state)
    bundle = next(value["value"] for key, value in saved.items()
                  if isinstance(value, dict) and str(value.get("value", "")).endswith("-case.zip"))
    viewer = callbacks["open_saved_case"](bundle)
    opened = next(value for value in viewer.values() if hasattr(value, "sha256"))
    assert (backend.stages.calls, len(backend.requests)) == (calls, requests)
    assert opened.session.cache_token is None and opened.session.result.ranking == result.ranking
    assert "Read-only" in next(v for v in viewer.values() if isinstance(v, str) and "read-only-banner" in v)

    follow = callbacks["begin_follow_up"](state, opened, first_case, "Dish soap bottle", "B-12", "Operator B")
    fresh = live(follow)
    assert fresh.views == () and fresh.result is None and fresh.case.case_id != first_case
    assert fresh.case.follow_up_of["case_id"] == first_case and fresh.case.operator == "Operator B"
    assert fresh.case.item == "Dish soap bottle" and fresh.case.batch == "B-12"
    assert "Follow-up of case" in component(follow, "case-banner")
    assert backend.stages.calls == calls


def test_ui_reads_case_fields_at_each_action_without_relying_on_blur(tmp_path):
    from visual_peer_inspector.web.ui import build_app

    app = build_app(SyntheticBackend(tmp_path / "cache"))
    callbacks = {f.fn.__name__: f.fn for f in app.fns.values() if f.fn}
    state = live(callbacks["prepare_uploads"](scene(tmp_path), Session(), "LOT-7", "Bottles", "B-1", "Operator A"))
    assert (state.case.case_id, state.case.batch, state.case.operator) == ("LOT-7", "B-1", "Operator A")
    output = callbacks["edit_case"](state, "no spaces allowed", "Bottles", "B-2", "Operator C")
    assert (state.case.case_id, state.case.batch, state.case.operator) == ("LOT-7", "B-2", "Operator C")
    assert "Case IDs use" in component(output, "case-banner")

    names = {index: fn.fn.__name__ for index, fn in app.fns.items() if fn.fn}
    button = next(c["id"] for c in app.config["components"] if c.get("props", {}).get("elem_id") == "inspect-button")
    first = next(d for d in app.config["dependencies"] if [button, "click"] in [list(t) for t in d["targets"]])
    then = next(d for d in app.config["dependencies"] if d["trigger_after"] == first["id"])
    run = next(d for d in app.config["dependencies"] if d["trigger_after"] == then["id"])
    assert (names[first["id"]], names[then["id"]], names[run["id"]]) == (
        "edit_case", "prepare_inspection", "do_inspect"
    )
    assert then["trigger_only_on_success"] and run["trigger_only_on_success"]

    fresh = live(callbacks["start_new_case"](state, "LOT-7", "Bottles", "B-2", "Operator D"))
    assert fresh.case.case_id != "LOT-7" and fresh.case.operator == "Operator D" and fresh.case.item == ""


def test_ui_tied_result_stays_open_and_bad_case_file_is_reported(tmp_path, cleanup):
    from visual_peer_inspector.web.ui import build_app

    backend = SyntheticBackend(tmp_path / "cache")
    callbacks = {f.fn.__name__: f.fn for f in build_app(backend).fns.values() if f.fn}
    state = live(callbacks["prepare_uploads"](scene(tmp_path, odd=False), Session()))
    callbacks["do_inspect"](state)
    callbacks["confirm_and_rerun"](state, state.corrections, True)
    result = state.result
    assert result.selected_identity_id is None and result.review_state == "review_required"
    assert "No unique selection" in summary_html(state)
    saved = callbacks["save_case_files"](state, "", state.case.case_id, "Bottles", "", "Operator A")
    cleanup(state)
    assert "Open · awaiting operator decision" in component(saved, "case-banner")

    broken = tmp_path / "broken.zip"
    broken.write_bytes(b"PK\x03\x04 not really a zip")
    viewer = callbacks["open_saved_case"](str(broken))
    assert "could not be opened" in next(v for v in viewer.values() if isinstance(v, str) and v)
    assert all(value is None for key, value in viewer.items() if key.get_block_name() == "state")
