"""Physical-object presentation and assisted-review checks with generated pixels."""

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

HAS_WEB = importlib.util.find_spec("gradio") is not None
if HAS_WEB:
    from PIL import Image, ImageDraw
    from visual_peer_inspector.web.core import (
        Box, InspectionResult, Observation, RankedObject, Session, View,
    )
    from visual_peer_inspector.web.presentation import (
        correspondence_label, discard_exports, draw_boxes, evidence_gallery,
        export_report, observation_rows, ranking_rows, selected_identity, summary_html,
        identity_aliases, _label_layout,
    )


@unittest.skipUnless(HAS_WEB, "Install the web requirements to run presentation checks")
class PhysicalObjectPresentationTests(unittest.TestCase):
    def setUp(self):
        self.views = tuple(View(f"view_{i}", Image.new("RGB", (180, 120), colour), f"digest-{i}")
                           for i, colour in enumerate(("#ddeedd", "#eeeecc"), 1))
        mask = Image.new("L", self.views[0].image.size)
        ImageDraw.Draw(mask).rectangle((25, 50, 70, 95), fill=255)
        self.observations = tuple(
            Observation(Box(v.view_id, "candidate_a", "object_a", (20, 45, 75, 100)),
                        .2, mask.copy(), Image.new("RGB", v.image.size, "orange"),
                        score=.35 + index * .1, score_method="bank_top2/v1",
                        component_scores={"bank_mean": .35 + index * .1, "p1": .2},
                        normalized_rank=.8, identity_status="confirmed", identity_source="user_confirmed")
            for index, v in enumerate(self.views)
        )
        self.contributors = tuple({"view_id": o.box.view_id, "candidate_id": o.box.object_id,
                                   "normalized_rank": .8, "used_in_fusion": True}
                                  for o in self.observations)
        self.result = InspectionResult(
            (RankedObject("object_a", .8, tuple(v.view_id for v in self.views), self.contributors, "bank_top2/v1"),),
            self.observations, "review_required", reasons=("Check object matches.",),
            method_version="bank_top2/v1", policy_version="test/v1",
            mode="assisted", selected_identity_id="object_a", cache_token="private-cache-handle",
            score_metadata={"selection_status": "unique", "method_label": "Cross-view object banks · strongest two views"},
        )
        self.session = Session(self.views, self.result, cache_token="private-cache-handle")
        self.addCleanup(discard_exports, self.session)

    def test_unique_selection_requires_an_explicit_supported_winner(self):
        self.assertEqual(selected_identity(self.result), "object_a")
        self.assertIsNone(selected_identity(replace(self.result, selected_identity_id=None)))
        self.assertIsNone(selected_identity(replace(self.result, selected_identity_id="other")))

    def test_overlay_aliases_are_compact_stable_and_preserve_user_ids(self):
        ids = ["winner-consensus-001", "winner-consensus-012", "unlinked_17", "A", "B"]
        expected = {"winner-consensus-001": "W1", "winner-consensus-012": "W12",
                    "unlinked_17": "?17", "A": "A", "B": "B"}
        self.assertEqual(identity_aliases(ids), expected)
        self.assertEqual(identity_aliases(reversed(ids)), expected)
        collision = identity_aliases(["W1", "winner-consensus-001"])
        self.assertEqual(collision["W1"], "W1")
        self.assertNotEqual(collision["W1"], collision["winner-consensus-001"])

    def test_labels_stay_inside_small_images_at_every_edge(self):
        draw = ImageDraw.Draw(Image.new("RGB", (64, 64)))
        for location in ((0, 0), (63, 0), (0, 63), (63, 63)):
            text, bounds, origin = _label_layout(draw, "a very long label that must fit", location, (64, 64))
            self.assertTrue(0 <= bounds[0] <= bounds[2] < 64)
            self.assertTrue(0 <= bounds[1] <= bounds[3] < 64)
            text_bounds = draw.textbbox(origin, text)
            self.assertTrue(0 <= text_bounds[0] <= text_bounds[2] < 64)
            self.assertTrue(0 <= text_bounds[1] <= text_bounds[3] < 64)

    def test_full_object_ids_and_aliases_are_available_in_tables_and_exports(self):
        identity = "winner-consensus-001"
        observations = tuple(replace(o, box=replace(o.box, identity_id=identity)) for o in self.observations)
        self.session.result = replace(self.result, observations=observations,
                                      ranking=(replace(self.result.ranking[0], identity_id=identity),),
                                      selected_identity_id=identity, mode="automatic",
                                      score_metadata={"method_label": "Automatic winner consensus"})
        self.assertEqual(observation_rows(self.session)[0][2], f"W1 · {identity}")
        self.assertEqual(ranking_rows(self.session)[0][1], f"W1 · {identity}")
        self.assertNotIn("Automatic · Automatic", summary_html(self.session))
        report, data = export_report(self.session, "Compact labels")
        record = json.loads(Path(data).read_text())
        self.assertEqual(record["overlay_aliases"], {identity: "W1"})
        self.assertIn("Observation score", Path(report).read_text())

    def test_tie_has_no_unique_highlight_or_export_and_competition_ranks(self):
        tied = replace(self.result, ranking=(*self.result.ranking,
                       RankedObject("object_b", .8, (self.views[0].view_id,))),
                       score_metadata={"selection_status": "tied"})
        self.session.result = tied
        self.assertIsNone(selected_identity(tied))
        self.assertIn("No unique selection", summary_html(self.session))
        self.assertEqual([row[0] for row in ranking_rows(self.session)], [1, 1])
        with patch("visual_peer_inspector.web.presentation.draw_boxes", wraps=draw_boxes) as render:
            evidence_gallery(self.session, "All objects", "Boxes")
            self.assertTrue(all(call.args[2] is None for call in render.call_args_list))
        _, data = export_report(self.session, "Tied scene")
        self.assertIsNone(json.loads(Path(data).read_text())["selected_identity_id"])

    def test_missing_match_keeps_original_view_and_does_not_claim_absence(self):
        self.session.result = replace(self.result, observations=self.observations[:1])
        gallery = evidence_gallery(self.session, "object_a", "Boxes")
        self.assertEqual(len(gallery), 2)
        self.assertIn("Not linked", gallery[1][1])
        self.assertNotIn("absent", gallery[1][1])
        self.assertEqual(gallery[1][0].tobytes(), self.views[1].image.tobytes())

    def test_black_background_and_heatmap_use_original_mask_coordinates(self):
        before = self.views[0].image.tobytes()
        black = evidence_gallery(self.session, "All objects", "Objects on black")[0][0]
        self.assertEqual(black.getpixel((5, 110)), (0, 0, 0))
        self.assertEqual(black.getpixel((45, 80)), self.views[0].image.getpixel((45, 80)))
        heatmap = evidence_gallery(self.session, "All objects", "Heatmaps")[0][0]
        self.assertEqual(heatmap.getpixel((5, 110)), self.views[0].image.getpixel((5, 110)))
        self.assertNotEqual(heatmap.getpixel((45, 80)), self.views[0].image.getpixel((45, 80)))
        self.assertEqual(before, self.views[0].image.tobytes())

    def test_scores_show_real_components_and_fusion_contributors(self):
        rows = observation_rows(self.session)
        self.assertEqual(rows[0][4:7], [.35, .8, "used"])
        self.assertIn("p1: 0.2", rows[0][-1])
        self.assertIn("bank_mean: 0.35", rows[0][-1])
        self.assertEqual(ranking_rows(self.session)[0][4], "View 1, View 2")
        unknown = replace(self.observations[0], score=None, normalized_rank=None, identity_status="unknown")
        self.session.result = replace(self.result, observations=(unknown,))
        self.assertIsNone(observation_rows(self.session)[0][4])

    def test_report_contains_evidence_and_provenance_without_cache_or_tensors(self):
        self.session.result = replace(self.result, provenance={
            "input_source": "recorded_development", "identity_source": "dataset_reference",
            "checkpoint_hash": "abc123", "session_id": "secret-session",
            "nested": {"cache_token": "private-cache-handle", "patches": [[1, 2]],
                       "source_path": "/Users/private/file", "runtime_seconds": 1.25},
        })
        report, data = export_report(self.session, '<script>bad</script>')
        document, raw = Path(report).read_text(), Path(data).read_text()
        record = json.loads(raw)
        self.assertIn("Dataset-supplied matches", document)
        self.assertIn("Recorded development example", document)
        self.assertIn("&lt;script&gt;bad&lt;/script&gt;", document)
        self.assertNotIn("<script>bad", document)
        self.assertEqual(document.count("data:image/png;base64,"), 10)
        for private in ("private-cache-handle", "secret-session", "/Users/private", '"patches"'):
            self.assertNotIn(private, document + raw)
        self.assertEqual(record["observations"][0]["score"], .35)
        self.assertEqual(record["observations"][0]["p1_score"], .2)
        self.assertEqual(record["ranking"][0]["contributors"], list(self.contributors))
        self.assertEqual(record["provenance"]["nested"], {"runtime_seconds": 1.25})
        self.assertEqual(len(record["selected_locations"]), 2)
        self.assertEqual(record["missing_links"], [])

    def test_report_names_missing_selected_link_and_ranking_confidence(self):
        result = replace(self.result,
            observations=self.observations[:1],
            ranking=(replace(self.result.ranking[0], view_ids=(self.views[0].view_id,)),),
            provenance={"review_signals": {"confidence": .42, "winner_support": 1, "view_count": 2}},
        )
        self.session.result = result
        document, data = export_report(self.session, "Partial match")
        record = json.loads(Path(data).read_text())
        self.assertEqual(record["selected_locations"][0]["view_id"], self.views[0].view_id)
        self.assertEqual(record["missing_links"][0]["view_id"], self.views[1].view_id)
        self.assertIn("not linked", record["missing_links"][0]["explanation"])
        self.assertIn("Ranking confidence index", Path(document).read_text())
        self.assertIn("1 / 2", Path(document).read_text())

    def test_human_confirmation_is_not_dataset_correspondence(self):
        self.assertEqual(correspondence_label(self.result), "User-confirmed matches")
        text = summary_html(self.session)
        self.assertNotIn("62.8", text)
        self.assertNotIn("60%", text)

    def _app(self):
        from visual_peer_inspector.web.ui import build_app
        expected = self.result
        class Backend:
            def __init__(self):
                self.requests, self.cleared = [], []
            def inspect(self, request, progress):
                self.requests.append(request)
                return replace(expected, mode=request.decision_mode, session_id=request.session_id,
                               score_metadata={**expected.score_metadata, "suggested_identities": [
                                   {"view_id": o.box.view_id, "candidate_id": o.box.object_id,
                                    "identity_id": "suggested_a", "status": "proposed"}
                                   for o in expected.observations]})
            def clear_session(self, session_id, cache_token=None):
                self.cleared.append((session_id, cache_token))
        backend = Backend()
        app = build_app(backend)
        callbacks = {f.fn.__name__: f.fn for f in app.fns.values() if f.fn}
        return app, backend, callbacks

    def test_mode_and_confirmed_rerank_reuse_session_cache_and_all_candidates(self):
        _, backend, callbacks = self._app()
        state = Session(self.views)
        callbacks["do_inspect"](state)
        self.assertEqual(backend.requests[0].decision_mode, "automatic")
        self.assertEqual(state.cache_token, "private-cache-handle")
        self.assertEqual({row[2] for row in state.corrections}, {"suggested_a"})
        with self.assertRaises(Exception):
            callbacks["confirm_and_rerun"](state, state.corrections, False)
        self.assertEqual(len(backend.requests), 1)
        callbacks["confirm_and_rerun"](state, state.corrections, True)
        request = backend.requests[-1]
        self.assertEqual(request.mode, "assisted")
        self.assertEqual(request.session_id, backend.requests[0].session_id)
        self.assertEqual(request.cache_token, "private-cache-handle")
        self.assertEqual(len(request.corrections), len(self.observations))
        self.assertEqual({box.identity_id for box in request.corrections}, {"suggested_a"})

    def test_edits_reset_confirmation_and_clear_images_releases_server_cache(self):
        _, backend, callbacks = self._app()
        state = Session(self.views, self.result, cache_token="private-cache-handle")
        rows = [[o.box.view_id, o.box.object_id, "", *o.box.xyxy] for o in self.observations]
        edits = callbacks["edit_identities"](state, self.views[0].view_id, rows)
        self.assertFalse(edits[-2])
        self.assertFalse(edits[-1]["interactive"])
        session_id = state.session_id
        reset = callbacks["start_over"](state)
        fresh = next(value for key, value in reset.items() if key.get_block_name() == "state")
        self.assertEqual(backend.cleared, [(session_id, "private-cache-handle")])
        self.assertNotEqual(fresh.session_id, session_id)
        self.assertIsNone(fresh.cache_token)

    def test_assisted_first_pass_directs_review_and_preserves_existing_tabs(self):
        app, backend, callbacks = self._app()
        state = Session(self.views, mode="assisted")
        updates = callbacks["do_inspect"](state)
        self.assertEqual(backend.requests[0].mode, "assisted")
        self.assertIsNone(backend.requests[0].corrections)
        self.assertTrue(any("Open **Review**" in str(value) for value in updates.values()))
        self.assertFalse(any(component.get_block_name() == "tabs" for component in updates))
        labels = [component.get("props", {}).get("label") for component in app.config["components"]
                  if component["type"] == "tabitem"]
        self.assertEqual(labels, ["Images", "Results", "Review", "Decision", "Report", "Saved cases"])
        self.assertTrue(all(component["props"].get("render_children")
                            for component in app.config["components"]
                            if component["type"] == "tabitem"))


if __name__ == "__main__":
    unittest.main()
