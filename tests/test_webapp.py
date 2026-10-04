"""App tests use generated pixels only; no dataset or recorded model results."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

HAS_WEB = importlib.util.find_spec("gradio") is not None
if HAS_WEB:
    from PIL import Image

    from visual_peer_inspector.web.core import (
        Box,
        InspectionError,
        InspectionRequest,
        InspectionResult,
        Observation,
        RankedObject,
        Session,
        load_backend,
        load_views,
        parse_corrections,
        validate_result,
    )
    from visual_peer_inspector.web.presentation import (
        discard_exports,
        evidence_gallery,
        export_report,
    )


@unittest.skipUnless(HAS_WEB, "Install apps/web/requirements.txt to run the web tests")
class WebAppTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.paths = []
        for index, colour in enumerate(("#447766", "#887744")):
            path = self.root / f"private-source-{index}.png"
            Image.new("RGB", (128, 96), colour).save(path)
            self.paths.append(str(path))
        self.views = load_views(self.paths)
        self.observations = tuple(
            Observation(
                Box(v.view_id, "a", "peer_a", (10, 10, 80, 70)),
                0.4,
                mask=Image.new("L", v.image.size, 128),
                heatmap=Image.new("RGB", v.image.size, "orange"),
            )
            for v in self.views
        )
        self.result = InspectionResult(
            (RankedObject("peer_a", 0.4, tuple(v.view_id for v in self.views)),),
            self.observations,
            "result_ready",
            method_version="test-only",
            policy_version="test-only",
            selected_identity_id="peer_a",
        )

    def test_completion_restores_inputs_and_result_visibility_in_one_update(self):
        # Gradio can coalesce streamed updates; the final update must be complete.
        from visual_peer_inspector.web.backend import InspectionCancelled
        from visual_peer_inspector.web.ui import build_app

        for outcome in ("success", "failure", "cancelled"):
            with self.subTest(outcome=outcome):
                fixture_result = self.result

                class Backend:
                    def inspect(self, request, progress):
                        if outcome == "failure":
                            raise RuntimeError("private /source/path")
                        if outcome == "cancelled":
                            raise InspectionCancelled("Inspection cancelled.")
                        return replace(fixture_result, cache_token="test-token")

                app = build_app(Backend())
                callback = next(f.fn for f in app.fns.values()
                                if f.fn and f.fn.__name__ == "do_inspect")
                state = Session(self.views)
                prepare = next(f.fn for f in app.fns.values()
                               if f.fn and f.fn.__name__ == "prepare_inspection")
                first = prepare(state)
                terminal = callback(state)
                self.assertIsInstance(terminal, dict)

                def by_id(update, name):
                    return next(value for component, value in update.items()
                                if getattr(component, "elem_id", None) == name)

                self.assertFalse(any(getattr(c, "elem_id", None) == "result-controls"
                                     for c in first))
                self.assertFalse(by_id(first, "inspect-button")["interactive"])
                self.assertTrue(by_id(terminal, "inspect-button")["interactive"])
                self.assertIsNone(state.active_request_id)
                for name in ("result-controls", "review-controls", "report-controls", "decision-controls"):
                    self.assertEqual(by_id(terminal, name)["visible"], outcome == "success")
                self.assertEqual(state.result is not None, outcome == "success")
                self.assertNotIn("/source/path", str(terminal))

    def test_preparation_shows_cancel_before_starting_models(self):
        from visual_peer_inspector.web.ui import build_app

        class Backend:
            def inspect(self, request, progress):
                raise AssertionError("Preparation must not start inference")

            def cancel(self, request_id):
                return False

        app = build_app(Backend())
        callback = next(f.fn for f in app.fns.values()
                        if f.fn and f.fn.__name__ == "prepare_inspection")
        state = Session(self.views, self.result)
        update = callback(state)
        cancel = next(v for c, v in update.items() if getattr(c, "value", None) == "Cancel inspection")
        self.assertTrue(cancel["visible"])
        self.assertFalse(next(v for c, v in update.items()
                              if getattr(c, "elem_id", None) == "inspect-button")["interactive"])
        self.assertIs(state.result, self.result)
        rows = [[o.box.view_id, o.box.object_id, o.box.identity_id, *o.box.xyxy]
                for o in self.observations]
        with self.assertRaisesRegex(Exception, "confirm them before reranking"):
            callback(state, rows, False)
        prepared = callback(state, rows, True)
        for parent in ("result-controls", "review-controls", "report-controls", "decision-controls"):
            self.assertFalse(any(getattr(c, "elem_id", None) == parent for c in prepared),
                             "Preparation must not unmount controls needed by later callbacks")
        downloads = [value for component, value in prepared.items()
                     if getattr(component, "label", None) in
                     ("Inspection case file", "HTML report", "JSON result")]
        self.assertEqual(len(downloads), 3)
        self.assertTrue(all(value["value"] is None and not value["visible"]
                            for value in downloads), "Reranking must retire stale download links")
        for label in ("Boxes and identities", "I checked the object IDs across all views. Blank IDs remain unconfirmed."):
            self.assertFalse(any(getattr(c, "label", None) == label for c in prepared),
                             "Preparation must preserve chained rerank inputs")
        self.assertFalse(any(c.get_block_name() == "tabs" for c in prepared),
                         "Model callbacks must not recreate the tab container")

    def test_upload_ids_are_stable_when_reordered_and_names_are_not_model_inputs(self):
        reversed_views = load_views(self.paths[::-1])
        self.assertEqual(self.views[0].view_id, reversed_views[1].view_id)
        self.assertEqual(set(vars(self.views[0])), {"view_id", "image", "digest"})
        self.assertEqual(self.views[0].image.info, {})

    def test_duplicate_pixels_are_rejected_even_with_different_filenames(self):
        copy = self.root / "different-name.png"
        copy.write_bytes(Path(self.paths[0]).read_bytes())
        with self.assertRaisesRegex(InspectionError, "same image"):
            load_views([self.paths[0], str(copy)])

    def test_upload_count(self):
        self.assertEqual(load_views(None), ())
        self.assertEqual(len(load_views(self.paths[:1])), 1)
        with self.assertRaisesRegex(InspectionError, "up to five"):
            load_views(self.paths * 3)

    def test_corrupt_image_rejected(self):
        path = self.root / "broken.png"
        path.write_text("not an image")
        with self.assertRaisesRegex(InspectionError, "could not be opened"):
            load_views([str(path)])

    def test_unsupported_and_tiny_images_rejected(self):
        path = self.root / "photo.gif"
        Image.new("RGB", (128, 128)).save(path)
        with self.assertRaisesRegex(InspectionError, "JPEG, PNG or WebP"):
            load_views([str(path)])
        path = self.root / "tiny.png"
        Image.new("RGB", (20, 20)).save(path)
        with self.assertRaisesRegex(InspectionError, "too small"):
            load_views([str(path)])

    def test_metadata_removed(self):
        path = self.root / "photo.jpg"
        exif = Image.Exif()
        exif[270] = "private identifying text"
        exif[274] = 6
        Image.new("RGB", (128, 96)).save(path, exif=exif)
        view = load_views([str(path)])[0]
        self.assertEqual(view.image.size, (96, 128))
        self.assertFalse(view.image.info)
        self.assertFalse(view.image.getexif())

    def test_corrections_require_in_bounds_integer_boxes_and_multiple_views(self):
        rows = [
            [o.box.view_id, o.box.object_id, o.box.identity_id, *o.box.xyxy]
            for o in self.observations
        ]
        self.assertEqual(len(parse_corrections(rows, self.views)), 2)
        for value in (-1, 200, 1.5, float("nan")):
            broken = [row.copy() for row in rows]
            broken[0][3] = value
            with self.subTest(value=value), self.assertRaises(InspectionError):
                parse_corrections(broken, self.views)
        with self.assertRaisesRegex(InspectionError, "two different views"):
            parse_corrections(rows[:1], self.views)

    def test_review_view_labels_follow_upload_order_and_reject_missing_views(self):
        views = tuple(reversed(self.views))
        rows = [[f"View {i}", "object_1", "A", 1, 1, 40, 40]
                for i in range(1, len(views) + 1)]
        boxes = parse_corrections(rows, views)
        self.assertEqual([b.view_id for b in boxes], [v.view_id for v in views])
        self.assertEqual(parse_corrections(
            [[b.view_id, b.object_id, b.identity_id, *b.xyxy] for b in boxes], views), boxes)
        rows[0][0] = "View 3"
        with self.assertRaisesRegex(InspectionError, "no longer uploaded"):
            parse_corrections(rows, views)

    def test_result_support_must_match_observations(self):
        validate_result(self.result, InspectionRequest(self.views))
        bad = replace(
            self.result,
            ranking=(
                replace(self.result.ranking[0], view_ids=(self.views[0].view_id,)),
            ),
        )
        with self.assertRaisesRegex(InspectionError, "view support"):
            validate_result(bad, InspectionRequest(self.views))

    def test_result_rejects_unversioned_nonfinite_and_unknown_states(self):
        for bad in (
            replace(self.result, method_version=""),
            replace(self.result, policy_version=""),
            replace(self.result, review_state="review_recommended"),
            replace(self.result, ranking=()),
            replace(
                self.result,
                ranking=(replace(self.result.ranking[0], relative_score=float("nan")),),
            ),
        ):
            with self.subTest(result=bad), self.assertRaises(InspectionError):
                validate_result(bad, InspectionRequest(self.views))

    def test_result_requires_review_reasons_and_accurate_mode(self):
        with self.assertRaisesRegex(InspectionError, "explain"):
            validate_result(
                replace(self.result, review_state="review_required"),
                InspectionRequest(self.views),
            )
        request = InspectionRequest(self.views, tuple(o.box for o in self.observations))
        with self.assertRaisesRegex(InspectionError, "assisted"):
            validate_result(self.result, request)
        validate_result(replace(self.result, mode="assisted"), request)

    def test_bad_overlay_dimensions_rejected(self):
        observations = (
            replace(self.observations[0], mask=Image.new("L", (5, 5))),
            self.observations[1],
        )
        with self.assertRaisesRegex(InspectionError, "dimensions"):
            validate_result(
                replace(self.result, observations=observations),
                InspectionRequest(self.views),
            )

    def test_missing_view_is_explicit_in_gallery(self):
        result = replace(
            self.result,
            observations=self.observations[:1],
            ranking=(
                replace(self.result.ranking[0], view_ids=(self.views[0].view_id,)),
            ),
        )
        gallery = evidence_gallery(Session(self.views, result), "peer_a", "Masks")
        self.assertEqual(len(gallery), 2)
        self.assertIn("No matching observation", gallery[1][1])

    def test_export_escapes_text_excludes_source_paths_and_cleans_up(self):
        session = Session(
            self.views, replace(self.result, reasons=("<script>alert(1)</script>",))
        )
        self.addCleanup(discard_exports, session)
        report, data = export_report(session, "<script>private title</script>")
        document = Path(report).read_text()
        record = json.loads(Path(data).read_text())
        self.assertNotIn("<script>", document)
        self.assertIn("&lt;script&gt;", document)
        self.assertIn("data:image/png;base64,", document)
        self.assertNotIn("private-source", Path(data).read_text())
        self.assertEqual(record["mode"], "automatic")
        discard_exports(session)
        self.assertFalse(Path(report).exists())

    def test_no_report_without_inference(self):
        with self.assertRaisesRegex(ValueError, "Run an inspection"):
            export_report(Session(self.views), "Scene")

    def test_backend_is_optional_and_invalid_config_fails(self):
        self.assertIsNone(load_backend(None))
        with self.assertRaisesRegex(ValueError, "module:factory"):
            load_backend("invalid")

    def test_ui_upload_and_reset_do_not_create_predictions(self):
        from visual_peer_inspector.web.ui import build_app

        app = build_app()
        functions = {f.fn.__name__: f.fn for f in app.fns.values() if f.fn}
        output = functions["prepare_uploads"](self.paths, Session())
        state = next(
            value for key, value in output.items() if key.get_block_name() == "state"
        )
        button = next(
            value for key, value in output.items() if key.elem_id == "inspect-button"
        )
        self.assertEqual(len(state.views), 2)
        self.assertIsNone(state.result)
        self.assertFalse(button["interactive"])
        reset = functions["start_over"](state)
        state = next(
            value for key, value in reset.items() if key.get_block_name() == "state"
        )
        self.assertFalse(state.views)

    def test_backend_callbacks_render_results_and_contain_failure(self):
        from visual_peer_inspector.web.ui import build_app

        expected = self.result

        class Backend:
            fail = False

            def inspect(self, request, progress):
                if self.fail:
                    raise RuntimeError("private /source/path")
                progress("Finding objects")
                return replace(
                    expected,
                    mode="assisted" if request.corrections is not None else "automatic",
                )

        backend = Backend()
        app = build_app(backend)
        callback = next(
            f.fn for f in app.fns.values() if f.fn and f.fn.__name__ == "do_inspect"
        )
        state = Session(self.views)
        callback(state)
        self.assertEqual(state.result, self.result)
        backend.fail = True
        outputs = callback(state)
        self.assertIsNone(state.result)
        self.assertNotIn("/source/path", str(outputs))
        self.assertTrue(any("Inspection failed" in str(output) for output in outputs.values()))


if __name__ == "__main__":
    unittest.main()
