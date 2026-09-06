from __future__ import annotations

import sys
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path

import numpy as np
from PIL import Image

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))

from visual_peer_inspector.qualitative_reporting import (
    EvaluatorObject,
    EvaluatorOverlay,
    EvaluatorView,
    PredictedObject,
    ProtectedFieldError,
    QualitativeResult,
    QualitativeView,
    render_evaluator_overlay_panel,
    render_prediction_panel,
    save_panel_png,
    select_publication_results,
)


def make_rgb(value: int) -> np.ndarray:
    image = np.full((48, 64, 3), value, dtype=np.uint8)
    image[12:36, 20:44] = (220, 150, 60)
    return image


def make_mask(offset: int = 0) -> np.ndarray:
    mask = np.zeros((48, 64), dtype=np.uint8)
    mask[12 + offset : 36 + offset, 20:44] = 1
    return mask


def make_view(view_id: str, value: int = 30) -> QualitativeView:
    return QualitativeView(
        view_id=view_id,
        rgb=make_rgb(value),
        objects=(
            PredictedObject(
                object_id="peer-a",
                box_xyxy=(18, 10, 46, 38),
                sam_mask=make_mask(),
                p1_score=0.82,
                p1_rank=1,
            ),
            PredictedObject(
                object_id="peer-b",
                box_xyxy=(3, 4, 17, 22),
                sam_mask=make_mask(2),
                p1_score=0.23,
                p1_rank=2,
            ),
        ),
        heatmap=np.arange(48 * 64, dtype=np.float32).reshape(48, 64),
    )


def make_result(result_id: str, view_count: int = 3) -> QualitativeResult:
    return QualitativeResult(
        result_id=result_id,
        views=tuple(
            make_view(f"view-{index}", 20 + 10 * index)
            for index in range(view_count)
        ),
        fused_odd_one_out="peer-a",
        agreement=0.8,
        margin=0.19,
        review_state="result_ready",
        publication_metadata={"pipeline_revision": "frozen-v1"},
    )


class QualitativeReportingTests(unittest.TestCase):
    def test_selection_and_rendering_are_deterministic(self) -> None:
        candidates = [make_result(f"scene-{index}") for index in range(8)]
        forward = select_publication_results(candidates, count=4)
        reverse = select_publication_results(list(reversed(candidates)), count=4)

        self.assertEqual(
            [item.result_id for item in forward],
            [item.result_id for item in reverse],
        )
        rendered_once = render_prediction_panel(forward[0], panel_size=128)
        reordered = QualitativeResult(
            result_id=forward[0].result_id,
            views=tuple(reversed(forward[0].views)),
            fused_odd_one_out=forward[0].fused_odd_one_out,
            agreement=forward[0].agreement,
            margin=forward[0].margin,
            review_state=forward[0].review_state,
            publication_metadata=forward[0].publication_metadata,
        )
        rendered_twice = render_prediction_panel(reordered, panel_size=128)
        self.assertEqual(
            sha256(rendered_once.tobytes()).hexdigest(),
            sha256(rendered_twice.tobytes()).hexdigest(),
        )
        with tempfile.TemporaryDirectory() as directory:
            first_path = Path(directory) / "first.png"
            second_path = Path(directory) / "second.png"
            save_panel_png(rendered_once, first_path)
            save_panel_png(rendered_twice, second_path)
            self.assertEqual(first_path.read_bytes(), second_path.read_bytes())

    def test_missing_view_renders_explicit_placeholders(self) -> None:
        result = QualitativeResult(
            result_id="scene-missing",
            views=(
                make_view("view-0"),
                QualitativeView(
                    view_id="view-1",
                    rgb=None,
                    missing_reason="upload was not supplied",
                ),
            ),
            fused_odd_one_out="peer-a",
            agreement=0.5,
            margin=0.1,
            review_state="review_required",
        )

        panel = render_prediction_panel(result, panel_size=128)

        self.assertGreater(panel.width, 256)
        self.assertGreater(panel.height, 500)

    def test_supports_every_web_input_count_from_two_to_five_views(self) -> None:
        for view_count in range(2, 6):
            with self.subTest(view_count=view_count):
                panel = render_prediction_panel(
                    make_result(f"scene-{view_count}", view_count),
                    panel_size=128,
                )
                self.assertEqual(panel.mode, "RGB")

        with self.assertRaisesRegex(ValueError, "between 2 and 5"):
            make_result("too-small", 1)
        with self.assertRaisesRegex(ValueError, "between 2 and 5"):
            make_result("too-large", 6)

    def test_empty_proposals_produce_a_valid_prediction_panel(self) -> None:
        result = QualitativeResult(
            result_id="scene-empty",
            views=(
                QualitativeView(view_id="view-0", rgb=make_rgb(10)),
                QualitativeView(view_id="view-1", rgb=make_rgb(20)),
            ),
            fused_odd_one_out=None,
            agreement=None,
            margin=None,
            review_state="insufficient_evidence",
        )

        panel = render_prediction_panel(result, panel_size=128)

        self.assertEqual(panel.mode, "RGB")

    def test_panel_contains_binary_mask_and_masked_object_rows(self) -> None:
        panel_size = 128
        panel = render_prediction_panel(make_result("mask-evidence", 2), panel_size=panel_size)
        header_height = 94
        label_height = 42
        row_gap = 26
        margin = 18

        def row_crop(row_index: int) -> np.ndarray:
            x = margin
            y = header_height + row_index * (panel_size + label_height + row_gap)
            return np.asarray(panel.crop((x, y, x + panel_size, y + panel_size)))

        binary = row_crop(2)
        isolated = row_crop(3)
        binary_content = binary[1:-1, 1:-1]
        isolated_content = isolated[1:-1, 1:-1]
        self.assertTrue(np.all(np.isin(np.unique(binary_content), [0, 255])))
        self.assertGreater(int(np.count_nonzero(binary)), 0)
        self.assertTrue(np.any(np.all(isolated_content == 0, axis=2)))
        self.assertTrue(np.any(np.any(isolated_content != 0, axis=2)))
        self.assertNotEqual(binary_content.tobytes(), isolated_content.tobytes())

    def test_missing_fused_object_has_explicit_black_evidence_rows(self) -> None:
        result = QualitativeResult(
            result_id="fused-missing-from-view",
            views=(
                make_view("view-0"),
                QualitativeView(
                    view_id="view-1",
                    rgb=make_rgb(50),
                    objects=(
                        PredictedObject(
                            object_id="peer-c",
                            box_xyxy=(18, 10, 46, 38),
                            sam_mask=make_mask(),
                            p1_score=0.4,
                            p1_rank=1,
                        ),
                    ),
                ),
            ),
            fused_odd_one_out="peer-a",
            agreement=0.5,
            margin=0.1,
            review_state="review_required",
        )

        panel = render_prediction_panel(result, panel_size=128)

        self.assertEqual(panel.mode, "RGB")
        self.assertGreater(panel.height, 900)

    def test_prediction_metadata_rejects_protected_fields_recursively(self) -> None:
        with self.assertRaises(ProtectedFieldError):
            QualitativeResult(
                result_id="unsafe",
                views=(make_view("view-0"), make_view("view-1")),
                fused_odd_one_out="peer-a",
                agreement=0.8,
                margin=0.2,
                review_state="result_ready",
                publication_metadata={
                    "summary": {"ground_truth_label": "peer-a"}
                },
            )

    def test_private_overlay_is_explicit_and_separate_from_prediction(self) -> None:
        result = make_result("scene-overlay", 2)
        evaluator = EvaluatorOverlay(
            views=(
                EvaluatorView(
                    view_id="view-0",
                    objects=(
                        EvaluatorObject(
                            object_id="peer-a",
                            box_xyxy=(17, 9, 47, 39),
                            mask=make_mask(),
                        ),
                    ),
                ),
            ),
            reference_odd_one_out="peer-a",
        )

        prediction = render_prediction_panel(result, panel_size=128)
        private = render_evaluator_overlay_panel(
            result,
            evaluator,
            panel_size=128,
        )

        self.assertNotEqual(prediction.tobytes(), private.tobytes())

    def test_private_header_lists_every_acceptable_anomaly_identity(self) -> None:
        evaluator = EvaluatorOverlay(
            views=(),
            acceptable_reference_ids=("peer-a", "peer-c", "peer-d"),
        )

        self.assertEqual(evaluator.reference_header, "peer-a, peer-c, peer-d")

        with self.assertRaisesRegex(ValueError, "must be unique"):
            EvaluatorOverlay(
                views=(),
                acceptable_reference_ids=("peer-a", "peer-a"),
            )

    def test_accepts_rgb_and_masks_from_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rgb_path = root / "rgb.png"
            mask_path = root / "mask.png"
            Image.fromarray(make_rgb(40), mode="RGB").save(rgb_path)
            Image.fromarray(make_mask() * 255, mode="L").save(mask_path)
            view = QualitativeView(
                view_id="view-0",
                rgb=rgb_path,
                objects=(
                    PredictedObject(
                        object_id="peer-a",
                        box_xyxy=(18, 10, 46, 38),
                        sam_mask=mask_path,
                        p1_score=0.75,
                        p1_rank=1,
                    ),
                ),
            )
            result = QualitativeResult(
                result_id="path-input",
                views=(view, make_view("view-1")),
                fused_odd_one_out="peer-a",
                agreement=0.7,
                margin=0.2,
                review_state="result_ready",
            )
            output = root / "panel.png"

            save_panel_png(render_prediction_panel(result, panel_size=128), output)

            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 100)


if __name__ == "__main__":
    unittest.main()
