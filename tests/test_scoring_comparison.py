"""Contracts for scoring, isolation, denominators and checkpoint recovery."""

import json
from pathlib import Path
import shutil
import sys
import tempfile

import nbformat
import numpy as np
import pytest

from visual_peer_inspector.scoring.forward_patch import forward_patch_scores
from visual_peer_inspector.scoring.peer import p1_scores
from visual_peer_inspector.workflows.comparison_io import atomic_json, read_json, seal, verify_seal
from visual_peer_inspector.workflows.comparison_evaluation import metrics, normalized_scores, track_references


def scoring_notebooks() -> list[Path]:
    root = Path(__file__).resolve().parents[1]
    organised = root / "notebooks" / "representation-and-scoring" / "development-scoring-comparison"
    if organised.is_dir():
        return [
            organised / "08g-development-scoring-comparison.ipynb",
            organised / "08h-development-scoring-evaluation.ipynb",
        ]
    return [
        root / "notebooks" / "08g_development_scoring_comparison.ipynb",
        root / "notebooks" / "08h_development_scoring_evaluation.ipynb",
    ]


@pytest.mark.parametrize("count", [0, 1, 2])
def test_forward_retains_insufficient_peers(count):
    result = forward_patch_scores([np.ones((4, 3))]*count, [np.ones((2,2), bool)]*count, top_fraction=.02)
    assert result.status == "insufficient_peers"
    assert len(result.scores) == count


def test_forward_lower_median_matches_manual_rule():
    angles = [0, .3, 1.4]
    features = [np.array([[np.cos(a), np.sin(a)]]) for a in angles]
    result = forward_patch_scores(features, [np.ones((1,1), bool)]*3, top_fraction=.02)
    expected = [min(1-np.cos(a-b) for b in angles if a != b) for a in angles]
    np.testing.assert_allclose(result.scores, expected)


def test_forward_floor_tail_and_different_current_reduction():
    rng = np.random.default_rng(4)
    features = [rng.normal(size=(16,6)) for _ in range(4)]
    valid = [np.ones((4,4), bool)]*4
    output = forward_patch_scores(features, valid, top_fraction=.10)
    for score, heatmap in zip(output.scores, output.heatmaps, strict=True):
        assert score == pytest.approx(heatmap.max())  # floor(16 * .10) = 1
    assert not np.allclose(output.scores, p1_scores(features, valid).scores)


def test_forward_permutation_equivariance():
    rng = np.random.default_rng(8)
    features = [rng.normal(size=(9,6)) for _ in range(5)]
    valid = [np.ones((3,3), bool)]*5
    permutation = [4,1,3,0,2]
    a = forward_patch_scores(features, valid, top_fraction=.3)
    b = forward_patch_scores([features[i] for i in permutation], valid, top_fraction=.3)
    np.testing.assert_allclose(a.scores[permutation], b.scores)


@pytest.mark.parametrize("fraction", [0, -1, 1.1])
def test_invalid_tail_rejected(fraction):
    with pytest.raises(ValueError):
        forward_patch_scores([], [], top_fraction=fraction)


def test_missing_positive_and_duplicate_candidate_do_not_inflate_ap():
    items = [{"id":"a", "score":3, "reference":"odd1"}, {"id":"b", "score":2, "reference":"odd1"}, {"id":"c", "score":1, "reference":None}]
    result = metrics(items, {"odd1":1, "odd2":1, "normal":0})
    assert result["ap"] == .5
    assert result["recall_at_2"] == .5


def test_empty_prediction_remains_zero_not_nan():
    result = metrics([], {"odd":1})
    assert result["ap"] == result["top1"] == result["recall_at_2"] == result["auroc"] == 0


def test_missing_positive_counts_as_lost_auc_pairs():
    result = metrics(
        [{"id": "normal", "score": 1.0, "reference": "normal"}],
        {"odd": 1, "normal": 0},
    )
    assert result["auroc"] == 0.0


def test_tie_sensitivity():
    result = metrics([{"id":"a", "score":1, "reference":"normal"}, {"id":"b", "score":1, "reference":"odd"}], {"normal":0,"odd":1})
    assert result["top1"] == 0
    assert result["top1_tie_expected"] == .5
    assert result["top1_tie_best"] == 1
    assert result["top1_tie_worst"] == 0


def test_normalized_score_ties_have_equal_rank():
    values = [{"candidate_id":str(i), "scores":{"P1_current":v}} for i,v in enumerate([1,1,3])]
    result = normalized_scores(values, "P1_current")
    assert result == {"0":.25,"1":.25,"2":1.0}


def test_mixed_identity_is_not_credited_for_any_positive_member():
    groups = [("track", [("V00","a"),("V01","b")])]
    mapping = {"automatic/V00/a":"odd", "automatic/V01/b":"normal"}
    assert track_references(groups, mapping, "automatic", {"odd":1,"normal":0})["track"][0] is None


def test_duplicate_tracks_cannot_both_claim_one_object():
    groups = [("t1", [("V00","a")]), ("t2", [("V01","b")])]
    mapping = {"automatic/V00/a":"odd", "automatic/V01/b":"odd"}
    result = track_references(groups, mapping, "automatic", {"odd":1})
    assert sum(ref == "odd" for ref, purity in result.values()) == 1


def test_checksum_tamper_rejected(tmp_path):
    atomic_json(tmp_path / "one.json", {"valid":1})
    seal(tmp_path, [tmp_path / "one.json"])
    atomic_json(tmp_path / "one.json", {"valid":2})
    with pytest.raises(ValueError, match="Checksum"):
        verify_seal(tmp_path)


def test_path_escape_rejected(tmp_path):
    from visual_peer_inspector.workflows.comparison_io import safe_path
    with pytest.raises(ValueError):
        safe_path(tmp_path, "../../etc/passwd")


def test_notebooks_are_valid_python_and_unexecuted():
    for path in scoring_notebooks():
        assert path.is_file()
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        assert len(notebook.cells) >= 20
        for cell in notebook.cells:
            if cell.cell_type == "code":
                compile(cell.source, path.name, "exec")
                assert cell.execution_count is None and cell.outputs == []
            assert "\u2014" not in cell.source


def test_notebooks_support_current_kaggle_mounts_without_venv():
    for path in scoring_notebooks():
        assert path.is_file()
        notebook = nbformat.read(path, as_version=4)
        source = "\n".join(cell.source for cell in notebook.cells)
        assert "datasets/*/*" in source
        assert "comparison-packages" in source
        assert "'-m', 'venv'" not in source


def test_prediction_module_does_not_import_evaluator():
    import visual_peer_inspector.workflows.scoring_comparison as module
    source = Path(module.__file__).read_text()
    assert "comparison_evaluation" not in source
    assert "labels_by_object_id" not in source
    assert "evaluator_root" not in source


def test_runtime_timeout(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from scoring_comparison_runtime import run_bounded
    with pytest.raises(TimeoutError):
        run_bounded([sys.executable, "-c", "import time; time.sleep(10)"], tmp_path / "worker.log", idle_seconds=.1, total_seconds=1)


def test_runtime_nonzero_exit(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from scoring_comparison_runtime import run_bounded
    with pytest.raises(RuntimeError):
        run_bounded([sys.executable, "-c", "raise SystemExit(3)"], tmp_path / "worker.log", total_seconds=5)


def test_supplied_mask_ignores_nonzero_rendered_background(tmp_path):
    from PIL import Image
    from visual_peer_inspector.workflows.comparison_io import digest_file
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from prepare_scoring_comparison import supplied_mask
    pixels = np.ones((256,256,4), dtype=np.uint8)
    pixels[:,:,3] = 255
    pixels[20:40,30:50,:3] = 206
    path = tmp_path / "rendered.png"
    Image.fromarray(pixels).save(path)
    ref = {"mask_sha256":digest_file(path),"foreground_pixels":400,"box_xyxy":[30,20,50,40]}
    assert supplied_mask(path, ref).sum() == 400
    ref["foreground_pixels"] = 399
    with pytest.raises(ValueError, match="geometry"):
        supplied_mask(path, ref)


def test_nonfinite_score_rejected():
    with pytest.raises(ValueError, match="finite"):
        metrics([{"id":"a", "score":np.nan, "reference":"odd"}], {"odd":1})


def test_duplicate_rank_token_rejected():
    with pytest.raises(ValueError, match="unique"):
        metrics([{"id":"a", "score":1}, {"id":"a", "score":2}], {"odd":1})
