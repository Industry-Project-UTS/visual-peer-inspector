"""Focused checks for peer support, geometry and the learned context ranker."""

import json
import sys
from hashlib import sha256
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from visual_peer_inspector.evaluation.context_ranker import (
    ContextGroup,
    assign_asset_folds,
    context_features,
    fit_context_ranker,
    score_context_group,
)
from visual_peer_inspector.scoring.peer_support import (
    geometry_peer_scores,
    mask_shape_descriptor,
    mutual_missing_scores,
    rank_consensus,
)


def test_mutual_missing_requires_peer_support():
    base = np.asarray([[1.0, 0.0], [1.0, 0.0]])
    complete = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    valid = [np.ones((1, 2), dtype=bool)] * 3
    result = mutual_missing_scores([base, complete, complete], valid, tail_fraction=0.5)
    assert result.status == "scored"
    assert result.scores[0] > result.scores[1]
    assert result.scores[0] > result.scores[2]


def test_geometry_score_marks_silhouette_outlier():
    normal = np.zeros((32, 32), dtype=bool)
    normal[8:24, 10:22] = True
    normal_two = np.zeros((32, 32), dtype=bool)
    normal_two[7:23, 9:21] = True
    outlier = np.zeros((32, 32), dtype=bool)
    outlier[14:18, 3:29] = True
    descriptors = [mask_shape_descriptor(mask) for mask in (normal, normal_two, outlier)]
    result = geometry_peer_scores(descriptors)
    assert result.scores[2] > result.scores[0]
    assert result.scores[2] > result.scores[1]


def test_rank_consensus_is_permutation_equivariant():
    arrays = ([0.2, 0.7, 0.4], [9.0, 1.0, 3.0], [2.0, 8.0, 4.0])
    permutation = [2, 0, 1]
    expected = rank_consensus(*arrays)
    changed = rank_consensus(*[[values[index] for index in permutation] for values in arrays])
    np.testing.assert_allclose(changed, expected[permutation])


def _context_object(value, identifier):
    methods = (
        "B0",
        "B1_control",
        "P1_current",
        "forward_10_cov0",
        "forward_02_cov70",
        "mutual_missing_05",
        "geometry_peer",
        "P2_consensus",
    )
    scores = {method: float(value + index * 0.01) for index, method in enumerate(methods)}
    return {
        "candidate_id": identifier,
        "scores": scores,
        "embedding": [1.0 - value, value, 0.2],
        "geometry": [2.0 + value, value, 0.5, value, 0.0, 1.0, 4.0 + value],
    }


def test_context_feature_schema_is_label_free_and_finite():
    methods = tuple(_context_object(0.0, "a")["scores"])
    matrix, names = context_features(
        [_context_object(0.1, "a"), _context_object(0.4, "b"), _context_object(0.9, "c")],
        methods,
    )
    assert matrix.shape == (3, len(names))
    assert np.isfinite(matrix).all()
    assert all("label" not in name and "category" not in name for name in names)


def test_scene_softmax_ranker_learns_a_repeatable_signal():
    groups = []
    for index in range(12):
        features = np.asarray([[0.0, 0.1], [0.2, 0.0], [1.5, 1.0]], dtype=np.float64)
        groups.append(ContextGroup(f"s{index}", "V00", "automatic", f"a{index}", ("n1", "n2", "odd"), features, np.asarray([0, 0, 1])))
    model = fit_context_ranker(groups, regularization=0.1)
    scores = score_context_group(groups[0], model)
    assert int(np.argmax(scores)) == 2
    assert model["training_groups_with_positive"] == 12


def test_asset_fold_assignment_is_complete_and_stable():
    assets = {f"A{index:02d}" for index in range(10)}
    first = assign_asset_folds(assets, 5)
    second = assign_asset_folds(set(reversed(sorted(assets))), 5)
    assert first == second
    assert set(first) == assets
    assert set(first.values()) == set(range(5))


def test_kaggle_mount_discovery_is_bounded(tmp_path):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    from kaggle_mounts import find_recovery, find_source

    source = tmp_path / "source-dataset" / "scoring-comparison"
    (source / "scripts").mkdir(parents=True)
    (source / "src/visual_peer_inspector").mkdir(parents=True)
    (source / "comparison_base.json").write_text("{}")
    (source / "scripts/run_scoring_comparison.py").write_text("")
    assert find_source(tmp_path) == source.resolve()
    assert find_recovery(tmp_path) is None


def test_kaggle_mount_discovery_accepts_zip_bin_recovery(tmp_path):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    from kaggle_mounts import find_recovery

    dataset = tmp_path / "recovery-dataset"
    dataset.mkdir()
    archive = dataset / "scoring-comparison-08g-recovery.zip.bin"
    manifest = dataset / "scoring-comparison-08g-recovery-manifest.json"
    archive.write_bytes(b"sealed recovery bytes")
    manifest.write_text("{}\n", encoding="utf-8")

    assert find_recovery(tmp_path) == (archive.resolve(), manifest.resolve())


def test_kaggle_mount_discovery_accepts_zip_recovery(tmp_path):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    from kaggle_mounts import find_recovery

    dataset = tmp_path / "recovery-dataset"
    dataset.mkdir()
    archive = dataset / "scoring-comparison-08g-recovery.zip"
    manifest = dataset / "scoring-comparison-08g-recovery-manifest.json"
    archive.write_bytes(b"sealed recovery bytes")
    manifest.write_text("{}\n", encoding="utf-8")

    assert find_recovery(tmp_path) == (archive.resolve(), manifest.resolve())


def test_kaggle_mount_discovery_rejects_manifest_without_archive(tmp_path):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    from kaggle_mounts import find_recovery

    dataset = tmp_path / "recovery-dataset"
    dataset.mkdir()
    (dataset / "scoring-comparison-08g-recovery-manifest.json").write_text("{}\n")

    try:
        find_recovery(tmp_path)
    except ValueError as error:
        assert "without a sealed archive" in str(error)
    else:
        raise AssertionError("A recovery manifest without an archive must fail closed")


def test_kaggle_mount_discovery_rejects_expanded_recovery(tmp_path):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    from kaggle_mounts import find_recovery

    dataset = tmp_path / "recovery-dataset"
    dataset.mkdir()
    (dataset / "scoring-comparison-08g-recovery").mkdir()
    (dataset / "scoring-comparison-08g-recovery-manifest.json").write_text("{}\n")

    try:
        find_recovery(tmp_path)
    except ValueError as error:
        assert "expanded by Kaggle" in str(error)
    else:
        raise AssertionError("An expanded recovery resource must fail closed")


def test_kaggle_mount_discovery_rejects_duplicate_pairs(tmp_path):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    from kaggle_mounts import find_recovery

    for index in range(2):
        dataset = tmp_path / f"recovery-dataset-{index}"
        dataset.mkdir()
        (dataset / f"run-{index}-recovery.zip.bin").write_bytes(b"sealed recovery bytes")
        (dataset / f"run-{index}-recovery-manifest.json").write_text("{}\n")

    try:
        find_recovery(tmp_path)
    except ValueError as error:
        assert "More than one" in str(error)
    else:
        raise AssertionError("Multiple recovery pairs must be rejected")


def test_kaggle_namespaced_mounts_find_expanded_resources(tmp_path):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    from kaggle_mounts import find_sealed_resource, find_source

    source = tmp_path / "datasets/user/source-slug/versions/2/scoring-comparison"
    (source / "scripts").mkdir(parents=True)
    (source / "src/visual_peer_inspector").mkdir(parents=True)
    (source / "comparison_base.json").write_text("{}")
    (source / "scripts/run_scoring_comparison.py").write_text("")

    model = tmp_path / "datasets/user/model-slug/versions/1/model-input"
    model.mkdir(parents=True)
    (model / "artifact_lock.json").write_text('{"metadata":{"kind":"model-input"}}\n')

    assert find_source(tmp_path) == source.resolve()
    assert find_sealed_resource(tmp_path, kind="model-input") == model.resolve()


def test_kaggle_mount_materializes_one_hash_bound_archive(tmp_path):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    from kaggle_mounts import materialize_sealed_resource

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    archive = dataset / "model-resource.zip"
    lock = b'{"metadata":{"kind":"model-input"}}\n'
    with ZipFile(archive, "w", compression=ZIP_DEFLATED) as handle:
        handle.writestr("model-input/artifact_lock.json", lock)
    result = materialize_sealed_resource(
        tmp_path,
        kind="model-input",
        archive_name=archive.name,
        archive_sha256=sha256(archive.read_bytes()).hexdigest(),
        destination=tmp_path / "working-resource",
        expected_lock_sha256=sha256(lock).hexdigest(),
    )
    assert result.name == "model-input"
    assert (result / "artifact_lock.json").read_bytes() == lock


def test_notebook_bootstrap_accepts_versioned_source_dataset_slugs():
    notebooks = Path(__file__).resolve().parents[1] / "notebooks"
    organised = notebooks / "representation-and-scoring" / "development-scoring-comparison"
    expected = "part.startswith('vpi-development-scoring-source')"
    paths = (
        [
            organised / "08g-development-scoring-comparison.ipynb",
            organised / "08h-development-scoring-evaluation.ipynb",
        ]
        if organised.is_dir()
        else [
            notebooks / "08g_development_scoring_comparison.ipynb",
            notebooks / "08h_development_scoring_evaluation.ipynb",
        ]
    )
    for path in paths:
        assert path.is_file()
        notebook = json.loads(path.read_text())
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )
        assert expected in source
