from __future__ import annotations

import ast
from pathlib import Path

import nbformat


def notebook_root() -> Path:
    repository_root = Path(__file__).parents[1]
    final_location = (
        repository_root
        / "notebooks/representation-and-scoring/development-learned-peer-view-ranker"
    )
    return final_location if final_location.is_dir() else repository_root / "notebooks"


def test_notebooks_are_clean_and_compile():
    root = notebook_root()
    names = [
        "08l-development-learned-peer-view-features.ipynb",
        "08m-development-learned-peer-view-ranker.ipynb",
    ]
    paths = [root / name for name in names]
    assert all(path.is_file() for path in paths)
    for path in paths:
        notebook = nbformat.read(path, as_version=4)
        assert len(notebook.cells) >= 10
        for cell in notebook.cells:
            assert cell.get("outputs", []) == []
            assert cell.get("execution_count") is None
            if cell.cell_type == "code":
                ast.parse(cell.source)
        joined = "\n".join(cell.source for cell in notebook.cells)
        assert path.name in joined
        assert "calibration" in joined.lower()
        assert "internal-lock" in joined.lower()
        assert "source-manifest.json" in joined
        assert "archive_sha256" in joined
        assert "Unsafe source archive member" in joined


def test_08l_has_no_evaluator_access():
    path = notebook_root() / "08l-development-learned-peer-view-features.ipynb"
    notebook = nbformat.read(path, as_version=4)
    code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code").lower()
    assert "evaluator_root" not in code
    assert "correspondence" not in code
    assert "labels" not in code


def test_08m_runs_tracking_after_evaluation_and_packaging():
    path = notebook_root() / "08m-development-learned-peer-view-ranker.ipynb"
    notebook = nbformat.read(path, as_version=4)
    text = "\n".join(cell.source for cell in notebook.cells)
    assert text.index("run_metric_evaluation.py") < text.index("package_metric_stage.py")
    assert text.index("package_metric_stage.py") < text.index("log_learned_metric_clearml.py")
