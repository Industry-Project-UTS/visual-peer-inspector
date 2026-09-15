#!/usr/bin/env python3
"""Build the two clean notebooks from the same local command interface."""
from pathlib import Path
import textwrap
import nbformat

ROOT = Path(__file__).resolve().parents[1]
NAMES = ('08n-object-registry-and-patch-preparation.ipynb', '08o-object-bank-evaluation.ipynb')


def code(text):
    return nbformat.v4.new_code_cell(textwrap.dedent(text).strip())


def markdown(text):
    return nbformat.v4.new_markdown_cell(textwrap.dedent(text).strip())


def setup_cell():
    return code('''
        from pathlib import Path
        import json
        import os
        import sys

        def is_source(path):
            return (path / "scripts/run_object_bank.py").is_file() and (path / "src/vpi_object_bank").is_dir()

        def locate_source():
            explicit = os.environ.get("VPI_OBJECT_BANK_ROOT")
            if explicit:
                root = Path(explicit).expanduser().resolve()
                if not is_source(root):
                    raise RuntimeError("The configured source root is incomplete")
                return root
            for root in (Path.cwd(), *list(Path.cwd().parents)[:6]):
                if is_source(root):
                    return root
            candidates = set()
            input_root = Path("/kaggle/input")
            if input_root.is_dir():
                for pattern in ("*", "datasets/*/*", "datasets/*/*/versions/*", "notebooks/*/*"):
                    for mount in input_root.glob(pattern):
                        if not mount.is_dir():
                            continue
                        if is_source(mount):
                            candidates.add(mount.resolve())
                        children = list(mount.iterdir())
                        if len(children) <= 200:
                            candidates.update(child.resolve() for child in children if child.is_dir() and is_source(child))
            if len(candidates) != 1:
                raise RuntimeError("Expected exactly one complete object-bank source resource")
            return candidates.pop()

        SOURCE_ROOT = locate_source()
        CONFIG_PATH = Path(os.environ.get("VPI_OBJECT_BANK_CONFIG", SOURCE_ROOT / ".local/config.json")).expanduser().resolve()
        if not CONFIG_PATH.is_file():
            raise RuntimeError("Provide the verified local or Kaggle configuration JSON")
        configuration = json.loads(CONFIG_PATH.read_text())
        sys.path.insert(0, str(SOURCE_ROOT / "scripts"))
        sys.path.insert(0, str(SOURCE_ROOT / "src"))
        from scoring_comparison_runtime import run_bounded
        os.environ["PYTHONPATH"] = str(SOURCE_ROOT / "src")
        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
        LOG_ROOT = SOURCE_ROOT / ".local/logs"
        if str(SOURCE_ROOT).startswith("/kaggle/input/"):
            LOG_ROOT = Path("/kaggle/working/object-bank-logs")
        LOG_ROOT.mkdir(parents=True, exist_ok=True)

        def run_stage(stage):
            run_bounded(
                [sys.executable, str(SOURCE_ROOT / "scripts/run_object_bank.py"), stage, "--config", str(CONFIG_PATH)],
                LOG_ROOT / (stage + ".log"), idle_seconds=900, total_seconds=18000,
            )
        print({"event": "notebook_ready", "python": sys.version.split()[0]})
    ''')


def method_contract():
    return markdown('''
        ## Frozen comparison

        LocateAnything-3B, SAM 2.1 and DINOv2-small remain fixed. The existing 100-scene development screen supplies 19 predefined schedules: five each with 1, 2 and 3 views, and four with 5 views. Three and five views are the application settings; one and two views are diagnostics. Every prediction uses only its selected views. This bounded stage does not claim results for 10 or 20 views.

        | Route | Object regions | Cross-view identity | Interpretation |
        | --- | --- | --- | --- |
        | `supplied_reference` | Supplied masks | Supplied neutral correspondence | Scoring diagnostic |
        | `automatic_reference` | Saved LocateAnything and SAM outputs | Supplied neutral correspondence | Region diagnostic; assisted identity |
        | `automatic_predicted` | Saved LocateAnything and SAM outputs | Existing automatic association | Automatic application route |

        Anomaly labels are available only to the evaluator, after predictions are sealed. All valid candidates remain represented. An unmatched observation stays visible in the output; its object-bank score is unavailable because exclusion of another view of itself cannot be guaranteed. Missing observations are never replaced with normal scores. One object cannot contain two detections from the same view.

        **Patch comparison.** For each target foreground patch, compute the minimum cosine distance to a peer object's patches across that peer's selected views. Each peer contributes one distance. Take the lower median across peers, then average the largest `max(1, floor(0.02 × target patch count))` patch distances. Exclude the target's complete identity from all peer banks. At least two usable peers are required.

        Compare this object bank with regenerated within-view scoring and cached forward scoring. `view_mean_control` first averages nearest-patch distances across each peer's usable views. It diagnoses the extra matching opportunities in a larger bank; it is not a guarantee of invariance to bank size.

        **One physical-object ranking.** Convert available observation scores into average within-view percentile ranks when at least three candidates are scored. Compare the mean of all valid ranks with the mean of the strongest two (or the single available rank). Retain the existing winner-consensus method as a comparison. The scorer's raw-score means are diagnostics; final rankings use the declared rank normalization.

        **Decision gate.** Choose between the two bank fusions using the supplied-reference primary three-view development result: the paired asset-bootstrap 95% lower bound for top-1 improvement must be positive, AP and Recall@2 must not regress, and the mean top-1 improvement across the five three-view schedules must be positive. Prefer mean fusion on a tie. If the gate fails, stop the scoring extension and retain the stronger existing method. Results remain development evidence; 60% is a goal, not a promised outcome.

        Report tie-aware top-1, Recall@2, reciprocal rank and average precision, with asset-grouped uncertainty and explicit candidate, association and ranking losses. A scene contributes one result per setting; its views are not independent successes.
    ''')


def preparation_summary():
    return code('''
        from IPython.display import display
        import pandas as pd
        from vpi_object_bank.protocol import PROTOCOL

        # The protocol has no evaluator labels or machine-specific paths.
        display(PROTOCOL)
        schedule_rows = [{"schedule": key, "view_count": len(views), "views": ", ".join(views)}
                         for key, views in PROTOCOL["view_sets"].items()]
        display(pd.DataFrame(schedule_rows))
        feature_root = Path(configuration["feature_root"])
        feature_lock = json.loads((feature_root / "feature_lock.json").read_text())
        binding = json.loads((feature_root / "run_binding.json").read_text())
        progress = json.loads((feature_root / "progress.json").read_text())
        metadata = feature_lock.get("metadata", {})
        display({"sealed_scene_count": len(metadata.get("scenes", [])),
                 "cached_view_count": len(metadata.get("views", [])),
                 "completed_scene_region_view_units": metadata.get("completed_units"),
                 "last_extraction_progress": {key: progress.get(key) for key in
                      ("completed", "total", "extracted", "reused", "elapsed_seconds")},
                 "device": binding.get("device"), "runtime": binding.get("runtime"),
                 "model_file_sha256": binding.get("model_files")})
        print("The extraction cell reports zero extracted units when an identical sealed cache is reused.")
    ''')


def evaluation_summary():
    return code('''
        from IPython.display import display
        import pandas as pd

        aggregate_root = Path(configuration["evaluation_root"]) / "shareable"
        metrics = pd.read_csv(aggregate_root / "metric_summary.csv")
        schedules = pd.read_csv(aggregate_root / "view_schedule_summary.csv")
        display(metrics)
        schedule_column = next((name for name in ("view_set_id", "view_schedule", "schedule") if name in schedules), None)
        primary = schedules[schedules[schedule_column].astype(str).str.endswith("primary")] if schedule_column else schedules
        # Display the long form as well as a compact primary-schedule pivot.
        display(primary)
        value_column = next((name for name in ("estimate", "value", "mean", "point_estimate") if name in primary), None)
        if "metric" in primary and value_column:
            index_columns = [name for name in ("route", "method", "view_count", schedule_column) if name and name in primary]
            display(primary.pivot_table(index=index_columns, columns="metric", values=value_column, aggfunc="first"))
        else:
            value_columns = [name for name in ("top1_tie_expected", "recall_at_2_tie_expected", "ap_tie_expected", "reciprocal_rank_tie_expected") if name in primary]
            index_columns = [name for name in ("route", "method", "view_count", schedule_column) if name and name in primary]
            if value_columns and index_columns:
                display(primary.set_index(index_columns)[value_columns])
        display(pd.read_csv(aggregate_root / "paired_asset_intervals.csv"))
        display(pd.read_csv(aggregate_root / "failure_summary.csv"))
        decision = json.loads((aggregate_root / "decision.json").read_text())
        display(decision)
    ''')


def figure_summary():
    return code('''
        from IPython.display import Image, display

        figure_root = Path(configuration["report_root"]) / "shareable/figures"
        for name in ("view_count_accuracy.png", "paired_improvement.png", "failure_waterfall.png"):
            figure = figure_root / name
            if figure.is_file():
                display(Image(filename=str(figure)))
    ''')


def private_gallery():
    return code('''
        # Private review evidence: the publishing helper removes these image outputs.
        from IPython.display import Image, display

        private_root = Path(configuration["report_root"]) / "private"
        gallery = json.loads((private_root / "gallery_manifest.json").read_text())
        if gallery:
            first_scene = gallery[0]["scene"]
            examples = [item for item in gallery if item["scene"] == first_scene and item["method"] == "object_bank_mean"]
            for example in examples:
                print({"route": example["route"], "selection_reason": example["selection_reason"],
                       "top_tie_count": example["top_tie_count"], "unlocated_views": example["unlocated_views"]})
                display(Image(filename=str(private_root / example["file"])))
        print("Gallery heatmaps are coarse patch evidence, not validated defect masks.")
    ''')


def build_notebooks(output_root=ROOT / 'notebooks'):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    definitions = [
        ('08N: Object registry and reusable patch preparation',
         'Reuse the fixed model revision and saved masks. Preserve all valid object candidates and prepare selected-view object banks. Neutral correspondence is an explicitly assisted route. Anomaly labels remain in the evaluator.',
         [('Input preflight', 'preflight'), ('Reusable patch extraction', 'extract'),
          ('Object registries and predictions', 'predict'), ('Verify sealed predictions', 'verify')]),
        ('08O: Object-bank evaluation',
         'Evaluate the sealed predictions on the declared development cohort. Keep every expected scene, compare the predefined methods and routes, and save scientific results before reporting or tracking.',
         [('Evaluate frozen predictions', 'evaluate'), ('Report evidence and limitations', 'report'),
          ('Verify scientific results and reports', 'verify')]),
    ]
    outputs = []
    for index, (title, intro, stages) in enumerate(definitions):
        nb = nbformat.v4.new_notebook()
        nb.metadata = {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                       'language_info': {'name': 'python'}}
        nb.cells = [markdown('# ' + title + '\n\n' + intro), method_contract(), markdown('## Runtime and configuration'), setup_cell()]
        for heading, stage in stages:
            nb.cells += [markdown('## ' + heading), code(f'run_stage("{stage}")')]
        if index == 0:
            nb.cells += [markdown('## Protocol and reusable cache evidence'), preparation_summary()]
        if index == 1:
            nb.cells += [markdown('## Aggregate results and method decision'), evaluation_summary(),
                         markdown('## Aggregate figures'), figure_summary(),
                         markdown('## Private multi-view gallery\n\nThe fixed first scene is an outcome-independent example. Any largest-gain or largest-loss illustrations in the local report are post-hoc explanations, not a demonstration accuracy estimate.'), private_gallery()]
            nb.cells += [markdown('''
                ## Publish aggregate evidence

                Tracking runs in a separate process after verified scientific results. A tracking failure leaves a retryable receipt and does not repeat computation. Private examples remain in the local review artifacts.
            '''), code('''
                import subprocess
                # Display only the publisher's compact status, never third-party output.
                try:
                    publication = subprocess.run(
                        [sys.executable, str(SOURCE_ROOT / "scripts/publish_clearml.py"),
                         "--config", str(CONFIG_PATH), "--bounded", "--timeout", "180"],
                        capture_output=True, text=True, timeout=195,
                    )
                    tracking = json.loads(publication.stdout.strip().splitlines()[-1])
                    print({"event": "tracking_finished", "status": tracking.get("status", "pending"),
                           "task_id": tracking.get("task_id")})
                except (ValueError, IndexError, OSError, subprocess.TimeoutExpired):
                    print({"event": "tracking_finished", "status": "pending"})
            ''')]
        if index == 1:
            nb.cells[-2:-2] = [markdown("## Selected-method multi-view evidence\n\nThese private examples use the method chosen by the predefined gate. They illustrate its decision and limitations; they are not a new evaluation subset."), code('''
                # Private selected-method evidence is excluded from repository publication.
                sys.path.insert(0, str(SOURCE_ROOT / "src"))
                from vpi_object_bank.selected_demo import build_selected_demo
                selected_receipt = build_selected_demo(configuration, SOURCE_ROOT)
                print(selected_receipt)
                selected_root = Path(configuration.get("selected_demo_root", Path(configuration["report_root"]).parent / "selected-demo"))
                selected_images = sorted((selected_root / "private").glob("*-automatic_reference-*.png"))
                if selected_images:
                    from IPython.display import Image, display
                    display(Image(filename=str(selected_images[0])))
            ''')]
        nbformat.validate(nb)
        target = output_root / NAMES[index]
        nbformat.write(nb, target)
        outputs.append(target)
    return outputs


if __name__ == '__main__':
    for path in build_notebooks():
        print(path.name)
