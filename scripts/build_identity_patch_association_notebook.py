"""Build Notebook 08K for the final bounded 2D identity comparison."""

from __future__ import annotations

from pathlib import Path
import textwrap

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "08k-development-patch-correspondence.ipynb"


def markdown(value: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(value).strip())


def code(value: str):
    return nbf.v4.new_code_cell(textwrap.dedent(value).strip())


def build() -> None:
    cells = [
        markdown(
            """
            # 08K: Development foreground patch correspondence

            This notebook tests one final 2D improvement to cross-view object identity. It keeps the existing LocateAnything, SAM 2.1, DINOv2-small and P1 pipeline fixed. It does not change the 2,000-scene experiment or use calibration or internal-lock data.

            The comparison uses the sealed 100-scene development screen. The current global DINO matching is compared with reciprocal foreground patches, dustbin Sinkhorn patch transport, and an order-free track assembly using the same transport cost.

            Notebook file: `08k-development-patch-correspondence.ipynb`.
            """
        ),
        markdown(
            """
            ## Scientific boundary

            DINO patch tokens are selected only where the automatic SAM mask covers at least half of a patch. Global and patch distances receive equal weight. Every candidate keeps the existing unmatched cost of 0.35. Sinkhorn uses an epsilon of 0.05, 30 iterations and an explicit dustbin state.

            The primary measure is cross-view identity-link F1. Average precision, top-1 accuracy, recall at two, track purity, identity switches, linked coverage, singleton rate, runtime and memory are reported alongside it. The promotion rule rejects gains caused by track fragmentation. There is no arbitrary accuracy gate.
            """
        ),
        markdown("## Runtime and source discovery"),
        code(
            """
            from pathlib import Path, PurePosixPath
            from zipfile import ZIP_DEFLATED, ZipFile
            import json
            import os
            import shutil
            import stat
            import subprocess
            import sys

            INPUT_ROOT = Path('/kaggle/input')
            WORKING_ROOT = Path('/kaggle/working')
            PACKAGE_ROOT = Path('/tmp/identity-patch-packages')
            PREDICTION_OUTPUT = WORKING_ROOT / 'identity-patch-predictions'
            EVALUATION_OUTPUT = WORKING_ROOT / 'identity-patch-evaluation'
            PYTHON = Path(sys.executable)
            EXPECT_RECOVERY = False
            os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

            mount_roots = []
            for pattern in ('*', 'datasets/*/*', 'datasets/*/*/versions/*', 'notebooks/*/*', 'notebooks/*/*/versions/*'):
                mount_roots.extend(path.resolve() for path in INPUT_ROOT.glob(pattern) if path.is_dir())
            mount_roots = sorted(set(mount_roots))
            source_archives = sorted({
                path for mount in mount_roots
                for path in mount.rglob('identity-patch-association-source-*.zip')
                if path.is_file()
            })
            if len(source_archives) > 1:
                raise ValueError(f'Expected at most one patch-association source archive, found {len(source_archives)}')

            if source_archives:
                extraction = WORKING_ROOT / 'identity-patch-source'
                temporary = extraction.with_name(extraction.name + '.incomplete')
                if extraction.exists() or temporary.exists():
                    raise FileExistsError('Source extraction destination is not empty')
                temporary.mkdir(parents=True)
                with ZipFile(source_archives[0]) as handle:
                    entries = handle.infolist()
                    names = [entry.filename for entry in entries]
                    if not entries or len(names) != len(set(names)) or handle.testzip() is not None:
                        raise ValueError('Source archive is empty, duplicated or corrupt')
                    for entry in entries:
                        relative = PurePosixPath(entry.filename)
                        mode = entry.external_attr >> 16
                        if entry.is_dir() or relative.is_absolute() or any(part in ('', '.', '..') for part in relative.parts) or stat.S_ISLNK(mode):
                            raise ValueError('Source archive contains an unsafe path')
                        destination = temporary.joinpath(*relative.parts)
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        with handle.open(entry) as source_stream, destination.open('xb') as target_stream:
                            shutil.copyfileobj(source_stream, target_stream, length=1024 * 1024)
                temporary.replace(extraction)
                search_roots = [extraction]
                source_label = source_archives[0].name
            else:
                search_roots = mount_roots
                source_label = 'expanded Kaggle dataset'

            source_candidates = sorted({
                path.parents[3]
                for root in search_roots
                for path in root.rglob('patch_correspondence.py')
                if (path.parents[3] / 'scripts/run_identity_patch_association.py').is_file()
            })
            if len(source_candidates) != 1:
                raise ValueError(f'Expected one source root, found {len(source_candidates)}')
            SOURCE = source_candidates[0]
            sys.path.insert(0, str(SOURCE / 'scripts'))
            from kaggle_mounts import find_prediction, find_recovery, materialize_sealed_resource
            from scoring_comparison_runtime import run_bounded, tracking_credentials
            print(json.dumps({'event': 'source_ready', 'source': source_label}, sort_keys=True))
            """
        ),
        markdown(
            """
            ## Method references

            The reciprocal comparison follows the mutual-nearest-neighbour idea used by Best-Buddies Similarity. The transport comparison combines foreground-weighted semantic correspondence with the dustbin assignment pattern used for partial matching. DINOv2-small remains the frozen visual representation used by the current project.

            - Dekel et al., *Best-Buddies Similarity for Robust Template Matching*, CVPR 2015.
            - Liu et al., *Semantic Correspondence as an Optimal Transport Problem*, CVPR 2020.
            - Sarlin et al., *SuperGlue: Learning Feature Matching with Graph Neural Networks*, CVPR 2020.
            - Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision*, 2023.
            """
        ),
        markdown(
            """
            ## Dependency and contract checks

            The notebook keeps Kaggle's CUDA-compatible PyTorch build. It installs only pinned add-ons into a run-local folder, compiles the relevant modules and runs the focused unit and workflow-contract tests before loading the model.
            """
        ),
        code(
            """
            PACKAGE_ROOT.mkdir(parents=True, exist_ok=True)
            subprocess.run([
                str(PYTHON), '-m', 'pip', 'install', '--disable-pip-version-check',
                '--retries', '1', '--timeout', '60', '--only-binary=:all:',
                '--upgrade', '--target', str(PACKAGE_ROOT),
                '-r', str(SOURCE / 'requirements/identity-patch-association.txt'),
            ], check=True, timeout=600)
            os.environ['PYTHONPATH'] = os.pathsep.join([
                str(SOURCE / 'src'), str(PACKAGE_ROOT), os.environ.get('PYTHONPATH', '')
            ]).rstrip(os.pathsep)
            os.environ['MPLCONFIGDIR'] = str(WORKING_ROOT / 'matplotlib')
            sys.path.insert(0, str(PACKAGE_ROOT))
            RUNNER = SOURCE / 'scripts/run_identity_patch_association.py'
            test_env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
            subprocess.run([
                str(PYTHON), '-m', 'pytest', '-p', 'no:cacheprovider', '-q',
                str(SOURCE / 'tests/test_patch_correspondence.py'),
                str(SOURCE / 'tests/test_identity_patch_association_contract.py'),
            ], check=True, cwd=WORKING_ROOT, env=test_env, timeout=300)
            for source_file in (
                SOURCE / 'src/visual_peer_inspector/representation/patch_correspondence.py',
                SOURCE / 'src/visual_peer_inspector/workflows/identity_patch_association_study.py',
                RUNNER,
            ):
                compile(source_file.read_text(encoding='utf-8'), str(source_file), 'exec')
            """
        ),
        markdown("## Locked development inputs"),
        code(
            """
            MODEL_INPUT = materialize_sealed_resource(
                INPUT_ROOT,
                kind='model-input',
                archive_name='scoring-comparison-model-input.zip',
                archive_sha256='6d6b45aa211ea595524729123870307ee5295ada48e6fd8f1898935f477b6901',
                destination=WORKING_ROOT / 'identity-patch-model-resource',
                expected_lock_sha256='9baa53454d2482caf8240e7a8b591a6e9e9646dd160c5b3abdc8bd17265ebda1',
            )
            from hashlib import sha256
            model_lock_sha256 = sha256((MODEL_INPUT / 'artifact_lock.json').read_bytes()).hexdigest()
            SCORING = find_prediction(INPUT_ROOT, expected_input_lock_sha256=model_lock_sha256)
            if not (SCORING / 'feature_lock.json').is_file():
                raise FileNotFoundError('Attach the completed 08G output with its feature lock')
            run_bounded([
                str(PYTHON), str(RUNNER), 'preflight',
                '--model-input', str(MODEL_INPUT), '--scoring', str(SCORING),
                '--output', str(PREDICTION_OUTPUT),
            ], WORKING_ROOT / 'identity-patch-preflight.log', idle_seconds=300, total_seconds=900)
            """
        ),
        markdown(
            """
            ## Checked recovery plan

            Every completed scene is written atomically with a checksum receipt. Recovery is accepted only when the source, inputs, dependencies and protocol match exactly. A fresh run ignores unrelated recovery files mounted by Kaggle.
            """
        ),
        code(
            """
            RECOVERY = find_recovery(INPUT_ROOT) if EXPECT_RECOVERY else None
            if EXPECT_RECOVERY and RECOVERY is None:
                raise FileNotFoundError('This continuation requires one verified recovery archive and manifest')
            recovery_arguments = [] if RECOVERY is None else [
                '--recovery-bundle', str(RECOVERY[0]),
                '--recovery-manifest', str(RECOVERY[1]),
            ]
            print(json.dumps({
                'event': 'recovery_plan',
                'expected': EXPECT_RECOVERY,
                'attached': RECOVERY is not None,
            }, sort_keys=True))
            """
        ),
        markdown(
            """
            ## Lock label-free identity predictions

            One GPU extracts DINOv2-small patch tokens from the eight fixed views. The existing global embeddings and anomaly scores are reused. A 15-minute silent limit stops a stalled process and a four-hour total limit protects GPU time. Completed scene checkpoints remain reusable after an operational interruption.
            """
        ),
        code(
            """
            run_bounded([
                str(PYTHON), str(RUNNER), 'predict',
                '--model-input', str(MODEL_INPUT), '--scoring', str(SCORING),
                '--output', str(PREDICTION_OUTPUT), *recovery_arguments,
            ], WORKING_ROOT / 'identity-patch-prediction.log', idle_seconds=900, total_seconds=14400)
            run_bounded([
                str(PYTHON), str(RUNNER), 'verify',
                '--model-input', str(MODEL_INPUT), '--scoring', str(SCORING),
                '--output', str(PREDICTION_OUTPUT),
            ], WORKING_ROOT / 'identity-patch-prediction-verification.log', idle_seconds=300, total_seconds=600)
            display(json.loads((PREDICTION_OUTPUT / 'run_status.json').read_text()))
            display(json.loads((PREDICTION_OUTPUT / 'progress.json').read_text()))
            """
        ),
        markdown(
            """
            ## Development-only evaluation

            The evaluator package is opened only after the prediction lock exists and verifies. It measures identity quality and odd-one-out ranking without changing any prediction.
            """
        ),
        code(
            """
            EVALUATOR = materialize_sealed_resource(
                INPUT_ROOT,
                kind='evaluator-only',
                archive_name='scoring-comparison-evaluator-only.zip',
                archive_sha256='86ac512005af5fdceb64c0fb666998832b8ee08ef96d1fccca23c6e35f420c12',
                destination=WORKING_ROOT / 'identity-patch-evaluator-resource',
                expected_lock_sha256='8012cd20c63b3788f861f2ba991b3181a0c1666f26e176beeee370228c8500c3',
            )
            run_bounded([
                str(PYTHON), str(RUNNER), 'evaluate',
                '--model-input', str(MODEL_INPUT), '--scoring', str(SCORING),
                '--predictions', str(PREDICTION_OUTPUT), '--evaluator', str(EVALUATOR),
                '--output', str(EVALUATION_OUTPUT),
            ], WORKING_ROOT / 'identity-patch-evaluation.log', idle_seconds=900, total_seconds=7200)
            run_bounded([
                str(PYTHON), str(RUNNER), 'verify',
                '--model-input', str(MODEL_INPUT), '--scoring', str(SCORING),
                '--output', str(EVALUATION_OUTPUT),
            ], WORKING_ROOT / 'identity-patch-evaluation-verification.log', idle_seconds=300, total_seconds=600)
            """
        ),
        markdown("## Results and decision"),
        code(
            """
            import pandas as pd
            for file_name in (
                'primary_candidate_ranking.csv',
                'patch_association_summary.csv',
                'paired_asset_intervals.csv',
                'identity_coverage_vs_error.csv',
                'patch_fallback_summary.csv',
                'category_summary.csv',
                'family_summary.csv',
            ):
                print(file_name)
                display(pd.read_csv(EVALUATION_OUTPUT / 'shareable' / file_name))
            display(json.loads((EVALUATION_OUTPUT / 'shareable/development_decision.json').read_text()))
            """
        ),
        markdown(
            """
            ## Figures and private examples

            Summary figures compare identity quality, odd-one-out accuracy, and the wrong-link rate as lower-confidence links are removed. The private gallery shows fixed improvements, regressions and representative cases. Red marks the selected identity, green marks the reference unusual object, and blue marks other automatic candidates.
            """
        ),
        code(
            """
            from IPython.display import Image as NotebookImage, display
            for figure in sorted((EVALUATION_OUTPUT / 'shareable/figures').glob('*.png')):
                print(figure.name)
                display(NotebookImage(filename=str(figure)))
            gallery = json.loads((EVALUATION_OUTPUT / 'private/gallery_manifest.json').read_text())
            for item in gallery['examples']:
                print(item['file'], item['selection_reason'])
                display(NotebookImage(filename=str(EVALUATION_OUTPUT / 'private/figures' / item['file'])))
            """
        ),
        markdown(
            """
            ## Post-compute ClearML upload

            ClearML runs only after the scientific outputs are sealed. It receives aggregate tables, decisions and summary figures. Private images, labels, per-scene records, model inputs and credentials are excluded. Tracking failure cannot trigger another GPU run.
            """
        ),
        code(
            """
            if tracking_credentials():
                try:
                    subprocess.run([
                        str(PYTHON), '-m', 'pip', 'install', '--retries', '1', '--timeout', '30',
                        '--only-binary=:all:', '--upgrade', '--target', str(PACKAGE_ROOT),
                        '-r', str(SOURCE / 'requirements/identity-patch-association-tracking.txt'),
                    ], check=True, timeout=180)
                    subprocess.run([
                        str(PYTHON), str(SOURCE / 'scripts/log_identity_patch_association.py'),
                        '--evidence', str(EVALUATION_OUTPUT),
                    ], check=True, timeout=300)
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
                    print(f'ClearML upload remains retryable: {type(error).__name__}')
            else:
                print('ClearML secrets are unavailable. Scientific outputs are complete and unchanged.')
            """
        ),
        markdown("## Download package"),
        code(
            """
            RESULT_ZIP = WORKING_ROOT / 'identity-patch-study-results.zip'
            with ZipFile(RESULT_ZIP, 'w', compression=ZIP_DEFLATED, compresslevel=6) as handle:
                for root in (PREDICTION_OUTPUT, EVALUATION_OUTPUT):
                    for path in sorted(root.rglob('*')):
                        if path.is_file():
                            handle.write(path, Path(root.name) / path.relative_to(root))
                for path in sorted(WORKING_ROOT.glob('identity-patch-*.log')):
                    handle.write(path, Path('logs') / path.name)
            with ZipFile(RESULT_ZIP) as handle:
                if handle.testzip() is not None:
                    raise ValueError('The result archive failed its integrity check')
            print(json.dumps({
                'result_zip': str(RESULT_ZIP),
                'size_bytes': RESULT_ZIP.stat().st_size,
                'prediction_output': str(PREDICTION_OUTPUT),
                'evaluation_output': str(EVALUATION_OUTPUT),
                'decision': json.loads((EVALUATION_OUTPUT / 'shareable/development_decision.json').read_text()),
            }, indent=2, sort_keys=True))
            """
        ),
        markdown(
            """
            ## Interpretation rule

            Promote a candidate only if the paired asset-group interval improves identity-link F1 and average precision, top-1 and recall at two do not regress, and linked coverage and singleton rate remain acceptable. Otherwise retain the current Hungarian method and use the confidence evidence to identify cases that need optional review.
            """
        ),
    ]
    notebook = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
    )
    nbf.validate(notebook)
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, NOTEBOOK)
    print(f"Built {NOTEBOOK.name} with {len(cells)} cells")


if __name__ == "__main__":
    build()
