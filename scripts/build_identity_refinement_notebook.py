"""Build the reviewed identity-refinement Kaggle notebook."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import textwrap

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = (
    ROOT
    / "notebooks"
    / "representation-and-scoring"
    / "development-identity-refinement"
    / "08i-development-identity-refinement.ipynb"
)


def markdown(value: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(value).strip())


def code(value: str):
    return nbf.v4.new_code_cell(textwrap.dedent(value).strip())


def build() -> None:
    cells = [
        markdown(
            """
            # 08I: Development cross-view identity refinement

            This notebook tests whether foreground DINO patch pooling improves object association across uploaded views. It uses the existing 100-scene development screen and does not change the 2,000-scene split.

            DINOv2-small, LocateAnything proposals, SAM 2.1 masks and all existing scorer outputs remain fixed. The notebook reuses the locked 08G global embeddings and only computes the foreground patch descriptors that were not retained after scoring.

            Notebook file: `08i-development-identity-refinement.ipynb`.
            """
        ),
        markdown(
            """
            ## Study boundary

            The automatic route is the main route because the web application will not receive reference boxes or masks. The notebook compares two, three and five selected views. It keeps the current global DINO identity method as the baseline and adds local and hybrid descriptors built from foreground patch tokens.

            Lock prediction-only identity candidates before any development correspondence is opened. Only then open development correspondence for evaluation. Labels, categories, asset groups and source object identities cannot enter descriptor extraction or matching.

            No arbitrary accuracy gate is used. Report actual identity precision, identity recall, identity F1, average precision, top-1 accuracy, recall at two and subgroup results. The highest development row is a candidate for review, not an automatic promotion or an untouched test result.
            """
        ),
        markdown(
            """
            ## Paths and runtime settings

            Attach four private resources: the identity-refinement source package, the existing scoring model-input package, the completed 08G screen output and the matching development evaluator. A recovery archive is optional for a fresh run and required only for an intentional continuation.
            """
        ),
        code(
            """
            from pathlib import Path, PurePosixPath
            from hashlib import sha256
            from zipfile import ZipFile
            import json
            import os
            import shutil
            import stat
            import subprocess
            import sys

            INPUT_ROOT = Path('/kaggle/input')
            WORKING_ROOT = Path('/kaggle/working')
            PACKAGE_ROOT = WORKING_ROOT / 'identity-packages'
            PREDICTION_OUTPUT = WORKING_ROOT / 'identity-refinement-predictions'
            EVALUATION_OUTPUT = WORKING_ROOT / 'identity-refinement-evaluation'
            PYTHON = Path(sys.executable)
            EXPECT_RECOVERY = False

            mount_roots = []
            for pattern in ('*', 'datasets/*/*', 'datasets/*/*/versions/*', 'notebooks/*/*', 'notebooks/*/*/versions/*'):
                mount_roots.extend(path.resolve() for path in INPUT_ROOT.glob(pattern) if path.is_dir())
            mount_roots = sorted(set(mount_roots))
            source_mounts = [path for path in mount_roots if any(part.startswith('vpi-identity-refinement-source') for part in path.parts[-4:])]
            source_markers = []
            for source_mount in source_mounts:
                source_markers.extend(path for path in source_mount.rglob('identity_refinement.py') if path.is_file())
            source_candidates = [path.parents[3] for path in sorted(set(source_markers)) if (path.parents[3] / 'scripts/run_identity_refinement.py').is_file()]
            if not source_candidates:
                archives = []
                for source_mount in source_mounts:
                    archives.extend(path for path in source_mount.rglob('identity-refinement-source-*.zip') if path.is_file())
                archives = sorted(set(archives))
                if len(archives) != 1:
                    raise ValueError(f'Expected one identity-refinement source archive, found {len(archives)}')
                extraction = WORKING_ROOT / 'identity-source'
                temporary = extraction.with_name(extraction.name + '.incomplete')
                if extraction.exists() or temporary.exists():
                    raise FileExistsError('Source extraction destination is not empty')
                temporary.mkdir(parents=True)
                with ZipFile(archives[0]) as handle:
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
                source_candidates = [extraction / 'identity-refinement']
            if len(source_candidates) != 1:
                raise ValueError(f'Expected one identity-refinement source root, found {len(source_candidates)}')
            SOURCE = source_candidates[0]
            sys.path.insert(0, str(SOURCE / 'scripts'))
            from kaggle_mounts import find_prediction, find_recovery, materialize_sealed_resource
            from scoring_comparison_runtime import run_bounded, tracking_credentials
            """
        ),
        markdown(
            """
            ## Runtime preparation

            Install the pinned add-ons into a run-local package directory. Retain Kaggle's CUDA-compatible PyTorch build. Package installation, tests and input validation are bounded and finish before model loading.
            """
        ),
        code(
            """
            PACKAGE_ROOT.mkdir(parents=True, exist_ok=True)
            subprocess.run([
                str(PYTHON), '-m', 'pip', 'install', '--disable-pip-version-check',
                '--retries', '1', '--timeout', '60', '--only-binary=:all:',
                '--upgrade', '--target', str(PACKAGE_ROOT),
                '-r', str(SOURCE / 'requirements/identity-refinement.txt'),
            ], check=True, timeout=600)
            os.environ['PYTHONPATH'] = os.pathsep.join([
                str(SOURCE / 'src'), str(PACKAGE_ROOT), os.environ.get('PYTHONPATH', '')
            ]).rstrip(os.pathsep)
            sys.path.insert(0, str(PACKAGE_ROOT))
            RUNNER = SOURCE / 'scripts/run_identity_refinement.py'
            test_env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', MPLCONFIGDIR=str(WORKING_ROOT / 'matplotlib'))
            subprocess.run([
                str(PYTHON), '-m', 'pytest', '-p', 'no:cacheprovider',
                str(SOURCE / 'tests/test_identity_patch_matching.py'),
                str(SOURCE / 'tests/test_identity_refinement_contract.py'), '-q',
            ], check=True, cwd=WORKING_ROOT, env=test_env, timeout=300)
            """
        ),
        markdown(
            """
            ## Locked input discovery

            Materialise the existing model-only package and locate the completed 08G screen output by its lock binding. The evaluator resource is not opened during this phase.
            """
        ),
        code(
            """
            MODEL_INPUT = materialize_sealed_resource(
                INPUT_ROOT,
                kind='model-input',
                archive_name='scoring-comparison-model-input.zip',
                archive_sha256='6d6b45aa211ea595524729123870307ee5295ada48e6fd8f1898935f477b6901',
                destination=WORKING_ROOT / 'identity-model-resource',
                expected_lock_sha256='9baa53454d2482caf8240e7a8b591a6e9e9646dd160c5b3abdc8bd17265ebda1',
            )
            model_lock_sha256 = sha256((MODEL_INPUT / 'artifact_lock.json').read_bytes()).hexdigest()
            SCORING = find_prediction(INPUT_ROOT, expected_input_lock_sha256=model_lock_sha256)
            if not (SCORING / 'feature_lock.json').is_file():
                raise FileNotFoundError('Attach the completed 08G screen output with its feature lock')
            run_bounded([
                str(PYTHON), str(RUNNER), 'preflight',
                '--model-input', str(MODEL_INPUT), '--scoring', str(SCORING),
                '--output', str(PREDICTION_OUTPUT),
            ], WORKING_ROOT / 'identity-input-validation.log', idle_seconds=300, total_seconds=900)
            """
        ),
        markdown(
            """
            ## Checked recovery

            Every completed scene has an atomic checkpoint and a matching receipt. The run binding covers inputs, source, dependencies and the fixed protocol. Kaggle recovery must be attached as the original `.zip.bin` bytes together with its manifest. Expanded, incomplete, duplicated or differently bound recovery data is rejected before inference.
            """
        ),
        code(
            """
            # A fresh run must ignore unrelated recovery files mounted by Kaggle.
            RECOVERY = find_recovery(INPUT_ROOT) if EXPECT_RECOVERY else None
            print(json.dumps({
                'event': 'recovery_discovery',
                'expected': EXPECT_RECOVERY,
                'attached': RECOVERY is not None,
                'archive': RECOVERY[0].name if RECOVERY else None,
            }, sort_keys=True))
            if EXPECT_RECOVERY and RECOVERY is None:
                raise FileNotFoundError('This continuation run requires one sealed recovery archive and manifest')
            recovery_arguments = [] if RECOVERY is None else [
                '--recovery-bundle', str(RECOVERY[0]),
                '--recovery-manifest', str(RECOVERY[1]),
            ]
            """
        ),
        markdown(
            """
            ## Lock prediction-only identity candidates

            Use one GPU. The runner downloads the pinned DINOv2-small revision, loads it on CPU, moves it to CUDA and reports each checkpoint. It recomputes foreground patch descriptors for the eight views used across the two, three and five-view comparisons. Global embeddings and all anomaly scores are reused from 08G.

            A 15-minute silent limit stops a stalled worker. A four-hour total limit prevents unattended GPU use. These are runtime limits, not accuracy gates. A failure preserves the newest recovery package; it does not justify repeating completed scenes.
            """
        ),
        code(
            """
            run_bounded([
                str(PYTHON), str(RUNNER), 'predict',
                '--model-input', str(MODEL_INPUT), '--scoring', str(SCORING),
                '--output', str(PREDICTION_OUTPUT), *recovery_arguments,
            ], WORKING_ROOT / 'identity-prediction.log', idle_seconds=900, total_seconds=14400)
            run_bounded([
                str(PYTHON), str(RUNNER), 'verify',
                '--model-input', str(MODEL_INPUT), '--scoring', str(SCORING),
                '--output', str(PREDICTION_OUTPUT),
            ], WORKING_ROOT / 'identity-prediction-validation.log', idle_seconds=300, total_seconds=600)
            display(json.loads((PREDICTION_OUTPUT / 'run_status.json').read_text()))
            display(json.loads((PREDICTION_OUTPUT / 'progress.json').read_text()))
            """
        ),
        markdown(
            """
            ## Only then open development correspondence

            The prediction lock now fixes every descriptor and match before labels are read. The evaluator checks true cross-view identity and the downstream odd-one-out ranking. The automatic route remains the denominator even when proposals are missing or tracks are mixed.
            """
        ),
        code(
            """
            EVALUATOR = materialize_sealed_resource(
                INPUT_ROOT,
                kind='evaluator-only',
                archive_name='scoring-comparison-evaluator-only.zip',
                archive_sha256='86ac512005af5fdceb64c0fb666998832b8ee08ef96d1fccca23c6e35f420c12',
                destination=WORKING_ROOT / 'identity-evaluator-resource',
                expected_lock_sha256='8012cd20c63b3788f861f2ba991b3181a0c1666f26e176beeee370228c8500c3',
            )
            run_bounded([
                str(PYTHON), str(RUNNER), 'evaluate',
                '--model-input', str(MODEL_INPUT), '--scoring', str(SCORING),
                '--predictions', str(PREDICTION_OUTPUT), '--evaluator', str(EVALUATOR),
                '--output', str(EVALUATION_OUTPUT),
            ], WORKING_ROOT / 'identity-evaluation.log', idle_seconds=900, total_seconds=7200)
            run_bounded([
                str(PYTHON), str(RUNNER), 'verify',
                '--model-input', str(MODEL_INPUT), '--scoring', str(SCORING),
                '--output', str(EVALUATION_OUTPUT),
            ], WORKING_ROOT / 'identity-evaluation-validation.log', idle_seconds=300, total_seconds=600)
            """
        ),
        markdown(
            """
            ## Summary tables

            Start with the three-view automatic route and the directed patch scorer. Compare top-1 accuracy with identity link F1 and inspect the paired asset-group intervals. Do not select a method from top-1 alone.
            """
        ),
        code(
            """
            import pandas as pd
            for file_name in (
                'primary_candidate_ranking.csv',
                'identity_refinement_summary.csv',
                'paired_asset_intervals.csv',
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
            ## Graphs and private gallery

            The first graph ranks the strongest three-view candidates. The second checks whether an odd-one-out improvement is accompanied by better identity links. The private gallery contains fixed samples, improvements and regressions. Red marks the selected identity, green marks the reference unusual object and blue marks other automatic candidates. A red mask is shown only where the final automatic route produced an object mask.
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
                print(item['file'], ':', item['selection_reason'])
                display(NotebookImage(filename=str(EVALUATION_OUTPUT / 'private/figures' / item['file'])))
            """
        ),
        markdown(
            """
            ## Experiment tracking

            ClearML runs only after the scientific outputs are sealed. It receives aggregate tables, decision records, summary graphs, metrics and checksums. It does not receive private galleries, labels, per-scene rows, raw images or credentials. A tracking failure can be retried without rerunning DINO.
            """
        ),
        code(
            """
            if tracking_credentials():
                try:
                    subprocess.run([
                        str(PYTHON), '-m', 'pip', 'install', '--retries', '1', '--timeout', '30',
                        '--only-binary=:all:', '--upgrade', '--target', str(PACKAGE_ROOT),
                        '-r', str(SOURCE / 'requirements/identity-refinement-tracking.txt'),
                    ], check=True, timeout=180)
                    subprocess.run([
                        str(PYTHON), str(SOURCE / 'scripts/log_identity_refinement.py'),
                        '--evidence', str(EVALUATION_OUTPUT),
                    ], check=True, timeout=180)
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                    print('ClearML upload is not verified. Preserve the outputs and retry only the tracking command.')
            else:
                print('ClearML secrets are unavailable. Preserve the outputs and enable the existing notebook secrets.')
            """
        ),
        markdown(
            """
            ## Evidence export and review

            Download the executed notebook, three log files, both complete output directories and the recovery archive with its manifest. Verify lock files after download. Keep private gallery images, per-scene metrics, model inputs and recovery data outside GitHub. A later repository package should contain the clean notebook, source, tests, safe tables, summary figures, sanitized executed notebook and verified run receipts.

            Review successes and failures before freezing a method. If no local or hybrid descriptor improves both downstream ranking and identity quality, retain the current method and keep optional human correction after the automatic result.
            """
        ),
        code(
            """
            print('Prediction evidence:', PREDICTION_OUTPUT)
            print('Evaluation evidence:', EVALUATION_OUTPUT)
            print('Recovery archive:', PREDICTION_OUTPUT.parent / f'{PREDICTION_OUTPUT.name}-recovery.zip')
            print('The development decision remains pending manual review.')
            """
        ),
        markdown(
            """
            ## Working-directory cleanup

            Remove only disposable dependency, extraction and plotting folders after outputs and optional tracking have finished. Prediction checkpoints, locks, logs, tables, graphs and galleries remain available for download.
            """
        ),
        code(
            """
            for path in (
                PACKAGE_ROOT,
                WORKING_ROOT / 'matplotlib',
                WORKING_ROOT / 'identity-source',
                WORKING_ROOT / 'identity-model-resource',
                WORKING_ROOT / 'identity-evaluator-resource',
            ):
                if path.exists():
                    shutil.rmtree(path)
            print('Disposable directories removed. Scientific evidence is unchanged.')
            """
        ),
    ]
    for index, cell in enumerate(cells):
        source = str(cell.get("source", ""))
        cell["id"] = sha256(
            f"{index}:{cell['cell_type']}:{source}".encode("utf-8")
        ).hexdigest()[:8]
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
