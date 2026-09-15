"""Build the reviewed CPU-only Notebook 08J for Kaggle."""

from __future__ import annotations

from pathlib import Path
import textwrap

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "08j-development-identity-association.ipynb"


def markdown(value: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(value).strip())


def code(value: str):
    return nbf.v4.new_code_cell(textwrap.dedent(value).strip())


def build() -> None:
    cells = [
        markdown(
            """
            # 08J: Development cross-view identity association

            This notebook tests one bounded change to the Visual Peer Inspector pipeline: how automatic object proposals are linked across views. It reuses the sealed DINOv2-small embeddings and anomaly scores from Notebook 08G. LocateAnything, SAM 2.1, DINOv2-small, the 2,000-scene split and the scoring methods remain unchanged.

            Notebook file: `08j-development-identity-association.ipynb`.
            """
        ),
        markdown(
            """
            ## Study boundary

            The existing Hungarian association is the baseline. The candidates use reciprocal nearest neighbours, cycle pruning and multi-view support. Only five predeclared configurations are evaluated across the same two, three and five-view sets used in Notebook 08I.

            The prediction stage cannot read labels, asset groups, categories or source identities. It locks all association candidates first. The evaluator is opened only after that lock exists. A candidate is promoted only if asset-grouped paired intervals support improvements in identity F1 and scene average precision together, without a mean regression in top-1 accuracy or recall at two. There is no arbitrary accuracy threshold.
            """
        ),
        markdown("## Runtime and source verification"),
        code(
            """
            from pathlib import Path, PurePosixPath
            from zipfile import ZipFile
            import json
            import os
            import shutil
            import stat
            import subprocess
            import sys

            INPUT_ROOT = Path('/kaggle/input')
            WORKING_ROOT = Path('/kaggle/working')
            SOURCE_ROOT = WORKING_ROOT / 'identity-association-source'
            PREDICTION_OUTPUT = WORKING_ROOT / 'identity-association-predictions'
            EVALUATION_OUTPUT = WORKING_ROOT / 'identity-association-evaluation'
            PYTHON = Path(sys.executable)

            mount_roots = []
            for pattern in ('*', 'datasets/*/*', 'datasets/*/*/versions/*'):
                mount_roots.extend(path.resolve() for path in INPUT_ROOT.glob(pattern) if path.is_dir())
            expanded_sources = sorted({
                root for root in mount_roots
                if (root / 'scripts/run_identity_association.py').is_file()
                and (root / 'src/visual_peer_inspector').is_dir()
            })
            if expanded_sources:
                if len(expanded_sources) != 1:
                    raise ValueError(f'Expected one expanded association source, found {len(expanded_sources)}')
                SOURCE_ROOT = expanded_sources[0]
            else:
                archives = sorted({path.resolve() for root in mount_roots for path in root.glob('identity-association-source-*.zip') if path.is_file()})
                if len(archives) != 1:
                    raise ValueError(f'Expected one expanded source or source archive, found {len(archives)} archives')
                temporary = SOURCE_ROOT.with_name(SOURCE_ROOT.name + '.incomplete')
                if SOURCE_ROOT.exists() or temporary.exists():
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
                temporary.replace(SOURCE_ROOT)
            sys.path.insert(0, str(SOURCE_ROOT / 'src'))
            sys.path.insert(0, str(SOURCE_ROOT / 'scripts'))
            from kaggle_mounts import find_prediction, materialize_sealed_resource
            from scoring_comparison_runtime import run_bounded, tracking_credentials
            RUNNER = SOURCE_ROOT / 'scripts/run_identity_association.py'
            """
        ),
        markdown(
            """
            ## Dependency and contract checks

            This is a CPU-only study. It does not download or load a neural model. The contract tests finish before any evidence is written.
            """
        ),
        code(
            """
            os.environ['PYTHONPATH'] = os.pathsep.join([str(SOURCE_ROOT / 'src'), os.environ.get('PYTHONPATH', '')]).rstrip(os.pathsep)
            test_environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', MPLCONFIGDIR=str(WORKING_ROOT / 'matplotlib'))
            subprocess.run([
                str(PYTHON), '-m', 'pytest', '-p', 'no:cacheprovider',
                str(SOURCE_ROOT / 'tests/test_identity_association.py'), '-q',
            ], check=True, cwd=WORKING_ROOT, env=test_environment, timeout=180)
            """
        ),
        markdown("## Locked input discovery"),
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
            from hashlib import sha256
            model_lock_sha256 = sha256((MODEL_INPUT / 'artifact_lock.json').read_bytes()).hexdigest()
            SCORING = find_prediction(INPUT_ROOT, expected_input_lock_sha256=model_lock_sha256)
            run_bounded([
                str(PYTHON), str(RUNNER), 'preflight',
                '--scoring', str(SCORING), '--output', str(PREDICTION_OUTPUT),
            ], WORKING_ROOT / 'association-preflight.log', idle_seconds=120, total_seconds=600)
            """
        ),
        markdown(
            """
            ## Lock label-free association candidates

            Each completed scene is stored atomically with a receipt bound to the exact inputs, protocol and implementation. Re-running the cell resumes valid scenes and rejects partial or mismatched checkpoints.
            """
        ),
        code(
            """
            run_bounded([
                str(PYTHON), str(RUNNER), 'predict',
                '--scoring', str(SCORING), '--output', str(PREDICTION_OUTPUT),
            ], WORKING_ROOT / 'association-prediction.log', idle_seconds=180, total_seconds=1800)
            run_bounded([
                str(PYTHON), str(RUNNER), 'verify',
                '--scoring', str(SCORING), '--output', str(PREDICTION_OUTPUT),
            ], WORKING_ROOT / 'association-prediction-validation.log', idle_seconds=120, total_seconds=300)
            display(json.loads((PREDICTION_OUTPUT / 'run_status.json').read_text()))
            display(json.loads((PREDICTION_OUTPUT / 'progress.json').read_text()))
            """
        ),
        markdown(
            """
            ## Open development correspondence and evaluate

            The evaluator measures identity precision, recall, F1 and track purity together with odd-one-out average precision, top-1 accuracy, recall at two and AUROC. Missing proposals remain in the denominator.
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
            ], WORKING_ROOT / 'association-evaluation.log', idle_seconds=300, total_seconds=1800)
            run_bounded([
                str(PYTHON), str(RUNNER), 'verify',
                '--scoring', str(SCORING), '--output', str(EVALUATION_OUTPUT),
            ], WORKING_ROOT / 'association-evaluation-validation.log', idle_seconds=120, total_seconds=300)
            """
        ),
        markdown("## Results and decision"),
        code(
            """
            import pandas as pd
            from IPython.display import Image as NotebookImage, display

            shareable = EVALUATION_OUTPUT / 'shareable'
            primary = pd.read_csv(shareable / 'primary_candidate_ranking.csv')
            display(primary[[
                'configuration_id', 'ap', 'top1', 'recall_at_2',
                'identity_link_precision', 'identity_link_recall', 'identity_link_f1',
                'linked_node_fraction', 'singleton_fraction',
            ]])
            display(json.loads((shareable / 'development_decision.json').read_text()))
            for path in sorted((shareable / 'figures').glob('*.png')):
                display(NotebookImage(filename=str(path)))
            """
        ),
        markdown(
            """
            ## Qualitative review

            These evaluator-only examples show the three selected views. Red marks the track chosen as unusual, green marks the reference unusual object and blue marks other automatic proposals. They are private review evidence and must not be copied into a public repository.
            """
        ),
        code(
            """
            gallery = json.loads((EVALUATION_OUTPUT / 'private' / 'gallery_manifest.json').read_text())
            display(pd.DataFrame(gallery['examples']))
            for row in gallery['examples'][:6]:
                display(NotebookImage(filename=str(EVALUATION_OUTPUT / 'private' / 'figures' / row['file'])))
            """
        ),
        markdown(
            """
            ## Optional ClearML upload

            Scientific outputs are already complete and sealed. Tracking runs last and cannot trigger recomputation. A failed upload leaves all local evidence intact.
            """
        ),
        code(
            """
            tracking_root = WORKING_ROOT / 'tracking-packages'
            tracking_root.mkdir(parents=True, exist_ok=True)
            tracking_receipt = EVALUATION_OUTPUT / 'tracking_receipt.json'
            if tracking_credentials():
                try:
                    subprocess.run([
                        str(PYTHON), '-m', 'pip', 'install', '--disable-pip-version-check',
                        '--retries', '1', '--timeout', '45', '--only-binary=:all:',
                        '--target', str(tracking_root), '-r',
                        str(SOURCE_ROOT / 'requirements/identity-association-tracking.txt'),
                    ], check=True, timeout=300)
                    tracking_environment = dict(os.environ)
                    tracking_environment['PYTHONPATH'] = os.pathsep.join([str(tracking_root), str(SOURCE_ROOT / 'src')])
                    subprocess.run([
                        str(PYTHON), str(SOURCE_ROOT / 'scripts/log_identity_association.py'),
                        '--evidence', str(EVALUATION_OUTPUT),
                    ], check=True, timeout=240, env=tracking_environment)
                except Exception as error:
                    tracking_receipt.write_text(json.dumps({'status': 'upload_not_verified', 'error_type': type(error).__name__}, sort_keys=True))
            else:
                tracking_receipt.write_text(json.dumps({'status': 'not_uploaded', 'reason': 'configured secrets unavailable'}, sort_keys=True))
            display(json.loads(tracking_receipt.read_text()))
            """
        ),
        markdown("## Output inventory"),
        code(
            """
            inventory = []
            for root in (PREDICTION_OUTPUT, EVALUATION_OUTPUT):
                for path in sorted(item for item in root.rglob('*') if item.is_file()):
                    inventory.append({'root': root.name, 'file': path.relative_to(root).as_posix(), 'bytes': path.stat().st_size})
            inventory_frame = pd.DataFrame(inventory)
            inventory_frame.to_csv(WORKING_ROOT / 'identity-association-output-inventory.csv', index=False)
            display(inventory_frame.groupby('root').agg(files=('file', 'size'), bytes=('bytes', 'sum')).reset_index())
            """
        ),
    ]
    notebook = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
    )
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, NOTEBOOK)


if __name__ == "__main__":
    build()
