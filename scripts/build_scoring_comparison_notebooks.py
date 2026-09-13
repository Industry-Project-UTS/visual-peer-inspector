"""Build the two notebooks from their reviewed cell definitions."""

from pathlib import Path
import textwrap
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def markdown(text):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())


def code(text):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip())


def common_cells(stage):
    return [
        markdown(f"# {stage}: Development scoring {'comparison' if stage == '08G' else 'evaluation'}\n\nCompare fixed peer-scoring methods and one lightweight peer-context ranker on the existing development screen. This is a development study, not an untouched test result."),
        markdown("""## Study boundary

        The original 2,000-scene split is unchanged. Use the existing 100-scene development screen, with all 20 views and two mask routes. The 20-scene smoke run is a fixed nested subset for runtime checks. It is not a new train/test split.

        Keep DINOv2-small, crop preparation, model revision and input masks identical across scoring methods. Supplied masks show scoring capability; existing LocateAnything and SAM masks show automatic-region performance. No detection or segmentation model is trained or rerun here.

        The main deployment-style comparison uses automatic masks, predicted identities and three views. Two and five views are secondary. Twenty-view reference-identity fusion is a capability diagnostic, not a web-app result. Protected calibration and internal-lock scenes are outside this study."""),
        markdown("## Paths and run settings\n\nSet each path to an attached, versioned resource. The code uses these exact paths and does not search the raw dataset. Keep the model-only and evaluator resources separate."),
        code(f"""
        from pathlib import Path
        import os
        import sys
        import subprocess
        import json
        import shutil
        import stat
        from hashlib import sha256
        from pathlib import PurePosixPath
        from zipfile import ZipFile

        INPUT_ROOT = Path('/kaggle/input')
        mount_roots = []
        for pattern in ('*', 'datasets/*/*', 'datasets/*/*/versions/*', 'notebooks/*/*', 'notebooks/*/*/versions/*'):
            mount_roots.extend(path.resolve() for path in INPUT_ROOT.glob(pattern) if path.is_dir())
        mount_roots = sorted(set(mount_roots))
        source_mounts = [
            path for path in mount_roots
            if any(part.startswith('vpi-development-scoring-source') for part in path.parts[-4:])
        ]
        source_markers = []
        for source_mount in source_mounts:
            source_markers.extend(path for path in source_mount.rglob('comparison_base.json') if path.is_file())
        source_candidates = [path.parent for path in sorted(set(source_markers)) if (path.parent / 'scripts/run_scoring_comparison.py').is_file()]
        if not source_candidates:
            source_archives = []
            for source_mount in source_mounts:
                source_archives.extend(path for path in source_mount.rglob('scoring-comparison-source-*.zip') if path.is_file())
            source_archives = sorted(set(source_archives))
            if len(source_archives) != 1:
                mounts = [path.as_posix() for path in mount_roots]
                raise ValueError(
                    'The attached source dataset does not contain a valid expanded source tree '
                    f'or one source archive. Mounts: {{mounts}}'
                )
            archive = source_archives[0]
            extraction = Path('/kaggle/working/scoring-source')
            temporary = extraction.with_name(extraction.name + '.incomplete')
            if extraction.exists() or temporary.exists():
                raise FileExistsError('Source extraction destination is not empty')
            temporary.mkdir(parents=True)
            with ZipFile(archive) as handle:
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
            source_candidates = [extraction / 'scoring-comparison']
            Path('/kaggle/working/source_archive_receipt.json').write_text(json.dumps({{'file': archive.name, 'sha256': sha256(archive.read_bytes()).hexdigest()}}, sort_keys=True))
        if len(source_candidates) != 1:
            raise ValueError(f'Expected one scoring source resource, found {{len(source_candidates)}}')
        SOURCE = source_candidates[0]
        sys.path.insert(0, str(SOURCE / 'scripts'))
        from kaggle_mounts import find_recovery, materialize_sealed_resource
        MODEL_INPUT = materialize_sealed_resource(
            INPUT_ROOT,
            kind='model-input',
            archive_name='scoring-comparison-model-input.zip',
            archive_sha256='6d6b45aa211ea595524729123870307ee5295ada48e6fd8f1898935f477b6901',
            destination=Path('/kaggle/working/scoring-model-resource'),
            expected_lock_sha256='9baa53454d2482caf8240e7a8b591a6e9e9646dd160c5b3abdc8bd17265ebda1',
        )
        OUTPUT = Path('/kaggle/working/scoring-comparison-{stage.lower()}')
        PYTHON = Path(sys.executable)
        PACKAGE_ROOT = Path('/kaggle/working/comparison-packages')
        # Use smoke20 first. Change only this value to screen100 after the smoke evidence passes review.
        COHORT = 'smoke20'
        # Set this to True only in a prepared continuation wrapper that attaches a sealed recovery archive.
        EXPECT_RECOVERY = False
        RECOVERY = find_recovery(INPUT_ROOT) if '{stage}' == '08G' else None
        recovery_state = {{
            'event': 'recovery_discovery',
            'expected': EXPECT_RECOVERY,
            'attached': RECOVERY is not None,
            'archive': RECOVERY[0].name if RECOVERY else None,
        }}
        print(json.dumps(recovery_state, sort_keys=True))
        if '{stage}' == '08G' and EXPECT_RECOVERY and RECOVERY is None:
            raise FileNotFoundError('This continuation run requires one sealed recovery archive and manifest')
        """),
        markdown("""## Runtime preparation

        Install the pinned study dependencies into a run-local package directory while retaining Kaggle's CUDA-compatible PyTorch. This avoids relying on `venv`, which is not available in every Kaggle container, and does not replace CUDA libraries. Package installation is bounded. Authentication, dependency and input errors must be fixed before inference; they are not reasons to retry a full GPU run."""),
        code("""
        if not (SOURCE / 'comparison_base.json').is_file():
            raise FileNotFoundError('Set SOURCE to the attached source package')
        if not (MODEL_INPUT / 'artifact_lock.json').is_file():
            raise FileNotFoundError('Set MODEL_INPUT to the attached model-only package')
        PACKAGE_ROOT.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            str(PYTHON), '-m', 'pip', 'install', '--disable-pip-version-check',
            '--retries', '1', '--timeout', '60', '--only-binary=:all:',
            '--upgrade', '--target', str(PACKAGE_ROOT),
            '-r', str(SOURCE / 'requirements/scoring-comparison.txt'),
        ], check=True, timeout=600)
        os.environ['PYTHONPATH'] = os.pathsep.join([
            str(SOURCE / 'src'), str(PACKAGE_ROOT), os.environ.get('PYTHONPATH', '')
        ]).rstrip(os.pathsep)
        sys.path.insert(0, str(PACKAGE_ROOT))
        from scoring_comparison_runtime import run_bounded
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        RUNNER = SOURCE / 'scripts/run_scoring_comparison.py'
        """),
        markdown("## Input validation\n\nCheck membership, file hashes, image decoding, mask dimensions, enclosing boxes and unique candidate IDs before allocating a model. Labels and reference object IDs are not accepted by the prediction schema."),
        code("""
        run_bounded([str(PYTHON), str(RUNNER), 'preflight', '--model-input', str(MODEL_INPUT), '--output', str(OUTPUT)], OUTPUT.parent / 'input-validation.log', idle_seconds=300, total_seconds=600)
        """),
        markdown("## Local contracts\n\nRun the scorer, notebook, integrity and timeout tests in the installed environment before continuing. A failed test stops the notebook before model loading."),
        code("""
        test_env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', MPLCONFIGDIR='/kaggle/working/matplotlib')
        subprocess.run([str(PYTHON), '-m', 'pytest', '-p', 'no:cacheprovider', str(SOURCE / 'tests/test_scoring_comparison.py'), str(SOURCE / 'tests/test_peer_support_and_context.py'), '-q'], check=True, env=test_env, cwd='/kaggle/working', timeout=300)
        """),
    ]


def tracking_cells(stage):
    return [
        markdown("""## Experiment tracking

        Scientific outputs are already saved. ClearML receives aggregate tables, charts, metrics and hash receipts, not labels, raw images, full feature caches or local source files. Tracking has a separate timeout and can be retried without inference. A completed SDK upload still requires remote artifact verification before a run receives the final tag."""),
        code(f"""
        from scoring_comparison_runtime import tracking_credentials
        if tracking_credentials():
            try:
                subprocess.run([
                    str(PYTHON), '-m', 'pip', 'install', '--retries', '1', '--timeout', '30',
                    '--only-binary=:all:', '--upgrade', '--target', str(PACKAGE_ROOT),
                    '-r', str(SOURCE / 'requirements/scoring-comparison-tracking.txt'),
                ], check=True, timeout=180)
                subprocess.run([str(PYTHON), str(SOURCE / 'scripts/log_scoring_comparison.py'), '--evidence', str(OUTPUT), '--notebook', '{stage}'], check=True, timeout=180)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                print('ClearML upload is not verified. Preserve the outputs and retry tracking separately.')
        else:
            print('ClearML secrets are unavailable. Preserve the outputs and enable the existing notebook secrets.')
        """),
        markdown("""## Evidence export

        Download the executed notebook, logs and complete output directory. Verify checksums after download. Keep private images, labels, per-scene evaluation rows and feature caches outside the GitHub-ready folder. Only reviewed, sanitized evidence belongs in a later repository package.

        Do not call a model final because a notebook completed. Read the paired development comparison, failure examples, tie sensitivity and identity support first. Protected evaluation remains paused until both team members reconcile which object assets have already been used."""),
        code("""
        print('Saved evidence:', OUTPUT)
        print('Download the complete output directory and verify its lock file.')
        """),
        markdown("""## Working-directory cleanup

        Remove only disposable dependency, extraction and plotting directories after the scientific output and optional tracking attempt are complete. Prediction checkpoints, evidence, logs and lock files remain in the stage output. This keeps Kaggle downloads compact and does not affect recovery."""),
        code("""
        disposable_paths = [
            PACKAGE_ROOT,
            Path('/kaggle/working/matplotlib'),
            Path('/kaggle/working/scoring-source'),
            Path('/kaggle/working/scoring-model-resource'),
            Path('/kaggle/working/scoring-evaluator-resource'),
        ]
        for disposable_path in disposable_paths:
            if disposable_path.exists():
                shutil.rmtree(disposable_path)
        print('Disposable Kaggle working directories removed. Scientific evidence is unchanged.')
        """),
    ]


def build():
    prediction = common_cells("08G") + [
        markdown("""## Scoring methods

        | Method | Comparison rule | Patch selection |
        |---|---|---|
        | B0 | Distance from the leave-one-out global feature mean | Global feature |
        | B1_control | Median distance to the other global object features | Global feature |
        | P1_current | Symmetric pairwise patch-tail distances, then median across peers | 50% foreground coverage, 10% tail, ceiling count |
        | forward_10_cov0 | Per-patch lower median across peers, then mean of worst patches | All patches, 10% tail, floor count |
        | forward_02_cov70 | Same directed reduction | 70% foreground coverage, 2% tail, floor count |
        | mutual_missing_05 | Features supported by peers but absent from the target | 50% foreground coverage, 5% tail |
        | geometry_peer | Robust silhouette deviation from peers | Seven mask geometry descriptors |
        | P2_consensus | Mean within-view rank of forward, missing-part and geometry evidence | No fitted weights |

        Forward variants retain the proposal's lower-median and floor-rounding choices. Its all-patch fallback is counted when no patch reaches the coverage cutoff. These are controlled common-crop comparisons, not exact reproductions of earlier notebooks that used different crop padding and background colours.

        Patch features are computed once per object. Temporary patch tensors support restart within an unfinished scene, then the sealed scene checkpoint retains the compact embeddings, component scores and heatmaps needed by 08H. Views with fewer than three objects remain recorded as insufficient peers."""),
        markdown("""## Checked recovery

        Each finished scene has an atomic result and checksum receipt. Individual feature files can be reused within an unfinished scene. A resumed run must match the model revision, source content, preprocessing, input hashes and installed library versions. A corrupt or incompatible cache stops the run; it is never silently accepted.

        Kaggle does not carry its working directory into a new committed run automatically. Download completed or failed-run outputs, reattach them as a private recovery resource, and set RECOVERY. If outputs are unavailable, those unsaved computations cannot be resumed."""),
        code("""
        recovery_arguments = []
        if RECOVERY is not None:
            recovery_arguments = ['--recovery-bundle', str(RECOVERY[0]), '--recovery-manifest', str(RECOVERY[1])]
        """),
        markdown("""## Feature extraction and scoring

        Use one CUDA device. The worker reports model loading, each completed route/view and each resumed scene. A 15-minute silent-worker limit stops a hang while retaining saved checkpoints. A five-hour total limit prevents an unattended job from consuming the entire GPU allowance. These are runtime limits, not accuracy gates.

        Failed model loading, incompatible libraries, corrupt files and GPU memory errors require diagnosis. Do not automatically repeat an entire failed run."""),
        code("""
        run_bounded([str(PYTHON), str(RUNNER), 'predict', '--model-input', str(MODEL_INPUT), '--output', str(OUTPUT), '--cohort', COHORT, *recovery_arguments], OUTPUT.parent / 'prediction.log', idle_seconds=900, total_seconds=18000)
        """),
        markdown("## Completion checks\n\nThe prediction lock binds every completed scene to its inputs and source. No accuracy numbers are calculated here because this notebook has no anomaly labels."),
        code("""
        run_bounded([str(PYTHON), str(RUNNER), 'verify', '--model-input', str(MODEL_INPUT), '--output', str(OUTPUT)], OUTPUT.parent / 'prediction-validation.log', idle_seconds=300, total_seconds=600)
        binding = json.loads((OUTPUT / 'run_binding.json').read_text())
        status = json.loads((OUTPUT / 'run_status.json').read_text())
        display(status)
        display(binding['runtime_versions'])
        """),
        markdown("## Masks, candidate picks and runtime\n\nThese label-free examples show both mask routes, black-background objects and each scorer's per-view pick. They do not claim that the picked object is correct. Notebook 08H performs that comparison using separate labels."),
        code("""
        from IPython.display import Image, display
        import pandas as pd
        display(pd.read_csv(OUTPUT / 'shareable/prediction_runtime.csv'))
        display(Image(filename=str(OUTPUT / 'shareable/prediction_runtime.png')))
        for item in json.loads((OUTPUT / 'private-previews/gallery_manifest.json').read_text()):
            display(Image(filename=str(OUTPUT / 'private-previews' / item['file'])))
        """),
    ] + tracking_cells("08G")
    evaluation = common_cells("08H") + [
        markdown("## Locked predictions and evaluator data\n\nAttach the exact prediction output from 08G and the matching development-only evaluator resource. This stage runs on CPU. Fixed methods are not tuned. The small peer-context ranker is trained only through asset-grouped cross-validation and produces out-of-fold scores for every reported development result."),
        code("""
        from kaggle_mounts import find_prediction
        import hashlib
        model_lock_sha256 = hashlib.sha256((MODEL_INPUT / 'artifact_lock.json').read_bytes()).hexdigest()
        PREDICTIONS = find_prediction(INPUT_ROOT, expected_input_lock_sha256=model_lock_sha256)
        EVALUATOR = materialize_sealed_resource(
            INPUT_ROOT,
            kind='evaluator-only',
            archive_name='scoring-comparison-evaluator-only.zip',
            archive_sha256='86ac512005af5fdceb64c0fb666998832b8ee08ef96d1fccca23c6e35f420c12',
            destination=Path('/kaggle/working/scoring-evaluator-resource'),
            expected_lock_sha256='8012cd20c63b3788f861f2ba991b3181a0c1666f26e176beeee370228c8500c3',
        )
        if not (PREDICTIONS / 'prediction_lock.json').is_file():
            raise FileNotFoundError('Attach the verified 08G output')
        if not (EVALUATOR / 'artifact_lock.json').is_file():
            raise FileNotFoundError('Attach the matching evaluator-only resource')
        """),
        markdown("""## Evaluation rules

        Report supplied masks with reference identity, automatic masks with reference identity, and automatic masks with predicted identity. The last route is closest to deployment. Reference identity is used only by the evaluator after predictions are locked.

        Missing detections and insufficient-peer views are not removed from the denominator. Average precision and recall use every positive object in the scene. An unmatched predicted track remains a false candidate. Mixed tracks receive identity credit only with a one-to-one, strict-majority reference match.

        Scores are fused by mean normalized rank across the views supporting a track. Single-view winners and track purity are reported because limited support can make fusion misleading. Exact ties use neutral IDs for deterministic order; expected, best and worst top-one tie results are also reported.

        The context ranker uses only relative component scores, silhouette deviations and peer distances. It cannot use scene names, categories, asset identifiers, object order or protected-role fields as predictive inputs. Every asset group stays within one fold. Each completed fold is checkpointed and can be reused after a reporting or tracking failure.

        Paired 95% intervals resample asset groups, not independent views. These development intervals are descriptive after prior experiments; they do not establish unseen-test generalization."""),
        code("""
        run_bounded([str(PYTHON), str(RUNNER), 'evaluate', '--model-input', str(MODEL_INPUT), '--predictions', str(PREDICTIONS), '--evaluator', str(EVALUATOR), '--output', str(OUTPUT)], OUTPUT.parent / 'evaluation.log', idle_seconds=900, total_seconds=7200)
        """),
        markdown("## Summary tables and paired differences"),
        code("""
        import pandas as pd
        for name in ['ranking_summary.csv', 'paired_asset_intervals.csv', 'context_fold_summary.csv', 'runtime_summary.csv', 'category_summary.csv', 'families_summary.csv', 'view_summary.csv']:
            print(name)
            display(pd.read_csv(OUTPUT / 'shareable' / name))
        """),
        markdown("## Ranking across routes and views"),
        code("""
        from IPython.display import Image, display
        display(Image(filename=str(OUTPUT / 'shareable/ranking_by_route_and_views.png')))
        """),
        markdown("""## Object masks and odd-one-out examples

        The private gallery contains up to 24 scenes: a fixed hash-selected sample, large improvements, large regressions and deterministic coverage fill. Each panel shows three views, mask-derived boxes, SAM foreground, objects on black, coarse patch evidence and the fused selected object.

        Red is the selected track; green is the reference unusual object; blue shows all automatic object boxes. Green boxes come from supplied reference masks, so missed automatic detections remain visible. Some views may have no red box when the chosen track is unsupported. The gallery manifest records exactly how every example was selected."""),
        code("""
        gallery = json.loads((OUTPUT / 'private/gallery_manifest.json').read_text())
        for item in gallery:
            print(item['file'], ':', item['selection_reason'])
            display(Image(filename=str(OUTPUT / 'private/figures' / item['file'])))
        """),
        markdown("## Evidence integrity and decision boundary\n\nInspect both successful and unsuccessful predictions before deciding whether the directed scorer improves the project. A better supplied-mask result alone is not enough to replace the deployed scoring rule. Retain the current method unless the automatic-route comparison supports a reviewed change."),
        code("""
        run_bounded([str(PYTHON), str(RUNNER), 'verify', '--model-input', str(MODEL_INPUT), '--output', str(OUTPUT)], OUTPUT.parent / 'evaluation-validation.log', idle_seconds=300, total_seconds=600)
        display(json.loads((OUTPUT / 'shareable/evaluation_receipt.json').read_text()))
        """),
    ] + tracking_cells("08H")
    (ROOT / "notebooks").mkdir(exist_ok=True)
    for name, cells in [("08g_development_scoring_comparison.ipynb", prediction), ("08h_development_scoring_evaluation.ipynb", evaluation)]:
        notebook = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.12"}})
        nbf.validate(notebook)
        nbf.write(notebook, ROOT / "notebooks" / name)
        print(f"Built {name}: {len(cells)} cells")


if __name__ == "__main__":
    build()
