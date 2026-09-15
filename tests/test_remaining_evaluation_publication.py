"""Strict aggregate-only publication and retry behavior, without external services."""
from pathlib import Path
import base64
import importlib.util
import json
import sys

from PIL import Image
import nbformat
import pytest

from vpi_object_bank.storage import atomic_json, digest_file, read_json, seal, verify_seal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import publish_remaining_evaluation as publication


def fixture_bundle(tmp_path):
    scientific, report, bundle = (tmp_path / name for name in ['scientific', 'report', 'bundle'])
    for name in publication.TABLES:
        path = scientific / 'shareable' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == '.json':
            atomic_json(path, {'status': 'prepared'})
        else:
            path.write_text('route,method,mean\nautomatic_predicted,winner_cluster_consensus,0.4\n')
    seal(scientific, list((scientific / 'shareable').iterdir()), name='scientific_lock.json')
    images = []
    for index, name in enumerate(sorted(publication.FIGURES)):
        path = report / 'shareable/figures' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (10, 10), (index, 30, 60)).save(path)
        images.append(base64.b64encode(path.read_bytes()).decode())
    atomic_json(report / 'run_binding.json', {'scientific_lock_sha256': digest_file(scientific / 'scientific_lock.json')})
    seal(report, [p for p in report.rglob('*') if p.is_file()], name='report_lock.json')
    clean = nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell('# 09E'),
        nbformat.v4.new_code_cell('display_aggregate_figures()'),
        nbformat.v4.new_code_cell("display_private_gallery('private/gallery.json')")])
    executed = nbformat.from_dict(json.loads(nbformat.writes(clean)))
    executed.cells[1].execution_count = 1
    executed.cells[1].outputs = [nbformat.v4.new_output('display_data', data={'image/png': value}) for value in images]
    executed.cells[2].execution_count = 2
    executed.cells[2].outputs = [nbformat.v4.new_output('display_data', data={
        'image/png': base64.b64encode(b'private photo').decode(), 'text/plain': 'S123456 object O1 score .9'})]
    clean_path, executed_path = tmp_path / 'clean.ipynb', tmp_path / 'executed.ipynb'
    nbformat.write(clean, clean_path)
    nbformat.write(executed, executed_path)
    args = (scientific, report, clean_path, executed_path, bundle)
    return args


def test_private_cells_removed_in_full_aggregates_verified_and_exact_resume(tmp_path):
    args = fixture_bundle(tmp_path)
    receipt = publication.prepare_bundle(*args)
    root = args[-1]
    assert receipt['removed_private_visuals'] == 1 and receipt['retained_safe_visuals'] == 7
    assert receipt['private_cell_outputs_removed'] is True
    published = read_json(root / publication.NOTEBOOK)
    assert published['cells'][-1]['outputs'] == []
    assert 'S123456' not in (root / publication.NOTEBOOK).read_text()
    assert 'object O1' not in (root / publication.LOG).read_text()
    before = {str(p): (digest_file(p), p.stat().st_mtime_ns) for p in root.rglob('*') if p.is_file()}
    assert publication.prepare_bundle(*args) == receipt
    assert before == {str(p): (digest_file(p), p.stat().st_mtime_ns) for p in root.rglob('*') if p.is_file()}
    files, manifest = publication.collect_artifacts(root, root, 'publication_lock.json')
    assert len(files) == 22 and len(manifest['artifacts']) == 22
    assert all('private' not in name for name in files)


def test_unapproved_file_and_unverified_image_block_publication(tmp_path):
    args = fixture_bundle(tmp_path)
    publication.prepare_bundle(*args)
    root = args[-1]
    (root / 'private.csv').write_text('scene\nS123456\n')
    with pytest.raises(ValueError, match='unsealed'):
        publication.collect_artifacts(root, root, 'publication_lock.json')
    (root / 'private.csv').unlink()
    notebook = read_json(root / publication.NOTEBOOK)
    notebook['cells'][2]['outputs'][0]['data']['image/png'] = base64.b64encode(b'unverified').decode()
    atomic_json(root / publication.NOTEBOOK, notebook)
    with pytest.raises(ValueError, match='outside the aggregate report lock'):
        publication.validate_notebook(root / publication.NOTEBOOK, {digest_file(root / 'figures' / name) for name in publication.FIGURES})


def test_tracking_failure_preserves_science_and_completed_retry_is_idempotent(tmp_path):
    args = fixture_bundle(tmp_path)
    publication.prepare_bundle(*args)
    root = args[-1]
    before = digest_file(root / 'publication_lock.json')
    class Broken:
        @staticmethod
        def get_tasks(**kwargs):
            raise ConnectionError('temporary external outage')
    receipt_path = tmp_path / 'tracking.json'
    receipt = publication.publish_bundle(root, receipt_path, task_api=Broken)
    assert receipt['status'] == 'pending' and receipt['error_type'] == 'ConnectionError'
    assert digest_file(root / 'publication_lock.json') == before
    verify_seal(root, name='publication_lock.json')

    class FakeTask:
        id = 'synthetic-task'
        name = 'synthetic'
        artifacts = {}
        def get_output_log_web_page(self): return 'https://example.invalid/task/synthetic'
        def mark_started(self, **kwargs): pass
        def mark_completed(self, **kwargs): pass
        def close(self): pass
        def upload_artifact(self, name, **kwargs): self.artifacts[name] = True; return True
    task = FakeTask()
    class Working:
        @staticmethod
        def get_tasks(**kwargs): return []
        @staticmethod
        def create(**kwargs): return task
        @staticmethod
        def get_task(**kwargs): return task
    complete = publication.publish_bundle(root, receipt_path, task_api=Working)
    assert complete['status'] == 'completed' and complete['remote_artifacts_verified'] is True
    assert publication.publish_bundle(root, receipt_path, task_api=Broken) == complete
    assert digest_file(root / 'publication_lock.json') == before
