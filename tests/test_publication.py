"""Publication must never invalidate scientific results or leak scene evidence."""
from pathlib import Path
import hashlib
import json
import re
import sys

import nbformat
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import build_notebooks
import execute_notebook
import publish_clearml as publication
import publish_executed_notebook as sanitiser


def write_lock(root, name='evaluation_lock.json'):
    files = {path.relative_to(root).as_posix(): publication.digest(path)
             for path in root.rglob('*') if path.is_file() and path.name != name}
    publication.atomic_json(root / name, {'files': files, 'metadata': {'status': 'completed'}})


@pytest.fixture
def evidence(tmp_path):
    root = tmp_path / 'evaluation'
    shareable = root / 'shareable'
    shareable.mkdir(parents=True)
    (shareable / 'metric_summary.csv').write_text('route,method,view_count,top1_tie_expected\nsupplied_reference,object_bank_mean,3,0.4\n')
    (shareable / 'decision.json').write_text(json.dumps({'status': 'development_only', 'selected_method': 'within_view_mean'}))
    (root / 'private').mkdir()
    (root / 'private/scene_metrics.csv').write_text('scene,score\nS000001,0.2\n')
    write_lock(root)
    return root, shareable, tmp_path / 'tracking/receipt.json'


class FakeTask:
    def __init__(self, identifier, name):
        self.id = identifier
        self.name = name
        self.artifacts = {}
        self.fail_once = False
        self.upload_calls = 0
        self.completed = False

    def mark_started(self, force=False):
        assert force is True

    def get_output_log_web_page(self):
        return 'https://app.clear.ml/projects/test/experiments/' + self.id + '/output/log'

    def upload_artifact(self, name, *, artifact_object, wait_on_upload, auto_pickle, retries):
        assert wait_on_upload is True and auto_pickle is False and retries == 0
        self.upload_calls += 1
        if self.fail_once:
            self.fail_once = False
            raise RuntimeError('secret-do-not-copy-provider-error')
        self.artifacts[name] = {'sha256': publication.digest(artifact_object)}
        return True

    def mark_completed(self, ignore_errors=True):
        assert ignore_errors is False
        self.completed = True

    def close(self):
        pass


class FakeTasks:
    def __init__(self):
        self.tasks = {}
        self.create_count = 0
        self.calls = 0
        self.fail_first_upload = False

    def get_tasks(self, *, project_name, task_filter):
        self.calls += 1
        assert project_name == publication.PROJECT
        return [task for task in self.tasks.values() if re.fullmatch(task_filter['name'], task.name)]

    def create(self, *, project_name, task_name, task_type, packages, detect_repository, add_task_init_call):
        self.calls += 1
        self.create_count += 1
        assert project_name == publication.PROJECT and task_type == 'data_processing'
        assert packages is detect_repository is add_task_init_call is False
        task = FakeTask('task-' + str(self.create_count), task_name)
        task.fail_once = self.fail_first_upload
        self.tasks[task.id] = task
        return task

    def get_task(self, *, task_id):
        self.calls += 1
        return self.tasks[task_id]


def test_only_verified_aggregate_files_are_collected(evidence):
    root, shareable, _ = evidence
    files, manifest = publication.collect_artifacts(root, shareable, 'evaluation_lock.json')
    assert set(files) == {'evaluation/metric_summary.csv', 'evaluation/decision.json'}
    assert 'S000001' not in json.dumps(manifest)
    assert all(path.parent == shareable for path in files.values())


@pytest.mark.parametrize('failure', ['unsealed', 'corrupt_private', 'extra_file', 'unsealed_table', 'symlink'])
def test_invalid_or_unapproved_evidence_fails_before_remote_access(evidence, failure):
    root, shareable, receipt = evidence
    if failure == 'unsealed':
        (root / 'evaluation_lock.json').unlink()
    elif failure == 'corrupt_private':
        (root / 'private/scene_metrics.csv').write_text('changed')
    elif failure == 'extra_file':
        (shareable / 'scene_metrics.csv').write_text('scene,score\nS000001,0.2\n')
        write_lock(root)
    elif failure == 'unsealed_table':
        (shareable / 'identity_summary.csv').write_text('metric,value\npurity,0.9\n')
    else:
        (shareable / 'metric_summary.csv').unlink()
        (shareable / 'metric_summary.csv').symlink_to(root / 'private/scene_metrics.csv')
        write_lock(root)
    api = FakeTasks()
    with pytest.raises((ValueError, FileNotFoundError)):
        publication.publish(root, shareable, receipt, task_api=api)
    assert api.calls == 0


@pytest.mark.parametrize('filename,content', [
    ('metric_summary.csv', 'scene_id,score\nan_opaque_id,0.4\n'),
    ('metric_summary.csv', 'name,score\nS000001,0.4\n'),
    ('decision.json', '{"scene_id": "another_opaque_id"}'),
    ('decision.json', '{"nested": [{"object_id": "opaque_id"}]}'),
    ('decision.json', '{"description": "/Users/person/private.csv"}'),
    ('decision.json', '{"api_key": "abcdefghijk-secret"}'),
])
def test_private_content_is_rejected_even_in_an_allowed_filename(evidence, filename, content):
    root, shareable, receipt = evidence
    (shareable / filename).write_text(content)
    write_lock(root)
    api = FakeTasks()
    with pytest.raises(ValueError):
        publication.publish(root, shareable, receipt, task_api=api)
    assert api.calls == 0


def test_failed_publication_resumes_same_task_without_error_text(evidence):
    root, shareable, receipt = evidence
    original_lock = (root / 'evaluation_lock.json').read_bytes()
    api = FakeTasks()
    api.fail_first_upload = True
    failed = publication.publish(root, shareable, receipt, task_api=api)
    assert failed['status'] == 'pending' and failed['task_id'] == 'task-1'
    assert failed['error_type'] == 'RuntimeError'
    assert 'secret-do-not-copy' not in receipt.read_text()
    assert (root / 'evaluation_lock.json').read_bytes() == original_lock
    completed = publication.publish(root, shareable, receipt, task_api=api)
    assert completed['status'] == 'completed' and completed['remote_artifacts_verified'] is True
    assert completed['task_url'].startswith('https://app.clear.ml/projects/test/experiments/')
    assert api.create_count == 1 and api.tasks['task-1'].completed
    calls = api.calls
    assert publication.publish(root, shareable, receipt, task_api=api) == completed
    assert api.calls == calls


def test_remote_name_recovers_create_before_receipt_crash(evidence):
    root, shareable, receipt = evidence
    _, manifest = publication.collect_artifacts(root, shareable, 'evaluation_lock.json')
    binding = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    name = '08O multi-view object-bank evaluation [' + binding[:16] + ']'
    api = FakeTasks()
    api.tasks['existing-task'] = FakeTask('existing-task', name)
    result = publication.publish(root, shareable, receipt, task_api=api)
    assert result['status'] == 'completed' and result['task_id'] == 'existing-task'
    assert api.create_count == 0


def test_changed_science_cannot_reuse_an_existing_publication(evidence):
    root, shareable, receipt = evidence
    api = FakeTasks()
    publication.publish(root, shareable, receipt, task_api=api)
    calls = api.calls
    (shareable / 'decision.json').write_text('{"status": "changed"}')
    write_lock(root)
    with pytest.raises(ValueError):
        publication.publish(root, shareable, receipt, task_api=api)
    assert api.calls == calls


def test_separately_sealed_aggregate_figures_are_optional(evidence, tmp_path):
    from PIL import Image
    root, shareable, _ = evidence
    report = tmp_path / 'report'
    figures = report / 'shareable/figures'
    figures.mkdir(parents=True)
    Image.new('RGB', (8, 8)).save(figures / 'view_count_accuracy.png')
    write_lock(report, 'report_lock.json')
    files, manifest = publication.collect_artifacts(root, shareable, 'evaluation_lock.json', report)
    assert 'report/view_count_accuracy.png' in files and 'report' in manifest['locks']
    Image.new('RGB', (8, 8)).save(figures / 'private_gallery.png')
    write_lock(report, 'report_lock.json')
    with pytest.raises(ValueError):
        publication.collect_artifacts(root, shareable, 'evaluation_lock.json', report)


def test_timeout_keeps_task_identity_and_completed_science(evidence):
    root, _, receipt = evidence
    before = (root / 'evaluation_lock.json').read_bytes()
    publication.atomic_json(receipt, {'status': 'uploading', 'task_id': 'existing-task'})
    result = publication.bounded([sys.executable, '-c', 'import time;time.sleep(10)'], receipt, timeout=0.05)
    assert result == {'status': 'pending', 'task_id': 'existing-task', 'error_type': 'TimeoutExpired'}
    assert (root / 'evaluation_lock.json').read_bytes() == before


def test_notebooks_preserve_contract_and_private_gallery_sanitisation(tmp_path):
    paths = build_notebooks.build_notebooks(tmp_path / 'notebooks')
    assert [path.name for path in paths] == list(build_notebooks.NAMES)
    first, second = [nbformat.read(path, as_version=4) for path in paths]
    for notebook in (first, second):
        assert all(cell.execution_count is None and not cell.outputs for cell in notebook.cells if cell.cell_type == 'code')
        for cell in notebook.cells:
            if cell.cell_type == 'code':
                compile(cell.source, '<notebook-cell>', 'exec')
        assert 'lower median' in notebook.cells[1].source
        assert '19 predefined schedules' in notebook.cells[1].source
    assert [cell.source for cell in first.cells if cell.cell_type == 'code' and cell.source.startswith('run_stage')] == [
        'run_stage("preflight")', 'run_stage("extract")', 'run_stage("predict")', 'run_stage("verify")']
    assert 'subprocess.TimeoutExpired' in second.cells[-1].source
    assert 'timeout=195' in second.cells[-1].source
    for count, cell in enumerate(second.cells):
        if cell.cell_type == 'code':
            cell.execution_count = count + 1
            if '# Private review evidence' in cell.source:
                cell.outputs = [nbformat.v4.new_output('display_data', data={'image/png': 'test-image', 'text/plain': 'private gallery'})]
    raw = tmp_path / 'raw.ipynb'
    nbformat.write(second, raw)
    safe = tmp_path / 'safe.ipynb'
    receipt = sanitiser.publish_notebook(clean_path=paths[1], executed_path=raw, output_path=safe,
                                         log_path=tmp_path / 'transcript.txt', label='08O test')
    assert 'image/png' not in safe.read_text()
    assert receipt


def test_actual_kernel_uses_current_runtime_and_preserves_clean_source(tmp_path):
    clean, raw, configuration = tmp_path / 'clean.ipynb', tmp_path / 'raw.ipynb', tmp_path / 'config.json'
    notebook = nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_markdown_cell('# Runtime probe'),
        nbformat.v4.new_code_cell('import os, sys\nassert os.environ["VPI_OBJECT_BANK_CONFIG"]\nprint(sys.version_info[:2])'),
        nbformat.v4.new_code_cell('print("completed probe")'),
    ])
    nbformat.write(notebook, clean)
    configuration.write_text('{}')
    clean_bytes = clean.read_bytes()
    receipt = execute_notebook.execute(clean, configuration, raw, cell_timeout=30)
    assert receipt['status'] == 'completed' and receipt['source_cells_unchanged'] is True
    assert clean.read_bytes() == clean_bytes
    executed = nbformat.read(raw, as_version=4)
    assert execute_notebook.signatures(executed) == execute_notebook.signatures(notebook)
    assert executed.cells[-1].outputs[0].text == 'completed probe\n'
    assert all(cell.execution_count is not None for cell in executed.cells if cell.cell_type == 'code')
