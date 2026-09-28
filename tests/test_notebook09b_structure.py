"""The current 09B evaluates fixed ranking; it does not claim fitted review policy."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
NOTEBOOK_PATH=ROOT/'notebooks/protected-evaluation/09b-toysad8k-calibration-review-policy.ipynb'


def notebook_source():
    notebook=json.loads(NOTEBOOK_PATH.read_text())
    return notebook,'\n'.join(''.join(cell['source']) for cell in notebook['cells'])


def test_notebook09b_has_current_boundary_outputs_and_honest_policy_gap():
    _,source=notebook_source()
    for phrase in ('600 original','automatic winner consensus','foreground patch-bank strongest-two',
        'nineteen schedules','before the evaluator package is opened','tie-aware top-1','Recall@2',
        '2,000 draws','Schedule averages are formed per scene','original scene denominator',
        'does **not** fit','native evaluator','protocol_lock.json','per-view scores',
        'correctly projected coarse patch evidence','missing masks','private'):
        assert phrase in source
    assert "readiness('09b',cfg,ROOT)" in source
    assert source.index("lambda:evaluate_predictions('09b'") < source.index('lambda:render_current_report(')
    assert 'run_calibration_review_policy.py' not in source


def test_notebook09b_clean_cells_and_explicit_runtime_provenance():
    notebook,source=notebook_source()
    assert source.count('_runtime.bootstrap(')==1
    assert len([c for c in notebook['cells'] if c['cell_type']=='markdown'])>=7
    for index,cell in enumerate(notebook['cells']):
        if cell['cell_type']=='code':
            assert cell['execution_count'] is None and cell['outputs']==[]
            compile(''.join(cell['source']),f'{NOTEBOOK_PATH}:cell-{index}','exec')


def test_notebook09b_metadata_requires_current_resources():
    metadata=json.loads((NOTEBOOK_PATH.parent/'kernel-metadata-09b.json').read_text())
    assert metadata['code_file']==NOTEBOOK_PATH.name and metadata['is_private'] is True
    assert metadata['enable_gpu'] is False and metadata['enable_internet'] is False
    assert metadata['kernel_sources']==[] and metadata['dataset_sources']==[]


def test_current_publisher_is_bounded_and_aggregate_only():
    source=(ROOT/'scripts/publish_next_stage.py').read_text()
    assert 'timeout=180' in source and 'tracking.bounded(' in source
    assert "'evaluation_lock.json'" in source and "'report_lock.json'" in source
    assert 'CURRENT_TABLES=frozenset' in source and 'CURRENT_FIGURES=frozenset' in source
    assert 'with current_allowlist():' in source
    assert 'private/galleries' not in source
    assert 'CLEARML_API_ACCESS_KEY' not in source and 'CLEARML_API_SECRET_KEY' not in source
    assert "status='pending'" in source


def test_current_adapters_and_builders_compile():
    for name in ('src/vpi_next_evaluation/original.py','src/vpi_next_evaluation/reporting.py',
                 'scripts/build_notebook09b.py','scripts/notebooks09_current.py','scripts/publish_next_stage.py'):
        path=ROOT/name;compile(path.read_text(),name,'exec')
