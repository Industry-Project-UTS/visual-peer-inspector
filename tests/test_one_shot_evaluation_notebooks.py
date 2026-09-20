"""Current one-shot notebook generation and scientific/report boundary checks."""
import ast
import json
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BUILDER=ROOT/'scripts/build_one_shot_evaluation_notebooks.py'
NOTEBOOKS={'09d':ROOT/'notebooks/protected-evaluation/09d-toysad8k-internal-lock-one-shot-evaluation.ipynb',
           '10b':ROOT/'notebooks/official-evaluation/10b-toysad8k-official-seen-unseen-evaluation.ipynb'}


def _sources(path):
    cells=json.loads(path.read_text())['cells']
    return ('\n'.join(''.join(c['source']) for c in cells if c['cell_type']=='markdown'),
            '\n'.join(''.join(c['source']) for c in cells if c['cell_type']=='code'),cells)


def test_builder_is_deterministic():
    before={stage:path.read_bytes() for stage,path in NOTEBOOKS.items()}
    subprocess.run([sys.executable,'-B',str(BUILDER)],check=True,cwd=ROOT)
    assert {stage:path.read_bytes() for stage,path in NOTEBOOKS.items()}==before


def test_notebooks_are_clean_and_call_current_evaluators():
    for stage,path in NOTEBOOKS.items():
        markdown,source,cells=_sources(path)
        assert path.name in markdown and source.count('_runtime.bootstrap(')==1
        assert 'run_one_shot_evaluation.py' not in source
        assert 'CLEARML_API_ACCESS_KEY' not in source and 'CLEARML_API_SECRET_KEY' not in source
        assert 'evaluate_predictions(' in source if stage=='09d' else 'evaluate_official(' in source
        for cell in cells:
            if cell['cell_type']=='code':
                assert cell['execution_count'] is None and cell['outputs']==[]
                ast.parse(''.join(cell['source']))


def test_09d_seals_before_reporting_and_does_not_retune():
    markdown,source,_=_sources(NOTEBOOKS['09d'])
    assert '400 original' in markdown and 'nineteen schedules are fixed' in markdown
    assert source.index("lambda:evaluate_predictions('09d'") < source.index('lambda:render_current_report(')
    assert "name='evaluation_lock.json'" in source and "name='protocol_lock.json'" in source
    assert 'original scene denominator' in markdown and 'seed 32040' in markdown


def test_10b_keeps_official_roles_and_provenance_separate():
    markdown,source,_=_sources(NOTEBOOKS['10b'])
    combined=(markdown+source).lower()
    assert 'seen' in combined and 'unseen' in combined
    assert 'provenance' in combined and 'do not select methods using official outcomes' in combined
    assert source.index('evaluate_official(') < source.index('render_official_report(')


def test_current_tracking_uses_scientific_and_report_lock_bindings():
    source=(ROOT/'scripts/publish_next_stage.py').read_text()
    assert 'tracking.verified_lock' in source and 'tracking.TABLES=frozenset' in source
    assert 'evaluation_lock_sha256' in source and 'timeout=180' in source


def test_kernel_metadata_uses_cpu_and_explicit_current_inputs():
    for stage,path in NOTEBOOKS.items():
        metadata=json.loads((path.parent/f'kernel-metadata-{stage}.json').read_text())
        assert metadata['enable_gpu'] is False and metadata['enable_internet'] is False
        assert metadata['dataset_sources']==[] and metadata['kernel_sources']==[]
