"""The prepared 09A/09C sources execute only current, sealed cached decisions."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_ROOT = ROOT / 'notebooks/protected-evaluation'


def test_notebooks_are_clean_prediction_only_and_compile():
    for stage in ('09a', '09c'):
        path = next(NOTEBOOK_ROOT.glob(stage + '-*.ipynb'))
        notebook = json.loads(path.read_text())
        source = '\n'.join(''.join(cell['source']) for cell in notebook['cells'])
        assert path.name in source
        assert f"readiness('{stage}',cfg,ROOT)" in source
        assert f"predict_current('{stage}',cfg,ROOT)" in source
        assert source.index('preflight=readiness(') < source.index('lambda:predict_current(')
        assert "preflight['status']!='prerequisites_present'" in source
        assert '_runtime.bootstrap(CODE_ROOT, NOTEBOOK_RELATIVE, runtime_profile=RUNTIME_PROFILE)' in source
        assert source.index('_runtime.bootstrap(') < source.index('import pandas as pd')
        assert 'prediction_lock.json' in source and 'per_view_scores.csv' in source
        assert 'evaluator should be mounted here' in source
        assert 'anomaly labels' in source and 'Two, three and five views' in source
        assert 'Scene checkpoints bind cache, method, exposure, source and schedule hashes' in source
        assert 'exported checkpoint files' in source
        assert 'run_protected_cohort_inference.py' not in source
        code = '\n'.join(''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type']=='code')
        assert "evaluate_predictions('" not in code and 'find_evaluator(' not in code
        for index, cell in enumerate(notebook['cells']):
            if cell['cell_type']=='code':
                assert cell['execution_count'] is None and cell['outputs']==[]
                compile(''.join(cell['source']),f'{path}:cell-{index}','exec')


def test_metadata_requires_explicit_current_resources_and_cpu():
    for stage in ('09a','09c'):
        metadata=json.loads((NOTEBOOK_ROOT/f'kernel-metadata-{stage}.json').read_text())
        assert metadata['is_private'] is True and metadata['enable_gpu'] is False
        assert metadata['enable_internet'] is False
        assert metadata['dataset_sources']==[] and metadata['kernel_sources']==[]
        assert metadata['code_file'].startswith(stage+'-')
