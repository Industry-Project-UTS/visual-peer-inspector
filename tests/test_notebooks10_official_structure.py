"""The current10 bodies must never dispatch the historical P1 runners."""
import json
import os
import sys
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
DIRECTORY=ROOT/'notebooks/official-evaluation'


def test_current10_notebooks_are_clean_and_compile():
    paths=sorted(DIRECTORY.glob('10*.ipynb'))
    assert len(paths)==5
    for path in paths:
        notebook=json.loads(path.read_text())
        source='\n'.join(''.join(c['source']) for c in notebook['cells'])
        assert path.name in source
        assert 'winner_cluster_consensus' in source and 'object_bank_top2' in source
        assert '_runtime.bootstrap(CODE_ROOT, NOTEBOOK_RELATIVE, runtime_profile=RUNTIME_PROFILE)' in source
        assert source.index('_runtime.bootstrap(') < source.index('import pandas as pd')
        assert 'current 09D' in source and 'ClearML' in source
        assert 'tie-aware top-1' in source and 'Recall@2' in source
        assert 'asset-grouped 95%' in source
        for forbidden in ('run_official_inference.py','run_official_release.py','run_one_shot_evaluation.py','pip install'):
            assert forbidden not in source
        for i,cell in enumerate(notebook['cells']):
            if cell['cell_type']=='code':
                assert cell['execution_count'] is None and not cell['outputs']
                compile(''.join(cell['source']),f'{path}:cell{i}','exec')


@pytest.mark.parametrize('stage,role',[('10a','seen'),('10a1','unseen')])
def test_current_official_notebook_rejects_conflicting_role_before_config(stage,role,monkeypatch):
    path=next(DIRECTORY.glob(stage+'-toysad8k-official-*.ipynb'))
    notebook=json.loads(path.read_text())
    source=next(''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code' and 'PINNED_OFFICIAL_COHORT' in ''.join(c['source']))
    guard=source.split('CONFIG = load_config()')[0]
    monkeypatch.setenv('VPI_OFFICIAL_COHORT','unseen' if role=='seen' else 'seen')
    with pytest.raises(ValueError,match='pinned'):
        exec(guard,{'os':os})


def test_cpu_prediction_metadata_has_no_evaluator_mounts():
    for stage in ('10a','10a1'):
        metadata=json.loads((DIRECTORY/f'kernel-metadata-{stage}.json').read_text())
        assert metadata['enable_gpu'] is False
        mounted=' '.join(metadata['dataset_sources']+metadata['kernel_sources'])
        assert 'evaluator' not in mounted and 'crosswalk' not in mounted
    notes=(DIRECTORY/'CURRENT_EXECUTION_READINESS.md').read_text()
    assert 'have not executed official experiments' in notes
    assert 'atomically release an already typed neutral official cache' in notes
    assert 'Typed current-schema evaluator' in notes


def test_both_official_regeneration_paths_keep_current_bodies(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    import build_notebooks10_official as official_builder
    import build_one_shot_evaluation_notebooks as shared_builder
    from notebook_runtime_cells import harden_source
    target=tmp_path/'notebooks/official-evaluation';target.mkdir(parents=True)
    for path in DIRECTORY.glob('kernel-metadata-10*.json'):
        (target/path.name).write_bytes(path.read_bytes())
    monkeypatch.setattr(official_builder,'NOTEBOOK_ROOT',target)
    official_builder.build()
    assert len(list(target.glob('10*.ipynb')))==5
    name='10b-toysad8k-official-seen-unseen-evaluation.ipynb'
    previous=json.loads((target/name).read_text())
    monkeypatch.setattr(shared_builder,'ROOT',tmp_path)
    shared_builder.build(next(c for c in shared_builder.CONFIGURATIONS if c['stage']=='10b'))
    regenerated=json.loads((target/name).read_text())
    normal=lambda cell:harden_source(''.join(cell['source'])) if cell['cell_type']=='code' else ''.join(cell['source'])
    assert [normal(cell) for cell in previous['cells']]==[normal(cell) for cell in regenerated['cells']]
    source='\n'.join(''.join(cell['source']) for cell in regenerated['cells'])
    assert 'evaluate_official(CONFIG, CODE_ROOT)' in source
    assert "('datasets', 2)" in source
    assert 'run_one_shot_evaluation.py' not in source
