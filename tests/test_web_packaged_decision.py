"""The deployable source tree must run the selected decision without GPU imports."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys


def test_packaged_current_decision_runs_without_neural_libraries(tmp_path):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('space_packager_qa', root / 'apps/web/package_space.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    target = tmp_path / 'space'
    module.package(target)
    program = '''
import sys
from visual_peer_inspector.web.decision import decide_cached
result = decide_cached({'a': {'objects': []}, 'b': {'objects': []}}, [], views=['a','b'], mode='automatic')
assert result['ranking'] == []
assert not {'torch','transformers','sam2'} & set(sys.modules)
from vpi_object_bank.scoring import score_scene
assert callable(score_scene)
'''
    process = subprocess.run([sys.executable, '-c', program], cwd=target,
        env={**os.environ, 'PYTHONPATH': str(target / 'src')}, capture_output=True, text=True, timeout=30)
    assert process.returncode == 0, process.stderr
    assert not (target / 'scripts/verify_web_cached_replay.py').exists()
    assert not (target / 'tests').exists()
