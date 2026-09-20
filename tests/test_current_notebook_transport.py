"""Local synthetic/package fixtures; never install, download or execute scientific stages."""
import hashlib
import json
from pathlib import Path
import platform
import sys
import zipfile
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import current_notebook_runtime as runtime
from current_notebook_cells import bootstrap_cell
from package_current_notebooks import build_package
from current_stage_recovery import export_state, restore_state
from current_stage_execution import run_stage


@pytest.fixture(scope='module')
def package(tmp_path_factory):
    root=tmp_path_factory.mktemp('source-package')
    result=build_package(ROOT,root/'package',base_commit='uncommitted-preparation-base')
    return root/'package',result


def test_complete_source_only_package_and_archive_hashes(package):
    root,result=package
    record=runtime.verify_source(root/'source','notebooks/official-evaluation/10b-toysad8k-official-seen-unseen-evaluation.ipynb')
    assert record['source_state']=='working_directory_snapshot' and record['git_clean_claim'] is False
    for required in ('src/vpi_object_bank/scoring.py','src/visual_peer_inspector/workflows/learned_peer_features.py',
                     'src/vpi_remaining_evaluation/cohort.py','scripts/current_stage_recovery.py'):
        assert required in record['files']
    assert 'src/vpi_next_evaluation/gpu_compatibility.py' not in record['files']
    assert 'notebooks/input-preparation/current-dino-gpu-compatibility.ipynb' not in record['files']
    assert not any('.local' in name.split('/') or name.startswith('tests/') for name in record['files'])
    archive=root/'current-evaluation-source.zip.bin'
    assert runtime.digest(archive)==result['archive_sha256']
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.testzip() is None
        assert set(bundle.namelist())==set(record['files'])|{runtime.MANIFEST}
    assert result['native_inputs_included'] is False and result['kaggle_execution_verified'] is False


def test_bootstrap_validates_mutable_source_before_scientific_imports(package,monkeypatch):
    root,_=package
    monkeypatch.setattr(runtime,'check_runtime',lambda contract:{'python':contract['python'],'packages':contract['packages']})
    result=runtime.bootstrap(root/'source','notebooks/official-evaluation/10a-toysad8k-official-seen-frozen-inference.ipynb',environ={})
    assert result['provenance']['git_clean_claim'] is False
    source=bootstrap_cell('notebooks/official-evaluation/example.ipynb').source
    compile(source,'bootstrap','exec')
    assert source.index('_runtime.bootstrap(')<source.index('import pandas as pd')
    assert 'bootstrap-failure.json' in source


def test_changed_source_is_rejected_before_import(package,tmp_path):
    import shutil
    root,_=package
    shutil.copytree(root/'source',tmp_path/'source')
    (tmp_path/'source/src/vpi_next_evaluation/official.py').write_text('raise RuntimeError("must not execute")')
    with pytest.raises(ValueError,match='changed'):
        runtime.verify_source(tmp_path/'source')


def test_dependency_mismatch_is_actionable_and_does_not_install(monkeypatch):
    contract={'python':'3.12.14','packages':{'numpy':'2.5.3'}}
    with pytest.raises(RuntimeError,match='expected_python.*package_differences') as error:
        runtime.check_runtime(contract,versions={'numpy':'0.1'},python_version='3.12.1')
    assert '0.1' in str(error.value) and '3.12.14' in str(error.value)


def test_flat_nested_version_mounts_and_ambiguous_config(tmp_path):
    mount=tmp_path/'input'; flat=mount/'flat'; nested=mount/'datasets/owner/data/versions/3'
    flat.mkdir(parents=True);nested.mkdir(parents=True)
    (flat/'09b-current-config.json').write_text('{"kind":"synthetic"}')
    assert runtime.resolve_config('09b',tmp_path,environ={},mount=mount)=={'kind':'synthetic'}
    assert flat in runtime.bounded_roots(mount) and nested in runtime.bounded_roots(mount)
    (nested/'configs').mkdir();(nested/'configs/09b-current-config.json').write_text('{}')
    with pytest.raises(ValueError,match='Ambiguous'):
        runtime.resolve_config('09b',tmp_path,environ={},mount=mount)
    assert runtime.resolve_config('09b',tmp_path,environ={'VPI_NEXT_CONFIG':str(flat/'09b-current-config.json')},mount=mount)=={'kind':'synthetic'}


def test_config_lookup_never_recurses_unknown_deep_trees(tmp_path):
    mount=tmp_path/'input'; deep=mount/'raw/huge/ignored/deeper'
    deep.mkdir(parents=True);(deep/'09b-current-config.json').write_text('{}')
    with pytest.raises(FileNotFoundError):
        runtime.resolve_config('09b',tmp_path,environ={},mount=mount)


def test_source_archive_rejects_traversal_before_materialization(tmp_path,monkeypatch):
    # Execute only the stdlib locator definitions; no notebook or scientific imports.
    from current_notebook_cells import LOCATOR
    namespace={}
    exec(LOCATOR.split('_matches =')[0],namespace)
    archive=tmp_path/'bad.zip.bin'
    with zipfile.ZipFile(archive,'w') as bundle:
        bundle.writestr('../escape.py','bad')
    monkeypatch.setenv('VPI_SOURCE_ARCHIVE',str(archive));monkeypatch.setenv('VPI_SOURCE_ARCHIVE_SHA256',runtime.digest(archive))
    monkeypatch.setenv('VPI_SOURCE_DEST',str(tmp_path/'extracted'))
    with pytest.raises(ValueError,match='Unsafe'):
        namespace['_archive_source']()
    assert not (tmp_path/'escape.py').exists() and not (tmp_path/'extracted').exists()


def test_verified_zip_and_automatically_expanded_tree(package,tmp_path,monkeypatch):
    from current_notebook_cells import LOCATOR
    namespace={};exec(LOCATOR.split('_matches =')[0],namespace)
    root,result=package
    monkeypatch.setenv('VPI_SOURCE_ARCHIVE',str(root/'current-evaluation-source.zip.bin'))
    monkeypatch.setenv('VPI_SOURCE_ARCHIVE_SHA256',result['archive_sha256']);monkeypatch.setenv('VPI_SOURCE_DEST',str(tmp_path/'expanded'))
    restored=namespace['_archive_source']()
    assert runtime.verify_source(restored)['snapshot_sha256']==result['snapshot_sha256']
    monkeypatch.setenv('VPI_CODE_ROOT',str(restored))
    assert namespace['_source_candidates']()==[restored]


def test_offline_setup_rejects_changed_wheel_without_calling_pip(tmp_path,monkeypatch):
    wheels=tmp_path/'wheels';wheels.mkdir();wheel=wheels/'numpy.whl';wheel.write_bytes(b'changed')
    contract={'python':platform.python_version(),'packages':{'numpy':'2.5.3'}}
    (wheels/'wheelhouse_manifest.json').write_text(json.dumps({'schema':'vpi-current-offline-wheels/v1','runtime_contract':contract,'files':{'numpy.whl':'0'*64}}))
    monkeypatch.setattr(runtime.subprocess,'run',lambda *a,**k:pytest.fail('Must not invoke pip'))
    with pytest.raises(ValueError,match='integrity'):
        runtime.install_offline(contract,wheels,tmp_path/'target')


def test_offline_command_is_bounded_target_only_without_network(tmp_path,monkeypatch):
    wheels=tmp_path/'wheels';wheels.mkdir();wheel=wheels/'fixture.whl';wheel.write_bytes(b'fixture')
    contract={'python':platform.python_version(),'packages':{'fixture':'1'}}
    (wheels/'wheelhouse_manifest.json').write_text(json.dumps({'schema':'vpi-current-offline-wheels/v1','runtime_contract':contract,'files':{'fixture.whl':runtime.digest(wheel)}}))
    for name in ('numpy','pandas','scipy','matplotlib','PIL'):
        monkeypatch.delitem(sys.modules,name,raising=False)
    called=[]
    def fake(command,**kwargs):
        called.append((command,kwargs))
        raise runtime.subprocess.TimeoutExpired(command,300)
    monkeypatch.setattr(runtime.subprocess,'run',fake)
    with pytest.raises(RuntimeError,match='Offline dependency setup failed'):
        runtime.install_offline(contract,wheels,tmp_path/'target')
    command,options=called[0]
    assert '--no-index' in command and '--no-deps' in command and '--target' in command
    assert options['timeout']==300 and options['stdout']==runtime.subprocess.DEVNULL
    assert not (tmp_path/'target').exists()
    assert list(tmp_path.glob('target.partial-*/failure_receipt.json'))


def test_verified_recovery_export_restore_and_corruption(tmp_path):
    state=tmp_path/'science';state.mkdir();(state/'run_binding.json').write_text('{"method":"synthetic"}')
    (state/'scene').mkdir();(state/'scene/checkpoint.json').write_text('{"status":"complete"}')
    bundle=tmp_path/'v1.zip.bin';manifest=tmp_path/'v1.json'
    export=export_state(state,bundle,manifest,stage='10b',provenance={'snapshot_sha256':'a'*64})
    restored=restore_state(bundle,manifest,tmp_path/'restored',expected_binding=export['binding'])
    assert (Path(restored['state_directory'])/'scene/checkpoint.json').read_bytes()==(state/'scene/checkpoint.json').read_bytes()
    with pytest.raises(ValueError,match='identity differs'):
        restore_state(bundle,manifest,tmp_path/'wrong',expected_binding={**export['binding'],'stage':'09b'})
    bundle.write_bytes(bundle.read_bytes()+b'corrupt')
    with pytest.raises(ValueError,match='identity differs'):
        restore_state(bundle,manifest,tmp_path/'corrupt',expected_binding=export['binding'])


def test_failure_preserves_original_error_and_pending_recovery(tmp_path):
    output=tmp_path/'science'
    def fail():
        output.mkdir();(output/'run_binding.json').write_text('{}')
        raise ValueError('expected synthetic failure')
    with pytest.raises(ValueError,match='expected synthetic'):
        run_stage('10a',{'output_root':output,'recovery_export_dir':tmp_path/'exports','recovery_max_bytes':0},fail,{'snapshot_sha256':'a'*64})
    receipt=json.loads(next((tmp_path/'runtime-receipts').glob('*.json')).read_text())
    assert receipt['status']=='failed' and receipt['recovery']['status']=='pending'
    assert 'expected synthetic failure' not in json.dumps(receipt)
    assert (output/'run_binding.json').exists()


def test_config_and_missing_output_leave_safe_failure_receipts(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError):
        runtime.resolve_config('09b',tmp_path,environ={},mount=tmp_path/'missing')
    with pytest.raises(KeyError):
        run_stage('10a',{},lambda:pytest.fail('must not run'),{'snapshot_sha256':'a'*64})
    receipts=list((tmp_path/'.local/runtime-receipts').glob('*configuration-failure.json'))
    assert len(receipts)==2
    assert all(json.loads(p.read_text())['scientific_run_started'] is False for p in receipts)
