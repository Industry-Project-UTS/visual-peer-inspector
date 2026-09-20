"""Current source/RGB release tested against tiny indexed archives only."""
from pathlib import Path
import json
import pytest
from vpi_next_evaluation import native_rgb as rgb
from vpi_object_bank.storage import digest_file, read_json, atomic_json, seal
from test_primary_resource_release import _fixture

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def fixture(tmp_path,monkeypatch):
    paths,contract=_fixture(tmp_path)
    monkeypatch.setattr(rgb,'CONTRACTS',{'calibration':contract})
    monkeypatch.setattr(rgb,'preparation_gates',lambda cfg,source:(contract,{'fixture_only':True}))
    admin=paths['administration'];selection=paths['selection']
    trusted=[json.loads(line) for line in (admin/'restricted_source_map.jsonl').read_text().splitlines()]
    for i,row in enumerate(trusted):row['source_base_asset_id']=f'asset-{i}'
    (admin/'restricted_source_map.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in trusted))
    for root in (admin,selection):
        files=sorted(p for p in root.iterdir() if p.name!='checksums.sha256')
        (root/'checksums.sha256').write_text(''.join(f'{digest_file(p)}  {p.name}\n' for p in files))
    monkeypatch.setattr(rgb,'EXPECTED_SELECTION_CHECKSUM_MANIFEST_SHA256',digest_file(selection/'checksums.sha256'))
    monkeypatch.setattr(rgb,'EXPECTED_ADMINISTRATIVE_CHECKSUM_MANIFEST_SHA256',digest_file(admin/'checksums.sha256'))
    cfg={'role':'calibration','raw_root':paths['source'],'selection_root':selection,'administrative_root':admin,
        'expected_source_checksums_sha256':digest_file(paths['source']/'package-checksums.sha256'),'output_root':tmp_path/'index'}
    return cfg,contract,paths


def test_current_rgb_index_and_release_real_indexed_archive(fixture,tmp_path,monkeypatch):
    cfg,contract,paths=fixture
    original=Path.read_text
    def forbidden(self,*args,**kwargs):
        if self.name=='calibration_labels.jsonl':raise AssertionError('Labels must not open')
        return original(self,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',forbidden)
    result=rgb.prepare_rgb_index(cfg,ROOT);assert result['scene_count']==2 and not result['labels_opened']
    index=read_json(cfg['output_root']/'rgb_index.json')
    assert 'anomaly_variants' not in json.dumps(index)
    release={**cfg,'rgb_index_root':cfg['output_root'],'output_root':tmp_path/'release','maximum_new_scenes':1}
    assert rgb.release_rgb(release,ROOT)['status']=='checkpointed'
    complete=rgb.release_rgb(release,ROOT);assert complete['status']=='rgb_complete'
    lock_hash=digest_file(release['output_root']/'release_lock.json')
    assert rgb.release_rgb(release,ROOT)==complete and digest_file(release['output_root']/'release_lock.json')==lock_hash
    package=release['output_root']/'model-input';rgb.verify_model_input_package(package,contract=contract)
    assert not any('source-' in p.name or 'label' in p.name for p in package.rglob('*'))


def test_index_precedes_exposure_gate(fixture,monkeypatch):
    cfg,_,_=fixture
    monkeypatch.setattr(rgb,'preparation_gates',lambda *args:(_ for _ in ()).throw(AssertionError('Index must not require future exposure')))
    assert rgb.prepare_rgb_index(cfg,ROOT)['status']=='private_index_complete'


def test_index_rejects_changed_source_checksum(fixture):
    cfg,_,_=fixture
    with pytest.raises(ValueError,match='reviewed mounted'):rgb.prepare_rgb_index({**cfg,'expected_source_checksums_sha256':'0'*64},ROOT)


def test_resealed_private_index_cannot_add_labels(fixture,tmp_path):
    cfg,_,_=fixture;rgb.prepare_rgb_index(cfg,ROOT);root=cfg['output_root']
    index=read_json(root/'rgb_index.json');index['scenes'][0]['labels']=[1,0,0];atomic_json(root/'rgb_index.json',index)
    binding=read_json(root/'run_binding.json');binding['index_sha256']=rgb.digest(index);atomic_json(root/'run_binding.json',binding)
    seal(root,[root/'run_binding.json',root/'rgb_index.json'])
    with pytest.raises(ValueError,match='Unexpected private'):rgb.release_rgb({**cfg,'rgb_index_root':root,'output_root':tmp_path/'release'},ROOT)


def test_rgb_checkpoint_corruption_stops_resume(fixture,tmp_path):
    cfg,_,_=fixture;rgb.prepare_rgb_index(cfg,ROOT)
    release={**cfg,'rgb_index_root':cfg['output_root'],'output_root':tmp_path/'release','maximum_new_scenes':1}
    rgb.release_rgb(release,ROOT)
    path=next((release['output_root']/'checkpoints').glob('*/media/*/rgb/*.png'));path.write_bytes(b'bad')
    with pytest.raises(ValueError):rgb.release_rgb(release,ROOT)


def test_rgb_output_budget_stops_before_image_read_including_completed_outputs(fixture,tmp_path,monkeypatch):
    cfg,_,_=fixture;rgb.prepare_rgb_index(cfg,ROOT)
    release={**cfg,'rgb_index_root':cfg['output_root'],'output_root':tmp_path/'release','maximum_new_scenes':2}
    def no_rgb(*args,**kwargs):raise AssertionError('Output budget must fail before reading RGB bytes')
    with monkeypatch.context() as scoped:
        scoped.setattr(rgb.IndexedArchive,'read',no_rgb)
        with pytest.raises(ValueError,match='maximum_output_bytes'):
            rgb.release_rgb({**release,'maximum_output_bytes':1},ROOT)
    assert not release['output_root'].exists()
    assert rgb.release_rgb(release,ROOT)['status']=='rgb_complete'
    with monkeypatch.context() as scoped:
        scoped.setattr(rgb.IndexedArchive,'read',no_rgb)
        with pytest.raises(ValueError,match='maximum_output_bytes'):
            rgb.release_rgb({**release,'maximum_output_bytes':1},ROOT)


def test_generated_notebooks_use_bootstrap_and_no_model_execution():
    import importlib.util,sys,nbformat
    sys.path.insert(0,str(ROOT/'scripts'))
    spec=importlib.util.spec_from_file_location('native_region_builder',ROOT/'scripts/build_native_regions_notebook.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    for relative in (module.RGB,module.REGIONS):
        notebook=nbformat.read(ROOT/relative,as_version=4);nbformat.validate(notebook)
        code=[c.source for c in notebook.cells if c.cell_type=='code']
        assert 'bootstrap' in code[0] and 'run_stage' in code[-1]
        assert all(c.execution_count is None and not c.outputs for c in notebook.cells if c.cell_type=='code')
        assert 'load_method_freeze' not in '\n'.join(code)
