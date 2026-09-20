"""Model asset provenance checks with synthetic bytes; no download or model load."""
from pathlib import Path
import importlib.util
import sys
import json
import pytest

ROOT=Path(__file__).resolve().parents[1]


def module():
    sys.path.insert(0,str(ROOT/'scripts'))
    spec=importlib.util.spec_from_file_location('current_model_assets_fixture',ROOT/'scripts/prepare_current_model_assets.py')
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result


def test_verified_model_snapshot_resume_never_downloads_or_loads(tmp_path,monkeypatch):
    asset=module();spec=asset.specs(ROOT)['locateanything'];root=tmp_path/'assets';target=root/'locateanything';target.mkdir(parents=True)
    (target/'model.safetensors').write_bytes(b'synthetic fixture weights, never loaded')
    asset.atomic_json(target/'model_snapshot.json',{'schema':'vpi-immutable-model-snapshot/v1','model_id':spec['model_id'],
        'revision':spec['revision'],'files':{'model.safetensors':asset.digest(target/'model.safetensors')}})
    monkeypatch.setattr(asset.subprocess,'Popen',lambda *args,**kwargs:(_ for _ in ()).throw(AssertionError('No download during verified resume')))
    receipt=asset.prepare(ROOT,root,only='locateanything')
    assert receipt['model_inference_executed'] is False and receipt['contains_dataset_images'] is False
    assert receipt['models']['locateanything']['model_loaded'] is False
    assert asset.prepare(ROOT,root,only='locateanything')==receipt
    (target/'model.safetensors').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='checksum'):asset.prepare(ROOT,root,only='locateanything')


def test_snapshot_rejects_an_unpinned_revision_before_download(tmp_path,monkeypatch):
    asset=module();spec=asset.specs(ROOT)['locateanything'];target=tmp_path/'locateanything';target.mkdir()
    (target/'model.safetensors').write_bytes(b'fixture')
    asset.atomic_json(target/'model_snapshot.json',{'schema':'vpi-immutable-model-snapshot/v1','model_id':spec['model_id'],
        'revision':'main','files':{'model.safetensors':asset.digest(target/'model.safetensors')}})
    monkeypatch.setattr(asset.subprocess,'Popen',lambda *args,**kwargs:(_ for _ in ()).throw(AssertionError('No remote retry for bad local snapshot')))
    with pytest.raises(ValueError,match='revision differs'):asset.prepare(ROOT,tmp_path,only='locateanything')
