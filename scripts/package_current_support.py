#!/usr/bin/env python3
"""Prepare the private Kaggle support dataset from two exact sealed allowlists."""
from pathlib import Path
import argparse
import shutil
from vpi_object_bank.storage import atomic_json,digest_file,read_json,seal,verify_seal
from vpi_next_evaluation.exposure import verify_exposure
import publish_remaining_evaluation as publication
from publish_executed_notebook import SECRET_TEXT,LOCAL_PATH_TEXT


def prepare(source_root,output_root):
    source_root,root=Path(source_root),Path(output_root)
    reference=source_root/'reports/remaining-cohort-workbench/development100'
    exposure=source_root/'.local/exposure-qualified-original2000'
    public_lock=verify_seal(reference,name='publication_lock.json');exposure_lock=verify_seal(exposure)
    expected={'run_binding.json',publication.NOTEBOOK,publication.LOG,publication.RECEIPT,
        *(f'tables/{name}' for name in publication.TABLES),*(f'figures/{name}' for name in publication.FIGURES)}
    if set(public_lock['files'])!=expected:raise ValueError('09E reference differs from its exact public allowlist')
    expected_exposure={'exposure_ledger.json','asset_records.json',*(f'indexes/{role}/scene_index.json' for role in ('development','calibration','internal_lock'))}
    if set(exposure_lock['files'])!=expected_exposure:raise ValueError('Exposure package contains undeclared data')
    exposure_checks={role:verify_exposure(role,exposure) for role in ('development','calibration','internal_lock')}
    publication.validate_notebook(reference/publication.NOTEBOOK,{public_lock['files'][f'figures/{name}'] for name in publication.FIGURES})
    for name in publication.TABLES:publication.tracking.validate_text_artifact(reference/'tables'/name)
    for base,paths in ((reference,expected),(exposure,expected_exposure)):
        for relative in paths:
            path=base/relative
            if path.is_symlink() or not path.resolve().is_relative_to(base.resolve()):raise ValueError('Support source must be a regular allowlisted file')
            if path.suffix!='.png' and (SECRET_TEXT.search(path.read_text()) or LOCAL_PATH_TEXT.search(path.read_text())):
                raise ValueError('Support source contains credential-shaped text or local paths')
    if root.exists():raise FileExistsError('Choose a fresh support output; never overwrite a prepared dataset')
    root.mkdir(parents=True)
    for base,label,paths,lockname in ((reference,'method-reference',expected,'publication_lock.json'),(exposure,'original-exposure',expected_exposure,'artifact_lock.json')):
        for relative in sorted(paths|{lockname}):
            target=root/label/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(base/relative,target)
            if digest_file(target)!=digest_file(base/relative):raise ValueError('Support copy changed bytes')
        verify_seal(root/label,name=lockname)
    (root/'README.md').write_text('# Current evaluation support\n\nPrivate prepared inputs; no Kaggle job has been run by preparing this dataset.\n\nmethod-reference is the unchanged, sanitized09E publication for the already selected methods. Its historical readiness statements describe that completed stage; the new launch kit defines current preparation. original-exposure contains only the reviewed neutral role/asset indexes and explicit uncertain exposure history. It does not certify untouched cohorts or independent confirmation. No raw images, object anomaly answers, credentials or recovery archives are included.\n')
    metadata={'id':'esreekarreddy/vpi-current-evaluation-support','title':'VPI Current Evaluation Support','licenses':[{'name':'other'}],
        'description':'Private sealed method reference and exposure-qualified original-role indexes for the fixed two-mode evaluation. No raw images or anomaly labels. Prepared locally; not uploaded.'}
    atomic_json(root/'dataset-metadata.json',metadata)
    files={p.relative_to(root).as_posix():digest_file(p) for p in sorted(root.rglob('*')) if p.is_file()}
    manifest={'schema':'vpi-current-evaluation-support/v1','files':files,
        'reference_publication_sha256':digest_file(reference/'publication_lock.json'),
        'exposure_lock_sha256':digest_file(exposure/'artifact_lock.json'),
        'exposure_verification':exposure_checks,'raw_anomaly_labels_included':False,'raw_images_included':False,
        'aggregate_development_evidence_included':True,'uploaded':False,'scientific_runs_started':False}
    atomic_json(root/'support_manifest.json',manifest)
    seal(root,[root/relative for relative in files]+[root/'support_manifest.json'])
    return {'status':'prepared','files':len(files)+2,'bytes':sum(p.stat().st_size for p in root.rglob('*') if p.is_file()),
        'support_manifest_sha256':digest_file(root/'support_manifest.json'),'artifact_lock_sha256':digest_file(root/'artifact_lock.json'),'uploaded':False}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-root',type=Path,default=Path(__file__).resolve().parents[1]);parser.add_argument('--output-root',type=Path,required=True)
    args=parser.parse_args();print(prepare(args.source_root,args.output_root))
