"""Selected-method visual evidence, separate from the sealed comparison report."""
from pathlib import Path
import pandas as pd
from .reporting import gallery, verified_report_inputs
from .storage import atomic_json, bind, digest_file, read_json, seal, source_hashes, verify_seal


def build_selected_demo(cfg, source_root):
    hashes = verified_report_inputs(cfg)
    decision = read_json(Path(cfg['evaluation_root'])/'shareable/decision.json')
    method = decision['selected_method']
    root = Path(cfg.get('selected_demo_root', Path(cfg['report_root']).parent/'selected-demo'))
    binding = {'schema':'vpi-selected-object-evidence/v1','input_locks':hashes,'method':method,
               'source':source_hashes(source_root,['src/vpi_object_bank/selected_demo.py','src/vpi_object_bank/reporting.py'])}
    bind(root,binding)
    if (root/'artifact_lock.json').exists():
        verify_seal(root)
        return read_json(root/'receipt.json')
    manifest=read_json(Path(cfg['prepared_root'])/'input_manifest.json')
    metrics=pd.read_csv(Path(cfg['evaluation_root'])/'private/scene_metrics.csv')
    selected=metrics[(metrics['route']=='automatic_reference')&(metrics['view_set_id']=='three-primary')]
    bank=selected[selected['method']==method].set_index('scene')['top1_tie_expected']
    anchor=selected[selected['method']=='within_view_mean'].set_index('scene')['top1_tie_expected']
    delta=(bank-anchor).sort_values(kind='stable')
    choices=[(manifest['scenes'][0],'fixed first scene'),(str(delta.index[0]),'largest selected-method loss; post-hoc illustration'),
             (str(delta.index[-1]),'largest selected-method gain; post-hoc illustration')]
    entries=[]
    for scene,reason in dict(choices).items():
        for route in ('supplied_reference','automatic_reference','automatic_predicted'):
            destination=root/'private'/f'{scene}-{route}-{method}.png'
            item=gallery(cfg,scene,route,'three-primary',method,destination)
            entries.append(item|{'selection_reason':reason})
    atomic_json(root/'private/gallery_manifest.json',entries)
    receipt={'status':'completed','method':method,'gallery_count':len(entries),
             'scope':'private development examples; separate from evaluation denominator'}
    atomic_json(root/'receipt.json',receipt)
    seal(root,[p for p in root.rglob('*') if p.is_file()],metadata=receipt)
    return receipt
