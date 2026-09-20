"""Repairable aggregate charts and private, coordinate-checked evidence panels."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from PIL import Image

from vpi_object_bank.storage import atomic_json, atomic_unit, bind, digest_file, read_json, seal, verify_seal, verified_unit
from visual_peer_inspector.models.sam2 import decode_binary_mask
from visual_peer_inspector.representation.crops import prepare_masked_crop, project_crop_heatmap
from .contracts import METRICS
from .original import _required
from . import diagnostic_figures


def _cache_file(root, relative, lock):
    path = Path(root) / relative
    if relative not in lock['files'] or path.is_symlink() or not path.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError('Gallery media must be a sealed relative cache artifact')
    if digest_file(path) != lock['files'][relative]:
        raise ValueError('Gallery source file changed')
    return path


def scene_panel(payload, setting, cache_root, cache_lock, output_path):
    """Use cached masks and original image coordinates; never infer missing pixels."""
    views = setting['views']; objects = setting['rankings'][setting['method']]
    ranked = sorted((o for o in objects if o['score'] is not None), key=lambda o: (-o['score'], o['id']))
    tied = [o for o in ranked if np.isclose(o['score'], ranked[0]['score'], atol=1e-12, rtol=0)] if ranked else []
    selected = tied[0] if len(tied) == 1 else None
    ids = {o['id']: f'O{i+1}' for i, o in enumerate(sorted(objects, key=lambda o: o['id']))}
    owners = {(m['view_id'], m['candidate_id']): o for o in objects for m in o['members']}
    observations = {(o['view_id'], o['candidate_id']): o for o in setting['observations']}
    fig, axes = plt.subplots(4, len(views), figsize=(4 * len(views), 12), squeeze=False, constrained_layout=True)
    missing, heatmaps, unlocated, located_count = [], 0, [], 0
    for col, view in enumerate(views):
        projected_here = False
        for ax in axes[:, col]: ax.set_axis_off()
        meta = payload.get('evidence_views', {}).get(view)
        if meta is None:
            missing.append(view); unlocated.append(view); axes[0, col].text(.05, .5, f'{view}: sealed RGB/mask evidence missing')
            continue
        rgb = np.asarray(Image.open(_cache_file(cache_root, meta['rgb_relative'], cache_lock)).convert('RGB'))
        h, w = rgb.shape[:2]; overlay = rgb.astype(float); black = np.zeros_like(rgb)
        axes[0, col].imshow(rgb); axes[0, col].set_title(f'{view}: original uploaded view')
        selected_member = next((m['candidate_id'] for m in selected['members'] if m['view_id'] == view), None) if selected else None
        geometry_by_candidate = {candidate: geometry for candidate, geometry in meta.get('candidates', {}).items()
            if geometry.get('mask') is not None and geometry.get('box') is not None}
        for candidate, geometry in geometry_by_candidate.items():
            mask = decode_binary_mask(geometry['mask'])
            if mask.shape != (h, w): raise ValueError('Mask shape differs from original image')
            x1, y1, x2, y2 = geometry['box']
            if not (0 <= x1 <= x2 <= w and 0 <= y1 <= y2 <= h): raise ValueError('Object box lies outside original image')
            black[mask] = rgb[mask]
            owner = owners.get((view, candidate))
            colour = np.asarray(plt.get_cmap('tab10')((int(ids[owner['id']][1:])-1) % 10)[:3]) if owner else np.array([.6,.6,.6])
            overlay[mask] = .65 * overlay[mask] + .35 * colour * 255
        axes[1, col].imshow(overlay.astype('uint8')); axes[1, col].set_title('SAM masks, object IDs and boxes')
        axes[2, col].imshow(black); axes[2, col].set_title('Masked objects on black')
        axes[3, col].imshow(rgb)
        for candidate, geometry in geometry_by_candidate.items():
            owner = owners.get((view, candidate)); x1, y1, x2, y2 = geometry['box']
            colour = 'red' if candidate == selected_member else 'cyan'
            axes[1, col].add_patch(Rectangle((x1,y1),x2-x1,y2-y1,fill=False,edgecolor=colour,linewidth=2))
            axes[1, col].text(x1,y1,ids[owner['id']] if owner else 'unmatched',color='white',fontsize=9)
        if selected_member is None:
            axes[3, col].set_title('No unique selection' if len(tied) > 1 else 'Selected object not linked in this view')
            unlocated.append(view)
            continue
        if selected_member not in geometry_by_candidate:
            axes[3, col].set_title('Selected object cannot be located in this view: geometry missing')
            unlocated.append(view)
            continue
        geometry = geometry_by_candidate[selected_member]; mask = decode_binary_mask(geometry['mask'])
        located_count += 1
        x1,y1,x2,y2 = geometry['box']
        axes[3, col].add_patch(Rectangle((x1,y1),x2-x1,y2-y1,fill=False,edgecolor='red',linewidth=3))
        patch_scores = observations[(view, selected_member)].get('object_bank', {}).get('patch_scores', [])
        if patch_scores and all(key in geometry for key in ('valid70','crop_sha256','padded_box_xyxy')):
            valid = np.asarray(geometry['valid70'], dtype=bool)
            if valid.shape != (16,16) or int(valid.sum()) != len(patch_scores): raise ValueError('Patch scores differ from the cached foreground grid')
            crop = prepare_masked_crop(rgb, mask)
            if crop.sha256 != geometry['crop_sha256'] or list(crop.padded_box_xyxy) != geometry['padded_box_xyxy']:
                raise ValueError('Cached crop transform or pixels changed')
            heatmap = np.full((16,16), np.nan, dtype=np.float32); heatmap[valid] = patch_scores
            projected = project_crop_heatmap(heatmap, crop, mask)
            axes[3, col].imshow(np.ma.masked_invalid(projected),cmap='magma',alpha=.6); heatmaps += 1; projected_here = True
        axes[3, col].set_title('Selected object; coarse patch evidence' if projected_here else 'Selected object; heatmap unavailable')
    fig.suptitle(f"{setting['mode']} | {setting['method']} | {len(views)} views | {setting.get('correspondence_provenance','unspecified')}\nObject boxes are not validated defect boundaries\n{setting.get('evaluation_caption','')}")
    fig.savefig(output_path,dpi=120);plt.close(fig)
    matrix = np.full((len(objects),len(views)),np.nan)
    ordered = sorted(objects,key=lambda o:o['id'])
    for i,obj in enumerate(ordered):
        for member in obj['members']:
            value=observations[(member['view_id'],member['candidate_id'])]['normalized_score']
            if value is not None: matrix[i,views.index(member['view_id'])]=value
    fig, ax = plt.subplots(figsize=(max(5,len(views)),max(3,len(objects)*.4)),constrained_layout=True)
    if len(objects):
        shown=ax.imshow(np.ma.masked_invalid(matrix),vmin=0,vmax=1,cmap='viridis',aspect='auto');fig.colorbar(shown,ax=ax,label='Per-view percentile score')
        ax.set_yticks(range(len(objects)),[f"{ids[o['id']]} · combined {o['score'] if o['score'] is not None else 'unavailable'} · support {o['support']}" for o in ordered])
    else: ax.text(.05,.5,'No rankable objects')
    ax.set_xticks(range(len(views)),views);ax.set_title('Every retained object; missing observations stay blank')
    fig.savefig(Path(output_path).with_name('score_matrix.png'),dpi=120);plt.close(fig)
    score_rows=[]
    for obj in ordered:
        values=[(observations[(m['view_id'],m['candidate_id'])]['normalized_score'],m['view_id'],m['candidate_id']) for m in obj['members']]
        usable=[item for item in values if item[0] is not None]
        cutoff=sorted((item[0] for item in usable),reverse=True)[min(2,len(usable))-1] if usable and setting['mode']=='assisted' else None
        above=sum(item[0]>cutoff for item in usable) if cutoff is not None else 0
        cutoff_ties=sum(item[0]==cutoff for item in usable) if cutoff is not None else 0
        for view in views:
            member=next((m for m in obj['members'] if m['view_id']==view),None)
            observed=observations[(view,member['candidate_id'])] if member else None
            value=observed['normalized_score'] if observed else None
            weight=None
            if value is not None and setting['mode']=='assisted':
                if value>cutoff:weight=1.0/min(2,len(usable))
                elif value==cutoff:weight=(min(2,len(usable))-above)/cutoff_ties/min(2,len(usable))
                else:weight=0.0
            score_rows.append({'object_id':obj['id'],'display_id':ids[obj['id']],'view_id':view,
                'candidate_id':member['candidate_id'] if member else None,'normalized_score':value,
                'combined_score':obj['score'],'support':obj['support'],'fusion_observations':obj.get('fusion_observations'),
                'fusion_weight_equivalent':weight,'fusion_tie_note':'equal contributors at the top2 cutoff share equivalent weight; no unique contributing view asserted',
                'fusion_role':'top2 percentile contributor' if setting['mode']=='assisted' else 'retained winner vote; cluster secondary score also applies',
                'raw_bank_score':observed.get('object_bank',{}).get('score') if observed else None,
                **(observed['component_scores'] if observed else {})})
    pd.DataFrame(score_rows).to_csv(Path(output_path).with_name('object_per_view_scores.csv'),index=False)
    atomic_json(Path(output_path).with_name('combined_ranking.json'),{'method':setting['method'],'mode':setting['mode'],
        'correspondence_provenance':setting.get('correspondence_provenance'),'ranking':objects,
        'association_tracks':setting.get('association_tracks',[]),
        'automatic_support_note':'winner votes and exact stored cluster diagnostics explain consensus; percentile averaging is not its full fusion rule'})
    return {'missing_media_views':missing,'projected_heatmap_count':heatmaps,'top_tie_count':len(tied),
            'selected_object_located_views':located_count,'unlocated_views':unlocated}


def render_current_report(cfg):
    evaluation, prediction, cache, root = (_required(cfg,key) for key in ('evaluation_root','prediction_root','cache_root','report_root'))
    verify_seal(evaluation,name='evaluation_lock.json');verify_seal(prediction,name='prediction_lock.json')
    cache_lock=verify_seal(cache)
    ev_binding=read_json(evaluation/'run_binding.json');pred_binding=read_json(prediction/'run_binding.json')
    if ev_binding['prediction_lock_sha256']!=digest_file(prediction/'prediction_lock.json') or pred_binding['cache_lock_sha256']!=digest_file(cache/'artifact_lock.json'):
        raise ValueError('Report mixes incompatible evaluation, prediction or cache inputs')
    binding={'evaluation_lock_sha256':digest_file(evaluation/'evaluation_lock.json'),'prediction_lock_sha256':digest_file(prediction/'prediction_lock.json'),
        'cache_lock_sha256':digest_file(cache/'artifact_lock.json'),'source_sha256':digest_file(__file__),
        'diagnostic_renderer_sha256':digest_file(diagnostic_figures.__file__)}
    binding_hash=bind(root,binding)
    if (root/'report_lock.json').exists(): verify_seal(root,name='report_lock.json');return read_json(root/'report_receipt.json')
    summaries=pd.read_csv(evaluation/'shareable/metric_summary.csv');figure_root=root/'shareable/figures';figure_root.mkdir(parents=True,exist_ok=True)
    caption=diagnostic_figures.exposure_caption(evaluation)
    figures=[]
    for metric in METRICS:
        fig,ax=plt.subplots(figsize=(9,5),constrained_layout=True)
        for (route,method,provenance),group in summaries[summaries.metric==metric].groupby(['route','method','correspondence_provenance']):
            group=group.sort_values('view_count')
            line=ax.plot(group.view_count,100*group['mean'],marker='o',label=f'{route}: {method} ({provenance})')[0]
            ax.vlines(group.view_count,100*group.ci_low,100*group.ci_high,color=line.get_color(),alpha=.7)
        ax.set(xlabel='Views (1 diagnostic; 2/3/5 core)',ylabel=f'{metric} (%)',ylim=(0,100),xticks=[1,2,3,5]);ax.legend(fontsize=8)
        ax.set_title(caption,fontsize=10)
        name=metric+'.png';fig.savefig(figure_root/name,dpi=130);plt.close(fig);figures.append(name)
    diagnostics=diagnostic_figures.render_diagnostic_figures(evaluation,root)
    figures.extend(item['file'] for item in diagnostics['figures'])
    frame=pd.read_csv(evaluation/'private/scene_metrics.csv');first=sorted(frame.scene.unique())[0]
    plans=[(first,mode,view,'fixed first scene; outcome-independent') for mode in ('automatic','assisted') for view in ('two-primary','three-primary','five-primary')]
    failures=frame[(frame.view_set_id=='three-primary')&(frame.top1_tie_expected<1)]
    if len(failures):
        row=failures.sort_values(['scene','mode']).iloc[0];plans.append((row.scene,row['mode'],'three-primary','post-hoc failure illustration; not an evaluation subset'))
    receipts=[]
    for i,(scene,mode,view,reason) in enumerate(plans):
        unit=root/'private/galleries'/f'example-{i+1:02d}'
        if not verified_unit(unit,binding_hash):
            payload=read_json(cache/'scenes'/f'{scene}.json');pred=read_json(prediction/'scenes'/scene/'predictions.json')
            setting=next(s for s in pred['settings'] if s['mode']==mode and s['view_set_id']==view)
            setting={**setting,'evaluation_caption':caption}
            with atomic_unit(unit,binding_hash) as temp:
                result=scene_panel(payload,setting,cache,cache_lock,temp/'objects_masks_selection.png')
                atomic_json(temp/'receipt.json',{'scene':scene,'mode':mode,'view_set_id':view,'reason':reason,**result})
        receipts.append(read_json(unit/'receipt.json'))
    atomic_json(root/'private/gallery_manifest.json',receipts)
    receipt={'status':'report_complete','aggregate_figures':figures,'gallery_count':len(receipts),
        'evaluation_caption':caption,
        'missing_media_examples':sum(bool(r['missing_media_views']) for r in receipts),
        'scientific_results_unchanged':True,'private_outputs':'scene panels and per-object scores stay private',
        'tracking_status':'not attempted; publish only an explicit aggregate allowlist after report seal'}
    atomic_json(root/'report_receipt.json',receipt)
    seal(root,[p for p in root.rglob('*') if p.is_file() and '.partial' not in str(p) and '.interrupted-' not in str(p)],name='report_lock.json')
    return receipt
