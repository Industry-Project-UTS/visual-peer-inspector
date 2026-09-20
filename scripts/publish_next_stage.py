#!/usr/bin/env python3
"""Bounded aggregate-only tracking for prepared current09/10 evaluators."""
from contextlib import redirect_stdout,redirect_stderr
from pathlib import Path
import argparse
import json
import os
import sys

import publish_clearml as tracking

tracking.TABLES=frozenset({'metric_summary.csv','view_schedule_summary.csv','failure_summary.csv',
    'identity_summary.csv','anomaly_count_summary.csv','majority_summary.csv','evaluation_receipt.json'})
tracking.FIGURES=frozenset({'failure_breakdown.png','candidate_coverage.png','identity_quality.png','identity_fragmentation.png','schedule_sensitivity.png'}) | frozenset({name+'.png' for name in ('top1_tie_expected','ap_tie_expected','recall_at_2_tie_expected','reciprocal_rank_tie_expected')})


def verify_publication(stage, evaluation_root, report_root):
    """Validate aggregate-only science/report compatibility without any service call."""
    evaluation_root,report_root=Path(evaluation_root),Path(report_root)
    evaluation,_=tracking.verified_lock(evaluation_root,'evaluation_lock.json')
    report,_=tracking.verified_lock(report_root,'report_lock.json')
    if 'run_binding.json' not in evaluation['files'] or 'run_binding.json' not in report['files']:
        raise ValueError('Scientific publication locks omit their input bindings')
    if tracking.read_receipt(evaluation_root/'run_binding.json').get('stage')!=stage:
        raise ValueError('Publication stage differs from completed scientific computation')
    if tracking.read_receipt(report_root/'run_binding.json').get('evaluation_lock_sha256')!=tracking.digest(evaluation_root/'evaluation_lock.json'):
        raise ValueError('Report figures belong to different scientific results')
    return tracking.collect_artifacts(evaluation_root,evaluation_root/'shareable','evaluation_lock.json',report_root)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',required=True,choices=['dev-evaluate','09b','09d','10b'])
    parser.add_argument('--evaluation-root',required=True,type=Path)
    parser.add_argument('--report-root',required=True,type=Path)
    parser.add_argument('--receipt',required=True,type=Path)
    parser.add_argument('--worker',action='store_true')
    args=parser.parse_args()
    try:
        verify_publication(args.stage, args.evaluation_root, args.report_root)
        from current_notebook_runtime import configure_tracking_runtime
        configure_tracking_runtime(Path(__file__).resolve().parents[1])
    except BaseException as error:
        receipt = tracking.read_receipt(args.receipt)
        receipt.update(status='pending', error_type=type(error).__name__, science_preserved=True, publication_started=False)
        tracking.atomic_json(args.receipt, receipt)
        print(json.dumps({'status': 'pending', 'error_type': type(error).__name__, 'science_preserved': True}))
        return
    if not args.worker:
        command=[sys.executable,'-B',str(Path(__file__).absolute()),'--stage',args.stage,
            '--evaluation-root',str(args.evaluation_root),'--report-root',str(args.report_root),
            '--receipt',str(args.receipt),'--worker']
        receipt=tracking.bounded(command,args.receipt,timeout=180)
    else:
        try:
            with open(os.devnull,'w') as sink,redirect_stdout(sink),redirect_stderr(sink):
                verify_publication(args.stage,args.evaluation_root,args.report_root)
                receipt=tracking.publish(args.evaluation_root,args.evaluation_root/'shareable',args.receipt,
                    report_root=args.report_root,task_name=f'{args.stage.upper()} current frozen multi-view evaluation')
        except BaseException as error:
            receipt=tracking.read_receipt(args.receipt)
            receipt.update(status='pending',error_type=type(error).__name__)
            tracking.atomic_json(args.receipt,receipt)
    print(json.dumps({key:receipt[key] for key in ('status','task_id','task_url','error_type') if key in receipt}))


if __name__=='__main__':main()
