"""Path-free cross-sequence evidence with failures and fixed recovery denominators."""
import argparse
import json
from pathlib import Path
import numpy as np
from radar_id_tracker.lidar_score_calibration import load_calibration_plan
from fit_calibration_folds import sha


def aggregate(metrics):
    keys=('tp','fp','fn','idtp','idfp','idfn','id_changes','fragmentations')
    total={k:sum(m.get(k,0) for m in metrics) for k in keys}
    ratio=lambda a,b:a/b if b else None
    total.update(precision=ratio(total['tp'],total['tp']+total['fp']),recall=ratio(total['tp'],total['tp']+total['fn']),
        idf1=ratio(2*total['idtp'],2*total['idtp']+total['idfp']+total['idfn']),
        sequence_count=len(metrics),macro_precision=float(np.mean([m['precision'] for m in metrics if m['precision'] is not None])) if any(m['precision'] is not None for m in metrics) else None,
        macro_recall=float(np.mean([m['recall'] for m in metrics if m['recall'] is not None])) if any(m['recall'] is not None for m in metrics) else None)
    return total


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();root=args.run_root
    plan,digest=load_calibration_plan(args.plan);complete=json.loads((root/'evaluation_complete.json').read_text())
    if complete['plan_sha256']!=digest or complete['rows']!=136 or complete['strict_rows']!=68:raise ValueError('incomplete cross-sequence evaluation')
    rows=json.loads((root/'benchmark.json').read_text());natural=[r for r in rows if r['scenario']=='natural']
    strict=json.loads((root/'strict_iou_results.json').read_text());fixed=json.loads((root/'fixed_recovery.json').read_text())
    reliability=json.loads((root/'heldout_reliability.json').read_text());geo=json.loads((root/'geometry_diagnostics.json').read_text())
    folds=[]
    for i,(train,calibrate,evaluate) in enumerate(plan['roles']):
        head=json.loads((root/f'fold_{i}'/'model/model.json').read_text());model=json.loads((root/f'fold_{i}'/'calibration/model.json').read_text())
        folds.append(dict(fold=i,train=plan['groups'][train],calibrate=plan['groups'][calibrate],evaluate=plan['groups'][evaluate],
            head_model_sha256=sha(root/f'fold_{i}'/'model/model.json'),head_weights_sha256=head['weights_sha256'],
            calibration_model_sha256=sha(root/f'fold_{i}'/'calibration/model.json'),calibration_parameters=model['models'],
            training_parameters=head['parameters'],calibration_label_sha256=model['labels_sha256']))
    profiles=('baseline','head_raw','calibrated_strong','calibrated_medium');totals={};by_fold={}
    for profile in profiles:
        subset=[r for r in natural if r['profile']==profile]
        totals[profile]=dict(observed=aggregate([r['evaluation']['observed_tracks'] for r in subset]),
            per_class={l:aggregate([r['evaluation']['per_class_observed'][l] for r in subset]) for l in ('Car','Pedestrian')},
            strict_observed=aggregate([r['evaluation']['observed_tracks'] for r in strict if r['profile']==profile]),
            strict_per_class={l:aggregate([r['evaluation']['per_class_observed'][l] for r in strict if r['profile']==profile]) for l in ('Car','Pedestrian')})
        by_fold[profile]={str(i):aggregate([r['evaluation']['observed_tracks'] for r in subset if r['fold']==i]) for i in range(3)}
    recovery={}
    for profile in ('head_raw','calibrated_strong','calibrated_medium'):
        values=[r[profile] for r in fixed.values() if r[profile] is not None]
        recovery[profile]={k:sum(r[k] for r in values) for k in ('eligible','baseline_same_id','candidate_same_id','candidate_no_pretrack','candidate_no_return','candidate_wrong_retained_identity')}
    failures=[]
    for r in natural:
        if r['profile']!='calibrated_medium':continue
        e=r['evaluation'];reference=next(x for x in natural if x['profile']=='baseline' and x['sequence']==r['sequence'])['evaluation']
        for label,m in [('overall',e['observed_tracks']),*e['per_class_observed'].items()]:
            if (m['tp']+m['fn']>0 and (m['precision'] is None or m['precision']<plan['acceptance_precision'] or m['recall']<plan['acceptance_recall'])) or (m['tp']+m['fn']==0 and m['fp']>0):
                failures.append(dict(sequence=r['sequence'],label=label,reason='precision/recall acceptance',precision=m['precision'],recall=m['recall'],fp=m['fp']))
        if e['observed_tracks']['id_changes']>reference['observed_tracks']['id_changes']:
            failures.append(dict(sequence=r['sequence'],reason='extra identity switches'))
        f=fixed[r['sequence']]['calibrated_medium']
        if f['candidate_same_id']<f['baseline_same_id'] or f['candidate_wrong_retained_identity']:
            failures.append(dict(sequence=r['sequence'],reason='fixed recovery loss or wrong identity',**{k:f[k] for k in ('eligible','baseline_same_id','candidate_same_id','candidate_wrong_retained_identity')}))
    audit=json.loads((root/'split_role_audit.json').read_text());latency=json.loads((root/'latency.json').read_text())
    review=json.loads((root/'review_audit.json').read_text())
    data=dict(format_version=1,experiment='whole-sequence three-role calibration rotations',plan=plan,plan_sha256=digest,
        folds=folds,default_model_changed=False,recommended=False,final_deployment_model_trained=False,
        outer_evaluation_is_development_only=True,reserved_sequence_opened=False,
        evaluation_protocol=json.loads((root/'evaluation_protocol.json').read_text()),
        natural_frames=sum(r['evaluation']['frames'] for r in natural if r['profile']=='baseline'),
        scored_frames=sum(r['evaluation']['scored_frames'] for r in natural if r['profile']=='baseline'),
        input_frames_missing=[dict(sequence=r['sequence'],frames=r['evaluation']['excluded_missing_scan_frames']) for r in natural if r['profile']=='baseline' and r['evaluation']['excluded_missing_scan_frames']],
        totals=totals,by_fold=by_fold,fixed_recovery_totals=recovery,fixed_recovery=fixed,
        failures=failures,heldout_reliability=reliability,geometry_diagnostics=geo,
        per_sequence=[dict(fold=r['fold'],sequence=r['sequence'],profile=r['profile'],observed=r['evaluation']['observed_tracks'],
                           per_class=r['evaluation']['per_class_observed']) for r in natural],
        strict_per_sequence=strict,
        split_role_audit=audit,latency={k:v for k,v in latency.items() if k!='records'},review_audit=review,
        outside_scope_parity=json.loads((root/'outside_scope_parity.json').read_text()),
        limitations=['No new final-test labels opened. Previously viewed development sequences only.',
            'Each fold trains only 5/6 sequences; not the full 17-sequence model and not directly comparable to the prior experiment.',
            'Monotone calibration cannot improve detector ranking or repair geometry.',
            'GPS ego distances and scan hashes do not prove physical-object/route/weather independence.',
            '3-second all-detection dropout is artificial, not natural physical occlusion.',
            'Local front ROI/IoU metrics are not official KITTI; rear and RoboSense remain unvalidated.',
            'No best fold or evaluation-threshold selection; final epoch/policies fixed.',
            'Calibration sequences are used to fit coefficients; their fitted reliability is not held-out evidence.'])
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(totals=totals,by_fold=by_fold,recovery=recovery,failure_count=len(failures)),indent=2))


if __name__=='__main__':main()
