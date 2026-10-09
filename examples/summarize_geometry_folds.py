"""Complete path-free matched-budget geometry evidence, including every failure."""
import argparse
import json
from pathlib import Path
from summarize_calibration_folds import aggregate
from fit_calibration_folds import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run-root','output'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();root=args.run_root
    complete=json.loads((root/'evaluation_complete.json').read_text())
    if complete['rows']!=136 or complete['strict_rows']!=68:raise ValueError('incomplete geometry evaluation')
    protocol=json.loads((root/'protocol.json').read_text());plan=protocol['parent_plan']
    inference=json.loads((root/'inference_protocol.json').read_text())
    rows=json.loads((root/'benchmark.json').read_text());natural=[r for r in rows if r['scenario']=='natural']
    strict=json.loads((root/'strict_iou_results.json').read_text());fixed=json.loads((root/'fixed_recovery.json').read_text())
    geo=json.loads((root/'geometry_diagnostics.json').read_text());totals={};by_fold={};recovery={}
    profiles=('baseline','head_raw',*protocol['variants'])
    for profile in profiles:
        values=[r for r in natural if r['profile']==profile];s=[r for r in strict if r['profile']==profile]
        totals[profile]=dict(observed=aggregate([r['evaluation']['observed_tracks'] for r in values]),
            per_class={l:aggregate([r['evaluation']['per_class_observed'][l] for r in values]) for l in ('Car','Pedestrian')},
            strict_observed=aggregate([r['evaluation']['observed_tracks'] for r in s]),
            strict_per_class={l:aggregate([r['evaluation']['per_class_observed'][l] for r in s]) for l in ('Car','Pedestrian')})
        by_fold[profile]={str(i):aggregate([r['evaluation']['observed_tracks'] for r in values if r['fold']==i]) for i in range(3)}
        if profile!='baseline':
            recovery[profile]={k:sum(r[profile][k] for r in fixed.values()) for k in ('eligible','baseline_same_id','candidate_same_id','candidate_no_pretrack','candidate_no_return','candidate_wrong_retained_identity')}
    failures=[];comparisons=[]
    for row in natural:
        if row['profile']!='weighted_axis_huber':continue
        e=row['evaluation'];sequence=row['sequence']
        for label,m in [('overall',e['observed_tracks']),*e['per_class_observed'].items()]:
            if (m['tp']+m['fn']>0 and (m['precision'] is None or m['precision']<protocol['acceptance_precision'] or m['recall']<protocol['acceptance_recall'])) or (m['tp']+m['fn']==0 and m['fp']>0):
                failures.append(dict(sequence=sequence,label=label,reason='precision/recall acceptance',precision=m['precision'],recall=m['recall'],fp=m['fp']))
        for reference in ('baseline','head_raw','continued_l1'):
            base=next(r for r in natural if r['sequence']==sequence and r['profile']==reference)['evaluation']
            if e['observed_tracks']['id_changes']>base['observed_tracks']['id_changes']:
                failures.append(dict(sequence=sequence,reason='extra identity switches',reference=reference))
        f=fixed[sequence]['weighted_axis_huber'];control=fixed[sequence]['continued_l1']
        if f['candidate_same_id']<max(f['baseline_same_id'],control['candidate_same_id']) or f['candidate_wrong_retained_identity']:
            failures.append(dict(sequence=sequence,reason='fixed recovery loss or wrong identity',eligible=f['eligible'],baseline_same_id=f['baseline_same_id'],control_same_id=control['candidate_same_id'],candidate_same_id=f['candidate_same_id'],candidate_wrong_retained_identity=f['candidate_wrong_retained_identity']))
        comparison=dict(sequence=sequence,fold=row['fold'])
        for profile in ('head_raw','continued_l1','weighted_axis_huber'):
            value=next(r for r in strict if r['sequence']==sequence and r['profile']==profile)['evaluation']['per_class_observed']['Pedestrian']
            comparison[profile]={k:value[k] for k in ('tp','fp','fn','precision','recall','id_changes')}
        comparisons.append(comparison)
    coverage={profile:{label:{k:sum(r[k] for r in geo['records'] if r['profile']==profile and r['label']==label) for k in ('gt','covered_025','covered_05')} for label in ('Car','Pedestrian')} for profile in profiles}
    latency=json.loads((root/'latency.json').read_text());models={}
    for key,value in inference['models'].items():
        fold,variant=key.split('/');path=root/f'fold_{fold}'/variant/'model/model.json'
        if sha(path)!=value['model_sha256']:raise ValueError('frozen geometry model changed before summary')
        models[key]={**value,'training_log':json.loads((path.parent/'training_log.json').read_text()),'frozen_tensor_audit':inference['frozen_tensor_audits'][key]}
    result=dict(format_version=1,experiment='matched-budget pedestrian geometry regression',protocol=protocol,
        protocol_sha256=sha(root/'protocol.json'),inference_protocol_sha256=sha(root/'inference_protocol.json'),
        evaluation_protocol=json.loads((root/'evaluation_protocol.json').read_text()),models=models,
        default_model_changed=False,recommended=False,final_deployment_model_trained=False,
        outer_evaluation_is_development_only=True,reserved_sequence_opened=False,
        natural_frames=sum(r['evaluation']['frames'] for r in natural if r['profile']=='baseline'),
        scored_frames=sum(r['evaluation']['scored_frames'] for r in natural if r['profile']=='baseline'),
        totals=totals,by_fold=by_fold,fixed_recovery_totals=recovery,fixed_recovery=fixed,failures=failures,
        strict_pedestrian_comparison=comparisons,raw_best_box_coverage=coverage,geometry_diagnostics=geo,
        per_sequence=[dict(fold=r['fold'],sequence=r['sequence'],profile=r['profile'],observed=r['evaluation']['observed_tracks'],per_class=r['evaluation']['per_class_observed']) for r in natural],
        strict_per_sequence=[dict(fold=r['fold'],sequence=r['sequence'],profile=r['profile'],observed=r['evaluation']['observed_tracks'],per_class=r['evaluation']['per_class_observed']) for r in strict],
        outside_scope_parity=json.loads((root/'outside_scope_parity.json').read_text()),review_audit=json.loads((root/'review_audit.json').read_text()),
        latency={k:v for k,v in latency.items() if k!='records'},
        limitations=['Previously reviewed development folds repeated; not a new final test or independent hardware evidence.',
            'No best fold, epoch or threshold selected after evaluation. Six fixed epochs and all frozen heads included.',
            'Compare against continued L1 at identical sample/optimization budget to attribute the loss effect.',
            'Heatmap weights frozen; geometry can still change proposal survival via center/NMS/ROI and tracking association.',
            'Prior score calibration is not reused on changed geometry; all profiles use raw scores and fixed 0.7/0.1 tracker.',
            'Local front +/-40deg 50m IoU metrics are not official KITTI. Rear and RoboSense remain unvalidated.',
            'Raw best-box coverage permits duplicate reuse and is not one-to-one recall.',
            'Fixed 3s all-detection dropout is artificial, not natural occlusion.',
            'Model/feature/pointcloud artifacts are local ignored files, not bundled deployments.'])
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(totals=totals,recovery=recovery,coverage=coverage,failure_count=len(failures)),indent=2))


if __name__=='__main__':main()
