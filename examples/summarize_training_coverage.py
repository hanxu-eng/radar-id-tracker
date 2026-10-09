"""Path-free coverage experiment evidence with explicit training/validation roles."""
import argparse
import hashlib
import json
from pathlib import Path
from radar_id_tracker.lidar_training_protocol import load_training_plan


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--split-plan',type=Path,required=True)
    p.add_argument('--previous-run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();root=args.run_root
    read=lambda name:json.loads((root/name).read_text())
    plan,digest=load_training_plan(args.split_plan);protocol=read('evaluation_protocol.json')
    if protocol['split_plan_sha256']!=digest or protocol['model_sha256']!=hashlib.sha256((root/'model/model.json').read_bytes()).hexdigest():
        raise ValueError('frozen protocol/model differs')
    previous=json.loads((args.previous_run/'model/model.json').read_text())
    if protocol['previous_head_model_sha256']!=hashlib.sha256((args.previous_run/'model/model.json').read_bytes()).hexdigest():
        raise ValueError('previous frozen model differs')
    current=read('model/model.json')
    if current['parameters']!=previous['parameters']:raise ValueError('training method changed')
    if set(plan['validation_sequences'])&(set(previous['training_sequences'])|set(current['training_sequences'])):
        raise ValueError('reference or candidate trained on validation')
    rows=read('benchmark.json')+read('previous_head_results.json')
    sequences=plan['diagnostic_replay_sequences']+plan['validation_sequences']
    if len(rows)!=len(sequences)*6 or len({(r['sequence'],r['scenario'],r['profile']) for r in rows})!=len(rows):
        raise ValueError('incomplete comparison')
    evidence=[];reasons=[]
    for sequence in sequences:
        role='validation' if sequence in plan['validation_sequences'] else 'training_replay'
        for scenario in ('natural','dropout_3s'):
            pair={r['profile']:r for r in rows if r['sequence']==sequence and r['scenario']==scenario}
            if set(pair)!={'baseline','candidate','previous_head'}:raise ValueError('profile missing')
            for profile,row in pair.items():
                if row['role']!=role or row['run']['ground_truth_used_as_input'] is not False:raise ValueError('invalid role/label-free input')
                expected={'baseline':None,'candidate':protocol['model_sha256'],'previous_head':protocol['previous_head_model_sha256']}[profile]
                if row['run'].get('head_model_sha256')!=expected:raise ValueError('comparison model hash differs')
            scores={name:{k:row['evaluation'][k] for k in
                ('observed_tracks','published_tracks','all_input_detections','per_class_observed')} for name,row in pair.items()}
            candidate=scores['candidate']['observed_tracks']
            passed=(candidate['precision'] is not None and candidate['recall'] is not None and
                candidate['precision']>=plan['acceptance_precision'] and candidate['recall']>=plan['acceptance_recall'])
            evidence.append(dict(sequence=sequence,role=role,scenario=scenario,scored_frames=pair['candidate']['evaluation']['scored_frames'],
                scores=scores,precision_recall_guard_pass=passed))
            if role=='validation' and scenario=='natural' and not passed:reasons.append(f'{sequence} fails frozen precision/recall guard')
    fixed=read('fixed_recovery.json');reference=read('fixed_previous_head_recovery.json');latency=read('latency.json')
    if any(fixed[s]['candidate_same_id']<fixed[s]['baseline_same_id'] for s in plan['validation_sequences']):
        reasons.append('3s fixed-baseline recovery regresses on fresh validation')
    if latency['stages']['total_ms']['p99']>latency['period_budget_ms']:reasons.append('P99 pipeline exceeds period budget')
    audit=read('split_input_audit.json')
    if audit['split_plan_sha256']!=digest or audit['ground_truth_labels_opened'] is not False or audit['exact_scan_overlap_count']!=0:
        raise ValueError('invalid label-free source split audit')
    result=dict(plan=plan,protocol=protocol,training_protocol=read('model/protocol.json'),model=current,
        previous_model=previous,evaluation_provenance=read('evaluation_provenance.json'),
        training_loss=read('model/training_log.json'),training_diversity=read('training_diversity.json'),
        previous_training_diversity=json.loads((args.previous_run/'training_diversity.json').read_text()),
        split_input_audit={k:v for k,v in audit.items() if k!='scan_hashes'},
        split_geography=read('split_geography.json'),
        scope=dict(range_m=50,half_angle_degrees=40,iou3d=.25,classes=['Car','Pedestrian'],rear_scored=False,
                   official_kitti_score=False,dontcare_2d_evaluation=False),
        rows=evidence,strict_iou_results=read('strict_iou_results.json'),
        proposal_coverage=read('proposal_coverage.json'),
        fixed_recovery={s:{k:v for k,v in r.items() if k!='targets'} for s,r in fixed.items()},
        previous_head_fixed_recovery={s:{k:v for k,v in r.items() if k!='targets'} for s,r in reference.items()},
        outside_scope_parity=read('outside_scope_parity.json'),review_audit=read('review_audit.json'),
        latency={k:v for k,v in latency.items() if k!='records'},
        adoption_decision='rejected' if reasons else 'requires_further_validation',rejection_reasons=reasons,
        default_changed=False,real_vehicle_ready=False,
        control_limit='same per-sequence schedule; total optimization steps rise with sequence count; no single-factor causal claim')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(args.output)


if __name__=='__main__':main()
