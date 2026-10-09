"""Path-free detector-head evidence; training replay is never held-out evidence."""
import argparse
import hashlib
import json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--diagnostic-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();root=args.run_root
    protocol=json.loads((root/'evaluation_protocol.json').read_text())
    training=json.loads((root/'model/protocol.json').read_text())
    model=json.loads((root/'model/model.json').read_text())
    assert protocol['model_sha256']==hashlib.sha256((root/'model/model.json').read_bytes()).hexdigest()
    rows=json.loads((root/'benchmark.json').read_text())
    sequences=protocol['training_sequences']+protocol['validation_sequences']
    assert len(rows)==len(sequences)*4
    assert len({(r['sequence'],r['scenario'],r['profile']) for r in rows})==len(rows)
    evidence=[]
    for sequence in sequences:
        role='training_replay' if sequence in protocol['training_sequences'] else 'validation'
        if role=='validation':assert sequence not in model['training_sequences']
        for scenario in ('natural','dropout_3s'):
            pair={r['profile']:r for r in rows if r['sequence']==sequence and r['scenario']==scenario}
            assert set(pair)=={'baseline','candidate'}
            assert all(r['role']==role and r['run']['ground_truth_used_as_input'] is False for r in pair.values())
            assert pair['candidate']['run']['head_model_sha256']==protocol['model_sha256']
            scores={profile:{k:r['evaluation'][k] for k in
                ('observed_tracks','published_tracks','all_input_detections','per_class_observed')} for profile,r in pair.items()}
            c=scores['candidate']['observed_tracks'];b=scores['baseline']['observed_tracks']
            pass_pr=(c['precision'] is not None and c['recall'] is not None and
                c['precision']>=protocol['acceptance_precision'] and c['recall']>=protocol['acceptance_recall'])
            evidence.append(dict(sequence=sequence,role=role,scenario=scenario,
                scored_frames=pair['candidate']['evaluation']['scored_frames'],scores=scores,
                precision_recall_guard_pass=pass_pr,id_changes_non_regressing=c['id_changes']<=b['id_changes']))
    fixed=json.loads((root/'fixed_recovery.json').read_text())
    latency=json.loads((root/'latency.json').read_text())
    validation=[row for row in evidence if row['role']=='validation' and row['scenario']=='natural']
    reasons=[]
    if not all(row['precision_recall_guard_pass'] for row in validation):
        reasons.append('Fresh validation fails the frozen precision/recall guard')
    if any(row['scores']['candidate']['per_class_observed']['Pedestrian']['recall'] is not None and
            row['scores']['candidate']['per_class_observed']['Pedestrian']['recall']<
            row['scores']['baseline']['per_class_observed']['Pedestrian']['recall'] for row in validation):
        reasons.append('Pedestrian recall regresses on fresh validation')
    if any(fixed[s]['candidate_same_id']<fixed[s]['baseline_same_id'] for s in protocol['validation_sequences']):
        reasons.append('Fixed-baseline 3s recovery regresses on fresh validation')
    if latency['stages']['total_ms']['p99']>latency['period_budget_ms']:
        reasons.append('P99 pipeline latency exceeds the period budget')
    result=dict(protocol=protocol,training_protocol=training,model=model,
        training_loss=json.loads((root/'model/training_log.json').read_text()),
        training_diversity=json.loads((root/'training_diversity.json').read_text()),
        scope=dict(range_m=50,half_angle_degrees=40,iou3d=.25,classes=['Car','Pedestrian'],
            rear_scored=False,official_kitti_score=False,dontcare_2d_evaluation=False),
        original_diagnostics=json.loads((args.diagnostic_root/'diagnostics.json').read_text()),
        rows=evidence,fixed_recovery={s:{k:v for k,v in r.items() if k!='targets'} for s,r in
            fixed.items()},
        previous_quality=[{k:r[k] for k in ('sequence','scenario','role','evaluation')} for r in
            json.loads((root/'previous_quality_results.json').read_text())],
        strict_iou_results=json.loads((root/'strict_iou_results.json').read_text()),
        outside_scope_parity=json.loads((root/'outside_scope_parity.json').read_text()),
        review_audit=json.loads((root/'review_audit.json').read_text()),
        latency={k:v for k,v in latency.items() if k!='records'},
        adoption_decision='rejected' if reasons else 'requires_further_validation',rejection_reasons=reasons,
        default_changed=False,real_vehicle_ready=False)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(args.output)

if __name__=='__main__':main()
