"""Export portable cross-sequence results and conservative acceptance guards."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();root=args.run_root
    protocol=json.loads((root/'protocol.json').read_text())
    rows=json.loads((root/'benchmark.json').read_text())
    ablation=json.loads((root/'input_ablation.json').read_text())
    fixed=json.loads((root/'fixed_recovery.json').read_text())
    reference_path=root/'reference_recovery.json'
    reference=json.loads(reference_path.read_text()) if reference_path.exists() else None
    sequences=protocol['development_sequences']+protocol['validation_sequences']
    expected={(s,scenario,profile) for s in sequences for scenario in ('natural','dropout_3s') for profile in ('baseline','candidate')}
    if len(rows)!=len(expected) or {(r['sequence'],r['scenario'],r['profile']) for r in rows}!=expected:
        raise ValueError('complete paired benchmark required')
    natural=[];failures=[]
    for sequence in sequences:
        paired={r['profile']:r for r in rows if r['sequence']==sequence and r['scenario']=='natural'}
        base,candidate=paired['baseline']['evaluation'],paired['candidate']['evaluation']
        for profile,row in paired.items():
            if row['run']['ground_truth_used_as_input'] is not False:raise ValueError('label-free input required')
        score=candidate['observed_tracks']
        if (score['precision'] or 0)<protocol['acceptance_precision'] or (score['recall'] or 0)<protocol['acceptance_recall']:
            failures.append(dict(sequence=sequence,reason='overall precision/recall guard failed'))
        if score['id_changes']>base['observed_tracks']['id_changes']:
            failures.append(dict(sequence=sequence,reason='identity switches increased'))
        for label,values in candidate['per_class_observed'].items():
            has_output=values['tp']+values['fp']>0
            has_truth=values['tp']+values['fn']>0
            if (has_output and (values['precision'] or 0)<protocol['acceptance_precision'] or
                    has_truth and (values['recall'] or 0)<protocol['acceptance_recall']):
                failures.append(dict(sequence=sequence,label=label,reason='class precision/recall guard failed'))
        natural.append(dict(sequence=sequence,role='development' if sequence in protocol['development_sequences'] else 'validation',
            frames=candidate['frames'],scored_frames=candidate['scored_frames'],
            excluded_missing_scan_frames=candidate['excluded_missing_scan_frames'],
            baseline={k:base[k] for k in ('observed_tracks','per_class_observed','all_input_detections')},
            candidate={k:candidate[k] for k in ('observed_tracks','per_class_observed','all_input_detections')}))
    if not protocol['grouped_selection']['fold_selection_agrees_with_final']:
        failures.append(dict(reason='input setting unstable across leave-one-sequence-out folds'))
    for sequence,result in fixed.items():
        if result['candidate_wrong_retained_identity'] or result['candidate_same_id']<result['baseline_same_id']:
            failures.append(dict(sequence=sequence,reason='fixed-denominator recovery regressed or wrong retained identity'))
    if not sum(v['eligible'] for v in fixed.values()):
        failures.append(dict(reason='no fixed-denominator 3s recovery opportunity'))
    portable_protocol={**protocol,'implementation_sha256':{Path(k).name:v for k,v in protocol['implementation_sha256'].items()}}
    result=dict(experiment='raw LiDAR input adaptation, sequence-grouped development + frozen validation',
        protocol=portable_protocol,scope=protocol['scope'],natural=natural,fixed_recovery=fixed,
        prior_pointrcnn_fixed_recovery=reference,
        development_ablation=[{k:r[k] for k in ('sequence','scale','sweeps','observed','raw_scores','output_sha256')} for r in ablation],
        acceptance=dict(passed=not failures,failures=failures,default_changed=False,real_vehicle_ready=False,
            limitations=['local front-only 3D IoU diagnostic, not official KITTI tracking',
                         'KITTI sensor-domain validation does not certify RoboSense or rear accuracy',
                         'one artificial all-detection dropout per sequence, not natural occlusion validation']))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result['acceptance'],indent=2))


if __name__=='__main__':main()
