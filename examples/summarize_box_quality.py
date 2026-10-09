"""Publish compact, path-free quality-model evidence from frozen replay artifacts."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();root=args.run_root
    protocol=json.loads((root/'models/protocol.json').read_text())
    rows=json.loads((root/'benchmark.json').read_text())
    sequences=protocol['training_sequences']+protocol['validation_sequences']
    assert len(rows)==len(sequences)*4
    assert len({(r['sequence'],r['scenario'],r['profile']) for r in rows})==len(rows)
    evidence=[]
    for sequence in sequences:
        role='out_of_fold' if sequence in protocol['training_sequences'] else 'validation'
        model=root/'models'/('fold_'+sequence if role=='out_of_fold' else 'full')/'model.json'
        model_data=json.loads(model.read_text())
        assert sequence not in model_data['training_sequences']
        digest=hashlib.sha256(model.read_bytes()).hexdigest()
        expected=next(f['model_sha256'] for f in protocol['folds'] if f['held_out_sequence']==(sequence if role=='out_of_fold' else None))
        assert digest==expected
        for scenario in ('natural','dropout_3s'):
            pair={r['profile']:r for r in rows if r['sequence']==sequence and r['scenario']==scenario}
            assert set(pair)=={'baseline','candidate'}
            for r in pair.values():assert r['role']==role and r['run']['ground_truth_used_as_input'] is False
            meta=json.loads((root/'quality_cache'/sequence/'inference_summary.json').read_text())
            assert meta['box_quality_model_sha256']==digest
            scores={}
            for profile,row in pair.items():
                result=row['evaluation']
                scores[profile]={k:result[k] for k in ('observed_tracks','published_tracks','all_input_detections','per_class_observed')}
            candidate=scores['candidate']['observed_tracks']
            baseline=scores['baseline']['observed_tracks']
            evidence.append(dict(sequence=sequence,role=role,scenario=scenario,
                scored_frames=pair['baseline']['evaluation']['scored_frames'],model_sha256=digest,
                scores=scores,precision_recall_guard_pass=(candidate['precision']>=protocol['acceptance_precision'] and candidate['recall']>=protocol['acceptance_recall']),
                id_changes_non_regressing=candidate['id_changes']<=baseline['id_changes']))
    targets={s:json.loads((root/'features'/s/'target_manifest.json').read_text())['targets'] for s in protocol['training_sequences']}
    fixed=json.loads((root/'fixed_recovery.json').read_text())
    recovery={s:{k:v for k,v in value.items() if k!='targets'} for s,value in fixed.items()}
    result=dict(protocol=protocol,scope=dict(range_m=50,half_angle_degrees=40,iou3d=.25,
        classes=['Car','Pedestrian'],rear_scored=False,official_kitti_score=False,
        dontcare_2d_evaluation=False,training_dontcare_negative_exclusion=True),
        acceptance=dict(default_configuration_changed=False,real_vehicle_ready=False,
        precision_recall_guard_pass=False,reasons=['fresh validation precision/recall guard failures',
        'pedestrian recall remains inadequate','0008 ID changes increased','12/100 frames miss 100ms deadline',
        '3s artificial-dropout evidence has only 3 eligible targets; rear and RoboSense data unvalidated']),
        training_target_counts=targets,rows=evidence,fixed_recovery=recovery,
        integrated_parity=json.loads((root/'integrated_parity.json').read_text()),
        latency=json.loads((root/'latency.json').read_text()))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(args.output)


if __name__=='__main__':main()
