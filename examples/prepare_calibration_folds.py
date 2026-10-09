"""Freeze three train/calibrate/evaluate rotations with verified cached features."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from radar_id_tracker.lidar_score_calibration import load_calibration_plan


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--features',type=Path,required=True)
    p.add_argument('--run-root',type=Path,required=True)
    args=p.parse_args();plan,digest=load_calibration_plan(args.plan)
    raw=(args.features/'manifest.json').read_bytes();source=json.loads(raw)
    sequences=[s for g in plan['groups'] for s in g]
    if sorted(sequences) != sorted(source['training_sequences']) or source['inference_config'] != plan['inference_config']:
        raise ValueError('verified development feature domain differs')
    frozen=args.run_root/'protocol.json';args.run_root.mkdir(parents=True,exist_ok=True)
    protocol=dict(plan=plan,plan_sha256=digest,feature_manifest_sha256=hashlib.sha256(raw).hexdigest(),
                  reserved_sequence_opened=False,evaluation_roles_are_development_only=True)
    if frozen.exists() and json.loads(frozen.read_text()) != protocol: raise ValueError('frozen protocol differs')
    frozen.write_text(json.dumps(protocol,indent=2)+'\n')
    for index,(train,calibrate,evaluate) in enumerate(plan['roles']):
        directory=args.run_root/f'fold_{index}'/'features';directory.mkdir(parents=True,exist_ok=True)
        path=directory/'manifest.json'
        training=plan['groups'][train];validation=plan['groups'][calibrate]+plan['groups'][evaluate]
        records=[r for r in source['records'] if r['sequence'] in training]
        manifest={**source,'training_sequences':training,'validation_sequences':validation,'records':records,
            'labels_sha256':{s:source['labels_sha256'][s] for s in training},
            'source_scans':[r for r in source['source_scans'] if r['sequence'] in training],
            'calibration_sequences':plan['groups'][calibrate],'evaluation_sequences':plan['groups'][evaluate],
            'calibration_plan_sha256':digest,'parent_feature_manifest_sha256':protocol['feature_manifest_sha256']}
        manifest.pop('split_plan_sha256',None)
        # Aggregate target counts from the parent describe all 17 sequences, not this fold.
        manifest.pop('targets',None)
        if path.exists():
            if json.loads(path.read_text()) != manifest: raise ValueError('existing fold feature split differs')
            continue
        for r in records:
            for kind in ('feature','targets'):
                original=args.features/r[kind]
                if hashlib.sha256(original.read_bytes()).hexdigest() != r[kind+'_sha256']:
                    raise ValueError('cached feature/target checksum differs')
                target=directory/r[kind];target.parent.mkdir(exist_ok=True)
                if not target.exists():os.link(original,target)
                if hashlib.sha256(target.read_bytes()).hexdigest() != r[kind+'_sha256']:
                    raise ValueError('linked fold artifact differs')
        path.write_text(json.dumps(manifest,indent=2)+'\n')
        print(json.dumps(dict(fold=index,train=training,calibrate=plan['groups'][calibrate],evaluate=plan['groups'][evaluate],features=len(records))),flush=True)


if __name__=='__main__':main()
