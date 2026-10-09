"""Label-free full-sequence inference for frozen three-role calibration folds."""
import argparse
import hashlib
import json
from pathlib import Path
from radar_id_tracker.lidar_score_calibration import load_calibration_plan
from radar_id_tracker.lidar_head_targets import load_head_manifest
from radar_id_tracker.lidar_raw import infer_sequence


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--input-audit',type=Path,required=True)
    args=p.parse_args();plan,digest=load_calibration_plan(args.plan);audit=json.loads(args.input_audit.read_text())
    protocol=json.loads((args.run_root/'protocol.json').read_text())
    if protocol['plan_sha256']!=digest or audit['exact_scan_overlap_count']!=0:raise ValueError('frozen plan or source audit differs')
    source_audit_sha=sha(args.input_audit)
    output=args.run_root/'cache_manifest.json';records={}
    existing=[]
    for path in Path('outputs').rglob('inference_summary.json'):
        m=json.loads(path.read_text())
        if (m.get('inference_config')==plan['inference_config'] and m.get('head_model_sha256') is None and
                m.get('box_quality_model_sha256') is None and m.get('score_calibration_sha256') is None and
                m.get('ground_truth_used_as_input') is False and (path.parent/'detections.jsonl').exists()):
            existing.append((path.parent,m))
    for fold,(train,calibrate,evaluate) in enumerate(plan['roles']):
        model=args.run_root/f'fold_{fold}'/'model/model.json';head,_=load_head_manifest(model)
        if head['training_sequences']!=plan['groups'][train] or head['validation_sequences']!=plan['groups'][calibrate]+plan['groups'][evaluate]:
            raise ValueError('head saw calibration/evaluation or fold role differs')
        for sequence in plan['groups'][calibrate]+plan['groups'][evaluate]:
            role='calibration' if sequence in plan['groups'][calibrate] else 'evaluation'
            expected=audit['scan_hashes'][sequence]
            source=next(r for r in audit['sequences'] if r['sequence']==sequence)
            for kind in ('oxts','calib'):
                if sha(args.dataset_root/'training'/kind/f'{sequence}.txt')!=source[kind+'_sha256']:
                    raise ValueError('pose or calibration changed')
            for profile in ('baseline','head_raw'):
                target=args.run_root/f'fold_{fold}'/'raw_cache'/sequence/profile
                if profile=='baseline':
                    matches=[path for path,m in existing if m['sequence']==sequence and
                             [{k:r[k] for k in ('frame_index','sha256')} for r in m['input_scans']]==expected]
                    if matches:target=matches[0]
                    elif (args.run_root/'baseline_cache'/sequence/'inference_summary.json').exists():target=args.run_root/'baseline_cache'/sequence
                    else:target=args.run_root/'baseline_cache'/sequence
                if not (target/'inference_summary.json').exists():
                    infer_sequence(args.dataset_root,sequence,target,args.checkpoint,head_model=model if profile=='head_raw' else None,**plan['inference_config'])
                metadata=json.loads((target/'inference_summary.json').read_text())
                if (metadata['inference_config']!=plan['inference_config'] or metadata.get('head_model_sha256')!=(sha(model) if profile=='head_raw' else None) or
                        metadata.get('box_quality_model_sha256') or metadata.get('score_calibration_sha256') or
                        metadata['ground_truth_used_as_input'] is not False or sha(target/'detections.jsonl')!=metadata['output_sha256'] or
                        [{k:r[k] for k in ('frame_index','sha256')} for r in metadata['input_scans']]!=expected):
                    raise ValueError('cached detector/model/source domain differs')
                records[f'{fold}/{sequence}/{profile}']=dict(cache=str(target.resolve()),role=role,
                    output_sha256=metadata['output_sha256'],head_model_sha256=metadata.get('head_model_sha256'))
                output.write_text(json.dumps(dict(plan_sha256=digest,input_audit_sha256=source_audit_sha,
                    inference_labels_opened=False,records=records),indent=2)+'\n')
    print('All frozen-fold inference complete; no calibration/evaluation detection labels opened.',flush=True)


if __name__=='__main__':main()
