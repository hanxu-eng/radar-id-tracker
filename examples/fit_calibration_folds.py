"""Fit monotone calibrators only on independent calibration-role sequences."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np
from radar_id_tracker.lidar_score_calibration import (load_calibration_plan,fit_sigmoid,
    ScoreCalibrationModel,calibration_targets)
from radar_id_tracker.kitti_evaluate import load_truth
from radar_id_tracker.lidar_scoped_evaluate import in_scope
from radar_id_tracker.lidar_tracker import BoxDetection


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def samples(dataset,sequence,cache,threshold):
    """Offline one-to-one TP targets in the existing front-ROI metric."""
    meta=json.loads((cache/'inference_summary.json').read_text())
    if meta['sequence']!=sequence or meta['ground_truth_used_as_input'] is not False or sha(cache/'detections.jsonl')!=meta['output_sha256']:
        raise ValueError('sample raw cache provenance differs')
    truth,ignored=load_truth(dataset,sequence);rows=[json.loads(s) for s in (cache/'detections.jsonl').read_text().splitlines()]
    if len(rows)!=len(truth):raise ValueError('sample frame count differs')
    scores={l:[] for l in ('Car','Pedestrian')};ys={l:[] for l in scores}
    for i,(row,gt,ig) in enumerate(zip(rows,truth,ignored)):
        if row['frame_index']!=i:raise ValueError('sample index differs')
        if not row['raw_scan_available']:continue
        inverse=np.linalg.inv(row['world_from_lidar']);keep=lambda b:in_scope(b,inverse)
        boxes=[b for b in (BoxDetection(**d) for d in row['detections']) if keep(b)]
        targets=calibration_targets([g for g in gt if keep(g.box)],boxes,[g for g in ig if keep(g)],threshold)
        for b,y in zip(boxes,targets):
            if y>=0:scores[b.label].append(b.confidence);ys[b.label].append(int(y))
    return {l:dict(scores=np.asarray(scores[l]),targets=np.asarray(ys[l])) for l in scores}


def reliability(scores,targets,weights=None):
    p=np.asarray(scores);y=np.asarray(targets);w=np.ones(len(p)) if weights is None else np.asarray(weights)
    if not len(p):return dict(samples=0,brier=None,ece=None,bins=[])
    w=w/w.sum();indices=np.minimum((p*10).astype(int),9);bins=[];ece=0.
    for i in range(10):
        use=indices==i;mass=float(w[use].sum())
        if not use.any():continue
        score=float(w[use]@p[use]/mass);rate=float(w[use]@y[use]/mass);ece+=mass*abs(score-rate)
        bins.append(dict(bin=i,count=int(use.sum()),weight=mass,mean_score=score,hit_rate=rate))
    return dict(samples=len(p),positives=int(y.sum()),brier=float(w@((p-y)**2)),ece=float(ece),bins=bins)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--run-root',type=Path,required=True);args=p.parse_args()
    plan,digest=load_calibration_plan(args.plan);manifest=json.loads((args.run_root/'cache_manifest.json').read_text())
    if manifest['plan_sha256']!=digest or manifest['inference_labels_opened'] is not False:raise ValueError('inference plan differs')
    split_audit=json.loads((args.run_root/'split_role_audit.json').read_text())
    if split_audit['plan_sha256']!=digest or split_audit['exact_scan_overlap_count']!=0:
        raise ValueError('calibration role scan isolation differs')
    for fold,(train,calibrate,evaluate) in enumerate(plan['roles']):
        directory=args.run_root/f'fold_{fold}'/'calibration';directory.mkdir(exist_ok=True)
        path=directory/'model.json';head_path=args.run_root/f'fold_{fold}'/'model/model.json'
        calibration=plan['groups'][calibrate];head=json.loads(head_path.read_text())
        if head['training_sequences']!=plan['groups'][train]:raise ValueError('detector role differs')
        sources={s:manifest['records'][f'{fold}/{s}/head_raw']['output_sha256'] for s in calibration}
        labels={s:sha(args.dataset_root/'training/label_02'/f'{s}.txt') for s in calibration}
        if path.exists():
            model=ScoreCalibrationModel.load(path)
            if model.data['plan_sha256']!=digest or model.data['calibration_raw_sha256']!=sources or model.data['labels_sha256']!=labels or model.data['head_model_sha256']!=sha(head_path):
                raise ValueError('frozen calibration inputs differ')
            continue
        datasets={s:samples(args.dataset_root,s,Path(manifest['records'][f'{fold}/{s}/head_raw']['cache']),plan['calibration']['target_iou']) for s in calibration}
        data=dict(format_version=1,method=plan['calibration']['method'],front_only=True,
            target_iou=plan['calibration']['target_iou'],models={},detector_training_sequences=plan['groups'][train],
            calibration_sequences=calibration,evaluation_sequences=plan['groups'][evaluate],
            evaluation_used_for_selection=False,plan_sha256=digest,calibration_raw_sha256=sources,labels_sha256=labels,
            head_model_sha256=sha(head_path),checkpoint_sha256=head['base_checkpoint_sha256'],
            inference_config=plan['inference_config'],recovery_raw_score_floor=plan['recovery_raw_score_floor'],
            calibration_parameters=plan['calibration'],identifiers_used_as_features=False,
            geometry_changed=False,sequence_equal_weight=True,score_is_validated_probability=False,
            implementation_sha256={n:sha(n) for n in ('examples/fit_calibration_folds.py','src/radar_id_tracker/lidar_score_calibration.py')})
        audit={}
        for label in ('Car','Pedestrian'):
            usable=[datasets[s][label] for s in calibration if len(datasets[s][label]['scores'])]
            if not usable:raise ValueError('calibration class has no predictions')
            x=np.concatenate([r['scores'] for r in usable]);y=np.concatenate([r['targets'] for r in usable]);w=np.concatenate([np.full(len(r['scores']),1/len(r['scores'])) for r in usable])
            fitted=fit_sigmoid(x,y,w,plan['calibration']['l2'],plan['calibration']['iterations']);data['models'][label]=fitted
            audit[label]=dict(samples=len(x),positives=int(y.sum()),represented_sequences=len(usable),raw=reliability(x,y,w))
        model=ScoreCalibrationModel(data)
        for label in audit:
            usable=[datasets[s][label] for s in calibration if len(datasets[s][label]['scores'])]
            x=np.concatenate([r['scores'] for r in usable]);y=np.concatenate([r['targets'] for r in usable]);w=np.concatenate([np.full(len(r['scores']),1/len(r['scores'])) for r in usable])
            audit[label]['calibrated']=reliability(model.predict(label,x),y,w)
        path.write_text(json.dumps(data,indent=2)+'\n');(directory/'training_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
        print(json.dumps(dict(fold=fold,models=data['models'],calibration_counts={l:{k:audit[l][k] for k in ('samples','positives','represented_sequences')} for l in audit},model_sha256=sha(path))),flush=True)


if __name__=='__main__':main()
