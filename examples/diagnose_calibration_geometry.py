"""Offline geometry decomposition of outer-evaluation raw boxes; never repair GT."""
import argparse
import json
from pathlib import Path
import numpy as np
from radar_id_tracker.box_geometry import iou3d
from radar_id_tracker.lidar_tracker import BoxDetection,axis_difference
from radar_id_tracker.lidar_score_calibration import load_calibration_plan,ScoreCalibrationModel
from radar_id_tracker.lidar_scoped_evaluate import in_scope
from radar_id_tracker.kitti_evaluate import load_truth
from fit_calibration_folds import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--geometry',action='store_true',help='diagnose frozen matched geometry controls, without score calibration')
    p.add_argument('--parent-run',type=Path)
    args=p.parse_args();root=args.run_root
    plan,digest=load_calibration_plan(args.plan);manifest=json.loads((root/'cache_manifest.json').read_text())
    frozen=json.loads((root/'evaluation_protocol.json').read_text())
    if not args.geometry and frozen['plan_sha256']!=digest:raise ValueError('frozen evaluation plan differs')
    profiles=('continued_l1','weighted_axis_huber') if args.geometry else ('baseline','head_raw')
    if args.geometry:
        if args.parent_run is None:raise ValueError('geometry diagnosis requires parent')
        protocol=json.loads((root/'protocol.json').read_text())
        if protocol['parent_calibration_plan_sha256']!=digest or frozen['protocol_sha256']!=sha(root/'protocol.json'):
            raise ValueError('geometry diagnostic protocol differs')
        records=json.loads((args.parent_run/'geometry_diagnostics.json').read_text())['records'].copy()
    else:records=[]
    for fold,(_,_,evaluate) in enumerate(plan['roles']):
        if not args.geometry:
            ScoreCalibrationModel.load(root/f'fold_{fold}'/'calibration/model.json')
            if sha(root/f'fold_{fold}'/'calibration/model.json')!=frozen['calibration_model_sha256'][str(fold)]:raise ValueError('frozen calibration differs')
        for sequence in plan['groups'][evaluate]:
            truth,_=load_truth(args.dataset_root,sequence)
            for profile in profiles:
                cache=Path(manifest['records'][f'{fold}/{sequence}/{profile}']['cache']);stats={l:dict(gt=0,covered_025=0,covered_05=0,errors=[],failed_strict_errors=[]) for l in ('Car','Pedestrian')}
                if sha(cache/'detections.jsonl')!=manifest['records'][f'{fold}/{sequence}/{profile}']['output_sha256']:raise ValueError('raw diagnostic cache differs')
                with (cache/'detections.jsonl').open() as source:
                    for i,line in enumerate(source):
                        row=json.loads(line)
                        if not row['raw_scan_available']:continue
                        inverse=np.linalg.inv(row['world_from_lidar'])
                        boxes=[b for b in (BoxDetection(**d) for d in row['detections']) if in_scope(b,inverse)]
                        for gt in truth[i]:
                            g=gt.box
                            if not in_scope(g,inverse):continue
                            values=stats[g.label];values['gt']+=1
                            candidates=[b for b in boxes if b.label==g.label]
                            if not candidates:continue
                            overlap=[iou3d(g,b) for b in candidates];index=int(np.argmax(overlap));best=overlap[index]
                            values['covered_025']+=best>=.25;values['covered_05']+=best>=.5
                            if best<.25:continue
                            b=candidates[index]
                            error=[float(np.hypot(b.x-g.x,b.y-g.y)),abs(b.z-g.z),
                                abs(np.log(b.length/g.length)),abs(np.log(b.width/g.width)),abs(np.log(b.height/g.height)),
                                abs(axis_difference(b.yaw,g.yaw))*180/np.pi]
                            values['errors'].append(error)
                            if best<.5:values['failed_strict_errors'].append(error)
                names=['center_xy_m','center_z_m','abs_log_length_ratio','abs_log_width_ratio','abs_log_height_ratio','axis_yaw_deg']
                for label,s in stats.items():
                    item=dict(fold=fold,sequence=sequence,profile=profile,label=label,gt=s['gt'],covered_025=int(s['covered_025']),covered_05=int(s['covered_05']))
                    for group in ('errors','failed_strict_errors'):
                        x=np.asarray(s[group]);item[group]=dict(samples=len(x),
                            median={name:float(v) for name,v in zip(names,np.median(x,axis=0))} if len(x) else None,
                            p90={name:float(v) for name,v in zip(names,np.quantile(x,.9,axis=0))} if len(x) else None)
                    records.append(item)
                print(json.dumps(dict(fold=fold,sequence=sequence,profile=profile,geometry_scored=True)),flush=True)
                (root/'geometry_diagnostics.json').write_text(json.dumps(dict(plan_sha256=digest,metric='per-GT best same-class front box; duplicate box use allowed; not one-to-one recall',
                    error_selection='best-box IoU>=.25; strict-failure subset .25<=IoU<.5',online_used=False,geometry_training_changed=args.geometry,records=records),indent=2)+'\n')


if __name__=='__main__':main()
