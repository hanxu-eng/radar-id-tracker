"""Common front-ROI diagnostics when comparing KITTI and surround detectors.

Rear predictions have no KITTI tracking ground truth and are not scored as FP.
This is not the KITTI benchmark: no official difficulty/2D DontCare evaluation.
"""
from dataclasses import asdict
import json
from math import atan2,hypot,radians
from pathlib import Path
import numpy as np
from .kitti_adapter import calibration_transforms,load_poses
from .kitti_evaluate import load_truth,BoxScore,dropout_recovery
from .lidar_tracker import BoxDetection,BoxTrack,LidarTrackerConfig

SCOPE=dict(labels=['Car','Pedestrian'],range_m=50.,half_angle_degrees=40.,
           coordinates='current KITTI LiDAR x-forward/y-left',
           gate='box center planar distance <= 50m and |atan2(y,x)| <= 40deg',
           rear_scored=False,cyclist_scored=False,
           note='Bicycle/Motorcycle are not mapped to KITTI rider+bike Cyclist')


def fixed_recovery(reference_dir,candidate_dir):
    """Freeze the baseline's eligible GT targets; candidate misses stay failures."""
    reference=json.loads((reference_dir/'evaluation.json').read_text())['dropout_recovery']
    if reference is None:return None
    summary=json.loads((candidate_dir/'summary.json').read_text())
    interval=summary['hidden_interval'];start=interval['start_frame'];end=start+interval['frames']
    rows=[json.loads(line) for line in (candidate_dir/'evaluation_frames.jsonl').read_text().splitlines()]
    window=max(1,int(round(.5/summary['period_s'])))
    before={};after={}
    for row in rows[max(0,start-window):start]:before.update(row['observed_tracks']['matches'])
    for row in rows[end:end+window]:
        for gid,pid in row['observed_tracks']['matches'].items():after.setdefault(gid,pid)
    owners={pid:gid for gid,pid in before.items()};records=[]
    for target in reference['targets']:
        gid=target['gt_id'];previous=before.get(gid);returned=after.get(gid)
        records.append(dict(gt_id=gid,baseline_same_id=target['same_id'],before_id=previous,after_id=returned,
            candidate_same_id=previous is not None and returned==previous,
            candidate_no_pretrack=previous is None,
            wrong_retained_identity=returned in owners and owners[returned]!=gid))
    return dict(eligibility='fixed baseline pretrack + post recovery-input GT match, both in common front ROI',
        kind='3s all-detection dropout, not natural occlusion',eligible=len(records),
        baseline_same_id=sum(r['baseline_same_id'] for r in records),
        candidate_same_id=sum(r['candidate_same_id'] for r in records),
        candidate_no_pretrack=sum(r['candidate_no_pretrack'] for r in records),
        candidate_no_return=sum(r['after_id'] is None for r in records),
        candidate_wrong_retained_identity=sum(r['wrong_retained_identity'] for r in records),targets=records)


def in_scope(box,lidar_from_world):
    p=lidar_from_world @ [box.x,box.y,box.z,1.]
    return (box.label in SCOPE['labels'] and hypot(p[0],p[1])<=SCOPE['range_m'] and
            abs(atan2(p[1],p[0]))<=radians(SCOPE['half_angle_degrees']))


def evaluate_scoped(root: Path,run_dir: Path):
    summary=json.loads((run_dir/'summary.json').read_text())
    if summary.get('ground_truth_used_as_input') is not False:raise ValueError('label-free replay required')
    sequence=summary['sequence']
    truth,ignored=load_truth(root,sequence)
    poses=load_poses(root/'training/oxts'/f'{sequence}.txt')
    _,lidar_to_imu=calibration_transforms(root/'training/calib'/f'{sequence}.txt')
    ds=[json.loads(line) for line in (run_dir/'detections.jsonl').read_text().splitlines()]
    ts=[json.loads(line) for line in (run_dir/'tracks.jsonl').read_text().splitlines()]
    if len(ds)!=len(ts) or len(ds)!=len(truth):raise ValueError('frame counts differ')
    scores=dict(observed_tracks=BoxScore(),published_tracks=BoxScore(),
                all_input_detections=BoxScore(False),birth_eligible_detections=BoxScore(False),
                recovery_eligible_detections=BoxScore(False))
    config=LidarTrackerConfig.from_dict(summary['tracker_config'])
    classes={label:BoxScore() for label in SCOPE['labels']}
    details=[];excluded_frames=[]
    for index,(d,t,gt,ig,pose) in enumerate(zip(ds,ts,truth,ignored,poses)):
        expected=index*summary['period_s']
        if any(r['frame_index']!=index or abs(r['timestamp_s']-expected)>1e-8 for r in (d,t)):
            raise ValueError('frame index/time mismatch')
        inverse=np.linalg.inv(pose @ lidar_to_imu)
        keep=lambda b:in_scope(b,inverse)
        gt=[g for g in gt if keep(g.box)];ig=[g for g in ig if keep(g)]
        detections=[b for b in (BoxDetection(**p) for p in d['detections']) if keep(b)]
        tracks=[b for b in (BoxTrack(**p) for p in t['tracks']) if keep(b)]
        if len({p.track_id for p in tracks})!=len(tracks):raise ValueError('duplicate ID')
        observed=[p for p in tracks if not p.predicted_only]
        proposals=dict(observed_tracks=observed,published_tracks=tracks,all_input_detections=detections,
            birth_eligible_detections=[p for p in detections if p.confidence>=config.score_policy(p.label).birth_confidence],
            recovery_eligible_detections=[p for p in detections if p.confidence>=config.score_policy(p.label).recovery_confidence])
        # Same absent raw scans are excluded from both detector comparisons.
        available=(root/'training/velodyne'/sequence/f'{index:06d}.bin').is_file()
        row=dict(frame_index=index,timestamp_s=expected,scored=available,
                 truth=[dict(gt_id=g.object_id,**asdict(g.box)) for g in gt])
        if not available:excluded_frames.append(index)
        for name,score in scores.items():
            row[name]=score.update(gt,proposals[name],ig,.25) if available else dict(matches={},matched_gt_ids=[],id_change_gt_ids=[],tp=0,fp=0,fn=0)
        if available:
            for label,score in classes.items():
                score.update([g for g in gt if g.box.label==label],[p for p in observed if p.label==label],
                             [b for b in ig if b.label==label],.25)
        details.append(row)
    result=dict(sequence=sequence,frames=len(ds),scored_frames=len(ds)-len(excluded_frames),
                excluded_missing_scan_frames=excluded_frames,scope=SCOPE,iou3d_threshold=.25,
                official_kitti_score=False,dontcare_2d_regions_handled=False,
                **{name:score.result() for name,score in scores.items()},
                per_class_observed={name:score.result() for name,score in classes.items()},
                dropout_recovery=dropout_recovery(details,summary['hidden_interval'],summary['period_s']))
    (run_dir/'evaluation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    (run_dir/'evaluation_frames.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in details),encoding='utf-8')
    return result
