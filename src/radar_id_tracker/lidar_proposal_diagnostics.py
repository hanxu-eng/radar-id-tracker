"""Offline per-GT loss attribution; never used by online detection or tracking."""
from collections import Counter
from dataclasses import asdict,replace
import hashlib
import json
from pathlib import Path
import numpy as np
from .box_geometry import iou3d
from .kitti_evaluate import load_truth
from .lidar_box_features import BoxFeatureIndex
from .lidar_scoped_evaluate import in_scope
from .lidar_tracker import BoxDetection,LidarTrackerConfig
from .pointcloud_review import read_scan


def best_overlap(target,boxes,same_class=True):
    """Conservative circle/height prefilter, then exact upright 3D overlap."""
    best=0.;chosen=None
    radius=np.hypot(target.length,target.width)/2
    for index,box in enumerate(boxes):
        if same_class and target.label!=box.label:continue
        if abs(target.z-box.z)>=(target.height+box.height)/2:continue
        if np.hypot(target.x-box.x,target.y-box.y)>radius+np.hypot(box.length,box.width)/2:continue
        value=iou3d(target,box)
        if value>best:best=value;chosen=index
    return best,chosen


def loss_reason(raw_iou,any_iou,continued_iou,birth_iou,observed,threshold=.25):
    if observed:return 'observed'
    if raw_iou<threshold:return 'wrong_class_geometry_available' if any_iou>=threshold else 'no_usable_raw_box'
    if continued_iou<threshold:return 'quality_below_continuation'
    if birth_iou<threshold:return 'quality_below_birth'
    return 'confirmation_association_or_filter_geometry'


def diagnostic(root,raw_cache,run,out):
    meta=json.loads((raw_cache/'inference_summary.json').read_text())
    raw_path=raw_cache/'detections.jsonl'
    digest=hashlib.sha256(raw_path.read_bytes()).hexdigest()
    if digest!=meta['output_sha256'] or meta['ground_truth_used_as_input'] is not False:raise ValueError('valid label-free raw cache required')
    if meta.get('box_quality_model_sha256'):raise ValueError('raw cache must precede quality scoring')
    summary=json.loads((run/'summary.json').read_text())
    if summary['sequence']!=meta['sequence'] or summary.get('hidden_interval'):raise ValueError('matching natural replay required')
    truth,_=load_truth(root,meta['sequence'])
    visibility={}
    for line in (root/'training/label_02'/f'{meta["sequence"]}.txt').read_text().splitlines():
        fields=line.split()
        if fields[2] in ('Car','Pedestrian') and int(fields[1])>=0:
            visibility[int(fields[0]),fields[1]]=dict(occlusion=int(fields[4]),truncation=float(fields[3]))
    raw=[json.loads(l) for l in raw_path.read_text().splitlines()]
    scored=[json.loads(l) for l in (run/'detections.jsonl').read_text().splitlines()]
    evaluations=[json.loads(l) for l in (run/'evaluation_frames.jsonl').read_text().splitlines()]
    if not len(raw)==len(scored)==len(evaluations)==len(truth):raise ValueError('frame count mismatch')
    config=LidarTrackerConfig.from_dict(summary['tracker_config'])
    details=[];counts={label:Counter() for label in ('Car','Pedestrian')};missing=[]
    for i,(frame,q,e,targets) in enumerate(zip(raw,scored,evaluations,truth)):
        if any(r['frame_index']!=i for r in (frame,q,e)):raise ValueError('frame index mismatch')
        scan=root/'training/velodyne'/meta['sequence']/f'{i:06d}.bin'
        if not scan.exists():missing.append(i);continue
        pose=np.asarray(frame['world_from_lidar']);inverse=np.linalg.inv(pose)
        targets=[g for g in targets if in_scope(g.box,inverse)]
        boxes=[BoxDetection(**b) for b in frame['detections']]
        quality=[BoxDetection(**b) for b in q['detections']]
        if len(boxes)!=len(quality):raise ValueError('quality pipeline changed proposal count')
        # The comparison must change scores only, with the same ordered geometry.
        for a,b in zip(boxes,quality):
            if {k:v for k,v in asdict(a).items() if k!='confidence'}!={k:v for k,v in asdict(b).items() if k!='confidence'}:
                raise ValueError('quality pipeline changed proposal geometry/class')
        if targets:
            source=hashlib.sha256(scan.read_bytes()).hexdigest()
            expected=next(s['sha256'] for s in meta['input_scans'] if s['frame_index']==i)
            if source!=expected:raise ValueError('source scan checksum differs')
            index=BoxFeatureIndex(read_scan(scan),pose)
        for target in targets:
            b=target.box;policy=config.score_policy(b.label)
            raw_iou,j=best_overlap(b,boxes);any_iou,other=best_overlap(b,boxes,False)
            continued,_=best_overlap(b,[p for p in quality if p.confidence>=policy.continuation_confidence])
            birth,_=best_overlap(b,[p for p in quality if p.confidence>=policy.birth_confidence])
            observed=target.object_id in e['observed_tracks']['matches']
            features=index.extract(b);distance=features[1];point_count=int(round(np.expm1(features[6])))
            same=[p for p in boxes if p.label==b.label]
            bev,_=best_overlap(b,[replace(p,z=b.z,height=b.height) for p in same])
            nearest=min((np.hypot(p.x-b.x,p.y-b.y) for p in same),default=None)
            geometry_issue=(None if raw_iou>=.25 else 'vertical_geometry' if bev>=.25 else
                'classification' if any_iou>=.25 else 'center_size_or_yaw' if nearest is not None and nearest<=2 else
                'no_nearby_same_class_center')
            row=dict(sequence=meta['sequence'],frame_index=i,gt_id=target.object_id,label=b.label,
                range_m=float(distance),points_in_gt_box=point_count,
                distance_bin='0-15m' if distance<15 else '15-30m' if distance<30 else '30-50m',
                point_bin='0-4' if point_count<5 else '5-19' if point_count<20 else '20+',
                raw_best_iou=raw_iou,any_class_best_iou=any_iou,
                raw_best_label=boxes[other].label if other is not None else None,
                raw_confidence=boxes[j].confidence if j is not None else None,
                quality_confidence=quality[j].confidence if j is not None else None,
                continuation_best_iou=continued,birth_best_iou=birth,observed=observed,
                loss=loss_reason(raw_iou,any_iou,continued,birth,observed),
                raw_best_bev_iou=bev,nearest_same_class_center_xy_m=None if nearest is None else float(nearest),
                raw_geometry_issue=geometry_issue,**visibility[i,target.object_id])
            for threshold in (.25,.5):row[f'raw_coverage_iou_{threshold}']=raw_iou>=threshold
            details.append(row);counts[b.label][row['loss']]+=1
        if i%50==0:print(json.dumps(dict(sequence=meta['sequence'],frame=i,targets=len(details))),flush=True)
    result=dict(sequence=meta['sequence'],frames=len(raw),excluded_missing_scan_frames=missing,
        raw_cache_sha256=digest,quality_model_sha256=summary.get('box_quality_model_sha256'),
        online_used=False,gt_used_only_for_offline_diagnosis=True,
        metric='per-GT best-box coverage, not one-to-one recall or official KITTI metric',
        classes={label:dict(gt=sum(c.values()),losses=dict(c)) for label,c in counts.items()},
        groups={})
    for label in counts:
        for field in ('distance_bin','point_bin','occlusion'):
            for group in sorted({r[field] for r in details if r['label']==label}):
                subset=[r for r in details if r['label']==label and r[field]==group]
                result['groups'][f'{label}/{field}/{group}']=dict(gt=len(subset),
                    raw_covered_025=sum(r['raw_coverage_iou_0.25'] for r in subset),
                    raw_covered_05=sum(r['raw_coverage_iou_0.5'] for r in subset),
                    observed=sum(r['observed'] for r in subset),losses=dict(Counter(r['loss'] for r in subset)))
    result['raw_geometry_issues']={label:dict(Counter(r['raw_geometry_issue'] for r in details
        if r['label']==label and not r['raw_coverage_iou_0.25'])) for label in counts}
    out.mkdir(parents=True,exist_ok=True)
    (out/'diagnostic.json').write_text(json.dumps(result,indent=2)+'\n')
    (out/'targets.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in details))
    print(json.dumps(result['classes']),flush=True)
    return result
