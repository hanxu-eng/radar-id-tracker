"""Extract current-scan features for cached detections; labels are a separate step.

Training targets use only front-scoped GT and skip ambiguous overlaps/DontCare.
Feature files contain no GT; validation feature extraction never reads labels.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from radar_id_tracker.lidar_box_features import BoxFeatureIndex,FEATURE_NAMES,FEATURE_VERSION
from radar_id_tracker.lidar_tracker import BoxDetection
from radar_id_tracker.pointcloud_review import read_scan


def extract(root,cache,out):
    meta=json.loads((cache/'inference_summary.json').read_text())
    data=cache/'detections.jsonl'
    if meta['ground_truth_used_as_input'] is not False or hashlib.sha256(data.read_bytes()).hexdigest()!=meta['output_sha256']:
        raise ValueError('valid label-free raw cache required')
    sequence=meta['sequence'];rows=[];features=[];labels=[];indices=[];timings=[];source=[]
    for line in data.read_text().splitlines():
        frame=json.loads(line);i=frame['frame_index'];start=perf_counter()
        path=root/'training/velodyne'/sequence/f'{i:06d}.bin'
        boxes=[BoxDetection(**d) for d in frame['detections']]
        chosen=[j for j,b in enumerate(boxes) if b.label in ('Car','Pedestrian')]
        if chosen:
            if not path.exists():raise ValueError('detections on missing scan')
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            expected=next(p['sha256'] for p in meta['input_scans'] if p['frame_index']==i)
            if digest!=expected:raise ValueError('scan differs from detector input')
            index=BoxFeatureIndex(read_scan(path),frame['world_from_lidar'])
            f=index.extract_many([boxes[j] for j in chosen])
            features.extend(f);labels.extend(boxes[j].label for j in chosen)
            indices.extend((i,j) for j in chosen);source.append(dict(frame_index=i,sha256=digest))
        timings.append((perf_counter()-start)*1000)
        if i%50==0:print(json.dumps(dict(sequence=sequence,frame=i,features=len(features))),flush=True)
    out.mkdir(parents=True,exist_ok=True)
    target=out/'features.npz'
    np.savez_compressed(target,x=np.asarray(features).reshape(-1,len(FEATURE_NAMES)),
                        labels=np.asarray(labels,dtype='<U16'),indices=np.asarray(indices,dtype=np.int64).reshape(-1,2))
    result=dict(sequence=sequence,feature_version=FEATURE_VERSION,features=list(FEATURE_NAMES),
                cache=str(cache),cache_sha256=meta['output_sha256'],ground_truth_used_as_features=False,
                current_scan_only=True,source_scans=source,
                source_sha256=hashlib.sha256(Path('src/radar_id_tracker/lidar_box_features.py').read_bytes()).hexdigest(),
                features_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                feature_count=len(features),median_ms=float(np.median(timings)),p99_ms=float(np.quantile(timings,.99)))
    (out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def image_exclusion(root,sequence):
    # Offline-only exclusion avoids labeling projected DontCare as negative.
    from radar_id_tracker.kitti_adapter import calibration_transforms,load_poses
    poses=load_poses(root/'training/oxts'/f'{sequence}.txt')
    camera_to_imu,_=calibration_transforms(root/'training/calib'/f'{sequence}.txt')
    camera_from_world=[np.linalg.inv(p@camera_to_imu) for p in poses]
    calibration={row.split(maxsplit=1)[0].rstrip(':'):np.fromstring(row.split(maxsplit=1)[1],sep=' ')
                 for row in (root/'training/calib'/f'{sequence}.txt').read_text().splitlines() if row.strip()}
    projection=calibration['P2'].reshape(3,4)
    ignored={i:[] for i in range(len(poses))}
    for line in (root/'training/label_02'/f'{sequence}.txt').read_text().splitlines():
        r=line.split()
        if r[2]=='DontCare' or (r[2] in ('Car','Pedestrian') and int(r[1])<0):
            ignored[int(r[0])].append(np.array(list(map(float,r[6:10]))))
    return camera_from_world,projection,ignored


def project_box(box,camera_from_world,projection):
    from radar_id_tracker.box_geometry import corners_xy
    xy=corners_xy(box)
    corners=np.vstack([np.column_stack([xy,np.full(4,box.z+sign*box.height/2),np.ones(4)]) for sign in (-1,1)])
    camera=corners@camera_from_world.T
    if (camera[:,2]<=.1).any():return None
    pixels=camera@projection.T;pixels=pixels[:,:2]/pixels[:,2,None]
    return np.r_[pixels.min(0),pixels.max(0)]


def training_targets(root,feature_dir):
    from radar_id_tracker.kitti_evaluate import load_truth
    from radar_id_tracker.box_geometry import iou3d
    from radar_id_tracker.lidar_scoped_evaluate import in_scope
    meta=json.loads((feature_dir/'manifest.json').read_text());sequence=meta['sequence']
    truth,ignored=load_truth(root,sequence)
    camera_from_world,projection,excluded=image_exclusion(root,sequence)
    frames=[json.loads(l) for l in (Path(meta['cache'])/'detections.jsonl').read_text().splitlines()]
    arrays=np.load(feature_dir/'features.npz',allow_pickle=False)
    y=np.full(len(arrays['x']),-1,dtype=np.int8);counts={}
    for row,(i,j) in enumerate(arrays['indices']):
        box=BoxDetection(**frames[i]['detections'][j]);inverse=np.linalg.inv(frames[i]['world_from_lidar'])
        if not in_scope(box,inverse):continue
        gt=[g.box for g in truth[i] if in_scope(g.box,inverse) and g.box.label==box.label]
        value=max((iou3d(box,b) for b in gt),default=0.)
        if value>=.25:y[row]=1
        elif value>=.1 or any(b.label==box.label and iou3d(box,b)>=.1 for b in ignored[i]):continue
        else:
            rect=project_box(box,camera_from_world[i],projection)
            if rect is None:continue
            area=max(0.,rect[2]-rect[0])*max(0.,rect[3]-rect[1])
            ambiguous=False
            for other in excluded[i]:
                intersection=np.maximum(0,np.minimum(rect[2:],other[2:])-np.maximum(rect[:2],other[:2])).prod()
                if area and intersection/area>=.5:ambiguous=True;break
            if not ambiguous:y[row]=0
    np.savez_compressed(feature_dir/'targets.npz',y=y)
    for label in ('Car','Pedestrian'):
        mask=arrays['labels']==label
        counts[label]={str(v):int(((y==v)&mask).sum()) for v in (-1,0,1)}
    result=dict(sequence=sequence,targets=counts,positive_iou=.25,ambiguous_iou=.1,
                front_only=True,dontcare_negative_exclusion=True,
                targets_sha256=hashlib.sha256((feature_dir/'targets.npz').read_bytes()).hexdigest(),
                label_sha256=hashlib.sha256((root/'training/label_02'/f'{sequence}.txt').read_bytes()).hexdigest())
    (feature_dir/'target_manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root',type=Path,required=True)
    p.add_argument('--cache-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--training-targets',action='store_true')
    args=p.parse_args()
    if not (args.output_dir/'manifest.json').exists():extract(args.dataset_root,args.cache_dir,args.output_dir)
    else:
        meta=json.loads((args.output_dir/'manifest.json').read_text())
        if Path(meta['cache']).resolve()!=args.cache_dir.resolve():raise ValueError('feature cache differs')
    if args.training_targets:training_targets(args.dataset_root,args.output_dir)


if __name__=='__main__':main()
