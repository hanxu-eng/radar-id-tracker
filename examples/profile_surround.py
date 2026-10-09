"""Isolated full-pipeline latency probe, including detection and stable-ID update."""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from radar_id_tracker.centerpoint_model import CenterPointDetector
from radar_id_tracker.centerpoint_geometry import KITTI_TO_MODEL,box_to_world
from radar_id_tracker.kitti_adapter import calibration_transforms,load_poses
from radar_id_tracker.lidar_raw import causal_cloud
from radar_id_tracker.lidar_tracker import LidarTracker,LidarTrackerConfig
from radar_id_tracker.lidar_inference_config import RawLidarConfig
from dataclasses import asdict
from radar_id_tracker.pointcloud_review import read_scan


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True,help='benchmark protocol with frozen_config')
    p.add_argument('--sequence',default='0001');p.add_argument('--start',type=int,default=100)
    p.add_argument('--frames',type=int,default=100)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--voxel-backend',choices=('numpy','numba'))
    args=p.parse_args();torch.set_num_threads(4)
    protocol=json.loads(args.config.read_text())
    inference=RawLidarConfig.from_dict(protocol.get('frozen_inference_config',{}))
    backend=args.voxel_backend or inference.voxel_backend
    poses=load_poses(args.dataset_root/'training/oxts'/f'{args.sequence}.txt')
    _,lidar_to_imu=calibration_transforms(args.dataset_root/'training/calib'/f'{args.sequence}.txt')
    transforms=[pose @ lidar_to_imu for pose in poses]
    detector=CenterPointDetector(args.checkpoint,score_threshold=inference.score_threshold,voxel_backend=backend)
    config=LidarTrackerConfig.from_dict(protocol['frozen_config'])
    tracker=LidarTracker(config);history=deque(maxlen=inference.sweeps);records=[]
    if args.start<0 or args.frames<1 or args.start>=len(poses):p.error('invalid latency frame range')
    warmup=max(5,inference.sweeps)
    for index in range(args.start-warmup,min(len(poses),args.start+args.frames)):
        if index<0:continue
        start=perf_counter();path=args.dataset_root/'training/velodyne'/args.sequence/f'{index:06d}.bin'
        while history and index-history[0][0]>inference.sweeps-1:history.popleft()
        if not path.exists():
            tracker.update(index*inference.period_s,[]);continue
        digest=hashlib.sha256(path.read_bytes()).hexdigest();points=read_scan(path)
        read_ms=(perf_counter()-start)*1000
        history.append((index,points,transforms[index]))
        before=perf_counter();cloud=causal_cloud(history,transforms[index],index,inference.period_s,inference.intensity_scale)
        compensate_ms=(perf_counter()-before)*1000
        detector.synchronize();before=perf_counter();decoded,stats=detector.predict(cloud);detector.synchronize()
        detector_ms=(perf_counter()-before)*1000
        before=perf_counter();matrix=transforms[index] @ KITTI_TO_MODEL.T
        tracks=tracker.update(index*inference.period_s,[box_to_world(box,s,c,matrix) for box,s,c in decoded])
        track_ms=(perf_counter()-before)*1000
        if index>=args.start:
            records.append(dict(frame=index,read_sha256_ms=read_ms,compensation_ms=compensate_ms,
                detector_voxel_decode_ms=detector_ms,world_box_tracker_ms=track_ms,
                total_ms=(perf_counter()-start)*1000,tracks=len(tracks),source_sha256=digest))
    if not records:raise ValueError('no available scans in latency frame range')
    result=dict(sequence=args.sequence,requested_frames=args.frames,measured_frames=len(records),voxel_backend=backend,
        inference_config={**asdict(inference),'voxel_backend':backend},
        period_budget_ms=inference.period_s*1000,gpu=torch.cuda.get_device_name(),torch_version=torch.__version__,
        stages={key:dict(median=float(np.median([r[key] for r in records])),p99=float(np.quantile([r[key] for r in records],.99)))
            for key in ('read_sha256_ms','compensation_ms','detector_voxel_decode_ms','world_box_tracker_ms','total_ms')},
        deadline_misses=sum(r['total_ms']>inference.period_s*1000 for r in records),
        no_other_repo_inference_process=True,warmup_frames=min(warmup,args.start),records=records)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))


if __name__=='__main__':main()
