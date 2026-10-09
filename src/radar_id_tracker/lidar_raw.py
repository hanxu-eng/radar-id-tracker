"""Causal KITTI raw-scan CenterPoint inference and cached stable-ID replay.

Training annotations are never opened by this module. KITTI times are nominal.
The nuScenes pretrained detector is experimental when transferred to KITTI.
"""
from __future__ import annotations
import argparse
from collections import deque
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from .centerpoint_geometry import KITTI_TO_MODEL, transform_points, box_to_world
from .kitti_adapter import calibration_transforms,load_poses
from .lidar_tracker import BoxDetection,LidarTracker,LidarTrackerConfig
from .pointcloud_review import read_scan
from .lidar_inference_config import RawLidarConfig


def causal_cloud(history,current_transform,frame,period_s,intensity_scale):
    """Compensate only available past scans into current LiDAR, then model axes.

    The fifth channel records actual nominal frame gaps, including missing scans.
    Current scan retains close points; past scans remove the 1m close square.
    """
    if not np.isfinite([period_s,intensity_scale]).all() or min(period_s,intensity_scale)<=0:
        raise ValueError('period and intensity scale must be positive and finite')
    if any(history[i][0]>=history[i+1][0] for i in range(len(history)-1)):
        raise ValueError('history frame indices must strictly increase')
    blocks=[]
    current_inverse=np.linalg.inv(current_transform)
    for index,points,world_from_lidar in reversed(history):
        if index>frame:raise ValueError('future scan is forbidden')
        if points.ndim!=2 or points.shape[1]!=4:raise ValueError('raw scans must be Nx4')
        if index<frame:
            points=points[~((np.abs(points[:,0])<1)&(np.abs(points[:,1])<1))]
        p=transform_points(points,KITTI_TO_MODEL @ current_inverse @ world_from_lidar)
        p[:,3]*=intensity_scale
        blocks.append(np.column_stack([p,np.full(len(p),(frame-index)*period_s,np.float32)]))
    return np.concatenate(blocks).astype(np.float32) if blocks else np.empty((0,5),np.float32)


def infer_sequence(root,sequence,output_dir,checkpoint,device='cuda',sweeps=5,
                   period_s=.1,intensity_scale=255.,score_threshold=.1,voxel_backend='numba',quality_model=None,head_model=None):
    inference_config=RawLidarConfig(sweeps,period_s,intensity_scale,score_threshold,voxel_backend)
    if len(sequence)!=4 or not sequence.isdecimal():raise ValueError('invalid sequence ID')
    if head_model is not None:
        from .lidar_head_targets import load_head_manifest
        head_metadata,_=load_head_manifest(head_model)
        if quality_model is not None:raise ValueError('old quality model cannot be combined with adapted detector heads')
        if any(asdict(inference_config).get(k)!=v for k,v in head_metadata['inference_config'].items() if k!='voxel_backend'):
            raise ValueError('head model detector input domain differs')
    from .centerpoint_model import CenterPointDetector,CHECKPOINT_SHA256,CHECKPOINT_URL
    quality=None
    if quality_model is not None:
        from .lidar_box_features import BoxFeatureIndex
        from .lidar_box_quality import BoxQualityModel
        quality=BoxQualityModel.load(quality_model)
        quality.validate_detector_inputs(CHECKPOINT_SHA256,asdict(inference_config))
    poses=load_poses(root/'training/oxts'/f'{sequence}.txt')
    _,lidar_to_imu=calibration_transforms(root/'training/calib'/f'{sequence}.txt')
    transforms=[pose @ lidar_to_imu for pose in poses]
    detector=CenterPointDetector(checkpoint,device,score_threshold,voxel_backend,head_model=head_model)
    import torch
    torch.set_num_threads(4)
    # Exclude explicit startup/warm-up from measured replay latency.
    warm=np.array([[0.,5.,-1.,128.,0.]],np.float32)
    detector.predict(warm);detector.synchronize()
    output_dir.mkdir(parents=True,exist_ok=True)
    target=output_dir/'detections.jsonl'
    if target.exists():raise ValueError(f'output exists: {target}; choose a new directory')
    history=deque(maxlen=sweeps)
    timing=[];missing=[];rear_count=0;class_counts={};input_files=[]
    with target.with_suffix('.part').open('w',encoding='utf-8') as f:
        for index,transform in enumerate(transforms):
            start=perf_counter()
            path=root/'training/velodyne'/sequence/f'{index:06d}.bin'
            available=path.exists()
            while history and (index-history[0][0])*period_s>(sweeps-1)*period_s+1e-8:
                history.popleft()
            inference_ms=None;detections=[];stats={};quality_details=[]
            if available:
                data=path.read_bytes()
                input_files.append(dict(frame_index=index,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
                points=read_scan(path)
                history.append((index,points,transform))
                cloud=causal_cloud(history,transform,index,period_s,intensity_scale)
                detector.synchronize();model_start=perf_counter()
                decoded,stats=detector.predict(cloud);detector.synchronize()
                inference_ms=(perf_counter()-model_start)*1000
                world_from_model=transform @ KITTI_TO_MODEL.T
                # Full homogeneous inverse: KITTI_TO_MODEL has zero translation.
                detections=[box_to_world(box,score,label,world_from_model) for box,score,label in decoded]
                if quality is not None:
                    detections,quality_details=quality.rescore(BoxFeatureIndex(points,transform),detections)
                rear_count+=sum(int(box[1]<0) for box,score,label in decoded)
                for detection in detections:
                    class_counts[detection.label]=class_counts.get(detection.label,0)+1
            else:missing.append(index)
            latency=(perf_counter()-start)*1000
            if available:timing.append(latency)
            f.write(json.dumps(dict(frame_index=index,timestamp_s=index*period_s,
                    coordinate_frame='sequence_enu_z_up',sensor_xyz=transform[:3,3].tolist(),
                    world_from_lidar=transform.tolist(),raw_scan_available=available,
                    synthetic_detector_dropout=False,detections=[asdict(d) for d in detections],
                    sweeps_used=[j for j,_,_ in reversed(history)] if available else [],
                    end_to_end_ms=latency,inference_ms=inference_ms,voxelization=stats,
                    box_quality_scores=quality_details),allow_nan=False)+'\n')
            if index%50==0 or index==len(transforms)-1:
                print(json.dumps(dict(sequence=sequence,frame=index,total=len(transforms),
                                     detections=len(detections),latency_ms=round(latency,2))),flush=True)
    target.with_suffix('.part').replace(target)
    summary=dict(sequence=sequence,frames=len(poses),dataset_root=str(root.resolve()),
        ground_truth_used_as_input=False,raw_pointcloud_detector_tested=True,
        detector='CenterPoint pillar02 SECOND SECFPN circle-NMS, dense PyTorch port',
        checkpoint_url=CHECKPOINT_URL,checkpoint_sha256=CHECKPOINT_SHA256,
        checkpoint_tensors_loaded_strictly=detector.weight_tensors,
        implementation_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ('centerpoint_model.py','centerpoint_geometry.py','centerpoint_voxel_numba.py','lidar_raw.py')},
        voxel_backend=voxel_backend,
        inference_config=asdict(inference_config),
        detector_score_threshold=score_threshold,confidence_is_calibrated=False,
        class_counts=class_counts,rear_detection_instances=rear_count,
        rear_detection_definition='current KITTI LiDAR x < 0',missing_scan_frames=missing,
        period_s=period_s,time_source='nominal_frame_period_not_measured_timestamps',
        sweeps=sweeps,max_history_s=(sweeps-1)*period_s,causal_past_only=True,
        ego_motion_compensated=True,point_deskewed=False,intensity_scale=intensity_scale,
        input_coordinate_transform='KITTI x-forward/y-left to model x-right/y-forward: (-y,x,z)',
        domain_transfer='nuScenes pretrained model on KITTI, unvalidated on RoboSense',
        median_detector_pipeline_ms=float(np.median(timing)),p99_detector_pipeline_ms=float(np.quantile(timing,.99)),
        latency_includes=('scan read + source SHA256 + sweep compensation + CPU voxelization + GPU + decode'
                          + (' + current-scan box features + quality scoring' if quality_model else '')),
        warmup_excluded=True,device=str(detector.device),torch_version=torch.__version__,
        gpu=torch.cuda.get_device_name() if detector.device.type=='cuda' else None,
        output_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),input_scans=input_files)
    if quality is not None:
        summary.update(box_quality_model_sha256=hashlib.sha256(quality_model.read_bytes()).hexdigest(),
            box_quality_training_sequences=quality.data['training_sequences'],quality_inference_ground_truth_used=False,
            box_quality_score_is_calibrated=False,
            quality_implementation_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                for name in ('lidar_box_features.py','lidar_box_quality.py')})
    if head_model is not None:
        summary.update(head_model_sha256=hashlib.sha256(head_model.read_bytes()).hexdigest(),
            head_weights_sha256=head_metadata['weights_sha256'],head_training_sequences=head_metadata['training_sequences'],
            head_front_only=True,head_inference_ground_truth_used=False,
            head_implementation_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                for name in ('centerpoint_head_adapter.py','lidar_head_targets.py')})
    (output_dir/'inference_summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    return summary


def replay_cache(cache_dir,output_dir,config,hide_start=None,hide_frames=30):
    if cache_dir.resolve()==output_dir.resolve():raise ValueError('cache and output directories must differ')
    metadata=json.loads((cache_dir/'inference_summary.json').read_text())
    if metadata.get('ground_truth_used_as_input') is not False or metadata.get('raw_pointcloud_detector_tested') is not True:
        raise ValueError('label-free raw detector cache required')
    cache=cache_dir/'detections.jsonl'
    if hashlib.sha256(cache.read_bytes()).hexdigest()!=metadata['output_sha256']:
        raise ValueError('cached detections checksum mismatch')
    frames=[json.loads(line) for line in cache.read_text().splitlines()]
    if len(frames)!=metadata['frames']:raise ValueError('cache frame count mismatch')
    for index,frame in enumerate(frames):
        if (frame['frame_index']!=index or not np.isfinite(frame['timestamp_s']) or
            abs(frame['timestamp_s']-index*metadata['period_s'])>1e-8):
            raise ValueError('cached frame index/time mismatch')
        if frame.get('raw_scan_available') is False and frame['detections']:
            raise ValueError('absent raw scan must not contain detections')
        for d in frame['detections']:BoxDetection(**d)
    if hide_start is not None and not (2<=hide_start<hide_start+hide_frames<len(frames)):
        raise ValueError('dropout must leave warm-up and recovery frames')
    output_dir.mkdir(parents=True,exist_ok=True)
    tracker=LidarTracker(config);timings=[]
    with (output_dir/'detections.jsonl').open('w',encoding='utf-8') as df, (output_dir/'tracks.jsonl').open('w',encoding='utf-8') as tf:
        for index,frame in enumerate(frames):
            if frame['frame_index']!=index or abs(frame['timestamp_s']-index*metadata['period_s'])>1e-8:
                raise ValueError('cached frame index/time mismatch')
            row=dict(frame)
            hidden=hide_start is not None and hide_start<=index<hide_start+hide_frames
            if hidden:row['detections']=[]
            row['synthetic_detector_dropout']=hidden
            ds=[BoxDetection(**d) for d in row['detections']]
            start=perf_counter();tracks=tracker.update(row['timestamp_s'],ds)
            timings.append((perf_counter()-start)*1000)
            df.write(json.dumps(row,allow_nan=False)+'\n')
            tf.write(json.dumps(dict(frame_index=index,timestamp_s=row['timestamp_s'],
                tracks=[asdict(t) for t in tracks],tentative=tracker.tentative_outputs,
                diagnostics=tracker.diagnostics,class_diagnostics=tracker.class_diagnostics),allow_nan=False)+'\n')
    summary={k:v for k,v in metadata.items() if k!='input_scans'}
    summary.update(tracker_config=asdict(config),box_convention='upright center xyz, ENU z-up, length along yaw',
        median_tracker_ms=float(np.median(timings)),p99_tracker_ms=float(np.quantile(timings,.99)),
        hidden_interval=None if hide_start is None else dict(start_frame=hide_start,frames=hide_frames,
            duration_s=hide_frames*metadata['period_s'],kind='all_detection_dropout'),
        inference_cache=str(cache_dir.resolve()))
    (output_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    return summary


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root',type=Path,required=True)
    p.add_argument('--sequence',required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    p.add_argument('--inference-config',type=Path,help='explicit raw-scan JSON; cannot be mixed with inference flags')
    p.add_argument('--sweeps',type=int)
    p.add_argument('--voxel-backend',choices=('numpy','numba'))
    p.add_argument('--intensity-scale',type=float)
    p.add_argument('--period-s',type=float)
    p.add_argument('--score-threshold',type=float)
    p.add_argument('--quality-model',type=Path,help='optional frozen JSON box quality model; explicit input config must match training domain')
    p.add_argument('--head-model',type=Path,help='optional frozen front-head manifest; incompatible with old quality models')
    p.add_argument('--track-output-dir',type=Path,help='optionally replay inference into a separate tracking directory')
    p.add_argument('--tracker-config',type=Path,help='explicit tracker JSON; required with --track-output-dir')
    args=p.parse_args(argv)
    if args.head_model and args.quality_model:
        p.error('--head-model cannot be combined with an old --quality-model')
    if bool(args.track_output_dir)!=bool(args.tracker_config):
        p.error('--track-output-dir and --tracker-config must be provided together')
    overrides={key:getattr(args,key) for key in RawLidarConfig.__dataclass_fields__ if getattr(args,key) is not None}
    if args.inference_config and overrides:
        p.error('--inference-config cannot be mixed with inference flags')
    try:
        inference=RawLidarConfig.from_dict(json.loads(args.inference_config.read_text()) if args.inference_config else overrides)
        config=LidarTrackerConfig.from_dict(json.loads(args.tracker_config.read_text())) if args.tracker_config else None
        if args.track_output_dir and args.track_output_dir.resolve()==args.output_dir.resolve():
            raise ValueError('cache and tracking output directories must differ')
        summary=infer_sequence(args.dataset_root,args.sequence,args.output_dir,args.checkpoint,
            device=args.device,quality_model=args.quality_model,head_model=args.head_model,**asdict(inference))
        if args.track_output_dir:
            summary['tracking_run']=replay_cache(args.output_dir,args.track_output_dir,config)
    except (OSError,ValueError,ImportError,KeyError,TypeError) as error:p.exit(2,f'raw LiDAR inference error: {error}\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='input_scans'},indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
