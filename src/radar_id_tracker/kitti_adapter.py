"""Label-free replay of published KITTI LiDAR detections in a fixed world frame.

This adapter reads only detection, calibration and OXTS files. Raw scans and
neural detector inference are outside this object-level replay. In the tracking
release used here no per-frame measurement timestamps are provided: frame time
is explicitly the configured nominal period (default 0.1 s), not a measured time.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from math import atan2, cos, exp, isfinite, pi, sin
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np

from .lidar_tracker import BoxDetection, LidarTracker, LidarTrackerConfig

LABEL_TYPES = {'Car': 2, 'Pedestrian': 1, 'Cyclist': 3}


@dataclass(frozen=True, slots=True)
class LidarFrame:
    frame_index: int
    timestamp_s: float
    sensor_xyz: tuple[float,float,float]
    detections: tuple[BoxDetection,...]


def _homogeneous(values: np.ndarray) -> np.ndarray:
    transform = np.eye(4)
    transform[:3,:] = values.reshape(3,4)
    return transform


def calibration_transforms(path: Path) -> tuple[np.ndarray,np.ndarray]:
    values = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        # Official tracking files omit the colon on the three extrinsics;
        # converted object/mini files use different names and include a colon.
        key,raw = line.split(maxsplit=1)
        key = key.rstrip(':')
        if key in values:
            raise ValueError(f'{path}: duplicate calibration field {key}')
        values[key] = np.fromstring(raw,sep=' ')

    def field(names: tuple[str,str],size: int) -> np.ndarray:
        present = [values[name] for name in names if name in values]
        if not present or any(v.size != size or not np.isfinite(v).all() for v in present):
            raise ValueError(f'{path}: expected {size} finite values for {names[0]} or {names[1]}')
        if len(present) == 2 and not np.array_equal(present[0],present[1]):
            raise ValueError(f'{path}: conflicting calibration aliases {names}')
        return present[0]

    rect = np.eye(4); rect[:3,:3] = field(('R0_rect','R_rect'),9).reshape(3,3)
    velo_to_cam = _homogeneous(field(('Tr_velo_to_cam','Tr_velo_cam'),12))
    imu_to_velo = _homogeneous(field(('Tr_imu_to_velo','Tr_imu_velo'),12))
    if not all(np.isfinite(m).all() for m in (rect,velo_to_cam,imu_to_velo)):
        raise ValueError('calibration must be finite')
    return np.linalg.inv(rect @ velo_to_cam @ imu_to_velo),np.linalg.inv(imu_to_velo)


def load_poses(path: Path) -> list[np.ndarray]:
    packets = np.loadtxt(path,ndmin=2)
    if packets.shape[1] != 30 or not np.isfinite(packets).all():
        raise ValueError('OXTS must contain 30 finite fields per frame')
    scale = cos(packets[0,0]*pi/180)
    radius = 6378137.
    origin = None
    poses = []
    for lat,lon,alt,roll,pitch,yaw in packets[:,:6]:
        position = np.array([scale*lon*pi*radius/180,
                             scale*radius*log_tan_lat(lat),alt])
        if origin is None:
            origin = position.copy()
        rx = np.array([[1,0,0],[0,cos(roll),-sin(roll)],[0,sin(roll),cos(roll)]])
        ry = np.array([[cos(pitch),0,sin(pitch)],[0,1,0],[-sin(pitch),0,cos(pitch)]])
        rz = np.array([[cos(yaw),-sin(yaw),0],[sin(yaw),cos(yaw),0],[0,0,1]])
        pose = np.eye(4); pose[:3,:3] = rz @ ry @ rx
        pose[:3,3] = position-origin
        poses.append(pose)
    return poses


def log_tan_lat(lat: float) -> float:
    if not -90 < lat < 90:
        raise ValueError('invalid OXTS latitude')
    return float(np.log(np.tan((90+lat)*pi/360)))


def camera_box_to_world(h: float,w: float,l: float,x: float,y: float,z: float,
                        rotation_y: float,label: str,confidence: float,
                        world_from_camera: np.ndarray) -> BoxDetection:
    # KITTI location is the bottom-center in rectified camera coordinates;
    # camera y points down, world z points up.
    center = world_from_camera @ np.array([x,y-h/2,z,1.])
    heading = world_from_camera[:3,:3] @ np.array([cos(rotation_y),0.,-sin(rotation_y)])
    return BoxDetection(*map(float,center[:3]),l,w,h,
                        atan2(heading[1],heading[0]),confidence,label)


def load_sequence(root: Path,sequence: str,period_s: float = .1,
                  labels: tuple[str,...] = ('Car','Pedestrian','Cyclist'),
                  detections_root: Path | None = None) -> tuple[list[LidarFrame],list[np.ndarray]]:
    if len(sequence) != 4 or not sequence.isdecimal():
        raise ValueError('sequence must be a four-digit ID')
    if not isfinite(period_s) or period_s <= 0:
        raise ValueError('period_s must be positive and finite')
    if not labels or not set(labels) <= LABEL_TYPES.keys():
        raise ValueError('unsupported detection labels')
    poses = load_poses(root/'training/oxts'/f'{sequence}.txt')
    camera_to_imu,velo_to_imu = calibration_transforms(root/'training/calib'/f'{sequence}.txt')
    transforms = [pose @ camera_to_imu for pose in poses]
    detections: list[list[BoxDetection]] = [[] for _ in poses]
    detections_root = detections_root if detections_root is not None else root/'detections'
    for label in labels:
        path = detections_root/label/f'{sequence}.txt'
        with path.open(newline='') as source:
            for line_number,row in enumerate(csv.reader(source),1):
                if not row:
                    continue
                a = np.array([float(v) for v in row])
                if len(a) != 15 or not np.isfinite(a).all():
                    raise ValueError(f'{path}:{line_number}: expected 15 finite detection fields')
                frame = int(a[0])
                if frame != a[0] or not 0 <= frame < len(poses):
                    raise ValueError(f'{path}:{line_number}: invalid frame index')
                if a[1] != LABEL_TYPES[label]:
                    raise ValueError(f'{path}:{line_number}: category does not match file')
                # Published detector scores are logits. Sigmoid is an interface
                # conversion, not a claim of probability calibration.
                score = 1/(1+exp(-float(np.clip(a[6],-60,60))))
                detections[frame].append(camera_box_to_world(*map(float,a[7:14]),
                                                            label,score,transforms[frame]))
    frames = []
    for i,(pose,ds) in enumerate(zip(poses,detections)):
        sensor = (pose @ velo_to_imu)[:3,3]
        frames.append(LidarFrame(i,i*period_s,tuple(map(float,sensor)),
                                 tuple(sorted(ds,key=lambda d:(d.label,d.x,d.y,d.z)))))
    return frames,transforms


def replay(root: Path,sequence: str,output_dir: Path,config: LidarTrackerConfig | None = None,
           period_s: float = .1,hide_start: int | None = None,hide_frames: int = 30,
           detections_root: Path | None = None) -> dict:
    frames,_ = load_sequence(root,sequence,period_s,detections_root=detections_root)
    if hide_start is not None and (hide_start < 2 or hide_frames < 1 or hide_start+hide_frames >= len(frames)):
        raise ValueError('hidden interval must leave warm-up and recovery frames')
    config = config or LidarTrackerConfig()
    tracker = LidarTracker(config)
    output_dir.mkdir(parents=True,exist_ok=True)
    timings = []; ids = set(); published = predicted = input_count = 0
    totals = dict(recovered=0,ambiguous_recoveries=0,capacity_drops=0,expired=0)
    with (output_dir/'detections.jsonl').open('w',encoding='utf-8') as df, \
         (output_dir/'tracks.jsonl').open('w',encoding='utf-8') as tf:
        for frame in frames:
            hidden = hide_start is not None and hide_start <= frame.frame_index < hide_start+hide_frames
            ds = [] if hidden else list(frame.detections)
            input_count += len(ds)
            start = perf_counter(); tracks = tracker.update(frame.timestamp_s,ds)
            timings.append((perf_counter()-start)*1000)
            ids.update(t.track_id for t in tracks)
            published += len(tracks); predicted += sum(t.predicted_only for t in tracks)
            for key in totals:
                totals[key] += tracker.diagnostics[key]
            df.write(json.dumps(dict(frame_index=frame.frame_index,timestamp_s=frame.timestamp_s,
                                     coordinate_frame='sequence_enu_z_up',sensor_xyz=frame.sensor_xyz,
                                     synthetic_detector_dropout=hidden,
                                     detections=[asdict(d) for d in ds]),allow_nan=False)+'\n')
            tf.write(json.dumps(dict(frame_index=frame.frame_index,timestamp_s=frame.timestamp_s,
                                     tracks=[asdict(t) for t in tracks],
                                     diagnostics=tracker.diagnostics),allow_nan=False)+'\n')
    summary = dict(dataset='KITTI tracking',sequence=sequence,frames=len(frames),
                   dataset_root=str(root.resolve()),
                   detections_root=str((detections_root if detections_root is not None else root/'detections').resolve()),
                   detection_source='published PointRCNN detections from AB3DMOT',
                   input_detection_count=input_count,
                   available_detection_count=sum(len(f.detections) for f in frames),
                   ground_truth_used_as_input=False,raw_pointcloud_detector_tested=False,
                   time_source='nominal_frame_period_not_measured_timestamps',period_s=period_s,
                   ego_motion_compensated=True,box_convention='upright center xyz, ENU z-up, length along yaw',
                   confidence_is_calibrated=False,tracker_config=asdict(config),
                   hidden_interval=(dict(start_frame=hide_start,frames=hide_frames,
                                         duration_s=hide_frames*period_s,
                                         kind='all_detection_dropout') if hide_start is not None else None),
                   unique_published_ids=len(ids),published_outputs=published,predicted_outputs=predicted,
                   median_tracker_ms=median(timings),p99_tracker_ms=float(np.quantile(timings,.99)),
                   diagnostics=totals)
    (output_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root',type=Path,required=True)
    p.add_argument('--detections-root',type=Path,help='directory containing Car/Pedestrian/Cyclist prediction files; default: DATASET_ROOT/detections')
    p.add_argument('--sequence',required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--period-s',type=float,default=.1)
    p.add_argument('--retention-seconds',type=float,default=4.)
    p.add_argument('--high-confidence',type=float,default=.98)
    p.add_argument('--low-confidence',type=float,default=.8)
    p.add_argument('--hide-start',type=int)
    p.add_argument('--hide-frames',type=int,default=30)
    args = p.parse_args(argv)
    try:
        report = replay(args.dataset_root,args.sequence,args.output_dir,
                        LidarTrackerConfig(retention_seconds=args.retention_seconds,
                                           high_confidence=args.high_confidence,
                                           low_confidence=args.low_confidence),
                        args.period_s,args.hide_start,args.hide_frames,args.detections_root)
    except (OSError,ValueError,KeyError) as error:
        p.exit(2,f'KITTI replay error: {error}\n')
    print(json.dumps(report,indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
