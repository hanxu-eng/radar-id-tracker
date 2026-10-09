"""Prepare original KITTI scans for local, frame-synchronous review.

Coordinates are rotated into world axes relative to the LiDAR origin. Only
display coordinates are quantized to 1 cm; original float32 scans stay intact.
No points or labels from this module enter tracking or its metrics.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path

import numpy as np

from .kitti_adapter import calibration_transforms, load_poses

PACKED_DTYPE=np.dtype([('xyz','<i2',(3,)),('reflectance','<u2')])


def read_scan(path: Path) -> np.ndarray:
    if not path.stat().st_size or path.stat().st_size%16:
        raise ValueError(f'{path}: expected float32 x,y,z,reflectance records')
    return np.fromfile(path,dtype='<f4').reshape(-1,4)


def pack_scan(points: np.ndarray,world_from_lidar: np.ndarray) -> tuple[bytes,dict]:
    if points.ndim!=2 or points.shape[1]!=4:
        raise ValueError('scan must have four columns')
    matrix=np.asarray(world_from_lidar,dtype=float)
    if (matrix.shape!=(4,4) or not np.isfinite(matrix).all() or
            not np.allclose(matrix[3],[0,0,0,1]) or
            not np.allclose(matrix[:3,:3].T@matrix[:3,:3],np.eye(3),atol=1e-5)):
        raise ValueError('expected a finite rigid LiDAR-to-world transform')
    valid=np.isfinite(points).all(axis=1)
    finite=points[valid]
    xyz=finite[:,:3].astype(float) @ matrix[:3,:3].T
    units=np.rint(xyz*100.)
    in_range=(np.abs(units)<=32767).all(axis=1)
    packed=np.empty(int(in_range.sum()),dtype=PACKED_DTYPE)
    packed['xyz']=units[in_range].astype('<i2')
    intensity=finite[in_range,3]
    packed['reflectance']=np.rint(np.clip(intensity,0,1)*65535).astype('<u2')
    return packed.tobytes(),dict(original_points=len(points),display_points=len(packed),
                                invalid_points=int((~valid).sum()),
                                outside_coordinate_range=int((~in_range).sum()),
                                clipped_reflectance_points=int(((intensity<0)|(intensity>1)).sum()),
                                sensor_xyz=matrix[:3,3].tolist(),
                                sensor_heading_rad=float(np.arctan2(matrix[1,0],matrix[0,0])))


def prepare_clouds(dataset_root: Path,output_dir: Path,sequences=('0000','0001','0005')) -> dict:
    assets=output_dir/'pointclouds';assets.mkdir(parents=True,exist_ok=True)
    manifest=dict(format_version=1,dataset_root=str(dataset_root.resolve()),
                  coordinate_frame='world_axes_relative_to_current_lidar',
                  coordinate_unit_m=.01,record_bytes=8,
                  layout='int16 xyz + uint16 normalized reflectance, little endian',
                  display_only=True,source_url='https://s3.eu-central-1.amazonaws.com/avg-kitti/data_tracking_velodyne.zip',
                  frames={},missing_frames=[],original_points=0,display_points=0)
    for sequence in sequences:
        if len(sequence)!=4 or not sequence.isdecimal(): raise ValueError('invalid sequence ID')
        poses=load_poses(dataset_root/'training/oxts'/f'{sequence}.txt')
        _,lidar_to_imu=calibration_transforms(dataset_root/'training/calib'/f'{sequence}.txt')
        manifest['frames'][sequence]={}
        directory=assets/sequence;directory.mkdir(exist_ok=True)
        for index,pose in enumerate(poses):
            source=dataset_root/'training/velodyne'/sequence/f'{index:06d}.bin'
            if not source.is_file():
                manifest['missing_frames'].append(dict(sequence=sequence,frame_index=index))
                continue
            packed,meta=pack_scan(read_scan(source),pose@lidar_to_imu)
            encoded=base64.b64encode(packed).decode('ascii')
            target=directory/f'{index:06d}.js'
            target.write_text(f'window.registerPointcloud({json.dumps(sequence)},{index},{json.dumps(encoded)});\n',encoding='ascii')
            meta['source_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
            meta['packed_sha256']=hashlib.sha256(packed).hexdigest()
            manifest['frames'][sequence][str(index)]=meta
            manifest['original_points']+=meta['original_points'];manifest['display_points']+=meta['display_points']
            if index%50==0:print(f'packed {sequence} frame {index}/{len(poses)}',flush=True)
    (assets/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root',type=Path,default=Path('datasets/kitti_tracking'))
    parser.add_argument('--output-dir',type=Path,default=Path('outputs/kitti_final'))
    parser.add_argument('--sequences',nargs='+',default=['0000','0001','0005'])
    args=parser.parse_args(argv)
    manifest=prepare_clouds(args.dataset_root,args.output_dir,args.sequences)
    from .lidar_visualization import add_stress_reviews,build_review
    cases=args.output_dir/'review_cases.json'
    rows=json.loads((cases if cases.exists() else args.output_dir/'benchmark.json').read_text())
    if (args.output_dir/'stress_grid.json').exists():add_stress_reviews(args.output_dir,rows)
    else:build_review(args.output_dir,rows)
    print(json.dumps(dict(scans=sum(len(v) for v in manifest['frames'].values()),
                          missing=manifest['missing_frames'],points=manifest['original_points'])),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
