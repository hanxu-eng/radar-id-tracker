"""Label-free pointcloud assets; reuse only source/packed checksum-verified data."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from radar_id_tracker.lidar_score_calibration import load_calibration_plan
from radar_id_tracker.kitti_adapter import load_poses,calibration_transforms
from radar_id_tracker.pointcloud_review import read_scan,pack_scan


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--run-root',type=Path,required=True);p.add_argument('--input-audit',type=Path,required=True)
    args=p.parse_args();plan,digest=load_calibration_plan(args.plan);audit=json.loads(args.input_audit.read_text())
    assets=args.run_root/'pointclouds';assets.mkdir(parents=True,exist_ok=True)
    if (assets/'manifest.json').exists():raise ValueError('cloud manifest exists; use review without preparing again')
    previous={}
    for path in Path('outputs').glob('*/pointclouds/manifest.json'):
        meta=json.loads(path.read_text())
        if meta.get('format_version')!=1 or meta.get('coordinate_frame')!='world_axes_relative_to_current_lidar' or meta.get('record_bytes')!=8:continue
        for sequence,frames in meta['frames'].items():
            for index,data in frames.items():previous.setdefault((sequence,int(index)),[]).append((path.parent/sequence/f'{int(index):06d}.js',data))
    manifest=dict(format_version=1,dataset_root=str(args.dataset_root.resolve()),coordinate_frame='world_axes_relative_to_current_lidar',
        coordinate_unit_m=.01,record_bytes=8,layout='int16 xyz + uint16 normalized reflectance, little endian',display_only=True,
        source_url='https://s3.eu-central-1.amazonaws.com/avg-kitti/data_tracking_velodyne.zip',frames={},missing_frames=[],original_points=0,display_points=0,
        plan_sha256=digest,ground_truth_labels_opened=False,reused_assets=0,new_assets=0)
    for sequence in [s for g in plan['groups'] for s in g]:
        poses=load_poses(args.dataset_root/'training/oxts'/f'{sequence}.txt');_,extrinsic=calibration_transforms(args.dataset_root/'training/calib'/f'{sequence}.txt')
        source=next(r for r in audit['sequences'] if r['sequence']==sequence)
        for kind in ('oxts','calib'):
            if hashlib.sha256((args.dataset_root/'training'/kind/f'{sequence}.txt').read_bytes()).hexdigest()!=source[kind+'_sha256']:raise ValueError('cloud pose/calibration changed')
        scans={r['frame_index']:r['sha256'] for r in audit['scan_hashes'][sequence]}
        directory=assets/sequence;directory.mkdir(exist_ok=True);manifest['frames'][sequence]={}
        for i,pose in enumerate(poses):
            if i not in scans:
                if (args.dataset_root/'training/velodyne'/sequence/f'{i:06d}.bin').exists():raise ValueError('previously missing scan appeared')
                manifest['missing_frames'].append(dict(sequence=sequence,frame_index=i));continue
            target=directory/f'{i:06d}.js';matrix=pose@extrinsic;data=None
            for path,meta in previous.get((sequence,i),[]):
                if meta['source_sha256']!=scans[i] or not path.exists() or not np.allclose(meta['sensor_xyz'],matrix[:3,3],rtol=0,atol=1e-8):continue
                text=path.read_text('ascii');prefix=f'window.registerPointcloud({json.dumps(sequence)},{i},'
                if not text.startswith(prefix) or not text.endswith(');\n'):continue
                packed=base64.b64decode(json.loads(text[len(prefix):-3]),validate=True)
                if hashlib.sha256(packed).hexdigest()!=meta['packed_sha256'] or len(packed)!=meta['display_points']*8:raise ValueError('cached cloud packed integrity differs')
                if not target.exists():os.link(path,target)
                elif target.read_bytes()!=path.read_bytes():raise ValueError('partial cloud target differs')
                data=meta;manifest['reused_assets']+=1;break
            if data is None:
                path=args.dataset_root/'training/velodyne'/sequence/f'{i:06d}.bin'
                if hashlib.sha256(path.read_bytes()).hexdigest()!=scans[i]:raise ValueError('cloud source scan changed')
                packed,data=pack_scan(read_scan(path),matrix);data.update(source_sha256=scans[i],packed_sha256=hashlib.sha256(packed).hexdigest())
                target.write_text(f'window.registerPointcloud({json.dumps(sequence)},{i},{json.dumps(base64.b64encode(packed).decode("ascii"))});\n',encoding='ascii')
                manifest['new_assets']+=1
            manifest['frames'][sequence][str(i)]=data;manifest['original_points']+=data['original_points'];manifest['display_points']+=data['display_points']
        print(json.dumps(dict(sequence=sequence,assets=len(manifest['frames'][sequence]),reused=manifest['reused_assets'],new=manifest['new_assets'])),flush=True)
    (assets/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
