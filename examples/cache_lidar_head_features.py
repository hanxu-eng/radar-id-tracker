"""Cache frozen label-free backbone features; build separate offline head targets."""
import argparse
from collections import deque
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch
from radar_id_tracker.centerpoint_model import CenterPointDetector,decorate_torch,CHECKPOINT_SHA256
from radar_id_tracker.centerpoint_geometry import KITTI_TO_MODEL
from radar_id_tracker.kitti_adapter import calibration_transforms,load_poses
from radar_id_tracker.kitti_evaluate import load_truth
from radar_id_tracker.lidar_head_targets import make_targets
from radar_id_tracker.lidar_raw import causal_cloud
from radar_id_tracker.lidar_tracker import BoxDetection
from radar_id_tracker.lidar_scoped_evaluate import in_scope
from radar_id_tracker.pointcloud_review import read_scan
from prepare_box_quality import image_exclusion

TRAIN=['0000','0001','0002','0003','0004','0005','0006','0008']
VALIDATION=['0009','0013','0017']
INPUT=dict(sweeps=5,period_s=.1,intensity_scale=64,score_threshold=.1,voxel_backend='numba')


def to_model(box,matrix):
    p=matrix@[*list((box.x,box.y,box.z)),1.]
    heading=matrix[:3,:3]@[np.cos(box.yaw),np.sin(box.yaw),0]
    return replace(box,x=float(p[0]),y=float(p[1]),z=float(p[2]),yaw=float(np.arctan2(heading[1],heading[0])))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--split-plan',type=Path)
    p.add_argument('--split-input-audit',type=Path)
    p.add_argument('--reuse-features',type=Path,help='checksum-verified previous frozen feature cache')
    args=p.parse_args();out=args.output_dir
    if (out/'manifest.json').exists():raise ValueError('feature manifest exists; choose a new directory')
    training,validation=TRAIN,VALIDATION;extra={};audit=None
    if args.split_plan:
        from radar_id_tracker.lidar_training_protocol import load_training_plan
        plan,digest=load_training_plan(args.split_plan)
        if plan['inference_config']!=INPUT:raise ValueError('training plan input differs from frozen extractor')
        if args.split_input_audit is None:raise ValueError('label-free split audit required with split plan')
        audit=json.loads(args.split_input_audit.read_text())
        if audit['split_plan_sha256']!=digest or audit['exact_scan_overlap_count']!=0 or audit['ground_truth_labels_opened'] is not False:
            raise ValueError('split input audit differs')
        training,validation=plan['training_sequences'],plan['validation_sequences']
        extra=dict(split_plan_sha256=digest,reserved_sequences=plan['reserved_sequences'],training_parameters=plan['training_parameters'],
            split_input_audit_sha256=hashlib.sha256(args.split_input_audit.read_bytes()).hexdigest())
    elif args.split_input_audit or args.reuse_features:
        raise ValueError('split plan required with audit or reuse')
    reuse=json.loads((args.reuse_features/'manifest.json').read_text()) if args.reuse_features else None
    if reuse and (reuse['inference_config']!=INPUT or reuse['base_checkpoint_sha256']!=CHECKPOINT_SHA256):
        raise ValueError('reused feature input domain differs')
    out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    detector=CenterPointDetector(args.checkpoint,'cuda',.1,'numba')
    records=[];source=[];target_counts=dict(Car=0,Pedestrian=0)
    with torch.inference_mode():
        for sequence in training:
            audited_scans={r['frame_index']:r['sha256'] for r in audit['scan_hashes'][sequence]} if audit else {}
            if audit:
                checked=next(r for r in audit['sequences'] if r['sequence']==sequence)
                for kind in ('oxts','calib'):
                    if hashlib.sha256((args.dataset_root/'training'/kind/f'{sequence}.txt').read_bytes()).hexdigest()!=checked[kind+'_sha256']:
                        raise ValueError('calibration/pose differs from input audit')
            if reuse and sequence in reuse['training_sequences']:
                label=args.dataset_root/'training/label_02'/f'{sequence}.txt'
                if hashlib.sha256(label.read_bytes()).hexdigest()!=reuse['labels_sha256'][sequence]:raise ValueError('reused training labels differ')
                prior_scans=[r for r in reuse['source_scans'] if r['sequence']==sequence]
                expected=audit['scan_hashes'][sequence]
                if [{k:r[k] for k in ('frame_index','sha256')} for r in prior_scans]!=expected:
                    raise ValueError('reused raw scan hashes differ')
                source.extend(prior_scans);(out/sequence).mkdir(exist_ok=True)
                for record in (r for r in reuse['records'] if r['sequence']==sequence):
                    for kind in ('feature','targets'):
                        original=args.reuse_features/record[kind]
                        if hashlib.sha256(original.read_bytes()).hexdigest()!=record[kind+'_sha256']:raise ValueError('reused artifact checksum differs')
                        os.link(original,out/record[kind])
                    records.append(record)
                    with np.load(out/record['targets'],allow_pickle=False) as targets:
                        for c,label in enumerate(('Car','Pedestrian')):target_counts[label]+=int(targets['valid'][c].sum())
                print(json.dumps(dict(sequence=sequence,reused=True,cached=len(records))),flush=True)
                continue
            poses=load_poses(args.dataset_root/'training/oxts'/f'{sequence}.txt')
            _,extrinsic=calibration_transforms(args.dataset_root/'training/calib'/f'{sequence}.txt')
            transforms=[pose@extrinsic for pose in poses]
            gt,ignored=load_truth(args.dataset_root,sequence)
            camera,projection,image_ignored=image_exclusion(args.dataset_root,sequence)
            history=deque(maxlen=5)
            (out/sequence).mkdir(exist_ok=True)
            for i,matrix in enumerate(transforms):
                while history and i-history[0][0]>4:history.popleft()
                path=args.dataset_root/'training/velodyne'/sequence/f'{i:06d}.bin'
                if not path.exists():continue
                data=path.read_bytes();points=read_scan(path);history.append((i,points,matrix))
                if audit and hashlib.sha256(data).hexdigest()!=audited_scans.get(i):
                    raise ValueError('training scan differs from input audit')
                source.append(dict(sequence=sequence,frame_index=i,sha256=hashlib.sha256(data).hexdigest()))
                inverse=np.linalg.inv(matrix)
                truth=[g.box for g in gt[i] if in_scope(g.box,inverse)]
                # Deterministic temporal subsampling, retaining every pedestrian frame.
                if i%2 and not any(b.label=='Pedestrian' for b in truth):continue
                cloud=causal_cloud(history,matrix,i,.1,64)
                voxels,counts,coors=detector.voxelize(cloud)
                coordinates=torch.from_numpy(coors).cuda()
                features=decorate_torch(torch.from_numpy(voxels).cuda(),torch.from_numpy(counts).cuda(),coordinates)
                shared=detector.network.extract_features(features,coordinates)[0].to(torch.float16).cpu().numpy()
                model_from_world=KITTI_TO_MODEL@inverse
                targets=make_targets([to_model(b,model_from_world) for b in truth],
                    [to_model(b,model_from_world) for b in ignored[i]])
                # Conservative offline image-ignore projection at three heights.
                yy,xx=np.mgrid[:128,:128]
                for z in (-1.73,-.7,.5):
                    model=np.column_stack([(xx.ravel()+.5)*.8-51.2,(yy.ravel()+.5)*.8-51.2,np.full(128*128,z),np.ones(128*128)])
                    world=model@np.linalg.inv(model_from_world).T
                    cam=world@camera[i].T;uv=cam@projection.T
                    depth=uv[:,2];pixels=uv[:,:2]/np.where(depth>0,depth,1)[:,None]
                    for rect in image_ignored[i]:
                        excluded=(depth>.1)&(pixels[:,0]>=rect[0])&(pixels[:,0]<=rect[2])&(pixels[:,1]>=rect[1])&(pixels[:,1]<=rect[3])
                        targets['heatmap_mask'][:,excluded.reshape(128,128)]=False
                # Retain supervised positive centers after conservative masking.
                targets['heatmap_mask'][targets['heatmap']==1]=True
                name=f'{sequence}/{i:06d}'
                np.save(out/(name+'.npy'),shared,allow_pickle=False)
                np.savez_compressed(out/(name+'.npz'),**targets)
                records.append(dict(sequence=sequence,frame_index=i,feature=name+'.npy',targets=name+'.npz',
                    feature_sha256=hashlib.sha256((out/(name+'.npy')).read_bytes()).hexdigest(),
                    targets_sha256=hashlib.sha256((out/(name+'.npz')).read_bytes()).hexdigest()))
                for c,label in enumerate(('Car','Pedestrian')):target_counts[label]+=int(targets['valid'][c].sum())
                if i%50==0:print(json.dumps(dict(sequence=sequence,frame=i,cached=len(records))),flush=True)
    result=dict(training_sequences=training,validation_sequences=validation,base_checkpoint_sha256=CHECKPOINT_SHA256,
        inference_config=INPUT,feature_dtype='float16',feature_shape=[64,128,128],
        feature_values_label_free=True,training_frame_selection_uses_pedestrian_labels=True,
        targets_offline_only=True,targets=target_counts,records=records,source_scans=source,
        labels_sha256={s:hashlib.sha256((args.dataset_root/'training/label_02'/f'{s}.txt').read_bytes()).hexdigest() for s in training},
        source_sha256={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in
            ['examples/cache_lidar_head_features.py','src/radar_id_tracker/lidar_head_targets.py','src/radar_id_tracker/centerpoint_model.py']},**extra)
    (out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(samples=len(records),targets=target_counts)),flush=True)

if __name__=='__main__':main()
