"""Label-free GPS route proximity audit; proximity is not proof of object leakage."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from radar_id_tracker.lidar_training_protocol import load_training_plan


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root',type=Path,required=True)
    p.add_argument('--split-plan',type=Path,required=True)
    p.add_argument('--run-root',type=Path,required=True)
    args=p.parse_args();plan,digest=load_training_plan(args.split_plan)
    audit=json.loads((args.run_root/'split_input_audit.json').read_text())
    if audit['split_plan_sha256']!=digest:raise ValueError('split source audit differs')
    coords={}
    for row in audit['sequences']:
        path=args.dataset_root/'training/oxts'/f'{row["sequence"]}.txt'
        raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=row['oxts_sha256']:raise ValueError('pose source differs')
        values=np.loadtxt(path)[:,:2]
        if not np.isfinite(values).all():raise ValueError('non-finite GPS coordinates')
        coords[row['sequence']]=values
    # A common equirectangular coordinate chart suffices for local meter bands.
    origin=coords[plan['training_sequences'][0]][0];scale=np.cos(np.deg2rad(origin[0]))
    project=lambda x:np.deg2rad(x-origin)*6378137*np.array([1.,scale])
    training=np.concatenate([project(coords[s]) for s in plan['training_sequences']])
    results=[]
    for sequence in plan['validation_sequences']:
        points=project(coords[sequence]);nearest=[]
        for start in range(0,len(points),128):
            distances=((points[start:start+128,None]-training[None])**2).sum(axis=2)
            nearest.extend(np.sqrt(distances.min(axis=1)).tolist())
        values=np.asarray(nearest)
        results.append(dict(sequence=sequence,frames=len(values),nearest_training_ego_pose_median_m=float(np.median(values)),
            within_m={str(band):int((values<=band).sum()) for band in (1,5,20)}))
    result=dict(split_plan_sha256=digest,ground_truth_labels_opened=False,
        metric='nearest training ego GPS position; ignores heading and time, does not prove shared target or scene',
        route_independence_verified=False,validation_sequences=results)
    (args.run_root/'split_geography.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
