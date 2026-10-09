"""Label-free scan duplication and geographic proximity audit of fold roles."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from radar_id_tracker.lidar_score_calibration import load_calibration_plan


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root',type=Path,required=True);p.add_argument('--input-audit',type=Path,required=True)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();plan,digest=load_calibration_plan(args.plan);audit=json.loads(args.input_audit.read_text())
    groups=plan['groups'];geo={};hashes={}
    for s in [s for group in groups for s in group]:
        path=args.dataset_root/'training/oxts'/f'{s}.txt';raw=path.read_bytes()
        record=next(r for r in audit['sequences'] if r['sequence']==s)
        if hashlib.sha256(raw).hexdigest()!=record['oxts_sha256']:raise ValueError('poses differ from source audit')
        geo[s]=np.loadtxt(path)[:,:2]*np.pi/180
        hashes[s]={r['sha256'] for r in audit['scan_hashes'][s]}
    latitude=float(np.mean(np.concatenate(list(geo.values()))[:,0]))
    chart={s:np.column_stack([a[:,1]*np.cos(latitude),a[:,0]])*6378137 for s,a in geo.items()}
    results=[];overlaps=0
    for fold,(train,calibrate,evaluate) in enumerate(plan['roles']):
        for left,right in ((train,calibrate),(train,evaluate),(calibrate,evaluate)):
            lh=set().union(*(hashes[s] for s in groups[left]));rh=set().union(*(hashes[s] for s in groups[right]))
            overlap=len(lh&rh);overlaps+=overlap
            reference=np.concatenate([chart[s] for s in groups[left]])
            for s in groups[right]:
                distances=[]
                for block in np.array_split(chart[s],max(1,len(chart[s])//100)):
                    distances.extend(np.sqrt(((block[:,None]-reference)**2).sum(2).min(1)).tolist())
                x=np.asarray(distances)
                results.append(dict(fold=fold,reference_group=left,query_group=right,query_sequence=s,frames=len(x),
                    median_nearest_reference_m=float(np.median(x)),minimum_m=float(x.min()),
                    frames_within_1m=int((x<=1).sum()),frames_within_5m=int((x<=5).sum()),frames_within_20m=int((x<=20).sum()),
                    exact_scan_overlap_count=overlap))
    data=dict(plan_sha256=digest,input_audit_sha256=hashlib.sha256(args.input_audit.read_bytes()).hexdigest(),
        ground_truth_labels_opened=False,reserved_sequence_opened=False,exact_scan_overlap_count=overlaps,
        geometry='local equirectangular GPS ego-position; ignores heading/time and target identity',
        route_independence_proven=False,results=results)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(dict(exact_scan_overlap_count=overlaps,query_frames_near_20m=sum(r['frames_within_20m'] for r in results),comparisons=len(results))))
    if overlaps:raise ValueError('identical raw scans cross development roles')


if __name__=='__main__':main()
