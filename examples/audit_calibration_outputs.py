"""Exact outside-scope parity and score-only geometry preservation audit."""
import argparse
import json
from pathlib import Path
import numpy as np
from radar_id_tracker.lidar_score_calibration import load_calibration_plan
from radar_id_tracker.lidar_tracker import BoxDetection
from radar_id_tracker.lidar_scoped_evaluate import in_scope
from fit_calibration_folds import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--run-root',type=Path,required=True)
    args=p.parse_args();plan,digest=load_calibration_plan(args.plan);root=args.run_root
    manifest=json.loads((root/'cache_manifest.json').read_text());results=[]
    for fold,(_,_,evaluate) in enumerate(plan['roles']):
        for sequence in plan['groups'][evaluate]:
            sources=[Path(manifest['records'][f'{fold}/{sequence}/{name}']['cache']) for name in ('baseline','head_raw')]
            sources.append(root/f'fold_{fold}'/'calibrated_cache'/sequence)
            for source in sources:
                meta=json.loads((source/'inference_summary.json').read_text())
                if sha(source/'detections.jsonl')!=meta['output_sha256']:raise ValueError('audited raw cache checksum differs')
            frames=0;outside=0;score_only=0
            with (sources[0]/'detections.jsonl').open() as left,(sources[1]/'detections.jsonl').open() as right,(sources[2]/'detections.jsonl').open() as calibrated:
                from itertools import zip_longest
                for a,b,c in zip_longest(left,right,calibrated):
                    if any(row is None for row in (a,b,c)):raise ValueError('parity frame counts differ')
                    rows=[json.loads(row) for row in (a,b,c)]
                    if any(r['frame_index']!=frames for r in rows):raise ValueError('parity frame index differs')
                    inverse=np.linalg.inv(rows[0]['world_from_lidar'])
                    select=lambda row:sorted(json.dumps(d,sort_keys=True) for d in row['detections'] if not in_scope(BoxDetection(**d),inverse))
                    if select(rows[0])!=select(rows[1]) or select(rows[1])!=select(rows[2]):raise ValueError('outside-scope detections changed')
                    outside+=len(select(rows[0]))
                    if len(rows[1]['detections'])!=len(rows[2]['detections']):raise ValueError('calibration added/removed proposals')
                    for raw,changed in zip(rows[1]['detections'],rows[2]['detections']):
                        if {k:v for k,v in raw.items() if k!='confidence'}!={k:v for k,v in changed.items() if k!='confidence'}:
                            raise ValueError('calibration altered box geometry/class')
                        score_only+=1
                    frames+=1
            results.append(dict(fold=fold,sequence=sequence,frames=frames,outside_scope_exact_instances=outside,score_only_instances=score_only))
    result=dict(plan_sha256=digest,source_labels_opened=False,outside_scope_exact_instances=sum(r['outside_scope_exact_instances'] for r in results),
        score_only_instances=sum(r['score_only_instances'] for r in results),rear_accuracy_proven=False,identical_integer_track_ids_required=False,records=results)
    (root/'outside_scope_parity.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='records'}))


if __name__=='__main__':main()
