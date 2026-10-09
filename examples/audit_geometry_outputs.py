"""Exact Car and outside-scope proposal preservation after pedestrian tuning."""
import argparse
from itertools import zip_longest
import json
from pathlib import Path
import numpy as np
from radar_id_tracker.lidar_tracker import BoxDetection
from radar_id_tracker.lidar_scoped_evaluate import in_scope
from fit_calibration_folds import sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-root',type=Path,required=True)
    args=p.parse_args();root=args.run_root
    protocol=json.loads((root/'protocol.json').read_text());plan=protocol['parent_plan']
    manifest=json.loads((root/'cache_manifest.json').read_text());records=[]
    for fold,(_,_,evaluate) in enumerate(plan['roles']):
        for sequence in plan['groups'][evaluate]:
            sources=[Path(manifest['records'][f'{fold}/{sequence}/{name}']['cache']) for name in ('head_raw',*protocol['variants'])]
            for cache in sources:
                if sha(cache/'detections.jsonl')!=json.loads((cache/'inference_summary.json').read_text())['output_sha256']:raise ValueError('parity cache checksum differs')
            frames=0;outside=0;car=0
            with (sources[0]/'detections.jsonl').open() as a,(sources[1]/'detections.jsonl').open() as b,(sources[2]/'detections.jsonl').open() as c:
                for lines in zip_longest(a,b,c):
                    if any(line is None for line in lines):raise ValueError('geometry parity frame counts differ')
                    rows=[json.loads(line) for line in lines]
                    if any(r['frame_index']!=frames for r in rows):raise ValueError('geometry frame index differs')
                    inverse=np.linalg.inv(rows[0]['world_from_lidar'])
                    select=lambda row,predicate:sorted(json.dumps(d,sort_keys=True) for d in row['detections'] if predicate(d))
                    for predicate,kind in ((lambda d:d['label']=='Car','car'),(lambda d:not in_scope(BoxDetection(**d),inverse),'outside')):
                        values=[select(row,predicate) for row in rows]
                        if values[0]!=values[1] or values[1]!=values[2]:raise ValueError(kind+' raw detections changed')
                        if kind=='car':car+=len(values[0])
                        else:outside+=len(values[0])
                    frames+=1
            records.append(dict(sequence=sequence,fold=fold,frames=frames,car_exact_instances=car,outside_scope_exact_instances=outside))
    data=dict(protocol_sha256=sha(root/'protocol.json'),records=records,
        car_exact_instances=sum(r['car_exact_instances'] for r in records),outside_scope_exact_instances=sum(r['outside_scope_exact_instances'] for r in records),
        integer_track_ids_required_identical=False,rear_accuracy_proven=False)
    (root/'outside_scope_parity.json').write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps({k:v for k,v in data.items() if k!='records'}))


if __name__=='__main__':main()
