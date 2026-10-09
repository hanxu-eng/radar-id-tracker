"""Offline front-ROI per-GT raw/high-score coverage, not one-to-one recall."""
import argparse
from dataclasses import fields
import json
from pathlib import Path
import numpy as np
from radar_id_tracker.lidar_tracker import BoxDetection,LidarTrackerConfig
from radar_id_tracker.lidar_scoped_evaluate import in_scope
from radar_id_tracker.lidar_proposal_diagnostics import best_overlap


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root',type=Path,required=True)
    args=p.parse_args();root=args.run_root
    rows=json.loads((root/'benchmark.json').read_text())+json.loads((root/'previous_head_results.json').read_text())
    names={f.name for f in fields(BoxDetection)};results=[]
    for row in rows:
        if row['role']!='validation' or row['scenario']!='natural':continue
        run=Path(row['run_dir']);config=LidarTrackerConfig.from_dict(row['run']['tracker_config'])
        data=[json.loads(x) for x in (run/'detections.jsonl').read_text().splitlines()]
        evaluation=[json.loads(x) for x in (run/'evaluation_frames.jsonl').read_text().splitlines()]
        counts={label:dict(gt=0,raw_covered_025=0,raw_covered_05=0,birth_score_covered_025=0,observed=0) for label in ('Car','Pedestrian')}
        if len(data)!=len(evaluation):raise ValueError('frame counts differ')
        for frame,detail in zip(data,evaluation):
            if frame['frame_index']!=detail['frame_index']:raise ValueError('frame indices differ')
            if not detail['scored']:continue
            inverse=np.linalg.inv(frame['world_from_lidar'])
            proposals=[BoxDetection(**b) for b in frame['detections']]
            proposals=[b for b in proposals if in_scope(b,inverse)]
            for target in detail['truth']:
                box=BoxDetection(**{k:v for k,v in target.items() if k in names})
                raw,_=best_overlap(box,proposals)
                threshold=config.score_policy(box.label).birth_confidence
                birth,_=best_overlap(box,[b for b in proposals if b.confidence>=threshold])
                count=counts[box.label];count['gt']+=1
                count['raw_covered_025']+=int(raw>=.25);count['raw_covered_05']+=int(raw>=.5)
                count['birth_score_covered_025']+=int(birth>=.25)
                count['observed']+=int(target['gt_id'] in detail['observed_tracks']['matches'])
        results.append(dict(sequence=row['sequence'],profile=row['profile'],classes=counts))
        print(json.dumps(results[-1]),flush=True)
    if len(results)!=9:raise ValueError('three validation sequences and three profiles required')
    result=dict(metric='per-GT best in-scope same-class proposal; duplicates allowed, not one-to-one recall',
                ground_truth_used_only_offline=True,rows=results)
    (root/'proposal_coverage.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
