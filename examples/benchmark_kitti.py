"""Reproduce label-free replay, independent scoring, and 3-second dropout tests."""
import argparse
import json
from dataclasses import replace
from pathlib import Path

from radar_id_tracker.kitti_adapter import load_sequence, replay
from radar_id_tracker.kitti_evaluate import evaluate
from radar_id_tracker.lidar_tracker import LidarTrackerConfig


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root',type=Path,default=Path('datasets/kitti_tracking'))
    p.add_argument('--detections-root',type=Path,help='directory containing Car/Pedestrian/Cyclist prediction files')
    p.add_argument('--output-dir',type=Path,default=Path('outputs/kitti_benchmark'))
    p.add_argument('--config',type=Path,help='JSON LidarTrackerConfig; overrides scalar defaults')
    p.add_argument('--sequences',nargs='+',default=['0000','0001','0005'])
    p.add_argument('--high-confidence',type=float)
    p.add_argument('--low-confidence',type=float)
    p.add_argument('--hide-start',type=int,default=50)
    p.add_argument('--tuning',action='store_true',help='0000-only confidence sweep, no dropout tests')
    p.add_argument('--stress-grid',action='store_true',help='extra 30-frame dropouts at fixed 30-frame intervals')
    args = p.parse_args()
    if args.config and any(v is not None for v in (args.high_confidence,args.low_confidence)):
        p.error('--config cannot be combined with scalar confidence flags')
    config=(LidarTrackerConfig.from_dict(json.loads(args.config.read_text())) if args.config else
            LidarTrackerConfig(high_confidence=args.high_confidence if args.high_confidence is not None else .98,
                               low_confidence=args.low_confidence if args.low_confidence is not None else .8))
    configs=dict(short_memory=replace(config,retention_seconds=1.5),stable=replace(config,retention_seconds=4.))
    if args.config and args.tuning:p.error('--config and --tuning cannot be combined')
    if args.tuning:
        configs = {f'high_{h}_low_{l}':LidarTrackerConfig(high_confidence=h,low_confidence=l)
                   for h,l in ((.6,.2),(.9,.6),(.95,.6),(.98,.8))}
    rows = []
    for seq in (['0000'] if args.tuning else args.sequences):
        for scenario,hide_start in ([('natural',None)] if args.tuning else [('natural',None),('dropout_3s',args.hide_start)]):
            for name,config in configs.items():
                out = args.output_dir/seq/scenario/name
                run = replay(args.dataset_root,seq,out,config,hide_start=hide_start,
                             detections_root=args.detections_root)
                evaluation = evaluate(args.dataset_root,out)
                sensitivity = {str(gate):evaluate(args.dataset_root,out,gate,write=False)['observed_tracks']
                               for gate in (.5,.7)}
                row = dict(sequence=seq,scenario=scenario,profile=name,run=run,
                           evaluation=evaluation,iou_sensitivity=sensitivity)
                rows.append(row)
                score = evaluation['observed_tracks']; recovery = evaluation['dropout_recovery']
                print(json.dumps(dict(sequence=seq,scenario=scenario,profile=name,
                                      precision=score['precision'],recall=score['recall'],idf1=score['idf1'],
                                      switches=score['id_changes'],
                                      recovery=None if recovery is None else f"{recovery['same_id']}/{recovery['eligible']}")),flush=True)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'benchmark.json').write_text(json.dumps(rows,indent=2)+'\n')
    if not args.tuning:
        from radar_id_tracker.lidar_visualization import build_review
        build_review(args.output_dir,rows)
    if args.stress_grid and not args.tuning:
        stress=[]
        for seq in args.sequences:
            frames,_=load_sequence(args.dataset_root,seq,detections_root=args.detections_root)
            for start in range(20,len(frames)-35,30):
                for name,config in configs.items():
                    out=args.output_dir/'stress_grid'/seq/f'start_{start}'/name
                    replay(args.dataset_root,seq,out,config,hide_start=start,
                           detections_root=args.detections_root)
                    result=evaluate(args.dataset_root,out)
                    recovery=result['dropout_recovery']
                    stress.append(dict(sequence=seq,start_frame=start,profile=name,recovery=recovery))
                print(f'grid {seq} start={start}: stable {recovery["same_id"]}/{recovery["eligible"]}',flush=True)
        totals={name:dict(trials=sum(r['profile']==name for r in stress),
                          eligible=sum(r['recovery']['eligible'] for r in stress if r['profile']==name),
                          same_id=sum(r['recovery']['same_id'] for r in stress if r['profile']==name),
                          changed_id=sum(r['recovery']['changed_id'] for r in stress if r['profile']==name),
                          wrong_retained_identity=sum(r['recovery']['wrong_retained_identity'] for r in stress if r['profile']==name),
                          no_track_after_return=sum(r['recovery']['no_track_after_return'] for r in stress if r['profile']==name),
                          not_matched_after_return=sum(r['recovery']['not_matched_after_return'] for r in stress if r['profile']==name))
                for name in configs}
        paired=[]
        for seq,start in sorted({(r['sequence'],r['start_frame']) for r in stress}):
            profiles={r['profile']:{t['gt_id']:t for t in r['recovery']['targets']}
                      for r in stress if r['sequence']==seq and r['start_frame']==start}
            common=profiles['short_memory'].keys() & profiles['stable'].keys()
            paired.extend(dict(sequence=seq,start_frame=start,gt_id=g,
                               short_same_id=profiles['short_memory'][g]['same_id'],
                               stable_same_id=profiles['stable'][g]['same_id']) for g in sorted(common))
        paired_totals=dict(eligible=len(paired),unique_physical_targets=len({(r['sequence'],r['gt_id']) for r in paired}),
                           short_same_id=sum(r['short_same_id'] for r in paired),
                           stable_same_id=sum(r['stable_same_id'] for r in paired))
        (args.output_dir/'stress_grid.json').write_text(json.dumps(dict(totals=totals,paired_totals=paired_totals,
                                                                      paired_targets=paired,trials=stress),indent=2)+'\n')
        print(json.dumps(totals),flush=True)
        print(json.dumps(paired_totals),flush=True)
        from radar_id_tracker.lidar_visualization import add_stress_reviews
        add_stress_reviews(args.output_dir,rows)


if __name__ == '__main__':
    main()
