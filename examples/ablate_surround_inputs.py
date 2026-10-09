"""Compare raw-input adaptations across explicit development sequences only."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from radar_id_tracker.lidar_raw import infer_sequence, replay_cache
from radar_id_tracker.kitti_adapter import load_poses, calibration_transforms
from radar_id_tracker.kitti_evaluate import load_truth, BoxScore
from radar_id_tracker.lidar_scoped_evaluate import in_scope, evaluate_scoped
from radar_id_tracker.lidar_tracker import BoxDetection, LidarTrackerConfig


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--baseline-cache',type=Path,help='optional existing 255/5 raw cache for 0000')
    p.add_argument('--sequences',nargs='+',default=['0000'])
    p.add_argument('--baseline-cache-root',type=Path)
    p.add_argument('--reuse-ablation',type=Path,help='optional complete 0000 ablation; checks cache settings')
    args=p.parse_args();out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    config=LidarTrackerConfig(high_confidence=.7,low_confidence=.1,
                              maximum_tentative_tracks=128,duplicate_birth_iou=.85)
    results=[]
    reused=json.loads(args.reuse_ablation.read_text()) if args.reuse_ablation else []
    multi=len(args.sequences)>1
    for sequence in args.sequences:
        truth,ignored=load_truth(args.dataset_root,sequence)
        poses=load_poses(args.dataset_root/'training/oxts'/f'{sequence}.txt')
        _,lti=calibration_transforms(args.dataset_root/'training/calib'/f'{sequence}.txt')
        settings=[(255,5),(128,5),(64,5),(16,5),(1,5)]
        if not multi:settings.extend([(255,1),(64,1)])
        for scale,sweeps in settings:
            name=f'intensity_{scale}_sweeps_{sweeps}'
            cache=out/'input_cache'/sequence/name
            if (scale,sweeps)==(255,5):
                if args.baseline_cache_root:cache=args.baseline_cache_root/sequence
                elif args.baseline_cache and sequence=='0000':cache=args.baseline_cache
            if sequence=='0000' and reused:
                cache=Path(next(r['cache'] for r in reused if (r['scale'],r['sweeps'])==(scale,sweeps)))
            if not (cache/'inference_summary.json').exists():
                infer_sequence(args.dataset_root,sequence,cache,args.checkpoint,intensity_scale=scale,sweeps=sweeps)
            meta=json.loads((cache/'inference_summary.json').read_text())
            if (meta['sequence'],meta['intensity_scale'],meta['sweeps'])!=(sequence,scale,sweeps):
                raise ValueError('cache input settings differ')
            run=out/'input_replay'/sequence/name
            # Validates the cache checksum and label-free provenance before scoring.
            replay_cache(cache,run,config)
            scores={t:BoxScore(False) for t in [.1,.2,.3,.4,.5,.6,.7,.8,.9]}
            for index,line in enumerate((cache/'detections.jsonl').read_text().splitlines()):
                if not (args.dataset_root/'training/velodyne'/sequence/f'{index:06d}.bin').exists():continue
                inverse=np.linalg.inv(poses[index]@lti)
                gt=[g for g in truth[index] if in_scope(g.box,inverse)]
                ig=[g for g in ignored[index] if in_scope(g,inverse)]
                ds=[BoxDetection(**d) for d in json.loads(line)['detections']]
                ds=[d for d in ds if in_scope(d,inverse)]
                for t,s in scores.items():s.update(gt,[d for d in ds if d.confidence>=t],ig,.25)
            evaluated=evaluate_scoped(args.dataset_root,run)
            result=dict(sequence=sequence,name=name,cache=str(cache),scale=scale,sweeps=sweeps,
                        raw_scores={t:s.result() for t,s in scores.items()},observed=evaluated['observed_tracks'],
                        tracker_config=asdict(config),output_sha256=meta['output_sha256'])
            results.append(result)
            print(json.dumps(dict(sequence=sequence,scale=scale,sweeps=sweeps,observed=result['observed'])),flush=True)
            (out/'input_ablation.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':main()
