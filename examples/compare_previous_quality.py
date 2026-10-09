"""Evaluate the previous frozen quality model on the same new raw caches."""
import argparse
import json
from pathlib import Path
from radar_id_tracker.lidar_raw import replay_cache
from radar_id_tracker.lidar_tracker import LidarTrackerConfig
from radar_id_tracker.lidar_scoped_evaluate import evaluate_scoped
from prepare_box_quality import extract
from apply_box_quality import apply

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root',type=Path,required=True)
    p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--quality-model',type=Path,required=True)
    args=p.parse_args();root=args.run_root
    protocol=json.loads((root/'evaluation_protocol.json').read_text())
    config=LidarTrackerConfig.from_dict(protocol['frozen_config']);rows=[]
    data=json.loads(args.quality_model.read_text())
    if set(protocol['validation_sequences'])&set(data['training_sequences']):raise ValueError('reference training overlaps new sequences')
    for sequence in protocol['validation_sequences']:
        cache=root/'raw_cache'/sequence/'baseline'
        features=root/'previous_quality/features'/sequence
        if not (features/'manifest.json').exists():extract(args.dataset_root,cache,features)
        quality_cache=root/'previous_quality/cache'/sequence
        apply(cache,features,args.quality_model,quality_cache)
        for scenario,start in [('natural',None),('dropout_3s',50)]:
            path=root/sequence/scenario/'previous_quality'
            run=replay_cache(quality_cache,path,config,hide_start=start)
            result=evaluate_scoped(args.dataset_root,path)
            rows.append(dict(sequence=sequence,scenario=scenario,profile='previous_quality',role='validation',run=run,evaluation=result))
            print(json.dumps(dict(sequence=sequence,scenario=scenario,**result['observed_tracks'])),flush=True)
            (root/'previous_quality_results.json').write_text(json.dumps(rows,indent=2)+'\n')

if __name__=='__main__':main()
