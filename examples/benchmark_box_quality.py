"""Evaluate fixed quality trees: OOF development and frozen new sequences."""
import argparse
import hashlib
import json
from pathlib import Path
from radar_id_tracker.lidar_raw import replay_cache
from radar_id_tracker.lidar_tracker import LidarTrackerConfig
from radar_id_tracker.lidar_scoped_evaluate import evaluate_scoped,fixed_recovery
from radar_id_tracker.lidar_visualization import build_review
from apply_box_quality import apply
from prepare_box_quality import extract
from benchmark_surround import label_review


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root',type=Path,required=True)
    p.add_argument('--run-root',type=Path,required=True)
    args=p.parse_args();out=args.run_root
    protocol=json.loads((out/'models/protocol.json').read_text())
    model=out/'models/full/model.json'
    if hashlib.sha256(model.read_bytes()).hexdigest()!=protocol['model_sha256']:raise ValueError('frozen model checksum differs')
    # Replays use the same fixed 0.7/0.1 settings, with no result-based retuning.
    config=LidarTrackerConfig.from_dict(protocol['frozen_config']);rows=[]
    for sequence in protocol['training_sequences']+protocol['validation_sequences']:
        training=sequence in protocol['training_sequences']
        features=out/'features'/sequence
        if training:
            manifest=json.loads((features/'manifest.json').read_text())
            cache=Path(manifest['cache']);quality_model=out/f'models/fold_{sequence}/model.json'
        else:
            cache=out/'raw_cache'/sequence;quality_model=model
            if not (features/'manifest.json').exists():extract(args.dataset_root,cache,features)
        data=json.loads(quality_model.read_text())
        if sequence in data['training_sequences']:raise ValueError('evaluation sequence leaked into model training')
        expected=next(f['model_sha256'] for f in protocol['folds'] if f['held_out_sequence']==(sequence if training else None))
        if hashlib.sha256(quality_model.read_bytes()).hexdigest()!=expected:raise ValueError('frozen fold model checksum differs')
        quality_cache=out/'quality_cache'/sequence
        apply(cache,features,quality_model,quality_cache)
        for scenario,start in [('natural',None),('dropout_3s',50)]:
            for profile,source in [('baseline',cache),('candidate',quality_cache)]:
                path=out/sequence/scenario/profile
                run=replay_cache(source,path,config,hide_start=start)
                result=evaluate_scoped(args.dataset_root,path)
                rows.append(dict(sequence=sequence,role='out_of_fold' if training else 'validation',scenario=scenario,profile=profile,run=run,evaluation=result))
                print(json.dumps(dict(sequence=sequence,scenario=scenario,profile=profile,**result['observed_tracks'])),flush=True)
                (out/'benchmark.json').write_text(json.dumps(rows,indent=2)+'\n')
    sequences=protocol['training_sequences']+protocol['validation_sequences']
    fixed={s:fixed_recovery(out/s/'dropout_3s/baseline',out/s/'dropout_3s/candidate') for s in sequences}
    (out/'fixed_recovery.json').write_text(json.dumps(fixed,indent=2)+'\n')
    build_review(out,rows);label_review(out/'review.html',comparison='quality')


if __name__=='__main__':main()
