"""Frozen-head raw inference: training replay diagnostics and unseen sequences."""
import argparse
import hashlib
import json
from pathlib import Path
from radar_id_tracker.lidar_raw import infer_sequence,replay_cache
from radar_id_tracker.lidar_head_targets import load_head_manifest
from radar_id_tracker.lidar_tracker import LidarTrackerConfig
from radar_id_tracker.lidar_scoped_evaluate import evaluate_scoped,fixed_recovery
from radar_id_tracker.lidar_visualization import build_review
from benchmark_surround import label_review


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--previous-quality-run',type=Path,required=True)
    args=p.parse_args();out=args.run_root
    model=out/'model/model.json';metadata,_=load_head_manifest(model)
    protocol=json.loads((out/'model/protocol.json').read_text())
    digest=hashlib.sha256(model.read_bytes()).hexdigest()
    frozen=out/'evaluation_protocol.json'
    evidence=dict(model_sha256=digest,weights_sha256=metadata['weights_sha256'],
        training_sequences=metadata['training_sequences'],validation_sequences=metadata['validation_sequences'],
        frozen_inference_config=metadata['inference_config'],frozen_config=protocol['tracker_config'],
        acceptance_precision=.96,acceptance_recall=.85,validation_used_for_selection=False)
    if frozen.exists() and json.loads(frozen.read_text())!=evidence:raise ValueError('evaluation protocol differs')
    frozen.write_text(json.dumps(evidence,indent=2)+'\n')
    config=LidarTrackerConfig.from_dict(protocol['tracker_config']);rows=[]
    for sequence in metadata['training_sequences']+metadata['validation_sequences']:
        training=sequence in metadata['training_sequences']
        if training:
            feature=json.loads((args.previous_quality_run/'features'/sequence/'manifest.json').read_text())
            baseline=Path(feature['cache'])
        else:
            baseline=out/'raw_cache'/sequence/'baseline'
            if not (baseline/'inference_summary.json').exists():
                infer_sequence(args.dataset_root,sequence,baseline,args.checkpoint,**metadata['inference_config'])
        candidate=out/'raw_cache'/sequence/'candidate'
        if not (candidate/'inference_summary.json').exists():
            infer_sequence(args.dataset_root,sequence,candidate,args.checkpoint,head_model=model,**metadata['inference_config'])
        if json.loads((candidate/'inference_summary.json').read_text())['head_model_sha256']!=digest:raise ValueError('candidate model differs')
        for scenario,start in [('natural',None),('dropout_3s',50)]:
            for profile,cache in [('baseline',baseline),('candidate',candidate)]:
                path=out/sequence/scenario/profile
                run=replay_cache(cache,path,config,hide_start=start)
                result=evaluate_scoped(args.dataset_root,path)
                rows.append(dict(sequence=sequence,role='training_replay' if training else 'validation',
                    scenario=scenario,profile=profile,run=run,evaluation=result))
                print(json.dumps(dict(sequence=sequence,role=rows[-1]['role'],scenario=scenario,profile=profile,
                    **result['observed_tracks'])),flush=True)
                (out/'benchmark.json').write_text(json.dumps(rows,indent=2)+'\n')
    fixed={s:fixed_recovery(out/s/'dropout_3s/baseline',out/s/'dropout_3s/candidate')
        for s in metadata['training_sequences']+metadata['validation_sequences']}
    (out/'fixed_recovery.json').write_text(json.dumps(fixed,indent=2)+'\n')
    build_review(out,rows);label_review(out/'review.html','heads')

if __name__=='__main__':main()
