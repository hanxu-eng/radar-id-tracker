"""Freeze development-selected input settings before regression/new-sequence tests.

Both sides use identical weights, tracker and front-only diagnostic scoring.
No validation result can change the frozen input settings.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from radar_id_tracker.lidar_inference_config import RawLidarConfig
from radar_id_tracker.lidar_input_selection import select_grouped_inputs
from radar_id_tracker.lidar_tracker import LidarTrackerConfig
from radar_id_tracker.lidar_raw import infer_sequence, replay_cache
from radar_id_tracker.lidar_scoped_evaluate import evaluate_scoped, fixed_recovery, SCOPE
from radar_id_tracker.lidar_visualization import build_review
from benchmark_surround import label_review


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--ablation',type=Path,required=True)
    p.add_argument('--baseline-cache-root',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--validation-sequences',nargs='+',default=['0002','0006'])
    args=p.parse_args();out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    candidates=json.loads(args.ablation.read_text())
    selection=select_grouped_inputs(candidates)
    development=selection['development_sequences']
    if set(development)&set(args.validation_sequences) or len(set(args.validation_sequences))!=len(args.validation_sequences):
        p.error('development and validation sequences must be distinct and unique')
    inference=RawLidarConfig.from_dict(selection['selected_inference_config'])
    config=LidarTrackerConfig(high_confidence=.7,low_confidence=.1,maximum_tentative_tracks=128,duplicate_birth_iou=.85)
    protocol=dict(development_sequences=development,
                  validation_sequences=args.validation_sequences,scope=SCOPE,
                  grouped_selection=selection,
                  development_acceptance_passed=selection['development_acceptance_passed'],acceptance_precision=.96,acceptance_recall=.85,
                  frozen_config=asdict(config),frozen_inference_config=asdict(inference),
                  baseline_inference_config=asdict(RawLidarConfig()),
                  development_ablation_sha256=hashlib.sha256(args.ablation.read_bytes()).hexdigest(),
                  implementation_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in
                    [Path(__file__),Path('src/radar_id_tracker/lidar_raw.py'),Path('src/radar_id_tracker/lidar_input_selection.py')]},
                  ground_truth_used_as_detector_or_tracker_input=False)
    protocol_path=out/'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text())!=protocol:
        raise ValueError('frozen protocol differs; use a new output directory')
    protocol_path.write_text(json.dumps(protocol,indent=2)+'\n')
    # Protocol is saved before any validation sequence is inferred or scored.
    rows=[]
    for sequence in development+args.validation_sequences:
        for profile,input_config in [('baseline',RawLidarConfig()),('candidate',inference)]:
            if sequence in development:
                source=next(r for r in candidates if r['sequence']==sequence and (r['scale'],r['sweeps'])==(input_config.intensity_scale,input_config.sweeps))
                cache=Path(source['cache'])
            else:
                existing=args.baseline_cache_root/sequence
                cache=existing if profile=='baseline' and (existing/'inference_summary.json').exists() else out/'validation_cache'/sequence/profile
            if not (cache/'inference_summary.json').exists():
                infer_sequence(args.dataset_root,sequence,cache,args.checkpoint,**asdict(input_config))
            meta=json.loads((cache/'inference_summary.json').read_text())
            if meta['checkpoint_sha256']!=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest():
                raise ValueError('cache checkpoint differs from requested checkpoint')
            expected=asdict(input_config)
            if meta['sequence']!=sequence or any(meta.get(k)!=v for k,v in expected.items() if k!='score_threshold') or meta['detector_score_threshold']!=input_config.score_threshold:
                raise ValueError('cache input settings differ from frozen protocol')
            for scenario,start in [('natural',None),('dropout_3s',50)]:
                path=out/sequence/scenario/profile
                run=replay_cache(cache,path,config,hide_start=start)
                result=evaluate_scoped(args.dataset_root,path)
                rows.append(dict(sequence=sequence,scenario=scenario,profile=profile,run=run,evaluation=result))
                print(json.dumps(dict(sequence=sequence,scenario=scenario,profile=profile,**result['observed_tracks'])),flush=True)
                (out/'benchmark.json').write_text(json.dumps(rows,indent=2)+'\n')
    fixed={s:fixed_recovery(out/s/'dropout_3s/baseline',out/s/'dropout_3s/candidate') for s in development+args.validation_sequences}
    (out/'fixed_recovery.json').write_text(json.dumps(fixed,indent=2)+'\n')
    build_review(out,rows);label_review(out/'review.html',comparison='input')


if __name__=='__main__':main()
