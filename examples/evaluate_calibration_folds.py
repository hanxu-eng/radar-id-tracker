"""Evaluate fixed strong/medium policies only after all calibrators are frozen."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from radar_id_tracker.lidar_score_calibration import (load_calibration_plan,ScoreCalibrationModel,calibrate_cache)
from radar_id_tracker.lidar_raw import replay_cache
from radar_id_tracker.lidar_tracker import LidarTrackerConfig,ClassScorePolicy
from radar_id_tracker.lidar_scoped_evaluate import evaluate_scoped,fixed_recovery
from fit_calibration_folds import samples,reliability,sha


def tracking_configs(plan,model):
    result={name:LidarTrackerConfig.from_dict(plan['tracker_config']) for name in ('baseline','head_raw')}
    for name in ('calibrated_strong','calibrated_medium'):
        values=dict(plan['tracker_config']);policies=[]
        p=plan['medium_policy']
        for label in ('Car','Pedestrian'):
            # Same raw >= .7 necessary condition for front and rear recovery.
            # Calibrated scores below .7 may impose a stricter additional gate.
            recovery=max(plan['recovery_raw_score_floor'],float(model.predict(label,[plan['recovery_raw_score_floor']])[0]))
            policies.append(ClassScorePolicy(label,p['continuation_confidence'],
                p['birth_confidence'] if name=='calibrated_medium' else p['strong_confidence'],p['strong_confidence'],recovery))
        values.update(score_policies=tuple(policies),medium_confirmation_hits=p['medium_confirmation_hits'])
        result[name]=LidarTrackerConfig(**values)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--run-root',type=Path,required=True);args=p.parse_args();root=args.run_root
    plan,digest=load_calibration_plan(args.plan);manifest=json.loads((root/'cache_manifest.json').read_text())
    if manifest['plan_sha256']!=digest:raise ValueError('inference manifest plan differs')
    models={i:ScoreCalibrationModel.load(root/f'fold_{i}'/'calibration/model.json') for i in range(3)}
    configs={i:tracking_configs(plan,m) for i,m in models.items()}
    intervals={}
    for fold,(_,_,evaluate) in enumerate(plan['roles']):
        for sequence in plan['groups'][evaluate]:
            cache=Path(manifest['records'][f'{fold}/{sequence}/baseline']['cache'])
            frames=json.loads((cache/'inference_summary.json').read_text())['frames']
            # Label-free deterministic clamp reserves the full 0.5s recovery
            # window on the short 78-frame sequence, instead of skipping it.
            start=min(plan['dropout_start_frame'],frames-plan['dropout_frames']-5)
            if start<2:raise ValueError('sequence too short for fixed dropout protocol')
            intervals[sequence]=dict(start_frame=start,frames=plan['dropout_frames'])
    frozen=dict(plan_sha256=digest,evaluation_used_for_selection=False,geometry_training_changed=False,
        calibration_model_sha256={str(i):sha(root/f'fold_{i}'/'calibration/model.json') for i in models},
        tracker_configs={str(i):{n:asdict(c) for n,c in configs[i].items()} for i in models},
        scope='front +/-40deg 50m Car/Pedestrian, not official KITTI',iou_thresholds=[.25,.5],
        adoption_guards=dict(precision=plan['acceptance_precision'],recall=plan['acceptance_recall'],
            require_each_sequence_and_supported_class=True,require_no_extra_id_changes=True,
            require_no_fixed_recovery_regression=True,require_no_extra_wrong_recovery=True),
        final_deployment_model_trained=False,reserved_sequence_opened=False,
        split_role_audit_sha256=sha(root/'split_role_audit.json'),dropout_intervals=intervals,
        short_sequence_dropout_policy='min(requested start, frame count - hidden frames - 5 recovery frames)')
    path=root/'evaluation_protocol.json'
    if path.exists() and json.loads(path.read_text())!=frozen:raise ValueError('frozen evaluation protocol differs')
    path.write_text(json.dumps(frozen,indent=2)+'\n')
    rows=[];strict=[];recovery={};reliability_rows=[]
    for fold,(train,calibrate,evaluate) in enumerate(plan['roles']):
        model=models[fold];model_path=root/f'fold_{fold}'/'calibration/model.json'
        if (model.data['plan_sha256']!=digest or model.data['detector_training_sequences']!=plan['groups'][train] or
                model.data['calibration_sequences']!=plan['groups'][calibrate] or model.data['evaluation_sequences']!=plan['groups'][evaluate]):
            raise ValueError('calibration model fold roles differ')
        for sequence in plan['groups'][evaluate]:
            original=Path(manifest['records'][f'{fold}/{sequence}/baseline']['cache'])
            raw=Path(manifest['records'][f'{fold}/{sequence}/head_raw']['cache']);model.validate_domain(json.loads((raw/'inference_summary.json').read_text()))
            calibrated=root/f'fold_{fold}'/'calibrated_cache'/sequence
            if not (calibrated/'inference_summary.json').exists():calibrate_cache(raw,calibrated,model_path)
            meta=json.loads((calibrated/'inference_summary.json').read_text())
            if meta['score_calibration_sha256']!=sha(model_path) or meta['raw_detector_output_sha256']!=manifest['records'][f'{fold}/{sequence}/head_raw']['output_sha256'] or sha(calibrated/'detections.jsonl')!=meta['output_sha256']:
                raise ValueError('frozen calibrated output differs')
            dataset=samples(args.dataset_root,sequence,raw,plan['calibration']['target_iou'])
            for label,data in dataset.items():
                reliability_rows.append(dict(fold=fold,sequence=sequence,label=label,raw=reliability(data['scores'],data['targets']),
                    calibrated=reliability(model.predict(label,data['scores']),data['targets'])))
            (root/'heldout_reliability.json').write_text(json.dumps(reliability_rows,indent=2)+'\n')
            for scenario,hide in (('natural',None),('dropout_3s',intervals[sequence]['start_frame'])):
                paths={}
                for profile,cache in (('baseline',original),('head_raw',raw),('calibrated_strong',calibrated),('calibrated_medium',calibrated)):
                    directory=root/f'fold_{fold}'/sequence/scenario/profile;paths[profile]=directory
                    summary_path=directory/'summary.json'
                    if summary_path.exists():
                        run=json.loads(summary_path.read_text())
                        if LidarTrackerConfig.from_dict(run['tracker_config'])!=configs[fold][profile] or run['output_sha256']!=json.loads((cache/'inference_summary.json').read_text())['output_sha256']:
                            raise ValueError('replayed parameters or cache differ')
                    else:run=replay_cache(cache,directory,configs[fold][profile],hide_start=hide,hide_frames=plan['dropout_frames'])
                    evaluation_path=directory/'evaluation.json'
                    result=json.loads(evaluation_path.read_text()) if evaluation_path.exists() else evaluate_scoped(args.dataset_root,directory)
                    row=dict(fold=fold,sequence=sequence,role='development_outer_evaluation',scenario=scenario,profile=profile,run=run,evaluation=result,run_dir=str(directory.resolve()))
                    rows.append(row);(root/'benchmark.json').write_text(json.dumps(rows,indent=2)+'\n')
                    if scenario=='natural':
                        result05=evaluate_scoped(args.dataset_root,directory,threshold=.5,write=False)
                        strict.append(dict(fold=fold,sequence=sequence,profile=profile,evaluation=result05))
                        (root/'strict_iou_results.json').write_text(json.dumps(strict,indent=2)+'\n')
                    print(json.dumps(dict(fold=fold,sequence=sequence,scenario=scenario,profile=profile,**result['observed_tracks'])),flush=True)
                if scenario=='dropout_3s':
                    recovery[sequence]={profile:fixed_recovery(paths['baseline'],paths[profile]) for profile in ('head_raw','calibrated_strong','calibrated_medium')}
                    (root/'fixed_recovery.json').write_text(json.dumps(recovery,indent=2)+'\n')
    (root/'evaluation_complete.json').write_text(json.dumps(dict(rows=len(rows),strict_rows=len(strict),sequences=17,plan_sha256=digest))+'\n')


if __name__=='__main__':main()
