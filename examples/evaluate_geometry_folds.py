"""Matched raw-score geometry evaluation; all models fixed before label scoring."""
import argparse
import json
from pathlib import Path
from radar_id_tracker.lidar_raw import replay_cache
from radar_id_tracker.lidar_tracker import LidarTrackerConfig
from radar_id_tracker.lidar_scoped_evaluate import evaluate_scoped,fixed_recovery
from fit_calibration_folds import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset-root','parent-run','run-root'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();root=args.run_root
    protocol=json.loads((root/'protocol.json').read_text());plan=protocol['parent_plan']
    manifest=json.loads((root/'cache_manifest.json').read_text());inference=json.loads((root/'inference_protocol.json').read_text())
    parent_protocol=json.loads((args.parent_run/'evaluation_protocol.json').read_text())
    parent_rows=json.loads((args.parent_run/'benchmark.json').read_text());parent_strict=json.loads((args.parent_run/'strict_iou_results.json').read_text())
    if sha(args.parent_run/'evaluation_protocol.json')!=protocol['parent_evaluation_protocol_sha256']:
        raise ValueError('parent evaluation protocol changed')
    labels={}
    for fold in range(3):
        model=json.loads((args.parent_run/f'fold_{fold}'/'calibration/model.json').read_text())
        labels.update(model['labels_sha256'])
    for sequence in (s for group in plan['groups'] for s in group):
        if sha(args.dataset_root/'training/label_02'/f'{sequence}.txt')!=labels[sequence]:
            raise ValueError('development labels differ from parent experiment')
    if len(manifest['records'])!=68 or manifest['inference_protocol_sha256']!=sha(root/'inference_protocol.json'):raise ValueError('incomplete geometry inference')
    for key,value in inference['models'].items():
        fold,variant=key.split('/');path=root/f'fold_{fold}'/variant/'model/model.json'
        if sha(path)!=value['model_sha256']:raise ValueError('frozen geometry model changed')
    config=LidarTrackerConfig.from_dict(plan['tracker_config']);intervals=parent_protocol['dropout_intervals']
    frozen=dict(protocol_sha256=sha(root/'protocol.json'),inference_protocol_sha256=sha(root/'inference_protocol.json'),
        cache_manifest_sha256=sha(root/'cache_manifest.json'),tracker_config=plan['tracker_config'],
        dropout_intervals=intervals,iou_thresholds=[.25,.5],score_calibration_applied=False,
        labels_previously_reviewed_development_only=True,reserved_sequence_opened=False,
        matched_control='continued_l1',candidate='weighted_axis_huber',selection_on_evaluation=False,
        source_sha256=sha(__file__),labels_sha256=labels,
        parent_benchmark_sha256=sha(args.parent_run/'benchmark.json'),parent_strict_results_sha256=sha(args.parent_run/'strict_iou_results.json'))
    path=root/'evaluation_protocol.json'
    if path.exists() and json.loads(path.read_text())!=frozen:raise ValueError('frozen geometry evaluation differs')
    path.write_text(json.dumps(frozen,indent=2)+'\n');rows=[];strict=[];recovery={}
    profiles=('baseline','head_raw',*protocol['variants'])
    for fold,(_,_,evaluate) in enumerate(plan['roles']):
        for sequence in plan['groups'][evaluate]:
            for scenario in ('natural','dropout_3s'):
                paths={}
                for profile in profiles:
                    cache=Path(manifest['records'][f'{fold}/{sequence}/{profile}']['cache'])
                    metadata=json.loads((cache/'inference_summary.json').read_text())
                    expected=manifest['records'][f'{fold}/{sequence}/{profile}']
                    if sha(cache/'detections.jsonl')!=expected['output_sha256'] or metadata['output_sha256']!=expected['output_sha256'] or metadata.get('head_model_sha256')!=expected['head_model_sha256']:
                        raise ValueError('geometry detector output/model changed')
                    if profile in ('baseline','head_raw'):
                        row=next(r for r in parent_rows if r['fold']==fold and r['sequence']==sequence and r['scenario']==scenario and r['profile']==profile)
                        directory=Path(row['run_dir']);run=row['run'];result=row['evaluation']
                        if LidarTrackerConfig.from_dict(run['tracker_config'])!=config:raise ValueError('parent tracker differs')
                    else:
                        directory=root/f'fold_{fold}'/sequence/scenario/profile
                        if (directory/'summary.json').exists():run=json.loads((directory/'summary.json').read_text())
                        else:run=replay_cache(cache,directory,config,hide_start=None if scenario=='natural' else intervals[sequence]['start_frame'],hide_frames=plan['dropout_frames'])
                        if LidarTrackerConfig.from_dict(run['tracker_config'])!=config or run['output_sha256']!=manifest['records'][f'{fold}/{sequence}/{profile}']['output_sha256']:raise ValueError('geometry replay domain differs')
                        result=json.loads((directory/'evaluation.json').read_text()) if (directory/'evaluation.json').exists() else evaluate_scoped(args.dataset_root,directory)
                        row=dict(fold=fold,sequence=sequence,role='development_outer_evaluation',scenario=scenario,profile=profile,run=run,evaluation=result,run_dir=str(directory.resolve()))
                    if run['output_sha256']!=expected['output_sha256']:raise ValueError('replayed source hash differs')
                    expected_interval=None if scenario=='natural' else dict(start_frame=intervals[sequence]['start_frame'],frames=plan['dropout_frames'],duration_s=plan['dropout_frames']*plan['inference_config']['period_s'],kind='all_detection_dropout')
                    if run['hidden_interval']!=expected_interval:raise ValueError('replay dropout interval differs')
                    paths[profile]=directory;rows.append(row)
                    (root/'benchmark.json').write_text(json.dumps(rows,indent=2)+'\n')
                    if scenario=='natural':
                        if profile in ('baseline','head_raw'):entry=next(r for r in parent_strict if r['fold']==fold and r['sequence']==sequence and r['profile']==profile)
                        else:entry=dict(fold=fold,sequence=sequence,profile=profile,evaluation=evaluate_scoped(args.dataset_root,directory,threshold=.5,write=False))
                        strict.append(entry);(root/'strict_iou_results.json').write_text(json.dumps(strict,indent=2)+'\n')
                    print(json.dumps(dict(sequence=sequence,scenario=scenario,profile=profile,**result['observed_tracks'])),flush=True)
                if scenario=='dropout_3s':
                    recovery[sequence]={profile:fixed_recovery(paths['baseline'],paths[profile]) for profile in profiles if profile!='baseline'}
                    (root/'fixed_recovery.json').write_text(json.dumps(recovery,indent=2)+'\n')
    (root/'evaluation_complete.json').write_text(json.dumps(dict(rows=len(rows),strict_rows=len(strict),sequences=17,evaluation_protocol_sha256=sha(path)))+'\n')


if __name__=='__main__':main()
