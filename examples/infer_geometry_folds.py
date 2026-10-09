"""Freeze all six geometry models, then infer only the development outer folds."""
import argparse
import json
from pathlib import Path
from radar_id_tracker.lidar_raw import infer_sequence
from radar_id_tracker.lidar_head_targets import load_head_manifest
from fit_calibration_folds import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset-root','checkpoint','parent-run','run-root','input-audit'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();root=args.run_root
    protocol=json.loads((root/'protocol.json').read_text());plan=protocol['parent_plan']
    parent=json.loads((args.parent_run/'cache_manifest.json').read_text())
    audit=json.loads(args.input_audit.read_text());models={};frozen_audits={}
    for fold,(train,_,evaluate) in enumerate(plan['roles']):
        for variant in protocol['variants']:
            path=root/f'fold_{fold}'/variant/'model/model.json';head,_=load_head_manifest(path)
            if head['training_sequences']!=plan['groups'][train] or head['parent_head_model_sha256']!=protocol['parent_head_sha256'][str(fold)] or head['geometry_protocol_sha256']!=sha(root/'protocol.json'):
                raise ValueError('geometry head split/parent/protocol differs')
            state=json.loads((path.parent/'frozen_tensor_audit.json').read_text())
            if not state['all_other_tensors_exactly_equal']:raise ValueError('frozen tensors changed')
            key=f'{fold}/{variant}';models[key]=dict(model_sha256=sha(path),weights_sha256=head['weights_sha256'])
            frozen_audits[key]=state
    frozen=dict(protocol_sha256=sha(root/'protocol.json'),models=models,frozen_tensor_audits=frozen_audits,
        inference_config=plan['inference_config'],input_audit_sha256=sha(args.input_audit),
        parent_cache_manifest_sha256=sha(args.parent_run/'cache_manifest.json'),
        labels_opened_as_input=False,score_calibration_applied=False,evaluation_used_for_selection=False)
    target=root/'inference_protocol.json'
    if target.exists() and json.loads(target.read_text())!=frozen:raise ValueError('frozen geometry inference differs')
    target.write_text(json.dumps(frozen,indent=2)+'\n');records={}
    for fold,(_,_,evaluate) in enumerate(plan['roles']):
        for sequence in plan['groups'][evaluate]:
            source=next(r for r in audit['sequences'] if r['sequence']==sequence)
            for kind in ('oxts','calib'):
                if sha(args.dataset_root/'training'/kind/f'{sequence}.txt')!=source[kind+'_sha256']:raise ValueError('source metadata changed')
            for profile in ('baseline','head_raw',*protocol['variants']):
                if profile in ('baseline','head_raw'):
                    record=parent['records'][f'{fold}/{sequence}/{profile}'];cache=Path(record['cache']);expected_head=record['head_model_sha256']
                else:
                    model=root/f'fold_{fold}'/profile/'model/model.json';expected_head=sha(model)
                    cache=root/f'fold_{fold}'/'raw_cache'/sequence/profile
                    if not (cache/'inference_summary.json').exists():
                        infer_sequence(args.dataset_root,sequence,cache,args.checkpoint,head_model=model,**plan['inference_config'])
                m=json.loads((cache/'inference_summary.json').read_text())
                if m['inference_config']!=plan['inference_config'] or m.get('head_model_sha256')!=expected_head or m.get('score_calibration_sha256') or m.get('box_quality_model_sha256') or m['ground_truth_used_as_input'] is not False or sha(cache/'detections.jsonl')!=m['output_sha256']:
                    raise ValueError('geometry cache domain/checksum differs')
                if [{k:r[k] for k in ('frame_index','sha256')} for r in m['input_scans']]!=audit['scan_hashes'][sequence]:raise ValueError('input scan hashes differ')
                records[f'{fold}/{sequence}/{profile}']=dict(cache=str(cache.resolve()),role='development_outer_evaluation',output_sha256=m['output_sha256'],head_model_sha256=expected_head)
                (root/'cache_manifest.json').write_text(json.dumps(dict(inference_protocol_sha256=sha(target),records=records),indent=2)+'\n')
    print('All geometry inference complete; labels were not detector inputs.',flush=True)


if __name__=='__main__':main()
