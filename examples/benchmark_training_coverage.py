"""Freeze expanded-data heads before fresh whole-sequence inference/evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
from radar_id_tracker.lidar_head_targets import load_head_manifest
from radar_id_tracker.lidar_training_protocol import load_training_plan
from radar_id_tracker.lidar_raw import infer_sequence, replay_cache
from radar_id_tracker.lidar_scoped_evaluate import evaluate_scoped, fixed_recovery
from radar_id_tracker.lidar_tracker import LidarTrackerConfig
from radar_id_tracker.lidar_visualization import build_review
from benchmark_surround import label_review


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--split-plan', type=Path, required=True)
    p.add_argument('--previous-run', type=Path, required=True)
    p.add_argument('--run-root', type=Path, required=True)
    p.add_argument('--phase', choices=('infer', 'evaluate'), required=True)
    args = p.parse_args(); root = args.run_root
    plan, plan_sha = load_training_plan(args.split_plan)
    models = {'candidate': root / 'model/model.json', 'previous_head': args.previous_run / 'model/model.json'}
    new, _ = load_head_manifest(models['candidate']); previous, _ = load_head_manifest(models['previous_head'])
    if new.get('split_plan_sha256') != plan_sha or new['training_sequences'] != plan['training_sequences'] or new['validation_sequences'] != plan['validation_sequences']:
        raise ValueError('model and frozen split differ')
    if set(previous['training_sequences']) & set(plan['validation_sequences']): raise ValueError('previous model saw validation')
    if new['inference_config'] != previous['inference_config'] or new['inference_config'] != plan['inference_config']:
        raise ValueError('comparison input domains differ')
    sequences = plan['diagnostic_replay_sequences'] + plan['validation_sequences']
    digests = {k: hashlib.sha256(v.read_bytes()).hexdigest() for k, v in models.items()}
    protocol = dict(model_sha256=digests['candidate'], weights_sha256=new['weights_sha256'],
        previous_head_model_sha256=digests['previous_head'], split_plan_sha256=plan_sha,
        training_sequences=plan['training_sequences'], validation_sequences=plan['validation_sequences'],
        reserved_sequences=plan['reserved_sequences'], diagnostic_replay_sequences=plan['diagnostic_replay_sequences'],
        frozen_inference_config=plan['inference_config'], frozen_config=plan['tracker_config'],
        acceptance_precision=plan['acceptance_precision'], acceptance_recall=plan['acceptance_recall'],
        validation_used_for_selection=False)
    frozen = root / 'evaluation_protocol.json'
    if frozen.exists() and json.loads(frozen.read_text()) != protocol: raise ValueError('frozen evaluation protocol differs')
    frozen.write_text(json.dumps(protocol, indent=2) + '\n')
    old_rows = json.loads((args.previous_run / 'benchmark.json').read_text())
    caches = {}
    audit = json.loads((root / 'split_input_audit.json').read_text())
    if audit['split_plan_sha256'] != plan_sha: raise ValueError('split audit differs')
    for sequence in sequences:
        source_audit=next(r for r in audit['sequences'] if r['sequence']==sequence)
        for kind in ('oxts','calib'):
            path=args.dataset_root/'training'/kind/f'{sequence}.txt'
            if hashlib.sha256(path.read_bytes()).hexdigest()!=source_audit[kind+'_sha256']:
                raise ValueError('calibration/pose differs from frozen input audit')
        known = sequence in plan['diagnostic_replay_sequences']
        baseline_row = next((r for r in old_rows if r['sequence'] == sequence and r['profile'] == 'baseline' and r['scenario'] == 'natural'), None)
        for profile in ('baseline', 'candidate', 'previous_head'):
            if known and profile == 'baseline': cache = Path(baseline_row['run']['inference_cache'])
            elif known and profile == 'previous_head': cache = args.previous_run / 'raw_cache' / sequence / 'candidate'
            else: cache = root / 'raw_cache' / sequence / profile
            if not (cache / 'inference_summary.json').exists():
                if args.phase != 'infer': raise ValueError('run inference before evaluation')
                infer_sequence(args.dataset_root, sequence, cache, args.checkpoint,
                               head_model=models.get(profile), **plan['inference_config'])
            meta = json.loads((cache / 'inference_summary.json').read_text())
            if meta['inference_config'] != plan['inference_config'] or meta.get('head_model_sha256') != digests.get(profile):
                raise ValueError('cached detector/model domain differs')
            if meta['ground_truth_used_as_input'] is not False: raise ValueError('label-free cache required')
            scans=[{k:r[k] for k in ('frame_index','sha256')} for r in meta['input_scans']]
            if scans != audit['scan_hashes'][sequence]: raise ValueError('inference source scans differ')
            caches[sequence, profile] = cache
    if args.phase == 'infer':
        print('Frozen model inference complete; no validation detection labels opened.'); return
    provenance=dict(model_sha256=digests['candidate'],split_plan_sha256=plan_sha,
        labels_sha256={s:hashlib.sha256((args.dataset_root/'training/label_02'/f'{s}.txt').read_bytes()).hexdigest() for s in sequences},
        label_hashes_recorded_after_model_freeze=True)
    provenance_path=root/'evaluation_provenance.json'
    if provenance_path.exists() and json.loads(provenance_path.read_text())!=provenance:
        raise ValueError('frozen scoring labels differ')
    provenance_path.write_text(json.dumps(provenance,indent=2)+'\n')
    config = LidarTrackerConfig.from_dict(plan['tracker_config']); rows = []; references = []
    for sequence in sequences:
        role = 'validation' if sequence in plan['validation_sequences'] else 'training_replay'
        for scenario, start in (('natural', None), ('dropout_3s', 50)):
            for profile in ('baseline', 'candidate', 'previous_head'):
                directory = root / sequence / scenario / profile
                if role == 'training_replay' and profile in ('baseline', 'previous_head'):
                    old_profile = 'candidate' if profile == 'previous_head' else 'baseline'
                    directory = args.previous_run / sequence / scenario / old_profile
                    run = json.loads((directory / 'summary.json').read_text())
                    if LidarTrackerConfig.from_dict(run['tracker_config']) != config:
                        raise ValueError('reused tracker parameters differ')
                    result = json.loads((directory / 'evaluation.json').read_text())
                else:
                    run = replay_cache(caches[sequence, profile], directory, config, hide_start=start)
                    result = evaluate_scoped(args.dataset_root, directory)
                row = dict(sequence=sequence, role=role, scenario=scenario, profile=profile,
                           run=run, evaluation=result, run_dir=str(directory.resolve()))
                (references if profile == 'previous_head' else rows).append(row)
                print(json.dumps(dict(sequence=sequence, role=role, scenario=scenario, profile=profile,
                                      **result['observed_tracks'])), flush=True)
                (root / 'benchmark.json').write_text(json.dumps(rows, indent=2) + '\n')
                (root / 'previous_head_results.json').write_text(json.dumps(references, indent=2) + '\n')
    paths = {(r['sequence'], r['profile']): Path(r['run_dir']) for r in rows if r['scenario'] == 'dropout_3s'}
    fixed = {s: fixed_recovery(paths[s, 'baseline'], paths[s, 'candidate']) for s in sequences}
    (root / 'fixed_recovery.json').write_text(json.dumps(fixed, indent=2) + '\n')
    reference_paths={r['sequence']:Path(r['run_dir']) for r in references if r['scenario']=='dropout_3s'}
    previous_fixed={s:fixed_recovery(paths[s,'baseline'],reference_paths[s]) for s in sequences}
    (root/'fixed_previous_head_recovery.json').write_text(json.dumps(previous_fixed,indent=2)+'\n')
    build_review(root, rows); label_review(root / 'review.html', 'coverage', plan['validation_sequences'])


if __name__ == '__main__': main()
