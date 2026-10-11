"""Generate portable full-sequence CenterPoint baselines on the target machine."""
import argparse
import json
from pathlib import Path

if __package__:
    from .benchmark_openpcdet import load_plan, write_frozen
else:
    from benchmark_openpcdet import load_plan, write_frozen
from radar_id_tracker.lidar_raw import infer_sequence
from radar_id_tracker.openpcdet_adapter import sha256


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--plan', type=Path, default=Path('configs/openpcdet_comparison.experimental.json'))
    args = parser.parse_args(argv)
    plan = load_plan(args.plan)
    config = dict(sweeps=5, period_s=.1, intensity_scale=64., score_threshold=.1, voxel_backend='numba')
    source_dir = Path(__file__).resolve().parents[1] / 'src/radar_id_tracker'
    sources = {name: sha256(source_dir / name) for name in
               ('centerpoint_model.py', 'centerpoint_geometry.py', 'centerpoint_voxel_numba.py', 'lidar_raw.py')}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_frozen(args.output_dir / 'baseline_protocol.json', dict(
        plan_sha256=sha256(args.plan), checkpoint_sha256=sha256(args.checkpoint),
        input_config=config, source_sha256=sources, labels_opened=False, reserved_sequence_opened=False))
    records = {}
    for sequence in plan['sequences']:
        cache = args.output_dir / 'raw_cache' / sequence
        if not (cache / 'inference_summary.json').exists():
            infer_sequence(args.dataset_root, sequence, cache, args.checkpoint, **config)
        meta = json.loads((cache / 'inference_summary.json').read_text())
        if (meta.get('ground_truth_used_as_input') is not False
                or meta['implementation_sha256'] != sources
                or meta['checkpoint_sha256'] != sha256(args.checkpoint)
                or meta['inference_config'] != config
                or sha256(cache / 'detections.jsonl') != meta['output_sha256']):
            raise ValueError('baseline cache differs from frozen protocol')
        records[f'0/{sequence}/baseline'] = dict(cache=cache.relative_to(args.output_dir).as_posix(),
                                               role='evaluation', output_sha256=meta['output_sha256'])
        (args.output_dir / 'cache_manifest.json').write_text(json.dumps(dict(
            format_version=2, cache_paths_relative_to_manifest=True, records=records), indent=2) + '\n')
    print(json.dumps(dict(sequences=len(records), manifest=str(args.output_dir / 'cache_manifest.json'))))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
