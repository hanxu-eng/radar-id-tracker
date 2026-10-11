"""Prepare/infer/evaluate a frozen, whole-sequence official-detector comparison.

Preparation and inference never read target annotations. Evaluation is an
explicit separate stage. Results remain development diagnostics until scene
overlap with checkpoint pretraining has been independently audited.
"""
import argparse
from collections import defaultdict
from dataclasses import asdict, fields
import json
import os
from pathlib import Path, PureWindowsPath
import sys

import numpy as np

from radar_id_tracker.kitti_adapter import load_poses
from radar_id_tracker.lidar_tracker import BoxDetection, LidarTrackerConfig
from radar_id_tracker.openpcdet_adapter import (
    MODEL_SPECS, UPSTREAM_COMMIT, OpenPCDetDetector, infer_sequence, sha256,
    validate_assets,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    'src/radar_id_tracker/openpcdet_adapter.py', 'examples/benchmark_openpcdet.py',
    'src/radar_id_tracker/lidar_raw.py', 'src/radar_id_tracker/lidar_tracker.py',
    'src/radar_id_tracker/kitti_adapter.py', 'src/radar_id_tracker/lidar_scoped_evaluate.py',
    'src/radar_id_tracker/kitti_evaluate.py', 'src/radar_id_tracker/lidar_detection_curve.py',
    'src/radar_id_tracker/box_geometry.py', 'src/radar_id_tracker/assignment.py',
)


def portable_path(value, manifest_dir=None, cache_root=None):
    """Relative manifests work on native Linux; legacy caches can be relocated.

    --cache-root names the copied repository's outputs directory. No source
    Windows drive is assumed to exist on a native Ubuntu deployment.
    """
    if cache_root is not None:
        parts = PureWindowsPath(value).parts if PureWindowsPath(value).drive else Path(value).parts
        if 'outputs' not in parts:
            raise ValueError('legacy cache relocation requires an outputs path component')
        relative = Path(*parts[parts.index('outputs') + 1:])
        result = (cache_root / relative).resolve()
        if not result.is_relative_to(cache_root.resolve()):
            raise ValueError('cache path escapes relocation root')
        return result
    if os.name != 'nt' and PureWindowsPath(value).drive:
        path = PureWindowsPath(value)
        result = Path('/mnt') / path.drive[0].lower() / Path(*path.parts[1:])
        if not result.exists():
            raise ValueError('Windows cache path unavailable on Linux; supply --cache-root or generate a native baseline')
        return result
    path = Path(value)
    return path if path.is_absolute() or manifest_dir is None else manifest_dir / path


def load_plan(path):
    plan = json.loads(path.read_text())
    sequences = plan['sequences']
    if (not sequences or len(sequences) != len(set(sequences))
            or any(len(s) != 4 or not s.isascii() or not s.isdecimal() for s in sequences)):
        raise ValueError('unique four-digit whole sequences required')
    forbidden = set(plan['reserved_sequences']) | set(plan['excluded_viewed_sequences']) | {'0020'}
    if forbidden.intersection(sequences):
        raise ValueError('reserved or excluded sequence must not be opened')
    expected = {f'{i:04d}' for i in range(20)} - {'0010', '0015', '0019'}
    if set(sequences) != expected:
        raise ValueError('all 17 fixed whole development sequences are required')
    if (plan['development_only'] is not True or plan['score_threshold_tuning'] is not False
            or plan['models'] != ['second_iou', 'pv_rcnn'] or plan['iou_thresholds'] != [.25, .5]
            or plan['dropout_start_frame'] != 50 or plan['dropout_frames'] != 30
            or plan['fixed_recovery_reference_profile'] != 'centerpoint'
            or plan['minimum_detector_precision'] != .96 or plan['target_detector_recall'] != .85):
        raise ValueError('frozen detector-comparison protocol changed')
    LidarTrackerConfig.from_dict(plan['tracker_config'])
    return plan


def baseline_caches(manifest, sequences, cache_root=None):
    records = json.loads(manifest.read_text())['records']
    result = {}
    for key, record in records.items():
        _, sequence, profile = key.split('/')
        if profile != 'baseline' or record.get('role') != 'evaluation' or sequence not in sequences:
            continue
        if sequence in result:
            raise ValueError('duplicate evaluation baseline sequence')
        cache = portable_path(record['cache'], manifest.resolve().parent, cache_root).resolve()
        metadata = json.loads((cache / 'inference_summary.json').read_text())
        if (metadata.get('ground_truth_used_as_input') is not False
                or metadata.get('raw_pointcloud_detector_tested') is not True
                or metadata.get('period_s') != .1 or metadata.get('sequence') != sequence
                or sha256(cache / 'detections.jsonl') != record['output_sha256']
                or metadata['output_sha256'] != record['output_sha256']):
            raise ValueError('baseline provenance/checksum mismatch: ' + sequence)
        result[sequence] = dict(cache=str(cache), output_sha256=record['output_sha256'],
                                summary_sha256=sha256(cache / 'inference_summary.json'))
    if set(result) != set(sequences):
        raise ValueError('complete development baseline required')
    return result


def dataset_inventory(root, sequences):
    records = {}
    for sequence in sequences:
        poses = root / 'training/oxts' / f'{sequence}.txt'
        calib = root / 'training/calib' / f'{sequence}.txt'
        frames = len(load_poses(poses))
        if frames < 37:
            raise ValueError('sequence too short for the fixed dropout interval')
        available = []
        for index in range(frames):
            scan = root / 'training/velodyne' / sequence / f'{index:06d}.bin'
            if scan.exists():
                size = scan.stat().st_size
                if size <= 0 or size % 16:
                    raise ValueError('invalid raw scan size: ' + str(scan))
                available.append(dict(frame_index=index, bytes=size))
        records[sequence] = dict(frames=frames, scans=available, dropout_start_frame=min(50, frames - 35),
                                 oxts_sha256=sha256(poses), calib_sha256=sha256(calib))
    return records


def write_frozen(path, value):
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError('frozen artifact changed; use a new experiment directory: ' + str(path))
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def check_cache(cache, sequence, protocol, profile):
    meta = json.loads((cache / 'inference_summary.json').read_text())
    if (meta.get('ground_truth_used_as_input') is not False
            or meta.get('raw_pointcloud_detector_tested') is not True
            or meta.get('sequence') != sequence or meta.get('period_s') != .1
            or sha256(cache / 'detections.jsonl') != meta['output_sha256']):
        raise ValueError('label-free inference cache checksum/domain mismatch')
    if profile != 'centerpoint':
        assets = protocol['asset_models'][profile]
        if (meta.get('model_name') != profile or meta.get('upstream_commit') != UPSTREAM_COMMIT
                or meta.get('checkpoint_sha256') != assets.get('sha256')
                or meta.get('source_sha256') != protocol['upstream_source_sha256']
                or meta.get('implementation_sha256') != protocol['source_sha256'][SOURCE_FILES[0]]):
            raise ValueError('official detector cache differs from frozen sources/checkpoint')
    return meta


def evaluate(args, plan, protocol):
    # Imports that can read labels stay confined to this explicit stage.
    from radar_id_tracker.kitti_evaluate import load_truth
    from radar_id_tracker.lidar_detection_curve import DetectionCurve
    from radar_id_tracker.lidar_raw import replay_cache
    from radar_id_tracker.lidar_scoped_evaluate import evaluate_scoped, fixed_recovery, in_scope
    from radar_id_tracker.lidar_visualization import build_review
    from radar_id_tracker.pointcloud_review import prepare_clouds
    config = LidarTrackerConfig.from_dict(plan['tracker_config'])
    profiles = ['centerpoint', *plan['models']]
    caches = {}
    # Require all detector sequences before creating any formal evaluation output.
    for sequence in plan['sequences']:
        reference = portable_path(protocol['baselines'][sequence]['cache'])
        reference_meta = check_cache(reference, sequence, protocol, 'centerpoint')
        for record in reference_meta['input_scans']:
            scan = args.dataset_root / 'training/velodyne' / sequence / f"{record['frame_index']:06d}.bin"
            if sha256(scan) != record['sha256']:
                raise ValueError('raw scan changed since baseline inference')
        caches[sequence, 'centerpoint'] = reference
        for profile in plan['models']:
            cache = args.output_dir / 'raw_cache' / sequence / profile
            meta = check_cache(cache, sequence, protocol, profile)
            if meta['input_scans'] != reference_meta['input_scans']:
                raise ValueError('current scan inventory/hash differs between detectors')
            caches[sequence, profile] = cache
    write_frozen(args.output_dir / 'evaluation_protocol.json', dict(
        protocol_sha256=sha256(args.output_dir / 'protocol.json'),
        tracker_config=asdict(config), labels_sha256={s: sha256(args.dataset_root / 'training/label_02' / f'{s}.txt')
                                                    for s in plan['sequences']},
        official_kitti_score=False, development_only=True, reserved_sequence_opened=False))
    rows, review_rows, curves, recovery, totals = [], [], {}, {}, defaultdict(lambda: defaultdict(int))
    aggregates = {}
    box_keys = {f.name for f in fields(BoxDetection)}
    for sequence in plan['sequences']:
        natural_dirs, dropout_dirs = {}, {}
        dropout_start = protocol['dataset_inventory'][sequence]['dropout_start_frame']
        _, ignored = load_truth(args.dataset_root, sequence)
        for profile in profiles:
            for scenario in ('natural', 'dropout_3s'):
                directory = args.output_dir / 'runs' / sequence / scenario / profile
                if directory.exists() and any(directory.iterdir()):
                    raise ValueError('evaluation output already exists; use a new evaluation directory')
                run = replay_cache(caches[sequence, profile], directory, config,
                                   hide_start=dropout_start if scenario == 'dropout_3s' else None,
                                   hide_frames=30)
                report = evaluate_scoped(args.dataset_root, directory)
                strict = evaluate_scoped(args.dataset_root, directory, .5, write=False)
                (directory / 'evaluation_iou05.json').write_text(json.dumps(strict, indent=2) + '\n')
                row = dict(sequence=sequence, profile=profile, scenario=scenario, run=run,
                           evaluation=report, strict=strict, run_dir=str(directory.resolve()))
                rows.append(row)
                if scenario == 'natural':
                    natural_dirs[profile] = directory
                    for threshold, score in ((.25, report), (.5, strict)):
                        for label, metric in score['per_class_observed'].items():
                            for key in ('tp', 'fp', 'fn', 'idtp', 'idfp', 'idfn', 'id_changes', 'fragmentations'):
                                if key in metric:
                                    totals[f'{profile}/{label}/{threshold}'][key] += metric[key]
                else:
                    dropout_dirs[profile] = directory
                print(json.dumps(dict(sequence=sequence, profile=profile, scenario=scenario,
                                      observed=report['observed_tracks'])), flush=True)
            directory = natural_dirs[profile]
            detections = [json.loads(line) for line in (directory / 'detections.jsonl').read_text().splitlines()]
            details = [json.loads(line) for line in (directory / 'evaluation_frames.jsonl').read_text().splitlines()]
            for threshold in (.25, .5):
                for label in ('Car', 'Pedestrian'):
                    key = f'{profile}/{label}/{threshold}'
                    curve = DetectionCurve(threshold)
                    aggregate = aggregates.setdefault(key, DetectionCurve(threshold))
                    for frame, detail in zip(detections, details):
                        if not detail['scored']:
                            continue
                        inverse = np.linalg.inv(frame['world_from_lidar'])
                        truth = [BoxDetection(**{k: v for k, v in b.items() if k in box_keys})
                                 for b in detail['truth'] if b['label'] == label]
                        proposals = [b for b in (BoxDetection(**d) for d in frame['detections'])
                                     if b.label == label and in_scope(b, inverse)]
                        ignore = [b for b in ignored[frame['frame_index']] if in_scope(b, inverse)]
                        curve.update(truth, proposals, ignore)
                        aggregate.update(truth, proposals, ignore)
                    curves[f'{sequence}/{key}'] = curve.result()
            for candidate in plan['models']:
                if profile in ('centerpoint', candidate):
                    for row in rows[-2:]:
                        review_rows.append(dict(row, scenario=row['scenario'] + '_' + candidate,
                                               profile='baseline' if profile == 'centerpoint' else 'candidate'))
        for profile in plan['models']:
            recovery[f'{sequence}/{profile}'] = fixed_recovery(dropout_dirs['centerpoint'], dropout_dirs[profile])
    (args.output_dir / 'benchmark.json').write_text(json.dumps(rows, indent=2) + '\n')
    (args.output_dir / 'detection_curves.json').write_text(json.dumps(dict(per_sequence=curves,
        aggregate={key: curve.result() for key, curve in aggregates.items()}), indent=2) + '\n')
    (args.output_dir / 'fixed_recovery.json').write_text(json.dumps(recovery, indent=2) + '\n')
    observed = {}
    for key, counts in totals.items():
        ratio = lambda a, b: a / b if b else None
        observed[key] = dict(counts, precision=ratio(counts['tp'], counts['tp'] + counts['fp']),
                             recall=ratio(counts['tp'], counts['tp'] + counts['fn']),
                             idf1=ratio(2 * counts['idtp'], 2 * counts['idtp'] + counts['idfp'] + counts['idfn']))
    (args.output_dir / 'observed_counts.json').write_text(json.dumps(observed, indent=2) + '\n')
    if args.with_pointclouds:
        prepare_clouds(args.dataset_root, args.output_dir, plan['sequences'])
    review = build_review(args.output_dir, review_rows)
    html = review.read_text(encoding='utf-8')
    html = html.replace('左侧为原分数策略，右侧为按类别候选确认策略；两侧身份均保留 4 秒',
                        '左侧 CenterPoint，右侧官方检测器（模型见场景名）；两侧使用同一跟踪配置')
    html = html.replace('原分数策略', 'CenterPoint 检测基线').replace('候选确认策略（实验）', '官方检测器对照')
    review.write_text(html, encoding='utf-8')
    (args.output_dir / 'evaluation_complete.json').write_text(json.dumps(dict(
        records=len(rows), expected_records=len(plan['sequences']) * len(profiles) * 2,
        protocol_sha256=sha256(args.output_dir / 'protocol.json'), development_only=True,
        pretrained_scene_overlap_status='not audited', deployment_accepted=False)) + '\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('prepare', 'infer', 'evaluate'), default='prepare')
    parser.add_argument('--plan', type=Path, default=REPO_ROOT / 'configs/openpcdet_comparison.experimental.json')
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--baseline-manifest', type=Path, required=True)
    parser.add_argument('--cache-root', type=Path, help='relocate legacy absolute paths under a copied outputs directory')
    parser.add_argument('--repo', type=Path, default=REPO_ROOT / 'outputs/vendor/OpenPCDet')
    parser.add_argument('--assets', type=Path, default=REPO_ROOT / 'outputs/models/openpcdet/assets.json')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--with-pointclouds', action='store_true', help='pack original clouds in the evaluation review')
    args = parser.parse_args(argv)
    try:
        plan = load_plan(args.plan)
        assets = json.loads(args.assets.read_text())
        if assets['upstream_commit'] != UPSTREAM_COMMIT:
            raise ValueError('unexpected upstream asset version')
        baseline = baseline_caches(args.baseline_manifest, plan['sequences'], args.cache_root)
        inventory = dataset_inventory(args.dataset_root, plan['sequences'])
        protocol = dict(plan=plan, plan_sha256=sha256(args.plan),
                        source_sha256={name: sha256(REPO_ROOT / name) for name in SOURCE_FILES},
                        asset_models=assets['models'], upstream_source_sha256=assets['source_sha256'],
                        baselines=baseline, dataset_inventory=inventory,
                        ground_truth_used_in_preparation=False, reserved_sequence_opened=False,
                        scene_overlap_audited=False, official_kitti_score=False)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_frozen(args.output_dir / 'protocol.json', protocol)
        if args.stage == 'prepare':
            verified = []
            for model in plan['models']:
                if assets['models'][model]['status'] == 'downloaded':
                    validate_assets(args.repo, args.assets, model)
                    verified.append(model)
            report = dict(status='prepared_only', gpu_inference_run=False, performance_evaluated=False,
                          full_sequences=len(inventory), frames=sum(r['frames'] for r in inventory.values()),
                          raw_scans=sum(len(r['scans']) for r in inventory.values()),
                          asset_status={name: row['status'] for name, row in assets['models'].items()},
                          source_and_checkpoint_checksums_verified=verified,
                          runtime_platform=sys.platform, runtime_ready=False,
                          runtime_note='official Linux CUDA runtime/operator build must be checked before inference',
                          labels_opened=False, reserved_sequence_opened=False)
            (args.output_dir / 'preparation.json').write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps(report, indent=2))
        elif args.stage == 'infer':
            for model in plan['models']:
                detector = OpenPCDetDetector(args.repo, args.assets, model)
                for sequence in plan['sequences']:
                    cache = args.output_dir / 'raw_cache' / sequence / model
                    if (cache / 'inference_summary.json').exists():
                        meta = check_cache(cache, sequence, protocol, model)
                    else:
                        meta = infer_sequence(args.dataset_root, sequence, cache, detector)
                    reference = check_cache(portable_path(baseline[sequence]['cache']), sequence, protocol, 'centerpoint')
                    if meta['input_scans'] != reference['input_scans']:
                        raise ValueError('input scan hash differs from baseline')
                del detector
            (args.output_dir / 'inference_complete.json').write_text(json.dumps(dict(
                records=len(plan['models']) * len(plan['sequences']),
                protocol_sha256=sha256(args.output_dir / 'protocol.json'))) + '\n')
        else:
            evaluate(args, plan, protocol)
    except (OSError, ValueError, RuntimeError, ImportError, KeyError) as error:
        parser.exit(2, f'OpenPCDet comparison error: {error}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
