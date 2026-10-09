"""Offline data diversity, exact unchanged-proposal parity and HTML asset audit."""
import argparse
from collections import Counter
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import numpy as np
from radar_id_tracker.centerpoint_geometry import KITTI_TO_MODEL
from radar_id_tracker.kitti_adapter import calibration_transforms, load_poses
from radar_id_tracker.kitti_evaluate import load_truth
from radar_id_tracker.lidar_head_targets import front_mask
from radar_id_tracker.lidar_scoped_evaluate import in_scope
from radar_id_tracker.lidar_tracker import BoxDetection


class JsonScripts(HTMLParser):
    def __init__(self):
        super().__init__(); self.active = None; self.data = {}

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if tag == 'script' and values.get('type') == 'application/json':
            self.active = values['id']; self.data[self.active] = ''

    def handle_endtag(self, tag):
        if tag == 'script': self.active = None

    def handle_data(self, data):
        if self.active: self.data[self.active] += data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    args = parser.parse_args(); root = args.run_root
    meta = json.loads((root / 'features/manifest.json').read_text())
    counts = {label: 0 for label in ('Car', 'Pedestrian')}
    objects = {label: set() for label in counts}; mask = front_mask()
    for sequence in meta['training_sequences']:
        truth, _ = load_truth(args.dataset_root, sequence)
        poses = load_poses(args.dataset_root / 'training/oxts' / f'{sequence}.txt')
        _, extrinsic = calibration_transforms(args.dataset_root / 'training/calib' / f'{sequence}.txt')
        frames = {r['frame_index'] for r in meta['records'] if r['sequence'] == sequence}
        for index in frames:
            inverse = np.linalg.inv(poses[index] @ extrinsic)
            for target in truth[index]:
                if not in_scope(target.box, inverse): continue
                box = target.box
                center = KITTI_TO_MODEL @ inverse @ [box.x, box.y, box.z, 1.]
                x, y = np.floor((center[:2] + 51.2) / .8).astype(int)
                if 0 <= x < 128 and 0 <= y < 128 and mask[y, x]:
                    counts[box.label] += 1
                    objects[box.label].add((sequence, target.object_id))
    if counts != meta['targets']: raise ValueError('training target count differs')
    diversity = dict(target_instances=counts, unique_annotated_objects={k: len(v) for k, v in objects.items()},
                     ids_used_only_for_offline_diversity_audit=True)
    (root / 'training_diversity.json').write_text(json.dumps(diversity, indent=2) + '\n')
    rows = json.loads((root / 'benchmark.json').read_text()); parity = []
    for sequence in sorted({r['sequence'] for r in rows}):
        pair = {r['profile']: r for r in rows if r['sequence'] == sequence and r['scenario'] == 'natural'}
        if set(pair) != {'baseline', 'candidate'}: raise ValueError('comparison pair missing')
        sources = {}
        for profile, row in pair.items():
            cache = Path(row['run']['inference_cache']); data = (cache / 'detections.jsonl').read_bytes()
            summary = json.loads((cache / 'inference_summary.json').read_text())
            if summary['output_sha256'] != hashlib.sha256(data).hexdigest(): raise ValueError('cache checksum differs')
            sources[profile] = [json.loads(line) for line in data.splitlines()]
        if len(sources['baseline']) != len(sources['candidate']): raise ValueError('frame counts differ')
        total = 0
        for index, (baseline, candidate) in enumerate(zip(sources['baseline'], sources['candidate'])):
            if baseline['frame_index'] != index or candidate['frame_index'] != index:
                raise ValueError('frame indices differ')
            if baseline['world_from_lidar'] != candidate['world_from_lidar']: raise ValueError('poses differ')
            inverse = np.linalg.inv(baseline['world_from_lidar'])
            def outside(frame):
                return Counter(tuple(sorted(item.items())) for item in frame['detections']
                               if not in_scope(BoxDetection(**item), inverse))
            a, b = outside(baseline), outside(candidate)
            if a != b: raise ValueError(f'outside-scope proposals differ: {sequence}/{index}')
            total += sum(a.values())
        parity.append(dict(sequence=sequence, frames=len(sources['baseline']), unchanged_detection_instances=total))
    preservation = dict(exact_outside_scope_preservation=True, includes_rear_and_other_classes=True,
                        sequences=parity, unchanged_detection_instances=sum(r['unchanged_detection_instances'] for r in parity))
    (root / 'outside_scope_parity.json').write_text(json.dumps(preservation, indent=2) + '\n')
    html = (root / 'review.html').read_text(encoding='utf-8'); reader = JsonScripts(); reader.feed(html)
    cases = json.loads(reader.data['data']); cloud = json.loads(reader.data['cloudMeta'])
    if len(cases) * 2 != len(rows): raise ValueError('HTML comparison count differs')
    frame_count = 0
    for case in cases:
        profiles = case['profiles']
        if set(profiles) != {'baseline', 'candidate'}: raise ValueError('HTML profiles differ')
        if len(profiles['baseline']['frames']) != len(profiles['candidate']['frames']): raise ValueError('HTML frame counts differ')
        frame_count += sum(len(p['frames']) for p in profiles.values())
    cloud_count = 0
    for sequence, frames in cloud['frames'].items():
        for index in frames:
            if not (root / 'pointclouds' / sequence / f'{int(index):06d}.js').is_file(): raise ValueError('pointcloud asset missing')
            cloud_count += 1
    for marker in ('训练序列回放只作诊断', '0009/0013/0017', '未通过部署验收', '前方 ±40°'):
        if marker not in html: raise ValueError('review scope statement missing')
    audit = dict(cases=len(cases), profile_frames=frame_count, pointclouds=cloud_count,
                 missing=cloud['missing_frames'], json_and_assets_valid=True, browser_render_not_tested=True)
    (root / 'review_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(dict(diversity=diversity, preservation=preservation, review=audit)))


if __name__ == '__main__': main()
