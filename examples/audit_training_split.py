"""Hash scans across whole-sequence splits without opening any detection labels."""
import argparse
import hashlib
import json
from pathlib import Path
from radar_id_tracker.lidar_training_protocol import load_training_plan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root', type=Path, required=True)
    p.add_argument('--split-plan', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); plan, digest = load_training_plan(args.split_plan)
    owners = {}; overlaps = []; sequences = []; hashes = {}
    for role in ('training', 'validation'):
        for sequence in plan[role + '_sequences']:
            oxts = args.dataset_root / 'training/oxts' / f'{sequence}.txt'
            calib = args.dataset_root / 'training/calib' / f'{sequence}.txt'
            frames = len(oxts.read_text().splitlines()); missing = []; scans = []
            for index in range(frames):
                scan = args.dataset_root / 'training/velodyne' / sequence / f'{index:06d}.bin'
                if not scan.exists(): missing.append(index); continue
                checksum = hashlib.sha256(scan.read_bytes()).hexdigest()
                previous = owners.setdefault(checksum, (role, sequence, index))
                if previous[0] != role:
                    overlaps.append(dict(first=previous, second=(role, sequence, index), sha256=checksum))
                scans.append(dict(frame_index=index, sha256=checksum))
            hashes[sequence] = scans
            sequences.append(dict(sequence=sequence, role=role, frames=frames, scans=len(scans), missing=missing,
                                  oxts_sha256=hashlib.sha256(oxts.read_bytes()).hexdigest(),
                                  calib_sha256=hashlib.sha256(calib.read_bytes()).hexdigest()))
            print(json.dumps(sequences[-1]), flush=True)
    if overlaps: raise ValueError('identical pointclouds cross training/validation split')
    result = dict(split_plan_sha256=digest, ground_truth_labels_opened=False,
                  exact_scan_overlap_count=0, drive_or_route_independence_verified=False,
                  reserved_sequences_not_opened=plan['reserved_sequences'], sequences=sequences, scan_hashes=hashes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and json.loads(args.output.read_text()) != result:
        raise ValueError('frozen split input audit differs')
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(args.output)


if __name__ == '__main__': main()
