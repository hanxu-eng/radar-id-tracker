"""Label-free raw-scan inference with pinned official OpenPCDet KITTI models."""
import argparse
import json
from pathlib import Path

from radar_id_tracker.openpcdet_adapter import MODEL_SPECS, OpenPCDetDetector, infer_sequence


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--sequence', required=True)
    parser.add_argument('--model', choices=MODEL_SPECS, required=True)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        detector = OpenPCDetDetector(args.repo, args.assets, args.model)
        summary = infer_sequence(args.dataset_root, args.sequence, args.output_dir, detector)
    except (OSError, ValueError, RuntimeError, ImportError, KeyError) as error:
        parser.exit(2, f'OpenPCDet inference unavailable: {error}\n')
    print(json.dumps({k: v for k, v in summary.items() if k not in ('input_scans', 'source_sha256')}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
