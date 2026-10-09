"""JSONL 3D detection-box replay, independent of ROS and dataset labels."""
from __future__ import annotations
import argparse
import json
import sys
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from .lidar_tracker import BoxDetection,LidarTracker,LidarTrackerConfig


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--config',type=Path,help='JSON LidarTrackerConfig; overrides scalar defaults')
    p.add_argument('--retention-seconds',type=float)
    p.add_argument('--high-confidence',type=float)
    p.add_argument('--low-confidence',type=float)
    args = p.parse_args(argv)
    if args.config and any(v is not None for v in (args.retention_seconds,args.high_confidence,args.low_confidence)):
        p.error('--config cannot be combined with --retention-seconds/--high-confidence/--low-confidence')
    if args.output and args.input.resolve() == args.output.resolve():
        p.exit(2,'input and output must be different files\n')
    try:
        config=(LidarTrackerConfig.from_dict(json.loads(args.config.read_text())) if args.config else
                LidarTrackerConfig(retention_seconds=args.retention_seconds if args.retention_seconds is not None else 4.,
                                  high_confidence=args.high_confidence if args.high_confidence is not None else .6,
                                  low_confidence=args.low_confidence if args.low_confidence is not None else .2))
        tracker=LidarTracker(config)
    except (OSError,ValueError,TypeError) as error:
        p.exit(2,f'invalid tracker configuration: {error}\n')
    temporary = None
    if args.output:
        destination = tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',delete=False,
                                                   dir=args.output.resolve().parent,suffix='.tmp')
        temporary = Path(destination.name)
    else:
        destination = sys.stdout
    try:
        with args.input.open(encoding='utf-8') as source:
            for index,line in enumerate(source,1):
                if not line.strip():
                    continue
                try:
                    frame = json.loads(line)
                    tracks = tracker.update(float(frame['timestamp_s']),
                                            [BoxDetection(**d) for d in frame['detections']])
                    destination.write(json.dumps(dict(timestamp_s=frame['timestamp_s'],
                                                      tracks=[asdict(t) for t in tracks],
                                                      tentative=tracker.tentative_outputs,
                                                      class_diagnostics=tracker.class_diagnostics,
                                                      diagnostics=tracker.diagnostics),allow_nan=False)+'\n')
                except (KeyError,TypeError,ValueError,ArithmeticError) as error:
                    p.exit(2,f'input line {index}: {error}\n')
        if args.output:
            destination.close()
            os.replace(temporary,args.output)
            temporary = None
    except OSError as error:
        p.exit(2,f'file error: {error}\n')
    finally:
        if args.output:
            destination.close()
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
