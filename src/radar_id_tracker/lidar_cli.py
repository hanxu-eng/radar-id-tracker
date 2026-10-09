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
    p.add_argument('--retention-seconds',type=float,default=4.)
    p.add_argument('--high-confidence',type=float,default=.6)
    p.add_argument('--low-confidence',type=float,default=.2)
    args = p.parse_args(argv)
    if args.output and args.input.resolve() == args.output.resolve():
        p.exit(2,'input and output must be different files\n')
    try:
        tracker = LidarTracker(LidarTrackerConfig(retention_seconds=args.retention_seconds,
                                                 high_confidence=args.high_confidence,
                                                 low_confidence=args.low_confidence))
    except ValueError as error:
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
