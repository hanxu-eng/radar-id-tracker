"""JSONL replay command for detector or radar target-list output."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .tracker import RadarTracker, TrackerConfig
from .types import Detection


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Track timestamped radar detections")
    parser.add_argument("input", type=Path, help="one JSON frame per line")
    parser.add_argument("--output", type=Path, help="write JSONL output; default stdout")
    parser.add_argument("--high-confidence", type=float, default=0.5)
    parser.add_argument("--low-confidence", type=float, default=0.2)
    args = parser.parse_args(argv)

    tracker = RadarTracker(
        TrackerConfig(
            high_confidence=args.high_confidence,
            low_confidence=args.low_confidence,
        )
    )
    destination = args.output.open("w", encoding="utf-8") if args.output else sys.stdout
    try:
        with args.input.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    frame = json.loads(line)
                    detections = [Detection(**item) for item in frame["detections"]]
                    sensor_xy = tuple(frame.get("sensor_xy", [0.0, 0.0]))
                    outputs = tracker.update(float(frame["timestamp_s"]), detections, sensor_xy)
                    destination.write(
                        json.dumps(
                            {
                                "timestamp_s": float(frame["timestamp_s"]),
                                "tracks": [asdict(output) for output in outputs],
                            },
                            ensure_ascii=False,
                            allow_nan=False,
                        ) + "\n"
                    )
                except (KeyError, TypeError, ValueError, ArithmeticError) as error:
                    parser.exit(2, f"input line {line_number}: {error}\n")
    finally:
        if args.output:
            destination.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
