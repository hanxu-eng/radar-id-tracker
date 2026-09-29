"""Convert one sequence of frame-level radar detections from CSV to JSONL.

This is a format adapter, not a detector. Coordinates and Doppler must already
follow the contract in README.md. Ground-truth identity columns are rejected.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict
from math import isfinite
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TextIO

from radar_id_tracker import Detection


REQUIRED = {"timestamp_s", "x", "y", "confidence"}
OPTIONAL = {
    "sensor_x", "sensor_y", "radial_velocity", "z", "length", "width",
    "height", "yaw",
}
DETECTION_OPTIONAL = OPTIONAL - {"sensor_x", "sensor_y"}


def _number(row: dict[str, str | None], key: str, line: int) -> float:
    raw = row.get(key)
    if raw is None or not raw.strip():
        raise ValueError(f"CSV line {line}: missing {key}")
    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError(f"CSV line {line}: invalid {key}: {raw!r}") from error
    if not isfinite(value):
        raise ValueError(f"CSV line {line}: {key} must be finite")
    return value


def convert(source: TextIO, destination: TextIO) -> int:
    reader = csv.DictReader(source)
    if not reader.fieldnames:
        raise ValueError("CSV header is missing")
    headers = set(reader.fieldnames)
    missing = REQUIRED - headers
    unknown = headers - REQUIRED - OPTIONAL
    if missing:
        raise ValueError(f"CSV missing columns: {', '.join(sorted(missing))}")
    if unknown:
        raise ValueError(f"CSV unsupported columns: {', '.join(sorted(unknown))}")
    if ("sensor_x" in headers) != ("sensor_y" in headers):
        raise ValueError("sensor_x and sensor_y must appear together")

    timestamp: float | None = None
    sensor_xy: tuple[float, float] | None = None
    detections: list[dict[str, float]] = []
    frames = 0

    def flush() -> None:
        nonlocal frames
        if timestamp is None or sensor_xy is None:
            return
        destination.write(json.dumps({
            "timestamp_s": timestamp,
            "sensor_xy": list(sensor_xy),
            "detections": detections,
        }, allow_nan=False) + "\n")
        frames += 1

    for line, row in enumerate(reader, 2):
        if None in row:
            raise ValueError(f"CSV line {line}: too many columns")
        current_timestamp = _number(row, "timestamp_s", line)
        current_sensor = (
            (_number(row, "sensor_x", line), _number(row, "sensor_y", line))
            if "sensor_x" in headers else (0.0, 0.0)
        )

        if timestamp is None:
            timestamp = current_timestamp
            sensor_xy = current_sensor
        elif current_timestamp != timestamp:
            if current_timestamp < timestamp:
                raise ValueError(f"CSV line {line}: timestamps must increase")
            flush()
            timestamp = current_timestamp
            sensor_xy = current_sensor
            detections = []
        elif current_sensor != sensor_xy:
            raise ValueError(f"CSV line {line}: sensor position differs within a frame")

        x = row.get("x") or ""
        y = row.get("y") or ""
        confidence = row.get("confidence") or ""
        if not (x.strip() or y.strip() or confidence.strip()):
            if any((row.get(key) or "").strip() for key in DETECTION_OPTIONAL):
                raise ValueError(f"CSV line {line}: empty frame has detection fields")
            continue

        values = {key: _number(row, key, line) for key in ("x", "y", "confidence")}
        for key in DETECTION_OPTIONAL:
            if (row.get(key) or "").strip():
                values[key] = _number(row, key, line)
        try:
            detection = Detection(**values)
        except ValueError as error:
            raise ValueError(f"CSV line {line}: {error}") from error
        detections.append({key: value for key, value in asdict(detection).items() if value is not None})

    flush()
    return frames


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="one sequence of sorted detections")
    parser.add_argument("output", type=Path, help="JSONL file for radar-id-track")
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.exit(2, "input and output must be different files\n")
    temporary: Path | None = None
    try:
        with args.input.open("r", encoding="utf-8-sig", newline="") as source:
            with NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=args.output.parent,
                prefix=f".{args.output.name}.", suffix=".tmp", delete=False,
            ) as destination:
                temporary = Path(destination.name)
                frames = convert(source, destination)
        os.replace(temporary, args.output)
    except (OSError, ValueError) as error:
        parser.exit(2, f"{error}\n")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f"converted {frames} frames to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
