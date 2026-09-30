"""RadarScenes single-sensor replay with a label-free clustering baseline.

This is an integration smoke test, not a validated object detector. Ground-truth
``track_id`` and ``label_id`` are never read when producing detections.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from math import cos, isfinite, sin
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np

from .tracker import RadarTracker
from .types import Detection
from .radarscenes_visualization import RadarScenesVisualizer


@dataclass(frozen=True, slots=True)
class ClusterConfig:
    min_abs_vr_mps: float = 0.5
    spatial_eps_m: float = 2.0
    doppler_eps_mps: float = 1.5
    min_points: int = 2
    confidence: float = 0.75
    doppler_sign: str = "as-is"

    def __post_init__(self) -> None:
        if not isfinite(self.min_abs_vr_mps) or self.min_abs_vr_mps < 0:
            raise ValueError("min_abs_vr_mps must be finite and nonnegative")
        if not isfinite(self.spatial_eps_m) or self.spatial_eps_m <= 0:
            raise ValueError("spatial_eps_m must be positive and finite")
        if not isfinite(self.doppler_eps_mps) or self.doppler_eps_mps <= 0:
            raise ValueError("doppler_eps_mps must be positive and finite")
        if self.min_points < 1:
            raise ValueError("min_points must be at least 1")
        if not isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be within [0, 1]")
        if self.doppler_sign not in {"as-is", "invert", "off"}:
            raise ValueError("doppler_sign must be as-is, invert, or off")


def cluster_scan(radar_data: np.ndarray, config: ClusterConfig) -> list[Detection]:
    """Cluster one scan without consulting RadarScenes annotation fields."""
    names = set(radar_data.dtype.names or ())
    required = {"x_seq", "y_seq", "vr_compensated"}
    if not required <= names:
        raise ValueError(f"radar_data missing fields: {', '.join(sorted(required - names))}")
    if len(radar_data) == 0:
        return []

    x = np.asarray(radar_data["x_seq"], dtype=np.float64)
    y = np.asarray(radar_data["y_seq"], dtype=np.float64)
    velocity = np.asarray(radar_data["vr_compensated"], dtype=np.float64)
    keep = (
        np.isfinite(x) & np.isfinite(y) & np.isfinite(velocity)
        & (np.abs(velocity) >= config.min_abs_vr_mps)
    )
    points = np.column_stack((x[keep], y[keep], velocity[keep]))
    count = len(points)
    if count < config.min_points:
        return []

    parent = list(range(count))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    spatial_gate_squared = config.spatial_eps_m**2
    for left in range(count):
        for right in range(left + 1, count):
            dx = points[left, 0] - points[right, 0]
            dy = points[left, 1] - points[right, 1]
            if (dx * dx + dy * dy <= spatial_gate_squared
                    and abs(points[left, 2] - points[right, 2]) <= config.doppler_eps_mps):
                parent[root(right)] = root(left)

    groups: dict[int, list[int]] = {}
    for index in range(count):
        groups.setdefault(root(index), []).append(index)

    detections: list[Detection] = []
    for indices in groups.values():
        if len(indices) < config.min_points:
            continue
        cluster = points[indices]
        radial = float(np.median(cluster[:, 2]))
        if config.doppler_sign == "invert":
            radial = -radial
        detections.append(Detection(
            x=float(np.median(cluster[:, 0])),
            y=float(np.median(cluster[:, 1])),
            confidence=config.confidence,
            radial_velocity=None if config.doppler_sign == "off" else radial,
        ))
    return sorted(detections, key=lambda item: (item.x, item.y))


def sensor_position(odometry: np.void, mounting: dict[str, float]) -> tuple[float, float]:
    """Transform the radar mounting location from car to sequence coordinates."""
    for key in ("x_seq", "y_seq", "yaw_seq"):
        if key not in (odometry.dtype.names or ()):
            raise ValueError(f"odometry missing field: {key}")
    x_car = float(odometry["x_seq"])
    y_car = float(odometry["y_seq"])
    yaw = float(odometry["yaw_seq"])
    x_mount = float(mounting["x"])
    y_mount = float(mounting["y"])
    if not all(isfinite(value) for value in (x_car, y_car, yaw, x_mount, y_mount)):
        raise ValueError("odometry or mounting contains non-finite coordinates")
    return (
        x_car + cos(yaw) * x_mount - sin(yaw) * y_mount,
        y_car + sin(yaw) * x_mount + cos(yaw) * y_mount,
    )


def find_sensors_json(scenes_path: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        if not explicit.is_file():
            raise FileNotFoundError(f"sensor mounting file not found: {explicit}")
        return explicit
    for directory in scenes_path.parents:
        for name in ("sensors.json", "sensor.json"):
            candidate = directory / name
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(
        "sensor mounting file not found; pass --sensors /path/to/sensors.json"
    )


def replay(
    scenes_path: Path,
    sensors_path: Path,
    output_dir: Path,
    sensor_id: int,
    config: ClusterConfig,
    limit: int | None = None,
    visualize: bool = False,
    vis_range_m: float = 60.0,
    trail_seconds: float = 2.0,
) -> dict[str, object]:
    """Read one RadarScenes sequence and write detections/tracks/diagnostics."""
    try:
        from radar_scenes.sensors import get_mounting
        from radar_scenes.sequence import Sequence
    except ImportError as error:
        raise RuntimeError(
            "RadarScenes reader missing; run: conda install -c conda-forge h5py "
            "and python -m pip install radar_scenes --no-deps"
        ) from error

    if sensor_id not in (1, 2, 3, 4):
        raise ValueError("sensor_id must be 1, 2, 3, or 4")
    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1")
    if visualize and (not isfinite(vis_range_m) or vis_range_m <= 0):
        raise ValueError("vis_range_m must be positive and finite")
    if visualize and (not isfinite(trail_seconds) or trail_seconds <= 0):
        raise ValueError("trail_seconds must be positive and finite")
    if not scenes_path.is_file():
        raise FileNotFoundError(f"scenes.json not found: {scenes_path}")
    if not (scenes_path.parent / "radar_data.h5").is_file():
        raise FileNotFoundError(f"radar_data.h5 not found beside: {scenes_path}")

    sequence = Sequence.from_json(str(scenes_path))
    timestamps = sorted(int(value) for value in sequence.timestamps)
    selected: list[int] = []
    available: set[int] = set()
    for timestamp in timestamps:
        scene = sequence.get_scene(timestamp)
        if scene is None:
            raise ValueError(f"scene unavailable at timestamp {timestamp}")
        available.add(scene.sensor_id)
        if scene.sensor_id == sensor_id:
            selected.append(timestamp)
    if not selected:
        raise ValueError(f"sensor {sensor_id} has no scans in this sequence; available: {sorted(available)}")
    mounting = get_mounting(sensor_id, json_path=str(sensors_path))
    if limit is not None:
        selected = selected[:limit]

    output_dir.mkdir(parents=True, exist_ok=True)
    detections_path = output_dir / "detections.jsonl"
    tracks_path = output_dir / "tracks.jsonl"
    summary_path = output_dir / "summary.json"
    tracker = RadarTracker()
    visualizer = (RadarScenesVisualizer(output_dir, sequence.sequence_name, sensor_id,
                                        vis_range_m, trail_seconds) if visualize else None)
    base_timestamp_us = selected[0]
    frame_times: list[float] = []
    processing_ms: list[float] = []
    point_count = detection_count = published_count = 0
    unique_ids: set[int] = set()

    with detections_path.open("w", encoding="utf-8") as detection_file:
        with tracks_path.open("w", encoding="utf-8") as track_file:
            for timestamp_us in selected:
                scene = sequence.get_scene(timestamp_us)
                assert scene is not None
                timestamp_s = (timestamp_us - base_timestamp_us) / 1_000_000.0
                frame_times.append(timestamp_s)
                start = perf_counter()
                detections = cluster_scan(scene.radar_data, config)
                sensor_xy = sensor_position(scene.odometry_data, mounting)
                tracks = tracker.update(timestamp_s, detections, sensor_xy)
                processing_ms.append((perf_counter() - start) * 1000.0)
                point_count += len(scene.radar_data)
                detection_count += len(detections)
                published_count += len(tracks)
                unique_ids.update(track.track_id for track in tracks)

                detection_file.write(json.dumps({
                    "timestamp_s": timestamp_s,
                    "sensor_xy": list(sensor_xy),
                    "detections": [asdict(item) for item in detections],
                }, allow_nan=False) + "\n")
                track_file.write(json.dumps({
                    "timestamp_s": timestamp_s,
                    "tracks": [asdict(item) for item in tracks],
                }, allow_nan=False) + "\n")
                if visualizer is not None:
                    visualizer.add_frame(scene.radar_data, detections, tracks,
                                         sensor_xy, timestamp_s)

    visualization_path = visualizer.finish() if visualizer is not None else None

    intervals = [right - left for left, right in zip(frame_times, frame_times[1:])]
    summary: dict[str, object] = {
        "sequence": sequence.sequence_name,
        "sensor_id": sensor_id,
        "scenes_file": str(scenes_path),
        "sensors_file": str(sensors_path),
        "frames": len(selected),
        "radar_points": point_count,
        "cluster_detections": detection_count,
        "published_track_outputs": published_count,
        "unique_published_ids": len(unique_ids),
        "median_interval_s": median(intervals) if intervals else None,
        "median_scan_rate_hz": 1.0 / median(intervals) if intervals else None,
        "median_cluster_and_track_ms": median(processing_ms),
        "ground_truth_used_as_input": False,
        "detection_method": "single_scan_spatial_doppler_connected_components",
        "confidence_is_calibrated": False,
        "validated_detection_or_tracking_accuracy": False,
        "config": asdict(config),
    }
    if visualization_path is not None:
        summary["visualization_file"] = str(visualization_path)
        summary["visualization_frames"] = len(visualizer.frames)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", required=True, type=Path, help="path to one sequence's scenes.json")
    parser.add_argument("--sensors", type=Path, help="path to sensors.json; auto-discovered by default")
    parser.add_argument("--sensor-id", required=True, type=int, choices=(1, 2, 3, 4))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--limit", type=int, help="first N scans of the selected sensor")
    parser.add_argument("--min-abs-vr", type=float, default=0.5)
    parser.add_argument("--spatial-eps", type=float, default=2.0)
    parser.add_argument("--doppler-eps", type=float, default=1.5)
    parser.add_argument("--min-points", type=int, default=2)
    parser.add_argument("--confidence", type=float, default=0.75)
    parser.add_argument("--doppler-sign", choices=("as-is", "invert", "off"), default="as-is")
    parser.add_argument("--visualize", action="store_true", help="write a local HTML replay and SVG BEV frames")
    parser.add_argument("--vis-range-m", type=float, default=60.0, help="BEV half-width/height in meters")
    parser.add_argument("--trail-seconds", type=float, default=2.0, help="visible track-history duration")
    args = parser.parse_args(argv)
    try:
        config = ClusterConfig(
            min_abs_vr_mps=args.min_abs_vr,
            spatial_eps_m=args.spatial_eps,
            doppler_eps_mps=args.doppler_eps,
            min_points=args.min_points,
            confidence=args.confidence,
            doppler_sign=args.doppler_sign,
        )
        sensors_path = find_sensors_json(args.scenes, args.sensors)
        summary = replay(args.scenes, sensors_path, args.output_dir, args.sensor_id, config,
                         args.limit, args.visualize, args.vis_range_m, args.trail_seconds)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.exit(2, f"RadarScenes input error: {error}\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
