"""Exercise the official RadarScenes reader against a small format-faithful fixture."""

from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
import unittest
from base64 import b64decode
from collections import deque
from dataclasses import replace
from pathlib import Path
from xml.etree import ElementTree

import h5py
import numpy as np

from radar_id_tracker.radarscenes_adapter import (
    ClusterConfig,
    cluster_scan,
    find_sensors_json,
    replay,
    sensor_position,
)
from radar_id_tracker.radarscenes_visualization import render_bev_svg
from radar_id_tracker.radarscenes_evaluate import evaluate
from radar_id_tracker.types import TrackOutput


RADAR_DTYPE = np.dtype([
    ("timestamp", "i8"), ("sensor_id", "i4"),
    ("x_seq", "f8"), ("y_seq", "f8"), ("vr_compensated", "f8"),
    ("label_id", "i4"), ("track_id", "S16"),
])
ODOMETRY_DTYPE = np.dtype([
    ("timestamp", "i8"), ("x_seq", "f8"), ("y_seq", "f8"),
    ("yaw_seq", "f8"),
])
TINY_PNG = b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/Z5kAAAAASUVORK5CYII="
)


class RadarScenesAdapterTests(unittest.TestCase):
    def test_clustering_ignores_ground_truth_labels(self):
        points = np.array([
            (0, 1, 10.0, 2.0, 2.0, 0, b"real-id-a"),
            (0, 1, 10.2, 2.1, 2.1, 0, b"real-id-a"),
            (0, 1, 20.0, 1.0, 0.0, 11, b""),
        ], dtype=RADAR_DTYPE)
        altered = points.copy()
        altered["label_id"] = [11, 3, 0]
        altered["track_id"] = [b"changed", b"different", b"another"]
        config = ClusterConfig()
        self.assertEqual(cluster_scan(points, config), cluster_scan(altered, config))
        detections = cluster_scan(points, config)
        self.assertEqual(len(detections), 1)
        self.assertAlmostEqual(detections[0].x, 10.1)
        self.assertAlmostEqual(detections[0].radial_velocity, 2.05)

    def test_sensor_mounting_transform(self):
        odometry = np.array([(0, 10.0, 20.0, math.pi / 2)], dtype=ODOMETRY_DTYPE)[0]
        x, y = sensor_position(odometry, {"x": 2.0, "y": 1.0})
        self.assertAlmostEqual(x, 9.0)
        self.assertAlmostEqual(y, 22.0)

    def test_official_reader_replay_keeps_real_intervals_and_one_sensor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sequence_dir = root / "data" / "sequence_001"
            sequence_dir.mkdir(parents=True)
            camera_dir = sequence_dir / "camera"
            camera_dir.mkdir()
            sensors_path = root / "sensors.json"
            sensors_path.write_text(json.dumps({
                "radar_1": {"x": 1.0, "y": 0.0, "yaw": 0.0},
                "radar_2": {"x": 0.0, "y": 1.0, "yaw": 0.0},
            }), encoding="utf-8")

            radar = np.array([
                (1_000_000, 1, 10.0, 2.0, 2.0, 0, b"truth-1"),
                (1_000_000, 1, 10.2, 2.1, 2.1, 0, b"truth-1"),
                (1_030_000, 2, 50.0, 1.0, 2.0, 0, b"truth-2"),
                (1_060_000, 1, 10.12, 2.0, 2.0, 0, b"truth-1"),
                (1_060_000, 1, 10.32, 2.1, 2.1, 0, b"truth-1"),
                (1_120_000, 1, 10.24, 2.0, 2.0, 0, b"truth-1"),
                (1_120_000, 1, 10.44, 2.1, 2.1, 0, b"truth-1"),
            ], dtype=RADAR_DTYPE)
            odometry = np.array([
                (1_000_000, 0.0, 0.0, 0.0),
                (1_030_000, 0.0, 0.0, 0.0),
                (1_060_000, 0.0, 0.0, 0.0),
                (1_120_000, 0.0, 0.0, 0.0),
            ], dtype=ODOMETRY_DTYPE)
            with h5py.File(sequence_dir / "radar_data.h5", "w") as source:
                source.create_dataset("radar_data", data=radar)
                source.create_dataset("odometry", data=odometry)

            scenes = {}
            for index, (timestamp, sensor, begin, end) in enumerate([
                (1_000_000, 1, 0, 2),
                (1_030_000, 2, 2, 3),
                (1_060_000, 1, 3, 5),
                (1_120_000, 1, 5, 7),
            ]):
                image_timestamp = 1_065_000 if timestamp == 1_120_000 else timestamp + 5_000
                scenes[str(timestamp)] = {
                    "sensor_id": sensor,
                    "radar_indices": [begin, end],
                    "odometry_index": index,
                    "odometry_timestamp": timestamp,
                    "image_name": f"{image_timestamp}.png",
                }
                (camera_dir / f"{image_timestamp}.png").write_bytes(TINY_PNG)
            scenes_path = sequence_dir / "scenes.json"
            scenes_path.write_text(json.dumps({
                "sequence_name": "sequence_001",
                "first_timestamp": 1_000_000,
                "last_timestamp": 1_120_000,
                "scenes": scenes,
            }), encoding="utf-8")

            self.assertEqual(find_sensors_json(scenes_path, None), sensors_path)
            output_dir = root / "output"
            summary = replay(scenes_path, sensors_path, output_dir, 1, ClusterConfig(),
                             visualize=True, with_camera=True)
            self.assertEqual(summary["frames"], 3)
            self.assertEqual(summary["radar_points"], 6)
            self.assertEqual(summary["cluster_detections"], 3)
            self.assertEqual(summary["unique_published_ids"], 1)
            self.assertAlmostEqual(summary["median_interval_s"], 0.06)
            self.assertFalse(summary["ground_truth_used_as_input"])
            self.assertEqual(summary["visualization_frames"], 3)
            self.assertEqual(summary["camera_images_copied"], 2)
            self.assertEqual(summary["median_abs_camera_offset_ms"], 5.0)
            self.assertEqual(summary["max_abs_camera_offset_ms"], 55.0)
            self.assertTrue(summary["camera_alignment_is_nearest_only"])
            self.assertTrue((output_dir / "visualization.html").is_file())
            html = (output_dir / "visualization.html").read_text(encoding="utf-8")
            self.assertIn('[[0.0, "frames/frame_000000.svg", "camera/camera_000000.png", 5.0]', html)
            self.assertIn('id="camera"', html)
            self.assertIn('frames[index + 1][0] - frames[index][0]', html)
            self.assertEqual((output_dir / "camera" / "camera_000000.png").read_bytes(), TINY_PNG)
            for frame_index in range(3):
                svg_path = output_dir / "frames" / f"frame_{frame_index:06d}.svg"
                root_element = ElementTree.parse(svg_path).getroot()
                markers = root_element.findall('.//*[@data-track-id]')
                self.assertEqual(len(markers), 0 if frame_index == 0 else 1)
                if markers:
                    self.assertEqual(markers[0].attrib['data-track-id'], '1')
                    self.assertEqual(markers[0].attrib['data-predicted-only'], 'false')

            detections = [json.loads(line) for line in
                          (output_dir / "detections.jsonl").read_text(encoding="utf-8").splitlines()]
            tracks = [json.loads(line) for line in
                      (output_dir / "tracks.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual([frame["timestamp_s"] for frame in detections], [0.0, 0.06, 0.12])
            self.assertEqual(detections[0]["sensor_xy"], [1.0, 0.0])
            self.assertEqual(tracks[0]["tracks"], [])
            self.assertEqual(tracks[1]["tracks"][0]["track_id"], 1)
            self.assertEqual(tracks[2]["tracks"][0]["track_id"], 1)

            evaluation = evaluate(scenes_path, output_dir, gate_m=2.0)
            self.assertEqual(evaluation["frames"], 3)
            self.assertEqual(evaluation["detections"]["f1"], 1.0)
            self.assertEqual(evaluation["observed_tracks"]["tp"], 2)
            self.assertEqual(evaluation["observed_tracks"]["fn"], 1)
            self.assertAlmostEqual(evaluation["observed_tracks"]["idf1"], 0.8)
            self.assertEqual(len((output_dir / "evaluation_frames.jsonl").read_text(
                encoding="utf-8").splitlines()), 3)

            eval_cli = subprocess.run(
                [sys.executable, "-m", "radar_id_tracker.radarscenes_evaluate",
                 "--scenes", str(scenes_path), "--run-dir", str(output_dir)],
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(eval_cli.returncode, 0, eval_cli.stderr)
            self.assertEqual(json.loads(eval_cli.stdout)["observed_tracks"]["fn"], 1)

            predicted = replace(TrackOutput(**tracks[1]["tracks"][0]),
                                predicted_only=True)
            svg = render_bev_svg(
                radar[3:5], [], [predicted],
                {1: deque([(0.06, predicted.x, predicted.y, True)])},
                (1.0, 0.0), 0.06, "sequence_001", 1, 60.0,
            )
            marker = ElementTree.fromstring(svg).find('.//*[@data-track-id]')
            self.assertEqual(marker.attrib['data-predicted-only'], 'true')
            self.assertIn('ID 1 (P)', svg)

            cli_output = root / "cli-output"
            result = subprocess.run(
                [
                    sys.executable, "-m", "radar_id_tracker.radarscenes_adapter",
                    "--scenes", str(scenes_path), "--sensor-id", "1",
                    "--output-dir", str(cli_output), "--limit", "2",
                    "--visualize", "--with-camera",
                ],
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["frames"], 2)
            self.assertTrue((cli_output / "visualization.html").is_file())
            self.assertEqual(json.loads(result.stdout)["camera_images_copied"], 2)
            self.assertEqual(
                len((cli_output / "tracks.jsonl").read_text(encoding="utf-8").splitlines()),
                2,
            )

            plain_output = root / "plain-output"
            plain_summary = replay(scenes_path, sensors_path, plain_output, 1,
                                   ClusterConfig())
            self.assertNotIn("visualization_file", plain_summary)
            self.assertFalse((plain_output / "visualization.html").exists())
            for filename in ("detections.jsonl", "tracks.jsonl"):
                self.assertEqual((plain_output / filename).read_bytes(),
                                 (output_dir / filename).read_bytes())

            radar_only_output = root / "radar-only-output"
            replay(scenes_path, sensors_path, radar_only_output, 1,
                   ClusterConfig(), limit=1, visualize=True)
            radar_only_html = (radar_only_output / "visualization.html").read_text(encoding="utf-8")
            self.assertNotIn('id="camera"', radar_only_html)
            self.assertFalse((radar_only_output / "camera").exists())

            with self.assertRaisesRegex(ValueError, "requires --visualize"):
                replay(scenes_path, sensors_path, root / "invalid", 1,
                       ClusterConfig(), with_camera=True)

            with self.assertRaisesRegex(ValueError, "no scans"):
                replay(scenes_path, sensors_path, root / "no-sensor", 3, ClusterConfig())

            (camera_dir / "1005000.png").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "camera image missing"):
                replay(scenes_path, sensors_path, root / "missing-camera", 1,
                       ClusterConfig(), visualize=True, with_camera=True)
            self.assertFalse((root / "missing-camera").exists())

            tracks[1]["timestamp_s"] = 0.5
            (output_dir / "tracks.jsonl").write_text(
                "\n".join(json.dumps(row) for row in tracks) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "timestamp mismatch"):
                evaluate(scenes_path, output_dir)


if __name__ == "__main__":
    unittest.main()
