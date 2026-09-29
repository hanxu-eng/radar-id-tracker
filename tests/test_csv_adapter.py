"""Smoke and validation tests for the detector-output CSV adapter."""

from __future__ import annotations

import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "examples" / "csv_to_jsonl.py"
spec = importlib.util.spec_from_file_location("csv_to_jsonl", SCRIPT)
assert spec is not None and spec.loader is not None
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


class CsvAdapterTests(unittest.TestCase):
    def test_multiple_detections_and_empty_frame(self):
        source = io.StringIO(
            "timestamp_s,x,y,confidence,radial_velocity\n"
            "0.0,10,2,0.9,3\n"
            "0.0,12,2,0.8,-1\n"
            "0.1,,, ,\n"
            "0.2,10.6,2,0.9,3\n"
        )
        output = io.StringIO()
        self.assertEqual(adapter.convert(source, output), 3)
        frames = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual([len(frame["detections"]) for frame in frames], [2, 0, 1])
        self.assertEqual(frames[0]["detections"][0]["radial_velocity"], 3.0)
        self.assertEqual(frames[1]["timestamp_s"], 0.1)

    def test_ground_truth_identity_rejected(self):
        source = io.StringIO("timestamp_s,x,y,confidence,track_id\n0,1,2,0.9,42\n")
        with self.assertRaisesRegex(ValueError, "unsupported columns: track_id"):
            adapter.convert(source, io.StringIO())

    def test_timestamp_and_sensor_validation(self):
        bad_timestamp = io.StringIO(
            "timestamp_s,x,y,confidence\n0.1,1,2,0.9\n0.0,1,2,0.9\n"
        )
        with self.assertRaisesRegex(ValueError, "timestamps must increase"):
            adapter.convert(bad_timestamp, io.StringIO())

        bad_sensor = io.StringIO(
            "timestamp_s,sensor_x,sensor_y,x,y,confidence\n"
            "0.0,0,0,1,2,0.9\n0.0,1,0,2,2,0.9\n"
        )
        with self.assertRaisesRegex(ValueError, "sensor position differs"):
            adapter.convert(bad_sensor, io.StringIO())

    def test_cli_rejects_same_input_and_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            original = "timestamp_s,x,y,confidence\n0,1,2,0.9\n"
            path.write_text(original, encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(SCRIPT), str(path), str(path)],
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(result.returncode, 2)
            self.assertEqual(path.read_text(encoding="utf-8"), original)

    def test_cli_does_not_replace_output_on_bad_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "bad.csv"
            output = Path(directory) / "tracks.jsonl"
            source.write_text("timestamp_s,x,y,confidence\n0,1,2,2.0\n", encoding="utf-8")
            output.write_text("previous result\n", encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(SCRIPT), str(source), str(output)],
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(result.returncode, 2)
            self.assertEqual(output.read_text(encoding="utf-8"), "previous result\n")
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
