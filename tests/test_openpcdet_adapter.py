from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from contextlib import redirect_stdout

import numpy as np

from radar_id_tracker.openpcdet_adapter import (
    UPSTREAM_COMMIT, UPSTREAM_URL, predictions_to_world, infer_sequence, validate_assets,
)
from radar_id_tracker.lidar_raw import replay_cache
from radar_id_tracker.lidar_tracker import LidarTrackerConfig
from examples.benchmark_openpcdet import load_plan, write_frozen, dataset_inventory


class OpenPCDetAdapterTests(unittest.TestCase):
    def test_native_center_dimensions_yaw_and_one_based_classes(self):
        transform = np.array([[0, -1, 0, 100], [1, 0, 0, 200],
                              [0, 0, 1, 3], [0, 0, 0, 1]], dtype=float)
        boxes = np.array([[10, 2, 1, 4, 2, 1.5, 0],
                          [20, 0, .2, .8, .6, 1.7, np.pi / 2],
                          [25, 0, 1, 1.7, .6, 1.7, 0]])
        result = predictions_to_world(boxes, [.8, .3, .6], [1, 2, 3], transform)
        self.assertEqual([d.label for d in result], ['Car', 'Pedestrian', 'Cyclist'])
        np.testing.assert_allclose([result[0].x, result[0].y, result[0].z], [98, 210, 4])
        self.assertEqual((result[0].length, result[0].width, result[0].height), (4, 2, 1.5))
        self.assertAlmostEqual(result[0].yaw, np.pi / 2)
        self.assertAlmostEqual(abs(result[1].yaw), np.pi)
        self.assertEqual(result[1].confidence, .3)
        self.assertAlmostEqual(result[1].z, 3.2)  # no bottom-center or CenterPoint-axis conversion

    def test_empty_predictions_and_malformed_outputs_fail_closed(self):
        self.assertEqual(predictions_to_world(np.empty((0, 7)), [], [], np.eye(4)), [])
        box = [10, 0, 0, .8, .6, 1.7, 0]
        for boxes, scores, labels in (([box], [.5], [0]), ([box], [.5], [2.5]),
                                     ([box], [1.1], [2]), ([box], [np.nan], [2]),
                                     ([box[:3] + [0] + box[4:]], [.5], [2]),
                                     ([box], [], [2])):
            with self.subTest(labels=labels, scores=scores), self.assertRaises(ValueError):
                predictions_to_world(boxes, scores, labels, np.eye(4))
        reflected = np.eye(4); reflected[0, 0] = -1
        with self.assertRaises(ValueError):
            predictions_to_world([box], [.5], [2], reflected)

    def test_raw_inference_is_label_free_single_scan_and_preserves_missing_frames(self):
        class FakeDetector:
            def __init__(self): self.inputs = []
            def provenance(self): return dict(model_name='second_iou')
            def synchronize(self): pass
            def predict(self, points):
                self.inputs.append(points.copy())
                return np.array([[10, 0, 0, .8, .6, 1.7, 0]]), np.array([.9]), np.array([2])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scan_dir = root / 'training/velodyne/0000'; scan_dir.mkdir(parents=True)
            scans = [np.array([[10, 0, -1, .25]], '<f4'), np.array([[20, 1, -1, .75]], '<f4')]
            scans[0].tofile(scan_dir / '000000.bin'); scans[1].tofile(scan_dir / '000002.bin')
            poses = [np.eye(4) for _ in range(3)]; poses[2][0, 3] = 5
            detector = FakeDetector()
            original_read = Path.read_text
            def guarded_read(path, *args, **kwargs):
                if 'label_02' in path.parts: raise AssertionError('annotations accessed')
                return original_read(path, *args, **kwargs)
            prefix = 'radar_id_tracker.openpcdet_adapter.'
            with patch(prefix + 'load_poses', return_value=poses), \
                 patch(prefix + 'calibration_transforms', return_value=(np.eye(4), np.eye(4))), \
                 patch.object(Path, 'read_text', guarded_read), redirect_stdout(io.StringIO()):
                summary = infer_sequence(root, '0000', root / 'cache', detector)
                replay_cache(root / 'cache', root / 'replay', LidarTrackerConfig())
            self.assertEqual(summary['missing_scan_frames'], [1])
            self.assertFalse(summary['ground_truth_used_as_input'])
            self.assertEqual(summary['sweeps'], 1)
            self.assertEqual(len(detector.inputs), 3)  # warm-up, frame 0, frame 2
            np.testing.assert_array_equal(detector.inputs[-1], scans[1])
            rows = [json.loads(x) for x in (root / 'cache/detections.jsonl').read_text().splitlines()]
            self.assertEqual(rows[1]['detections'], [])
            self.assertEqual(rows[2]['detections'][0]['x'], 15)
            self.assertEqual(rows[2]['sweeps_used'], [2])
            with self.assertRaisesRegex(ValueError, 'new or empty'):
                infer_sequence(root, '0000', root / 'cache', detector)

    def test_corrupt_scan_never_produces_completion_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); scans = root / 'training/velodyne/0000'; scans.mkdir(parents=True)
            (scans / '000000.bin').write_bytes(b'broken')
            with patch('radar_id_tracker.openpcdet_adapter.load_poses', return_value=[np.eye(4)]), \
                 patch('radar_id_tracker.openpcdet_adapter.calibration_transforms', return_value=(np.eye(4), np.eye(4))):
                detector = SimpleNamespace(provenance=lambda: {'model_name': 'second_iou'}, predict=lambda p: None)
                with self.assertRaises(ValueError): infer_sequence(root, '0000', root / 'cache', detector)
            self.assertFalse((root / 'cache/inference_summary.json').exists())

    def test_asset_hash_mismatch_is_rejected_before_deserialization(self):
        from radar_id_tracker.openpcdet_adapter import MODEL_SPECS
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); weights = root / 'weights.pth'; weights.write_bytes(b'fixture, never loaded')
            manifest = root / 'assets.json'
            record = dict(filename=weights.name, status='downloaded', sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
                          drive_id=MODEL_SPECS['second_iou']['drive_id'])
            manifest.write_text(json.dumps(dict(upstream_commit=UPSTREAM_COMMIT, upstream_url=UPSTREAM_URL,
                                                source_sha256={}, models={'second_iou': record})))
            with patch('radar_id_tracker.openpcdet_adapter.subprocess.check_output', return_value=UPSTREAM_COMMIT + '\n'):
                self.assertEqual(validate_assets(root, manifest, 'second_iou')[0], weights)
                weights.write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'checkpoint checksum'):
                    validate_assets(root, manifest, 'second_iou')

    def test_whole_sequence_plan_reserves_final_sequence_and_freezes_artifacts(self):
        path = Path(__file__).parents[1] / 'configs/openpcdet_comparison.experimental.json'
        plan = load_plan(path)
        self.assertEqual(len(plan['sequences']), 17)
        self.assertNotIn('0020', plan['sequences'])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'plan.json'
            for sequences in (plan['sequences'][:1], [*plan['sequences'], '0020']):
                output.write_text(json.dumps(dict(plan, sequences=sequences)))
                with self.assertRaises(ValueError): load_plan(output)
            artifact = Path(directory) / 'protocol.json'
            write_frozen(artifact, {'threshold': .7})
            with self.assertRaisesRegex(ValueError, 'frozen artifact changed'):
                write_frozen(artifact, {'threshold': .2})


if __name__ == '__main__':
    unittest.main()
