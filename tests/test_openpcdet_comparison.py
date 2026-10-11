"""Small known-answer fixtures exercise the complete *offline* evaluation path.

These synthetic cache fixtures never exercise a learned model or count as model
performance evidence. Real GPU inference remains an independent pending step.
"""
from contextlib import redirect_stdout
from dataclasses import asdict
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from examples.benchmark_openpcdet import evaluate, load_plan, SOURCE_FILES
from radar_id_tracker.openpcdet_adapter import UPSTREAM_COMMIT, sha256
from radar_id_tracker.lidar_tracker import BoxDetection


class OpenPCDetComparisonTests(unittest.TestCase):
    def test_complete_comparison_has_strict_curves_fixed_recovery_and_both_review_models(self):
        repo = Path(__file__).parents[1]
        plan = load_plan(repo / 'configs/openpcdet_comparison.experimental.json')
        # Keep the evaluator fixture small; plan loading above validates all 17
        # production sequences and the production CLI cannot select just one.
        plan = dict(plan, sequences=['0000'])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); dataset = root / 'dataset'; output = root / 'comparison'
            output.mkdir()
            for name in ('calib', 'oxts', 'label_02', 'velodyne/0000'):
                (dataset / 'training' / name).mkdir(parents=True)
            (dataset / 'training/calib/0000.txt').write_text(
                'R_rect 1 0 0 0 1 0 0 0 1\n'
                'Tr_velo_cam 0 -1 0 0 0 0 -1 0 1 0 0 0\n'
                'Tr_imu_velo 1 0 0 0 0 1 0 0 0 0 1 0\n')
            (dataset / 'training/oxts/0000.txt').write_text(('49 8 100 ' + ' '.join(['0'] * 27) + '\n') * 84)
            label = lambda i: f'{i} 1 Pedestrian 0 0 0 0 0 50 100 1.7 0.6 0.8 0 0.85 10 -1.5707963267948966\n'
            (dataset / 'training/label_02/0000.txt').write_text(''.join(label(i) for i in range(84)))
            inputs = []
            for index in range(84):
                scan = dataset / 'training/velodyne/0000' / f'{index:06d}.bin'
                np.array([[10, 0, 0, .5]], '<f4').tofile(scan)
                inputs.append(dict(frame_index=index, bytes=16, sha256=sha256(scan)))
            adapter_hash = sha256(repo / SOURCE_FILES[0])
            protocol = dict(baselines={}, asset_models={m: {'sha256': 'synthetic-fixture'} for m in plan['models']},
                            upstream_source_sha256={}, source_sha256={SOURCE_FILES[0]: adapter_hash},
                            dataset_inventory={'0000': {'dropout_start_frame': 50}})
            for model in ('centerpoint', *plan['models']):
                cache = root / 'baseline' if model == 'centerpoint' else output / 'raw_cache/0000' / model
                cache.mkdir(parents=True)
                proposal = asdict(BoxDetection(10, 0, 0, .8, .6, 1.7, 0, .9, 'Pedestrian'))
                rows = [dict(frame_index=i, timestamp_s=i * .1, sensor_xyz=[0, 0, 0],
                             world_from_lidar=np.eye(4).tolist(), raw_scan_available=True,
                             synthetic_detector_dropout=False, detections=[proposal]) for i in range(84)]
                (cache / 'detections.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
                meta = dict(sequence='0000', frames=84, period_s=.1, ground_truth_used_as_input=False,
                            raw_pointcloud_detector_tested=True, input_scans=inputs,
                            output_sha256=sha256(cache / 'detections.jsonl'), model_name=model,
                            upstream_commit=UPSTREAM_COMMIT, checkpoint_sha256='synthetic-fixture',
                            source_sha256={}, implementation_sha256=adapter_hash)
                (cache / 'inference_summary.json').write_text(json.dumps(meta))
                if model == 'centerpoint': protocol['baselines']['0000'] = {'cache': str(cache)}
            (output / 'protocol.json').write_text(json.dumps(protocol))
            args = SimpleNamespace(dataset_root=dataset, output_dir=output, with_pointclouds=False)
            with redirect_stdout(io.StringIO()): evaluate(args, plan, protocol)
            complete = json.loads((output / 'evaluation_complete.json').read_text())
            self.assertEqual(complete['records'], 6)
            self.assertFalse(complete['deployment_accepted'])
            curves = json.loads((output / 'detection_curves.json').read_text())['aggregate']
            for model in ('centerpoint', *plan['models']):
                self.assertAlmostEqual(curves[f'{model}/Pedestrian/0.5']['average_precision'], 1.)
                self.assertEqual(curves[f'{model}/Pedestrian/0.5']['gt'], 84)
            counts = json.loads((output / 'observed_counts.json').read_text())
            self.assertEqual(counts['pv_rcnn/Pedestrian/0.5']['id_changes'], 0)
            self.assertGreater(counts['pv_rcnn/Pedestrian/0.5']['idf1'], .95)
            recoveries = json.loads((output / 'fixed_recovery.json').read_text())
            for result in recoveries.values():
                self.assertEqual(result['eligible'], 1)
                self.assertEqual(result['candidate_same_id'], 1)
            html = (output / 'review.html').read_text(encoding='utf-8')
            self.assertIn('natural_second_iou', html)
            self.assertIn('natural_pv_rcnn', html)
            self.assertIn('官方检测器对照', html)


if __name__ == '__main__':
    unittest.main()
