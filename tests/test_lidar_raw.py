from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout,redirect_stderr
import io
import numpy as np
from radar_id_tracker.lidar_raw import replay_cache,main
from radar_id_tracker.lidar_scoped_evaluate import in_scope,fixed_recovery
from radar_id_tracker.lidar_tracker import BoxDetection,LidarTrackerConfig


class RawLidarTests(unittest.TestCase):
    def test_cli_replays_frozen_inference_settings_and_rejects_silent_override(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'input.json'
            path.write_text(json.dumps(dict(sweeps=3,intensity_scale=64,period_s=.2,score_threshold=.15)))
            argv=['--dataset-root',directory,'--sequence','0000','--output-dir',directory,
                  '--checkpoint','unused.pth','--inference-config',str(path)]
            with patch('radar_id_tracker.lidar_raw.infer_sequence',return_value={}) as infer,redirect_stdout(io.StringIO()):
                self.assertEqual(main(argv),0)
            self.assertEqual(infer.call_args.kwargs['intensity_scale'],64)
            self.assertEqual(infer.call_args.kwargs['sweeps'],3)
            self.assertEqual(infer.call_args.kwargs['period_s'],.2)
            with patch('radar_id_tracker.lidar_raw.infer_sequence') as infer,redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):main(argv+['--intensity-scale','255'])
                infer.assert_not_called()

    def test_common_scope_follows_current_sensor_and_excludes_other_classes(self):
        inverse=np.eye(4);inverse[0,3]=-100
        make=lambda x,y,label='Car':BoxDetection(x,y,0,4,2,1.5,0,.8,label)
        self.assertTrue(in_scope(make(110,0),inverse))
        self.assertFalse(in_scope(make(90,0),inverse))
        self.assertFalse(in_scope(make(151,0),inverse))
        self.assertFalse(in_scope(make(110,10),inverse))
        self.assertFalse(in_scope(make(110,0,'Bicycle'),inverse))

    def test_cache_missing_scan_and_dropout_do_not_reuse_stale_detections(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cache=root/'cache';cache.mkdir()
            b=asdict(BoxDetection(10,0,0,4,2,1.5,0,.8,'Car'))
            rows=[dict(frame_index=i,timestamp_s=i*.1,sensor_xyz=[0,0,0],
                       raw_scan_available=i!=4,synthetic_detector_dropout=False,
                       detections=[b] if i!=4 else []) for i in range(8)]
            data=''.join(json.dumps(row)+'\n' for row in rows).encode()
            (cache/'detections.jsonl').write_bytes(data)
            meta=dict(sequence='0000',frames=8,period_s=.1,output_sha256=hashlib.sha256(data).hexdigest(),
                      ground_truth_used_as_input=False,raw_pointcloud_detector_tested=True)
            (cache/'inference_summary.json').write_text(json.dumps(meta))
            result=replay_cache(cache,root/'run',LidarTrackerConfig(),hide_start=2,hide_frames=2)
            ds=[json.loads(line) for line in (root/'run/detections.jsonl').read_text().splitlines()]
            self.assertFalse(ds[4]['raw_scan_available'])
            for i in (2,3,4):self.assertEqual(ds[i]['detections'],[])
            self.assertEqual(result['hidden_interval']['duration_s'],.2)
            (cache/'detections.jsonl').write_bytes(data+b'\n')
            with self.assertRaisesRegex(ValueError,'checksum'):replay_cache(cache,root/'bad',LidarTrackerConfig())
            self.assertFalse((root/'bad').exists())

    def test_missing_pretrack_stays_in_fixed_recovery_denominator(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);reference=root/'baseline';candidate=root/'candidate'
            reference.mkdir();candidate.mkdir()
            (reference/'evaluation.json').write_text(json.dumps(dict(dropout_recovery=dict(
                targets=[dict(gt_id='11',same_id=True),dict(gt_id='31',same_id=True)]))))
            (candidate/'summary.json').write_text(json.dumps(dict(period_s=.1,
                hidden_interval=dict(start_frame=2,frames=30))))
            rows=[dict(observed_tracks=dict(matches={'11':5} if i<2 or i>=32 else {})) for i in range(38)]
            (candidate/'evaluation_frames.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            result=fixed_recovery(reference,candidate)
            self.assertEqual(result['eligible'],2)
            self.assertEqual(result['baseline_same_id'],2)
            self.assertEqual(result['candidate_same_id'],1)
            self.assertEqual(result['candidate_no_pretrack'],1)

    def test_same_directory_is_rejected_before_read(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            with self.assertRaisesRegex(ValueError,'directories must differ'):
                replay_cache(path,path,LidarTrackerConfig())


if __name__=='__main__':unittest.main()
