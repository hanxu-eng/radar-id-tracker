import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from radar_id_tracker.kitti_adapter import calibration_transforms, load_sequence, replay
from radar_id_tracker.kitti_evaluate import evaluate
from radar_id_tracker.lidar_cli import main as lidar_cli


class KittiAdapterTests(unittest.TestCase):
    def fixture(self,root):
        for name in ('training/calib','training/oxts','training/label_02',
                     'detections/Car','detections/Pedestrian','detections/Cyclist'):
            (root/name).mkdir(parents=True)
        # x_cam=-y_velo, y_cam=-z_velo, z_cam=x_velo. Identity IMU/LiDAR.
        (root/'training/calib/0000.txt').write_text(
            'R0_rect: 1 0 0 0 1 0 0 0 1\n'
            'Tr_velo_to_cam: 0 -1 0 0 0 0 -1 0 1 0 0 0\n'
            'Tr_imu_to_velo: 1 0 0 0 0 1 0 0 0 0 1 0\n')
        packet='49 8 100 0 0 0 '+' '.join(['0']*24)+'\n'
        (root/'training/oxts/0000.txt').write_text(packet*3)
        for label in ('Pedestrian','Cyclist'):
            (root/f'detections/{label}/0000.txt').write_text('')
        (root/'detections/Car/0000.txt').write_text(
            '0,2,0,0,10,10,8,2,2,4,0,1,10,-1.5707963267948966,0\n'
            '1,2,0,0,10,10,8,2,2,4,0,1,10,-1.5707963267948966,0\n')
        (root/'training/label_02/0000.txt').write_text(''.join(
            f'{i} 42 Car 0 0 0 0 0 10 10 2 2 4 0 1 10 -1.5707963267948966\n'
            for i in range(3)))

    def test_real_format_empty_frame_and_label_isolation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);self.fixture(root)
            frames,_=load_sequence(root,'0000')
            self.assertEqual(len(frames),3); self.assertEqual(frames[-1].detections,())
            self.assertAlmostEqual(frames[1].detections[0].x,10.)
            self.assertAlmostEqual(frames[1].detections[0].z,0.)
            replay(root,'0000',root/'first')
            score=evaluate(root,root/'first')
            self.assertEqual(score['observed_tracks']['tp'],1)
            self.assertEqual(score['published_tracks']['tp'],2)
            # Malformed/deleted GT is never opened by replay. IDs are identical.
            (root/'training/label_02/0000.txt').write_text('invalid and private identity\n')
            replay(root,'0000',root/'second')
            self.assertEqual((root/'first/tracks.jsonl').read_bytes(),(root/'second/tracks.jsonl').read_bytes())
            (root/'training/label_02/0000.txt').unlink()
            replay(root,'0000',root/'third')
            self.assertEqual((root/'first/tracks.jsonl').read_bytes(),(root/'third/tracks.jsonl').read_bytes())

    def test_saved_timestamp_mismatch_fails_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);self.fixture(root);replay(root,'0000',root/'out')
            path=root/'out/tracks.jsonl';rows=[json.loads(line) for line in path.read_text().splitlines()]
            rows[1]['timestamp_s']=.11
            path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with self.assertRaises(ValueError):evaluate(root,root/'out')

    def test_invalid_detector_category_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);self.fixture(root)
            path=root/'detections/Car/0000.txt';path.write_text(path.read_text().replace(',2,0,0',',1,0,0'))
            with self.assertRaises(ValueError):load_sequence(root,'0000')

    def test_calibration_with_vehicle_translation_rotation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);self.fixture(root)
            path=root/'training/calib/0000.txt'
            path.write_text(path.read_text().replace('Tr_imu_to_velo: 1 0 0 0 0 1 0 0 0 0 1 0',
                                                    'Tr_imu_to_velo: 0 -1 0 2 1 0 0 3 0 0 1 4'))
            camera_to_imu,velo_to_imu=calibration_transforms(path)
            np.testing.assert_allclose(camera_to_imu@np.array([0,0,10,1]),[ -3,-8,-4,1])
            np.testing.assert_allclose(velo_to_imu@np.array([10,0,0,1]),[-3,-8,-4,1])

    def test_official_tracking_calibration_without_colons(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);self.fixture(root)
            path=root/'training/calib/0000.txt'
            original=calibration_transforms(path)
            official=path.read_text().replace('R0_rect:', 'R_rect').replace(
                'Tr_velo_to_cam:', 'Tr_velo_cam').replace('Tr_imu_to_velo:', 'Tr_imu_velo')
            path.write_text(official+'\n')
            for expected,actual in zip(original,calibration_transforms(path)):
                np.testing.assert_allclose(actual,expected)
            path.write_text(official+'R0_rect: 0 0 0 0 0 0 0 0 0\n')
            with self.assertRaisesRegex(ValueError,'conflicting calibration'):
                calibration_transforms(path)

    def test_external_predictions_are_separate_from_official_dataset(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);self.fixture(root)
            (root/'detections').rename(root/'published_predictions')
            predictions=root/'published_predictions'
            with self.assertRaises(FileNotFoundError):load_sequence(root,'0000')
            summary=replay(root,'0000',root/'out',detections_root=predictions)
            self.assertEqual(summary['input_detection_count'],2)
            self.assertEqual(summary['detections_root'],str(predictions.resolve()))
            self.assertFalse(summary['ground_truth_used_as_input'])
            self.assertEqual(evaluate(root,root/'out')['observed_tracks']['tp'],1)

    def test_cli_preserves_files_on_bad_input_and_rejects_truth_id(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);self.fixture(root);replay(root,'0000',root/'out')
            source=root/'out/detections.jsonl'; output=root/'result.jsonl'
            self.assertEqual(lidar_cli([str(source),'--output',str(output)]),0)
            old=output.read_bytes()
            rows=[json.loads(line) for line in source.read_text().splitlines()]
            rows[1]['detections'][0]['track_id']='GT 42'
            source.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            with self.assertRaises(SystemExit):lidar_cli([str(source),'--output',str(output)])
            self.assertEqual(output.read_bytes(),old)
            with self.assertRaises(SystemExit):lidar_cli([str(source),'--output',str(source)])
            self.assertFalse(list(root.glob('*.tmp')))


if __name__ == '__main__': unittest.main()
