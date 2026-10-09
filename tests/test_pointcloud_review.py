import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
import zlib

import numpy as np

from examples.download_kitti_pointclouds import extract_member
from radar_id_tracker.pointcloud_review import PACKED_DTYPE, pack_scan, read_scan, prepare_clouds


class PointCloudReviewTests(unittest.TestCase):
    def test_world_rotation_and_origin_align_with_world_boxes(self):
        transform=np.eye(4)
        transform[:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]]
        transform[:3,3]=[100,200,300]
        points=np.array([[1,0,0,.5],[2,1,-1,1]],dtype='<f4')
        original=points.copy()
        data,meta=pack_scan(points,transform)
        decoded=np.frombuffer(data,dtype=PACKED_DTYPE)
        np.testing.assert_array_equal(decoded['xyz'],[[0,100,0],[-100,200,-100]])
        # The first point lands at the center of a box at (100,201,300).
        np.testing.assert_allclose(decoded['xyz'][0]*.01+meta['sensor_xyz'],[100,201,300])
        self.assertEqual(decoded['reflectance'].tolist(),[32768,65535])
        np.testing.assert_array_equal(points,original)

    def test_display_quantization_and_invalid_data_are_explicit(self):
        points=np.array([[1.234,-2.345,3.456,.2],[np.nan,0,0,1],[1000,0,0,1],[0,0,0,2]],dtype=float)
        data,meta=pack_scan(points,np.eye(4))
        decoded=np.frombuffer(data,dtype=PACKED_DTYPE)
        self.assertEqual(meta['original_points'],4);self.assertEqual(meta['display_points'],2)
        self.assertEqual(meta['invalid_points'],1);self.assertEqual(meta['outside_coordinate_range'],1)
        self.assertEqual(meta['clipped_reflectance_points'],1)
        self.assertLessEqual(np.max(np.abs(decoded['xyz'][0]*.01-points[0,:3])),.005001)

    def test_invalid_scan_or_non_rigid_pose_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'scan.bin';path.write_bytes(b'bad')
            with self.assertRaises(ValueError):read_scan(path)
        matrix=np.eye(4);matrix[0,0]=2
        with self.assertRaises(ValueError):pack_scan(np.zeros((1,4)),matrix)

    def test_missing_frame_never_reuses_another_scan(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);dataset=root/'dataset';output=root/'out'
            for folder in ('training/oxts','training/calib','training/velodyne/0000'):
                (dataset/folder).mkdir(parents=True)
            (dataset/'training/oxts/0000.txt').write_text(('49 8 100 0 0 0 '+' '.join(['0']*24)+'\n')*2)
            (dataset/'training/calib/0000.txt').write_text(
                'R0_rect: 1 0 0 0 1 0 0 0 1\n'
                'Tr_velo_to_cam: 0 -1 0 0 0 0 -1 0 1 0 0 0\n'
                'Tr_imu_to_velo: 1 0 0 0 0 1 0 0 0 0 1 0\n')
            np.array([[10,0,0,.5]],dtype='<f4').tofile(dataset/'training/velodyne/0000/000000.bin')
            manifest=prepare_clouds(dataset,output,['0000'])
            self.assertEqual(manifest['missing_frames'],[dict(sequence='0000',frame_index=1)])
            self.assertEqual(list(manifest['frames']['0000']),['0'])
            self.assertFalse((output/'pointclouds/0000/000001.js').exists())
            self.assertTrue((output/'pointclouds/0000/000000.js').read_text().startswith('window.registerPointcloud("0000",0,'))

    def test_range_extraction_validates_stored_deflate_and_zip64_payloads(self):
        payload=b'raw scan data'*1000
        for method in (zipfile.ZIP_STORED,zipfile.ZIP_DEFLATED):
            buffer=io.BytesIO()
            with zipfile.ZipFile(buffer,'w',compression=method) as archive:
                with archive.open('training/velodyne/0000/000000.bin','w',force_zip64=True) as target:
                    target.write(payload)
            content=buffer.getvalue()
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                info=archive.infolist()[0]
            self.assertEqual(extract_member(info,content,0),payload)
            corrupt=bytearray(content);corrupt[content.index(b'raw scan data') if method==zipfile.ZIP_STORED else 100]^=1
            with self.assertRaises((ValueError,zlib.error)):
                extract_member(info,bytes(corrupt),0)


if __name__=='__main__':unittest.main()
