import unittest
import numpy as np
from radar_id_tracker.centerpoint_geometry import (voxelize,decorate,circle_nms,
    box_to_world,transform_points,KITTI_TO_MODEL)
from radar_id_tracker.lidar_raw import causal_cloud


class CenterPointGeometryTests(unittest.TestCase):
    def test_first_encounter_caps_and_boundaries(self):
        points=np.array([[1,1,0,10,0],[0,0,0,20,0],[1,1,0,30,0],
                         [0,0,0,40,0],[51.2,0,0,50,0],[-51.2,0,-5,60,0]],np.float32)
        voxels,counts,coors=voxelize(points,max_voxels=2,max_points=1)
        np.testing.assert_array_equal(voxels[:,0,3],[10,20])
        np.testing.assert_array_equal(counts,[1,1])
        np.testing.assert_array_equal(coors[:,2:],[ [261,261],[256,256] ])

    def test_empty_and_reject_corrupt_input(self):
        voxels,counts,coors=voxelize(np.empty((0,5),np.float32))
        self.assertEqual(voxels.shape,(0,20,5))
        self.assertEqual(coors.shape,(0,4))
        with self.assertRaises(ValueError):voxelize(np.array([[np.nan,0,0,0,0]]))

    def test_feature_center_and_padding(self):
        points=np.array([[.02,.04,0,128,0],[.12,.14,1,200,.1]],np.float32)
        v,n,c=voxelize(points)
        f=decorate(v,n,c)
        np.testing.assert_allclose(f[0,:2,5:8],[[-.05,-.05,-.5],[.05,.05,.5]],atol=1e-6)
        np.testing.assert_allclose(f[0,:2,8:11],[[-.08,-.06,1],[.02,.04,2]],atol=1e-5)
        np.testing.assert_array_equal(f[0,2:],0)
        np.testing.assert_array_equal(f[0,:2,:5],points)

    def test_nms_uses_squared_distance_and_includes_boundary(self):
        centers=np.array([[0,0],[2,0],[3,0],[0,4]])
        self.assertEqual(circle_nms(centers,[.9,.8,.7,.6],4),[0,2,3])
        self.assertEqual(circle_nms(centers,[.9,.8,.7,.6],4,limit=1),[0])

    def test_scan_axes_and_gravity_center(self):
        points=np.array([[10,2,-1,.5]],np.float32)
        np.testing.assert_allclose(transform_points(points,KITTI_TO_MODEL),[[-2,10,-1,.5]])
        # Model y-forward maps to KITTI x-forward; height must not move center.
        b=box_to_world([-2,10,-1,4,2,1.5,np.pi/2,0,0],.8,0,KITTI_TO_MODEL.T)
        np.testing.assert_allclose([b.x,b.y,b.z,b.length,b.width,b.height,b.yaw],
                                   [10,2,-1,4,2,1.5,0],atol=1e-7)
        self.assertEqual(box_to_world([0,0,0,2,1,1,0],.9,7,np.eye(4)).label,'Bicycle')

    def test_causal_compensation_and_actual_gaps(self):
        before=np.eye(4);current=np.eye(4);current[0,3]=2
        p=np.array([[10,0,0,.5]],np.float32)
        cloud=causal_cloud([(0,p,before),(2,p,current)],current,2,.1,255.)
        np.testing.assert_allclose(cloud,[[0,10,0,127.5,0],[0,8,0,127.5,.2]],atol=1e-6)
        with self.assertRaises(ValueError):causal_cloud([(3,p,current)],current,2,.1,255)
        invalid=np.eye(4);invalid[0,0]=2
        with self.assertRaises(ValueError):transform_points(p,invalid)


if __name__=='__main__':unittest.main()
