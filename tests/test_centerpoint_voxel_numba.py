import importlib.util
import unittest
import numpy as np
from radar_id_tracker.centerpoint_geometry import voxelize


@unittest.skipUnless(importlib.util.find_spec('numba'),'optional numba inference environment required')
class NumbaVoxelTests(unittest.TestCase):
    def test_random_caps_boundaries_and_empty_are_exact(self):
        from radar_id_tracker.centerpoint_voxel_numba import voxelize_numba
        rng=np.random.default_rng(827)
        points=rng.uniform([-60,-60,-6,0,0],[60,60,4,255,.5],size=(30000,5)).astype(np.float32)
        points=np.concatenate([points,points[:3000],
            np.array([[-51.2,-51.2,-5,0,0],[51.2,0,0,0,0],[0,0,3,0,0]],np.float32)])
        for cloud in (points,np.empty((0,5),np.float32)):
            for maximum,pcount in ((40000,20),(100,1)):
                expected=voxelize(cloud,maximum,pcount)
                actual=voxelize_numba(cloud,maximum,pcount)
                for a,b in zip(actual,expected):np.testing.assert_array_equal(a,b)


if __name__=='__main__':unittest.main()
