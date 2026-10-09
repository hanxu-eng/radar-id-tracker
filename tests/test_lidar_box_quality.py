import unittest
import numpy as np
from radar_id_tracker.lidar_box_features import BoxFeatureIndex,FEATURE_NAMES
from radar_id_tracker.lidar_box_quality import BoxQualityModel
from radar_id_tracker.lidar_tracker import BoxDetection


class BoxQualityTests(unittest.TestCase):
    def test_features_are_rigid_transform_invariant_and_ignore_intensity(self):
        rng=np.random.default_rng(17);points=np.column_stack([rng.uniform([-2,-1,-1],[2,1,1],(300,3)),rng.random(300)])
        box=BoxDetection(0,0,0,4,2,2,0,.3,'Car')
        reference=BoxFeatureIndex(points,np.eye(4)).extract(box)
        pose=np.eye(4);a=.71;pose[:2,:2]=[[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]];pose[:3,3]=[101,-23,10]
        moved=BoxDetection(101,-23,10,4,2,2,a,.3,'Car')
        changed=points.copy();changed[:,3]=1000
        np.testing.assert_allclose(BoxFeatureIndex(changed,pose).extract(moved),reference,atol=1e-10)
        self.assertAlmostEqual(reference[6],np.log1p(300))
        flipped=BoxDetection(0,0,0,4,2,2,np.pi,.3,'Car')
        np.testing.assert_allclose(BoxFeatureIndex(points,np.eye(4)).extract(flipped),reference,atol=1e-10)

    def test_empty_box_support_stays_finite_and_rejects_bad_scan(self):
        box=BoxDetection(10,0,0,4,2,2,0,.3,'Car')
        f=BoxFeatureIndex(np.empty((0,4)),np.eye(4)).extract(box)
        self.assertEqual(f[6],0);self.assertTrue(np.isfinite(f).all())
        with self.assertRaises(ValueError):BoxFeatureIndex(np.array([[np.nan,0,0,0]]),np.eye(4))

    def test_sparse_queries_match_brute_force_support(self):
        rng=np.random.default_rng(28)
        points=np.column_stack([rng.uniform(-15,15,(5000,3)),np.zeros(5000)])
        # Include exact box/context boundaries and negative integer grid cells.
        points=np.vstack([points,[-3,-1,0,0],[-4,-1,0,0],[3,1,0,0]])
        index=BoxFeatureIndex(points,np.eye(4))
        for yaw in (0,.73,np.pi/2):
            for x,y in ((-2,-1),(1.1,-3.2),(100,0)):
                box=BoxDetection(x,y,0,4,2,2,yaw,.3,'Car')
                local=(points[:,:3]-[x,y,0])@np.array([
                    [np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]])/[2,1,1]
                inside=(np.abs(local)<=1+1e-10).all(1)
                context=(np.abs(local)<=1.5+1e-10).all(1)
                f=index.extract(box)
                self.assertAlmostEqual(f[6],np.log1p(inside.sum()))
                self.assertAlmostEqual(f[7],np.log1p(context.sum()-inside.sum()))

    def model(self):
        tree=[[0,.5,1,2,0,0],[0,0,0,0,1,-2],[0,0,0,0,1,2]]
        return dict(format_version=1,feature_version=1,features=list(FEATURE_NAMES),models={
            label:dict(intercept=0,trees=[tree]) for label in ('Car','Pedestrian')})

    def test_tree_threshold_and_forward_edge_validation(self):
        model=BoxQualityModel(self.model());x=np.zeros((2,len(FEATURE_NAMES)));x[:,0]=[.5,.6]
        np.testing.assert_allclose(model.predict('Car',x),[1/(1+np.exp(2)),1/(1+np.exp(-2))])
        data=self.model();data['models']['Car']['trees'][0][0][2]=0
        with self.assertRaises(ValueError):BoxQualityModel(data)

    def test_rescore_preserves_geometry_other_classes_and_original(self):
        model=BoxQualityModel(self.model());index=BoxFeatureIndex(np.empty((0,4)),np.eye(4))
        boxes=[BoxDetection(10,0,0,4,2,2,0,.2,label) for label in ('Car','Pedestrian','Truck')]
        result,details=model.rescore(index,boxes)
        self.assertEqual(result[2],boxes[2])
        self.assertEqual(boxes[0].confidence,.2)
        self.assertEqual(result[0].x,boxes[0].x)
        self.assertEqual([d['detection_index'] for d in details],[0,1])
        self.assertTrue(result[0].confidence<.2)

    def test_detector_domain_mismatch_is_explicit(self):
        data=self.model();data.update(checkpoint_sha256='fixed',inference_config=dict(intensity_scale=64))
        model=BoxQualityModel(data)
        model.validate_detector_inputs('fixed',dict(intensity_scale=64))
        with self.assertRaises(ValueError):model.validate_detector_inputs('fixed',dict(intensity_scale=255))


if __name__=='__main__':unittest.main()
