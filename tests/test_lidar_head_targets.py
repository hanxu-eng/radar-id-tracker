import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from radar_id_tracker.lidar_head_targets import make_targets,front_mask,load_head_manifest
from radar_id_tracker.lidar_tracker import BoxDetection
from radar_id_tracker.lidar_proposal_diagnostics import best_overlap,loss_reason

class HeadTargetTests(unittest.TestCase):
    def test_center_regression_round_trip_and_front_only_supervision(self):
        b=BoxDetection(-2,10,-.5,4,2,1.5,.7,.9,'Car')
        targets=make_targets([b,BoxDetection(0,-10,0,4,2,1.5,0,.9,'Car')])
        self.assertEqual(targets['valid'].sum(),1)
        index=targets['indices'][0,0];y,x=divmod(index,128)
        v=targets['values'][0,0]
        np.testing.assert_allclose((np.array([x,y])+v[:2])*.8-51.2,[b.x,b.y],atol=1e-5)
        np.testing.assert_allclose(np.exp(v[3:6]),[b.length,b.width,b.height],atol=1e-6)
        self.assertAlmostEqual(np.arctan2(v[6],v[7]),b.yaw,places=6)
        self.assertEqual(targets['heatmap'][0,y,x],1)
        self.assertFalse(front_mask()[10,64])

    def test_ignore_masks_and_valid_positive_priority(self):
        b=BoxDetection(0,10,0,.8,.6,1.7,0,.9,'Pedestrian')
        ignored=make_targets([], [b])
        target=make_targets([b],[b]);i=target['indices'][1,0];y,x=divmod(i,128)
        self.assertFalse(ignored['heatmap_mask'][1,y,x])
        self.assertTrue(target['heatmap_mask'][1,y,x])
        self.assertTrue(target['valid'][1,0])

    def test_manifest_rejects_overlap_and_changed_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);weights=root/'heads.pth';weights.write_bytes(b'weights')
            data=dict(format_version=1,adapted_classes=['Car','Pedestrian'],front_only=True,feature_dtype='float16',
                base_checkpoint_sha256='191a38226d2ce990f26cc3f78c454310cc562d81d29e86d8ca8fa39c0a44215d',
                inference_config=dict(intensity_scale=64),training_sequences=['0000'],validation_sequences=['0009'],
                weights_file=weights.name,weights_sha256=hashlib.sha256(weights.read_bytes()).hexdigest())
            path=root/'model.json';path.write_text(json.dumps(data));self.assertEqual(load_head_manifest(path)[1],weights)
            for split in ([],['0009','0009'],['../x'], '0009'):
                invalid=dict(data,validation_sequences=split);path.write_text(json.dumps(invalid))
                with self.assertRaises(ValueError):load_head_manifest(path)
            path.write_text(json.dumps(data))
            weights.write_bytes(b'changed')
            with self.assertRaises(ValueError):load_head_manifest(path)
            data['validation_sequences']=['0000'];path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'disjoint'):load_head_manifest(path)

    def test_loss_partition_and_geometry_prefilter(self):
        b=BoxDetection(0,0,0,4,2,2,0,.9,'Car')
        wrong=BoxDetection(0,0,0,4,2,2,0,.9,'Truck')
        self.assertEqual(best_overlap(b,[wrong]),(0,None))
        self.assertEqual(best_overlap(b,[wrong],False),(1,0))
        cases=[(0,.5,.5,.5,False,'wrong_class_geometry_available'),(0,0,0,0,False,'no_usable_raw_box'),
            (.5,.5,0,0,False,'quality_below_continuation'),(.5,.5,.5,0,False,'quality_below_birth'),
            (.5,.5,.5,.5,False,'confirmation_association_or_filter_geometry'),(0,0,0,0,True,'observed')]
        for *inputs,reason in cases:self.assertEqual(loss_reason(*inputs),reason)

if __name__=='__main__':unittest.main()
