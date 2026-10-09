"""Optional Torch checks; core installation remains NumPy-only."""
import unittest
import numpy as np
try:
    import torch
except ImportError:
    torch=None


@unittest.skipIf(torch is None,'optional PyTorch not installed')
class HeadAdapterTests(unittest.TestCase):
    def test_replacement_preserves_rear_other_classes_and_excludes_cone(self):
        from radar_id_tracker.centerpoint_head_adapter import HeadAdapter
        def head(classes):
            fields={name:torch.zeros(1,n,128,128) for name,n in
                [('heatmap',classes),('reg',2),('height',1),('dim',3),('rot',2),('vel',2)]}
            fields['heatmap'].fill_(-100);fields['heatmap'][0,0,80,64]=5
            if classes==2:fields['heatmap'][0,1].fill_(100)
            fields['dim'][:,0].fill_(np.log(4));fields['dim'][:,1].fill_(np.log(2));fields['dim'][:,2].fill_(np.log(1.5))
            fields['rot'][:,1].fill_(1)
            return lambda x:fields
        adapter=HeadAdapter.__new__(HeadAdapter);adapter.heads=[head(1),head(2)]
        def proposal(y,label):return (np.array([0,y,0,4,2,1.5,0,0,0],np.float32),.8,label)
        original=[proposal(-10,0),proposal(60,0),proposal(10,1),proposal(-10,8),proposal(10,0),proposal(10,8),proposal(10,9)]
        result=adapter.replace_front(original,torch.zeros(1,64,128,128),.1)
        for old in original[:4]+original[6:]:self.assertTrue(any(r is old for r in result))
        self.assertFalse(any(r is original[4] or r is original[5] for r in result))
        self.assertEqual([r[2] for r in result[-2:]],[0,8])
        self.assertEqual(len(result),7)

    def test_training_loss_is_finite_and_ignores_unsupervised_pixels(self):
        import importlib.util
        from pathlib import Path
        from radar_id_tracker.lidar_head_targets import make_targets
        from radar_id_tracker.lidar_tracker import BoxDetection
        spec=importlib.util.spec_from_file_location('train_lidar_heads',Path(__file__).parents[1]/'examples/train_lidar_heads.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        arrays=make_targets([BoxDetection(0,10,0,4,2,1.5,0,1,'Car')])
        targets={k:torch.from_numpy(v[None]) for k,v in arrays.items()}
        head={name:torch.zeros(1,n,128,128,requires_grad=True) for name,n in
            [('heatmap',1),('reg',2),('height',1),('dim',3),('rot',2)]}
        loss=module.loss_for(head,targets,0);loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(head['heatmap'].grad[0,0,0,0]),0)
        i=int(targets['indices'][0,0,0]);y,x=divmod(i,128)
        self.assertNotEqual(float(head['heatmap'].grad[0,0,y,x]),0)

if __name__=='__main__':unittest.main()
