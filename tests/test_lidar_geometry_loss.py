import unittest
try:
    import torch
except ImportError:
    torch=None


@unittest.skipIf(torch is None,'optional PyTorch training dependency absent')
class GeometryLossTests(unittest.TestCase):
    def fixture(self):
        values=torch.tensor([0.,0.,0.,0.,0.,0.,0.,1.]).reshape(1,1,8)
        target=dict(indices=torch.zeros((1,2,1),dtype=torch.int64),values=torch.stack([values,values],dim=1),valid=torch.ones((1,2,1),dtype=torch.bool))
        head={name:torch.zeros((1,n,1,1),requires_grad=True) for name,n in [('reg',2),('height',1),('dim',3),('rot',2)]}
        head['rot']=torch.tensor([0.,1.]).reshape(1,2,1,1).requires_grad_()
        return head,target

    def test_pi_axis_equivalence_without_accepting_quarter_turn(self):
        from radar_id_tracker.lidar_geometry_loss import pedestrian_geometry_loss
        head,target=self.fixture();head['rot']=torch.tensor([0.,-1.]).reshape(1,2,1,1)
        self.assertEqual(float(pedestrian_geometry_loss(head,target,'weighted_axis_huber')),0.)
        self.assertGreater(float(pedestrian_geometry_loss(head,target,'continued_l1')),0.)
        head['rot']=torch.tensor([1.,0.]).reshape(1,2,1,1)
        self.assertGreater(float(pedestrian_geometry_loss(head,target,'weighted_axis_huber')),0.)

    def test_invalid_targets_produce_no_geometry_gradient(self):
        from radar_id_tracker.lidar_geometry_loss import pedestrian_geometry_loss
        head,target=self.fixture();target['valid'][:,1]=False
        for variant in ('continued_l1','weighted_axis_huber'):
            loss=pedestrian_geometry_loss(head,target,variant);loss.backward()
            self.assertEqual(float(loss),0.)
            for value in head.values():
                self.assertTrue(torch.isfinite(value.grad).all());self.assertEqual(float(value.grad.abs().sum()),0.)
                value.grad.zero_()

    def test_continued_control_matches_existing_regression(self):
        from radar_id_tracker.lidar_geometry_loss import pedestrian_geometry_loss
        head,target=self.fixture();head['reg']=torch.tensor([.2,.3]).reshape(1,2,1,1)
        self.assertAlmostEqual(float(pedestrian_geometry_loss(head,target,'continued_l1')),.125,places=6)

    def test_robust_loss_has_finite_descent_direction_and_dimension_weight(self):
        from radar_id_tracker.lidar_geometry_loss import pedestrian_geometry_loss
        head,target=self.fixture();head['dim']=torch.tensor([.4,0.,0.]).reshape(1,3,1,1).requires_grad_()
        loss=pedestrian_geometry_loss(head,target,'weighted_axis_huber');loss.backward()
        self.assertGreater(float(head['dim'].grad[0,0,0,0]),0.)
        self.assertAlmostEqual(float(head['dim'].grad[0,0,0,0]),.5,places=6)
        self.assertTrue(torch.isfinite(loss))


if __name__=='__main__':unittest.main()
