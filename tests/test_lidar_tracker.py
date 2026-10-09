from dataclasses import asdict, replace
import math
import unittest

import numpy as np

from radar_id_tracker import BoxDetection, LidarTracker, LidarTrackerConfig
from radar_id_tracker.box_geometry import iou3d
from radar_id_tracker.kitti_adapter import camera_box_to_world
from radar_id_tracker.kitti_evaluate import BoxScore, TruthBox, dropout_recovery


def box(x=0.,y=0.,**changes):
    return replace(BoxDetection(x,y,1.,4.,2.,2.,0.,.99,'Car'),**changes)


class LidarTrackerTests(unittest.TestCase):
    def warmed(self,config=None,moving=False):
        tracker = LidarTracker(config)
        for i in range(20):
            output = tracker.update(i*.1,[box(x=2*i*.1 if moving else 0.)])
        return tracker,output[0].track_id

    def test_three_second_dropout_keeps_moving_and_stopped_id(self):
        for moving in (False,True):
            tracker,identifier = self.warmed(moving=moving)
            for i in range(20,50):
                tracks = tracker.update(i*.1,[])
                if i >= 25:
                    self.assertEqual(tracks,[])
            result = tracker.update(5.,[box(x=10. if moving else 0.)])
            self.assertEqual(result[0].track_id,identifier)
            self.assertFalse(result[0].predicted_only)
            self.assertEqual(tracker.diagnostics['recovered'],1)

    def test_expires_identity_after_retention(self):
        tracker,identifier = self.warmed()
        self.assertEqual(tracker.update(6.1,[box()]),[])
        result = tracker.update(6.2,[box()])
        self.assertNotEqual(result[0].track_id,identifier)

    def test_predictions_decay_and_uncertainty_grows(self):
        tracker,_ = self.warmed()
        a = tracker.update(2.,[])[0]; b = tracker.update(2.2,[])[0]
        self.assertTrue(a.predicted_only)
        self.assertLess(b.confidence,a.confidence)
        self.assertGreater(np.trace(np.array(b.position_covariance).reshape(3,3)),
                           np.trace(np.array(a.position_covariance).reshape(3,3)))
        self.assertAlmostEqual(b.last_observed_timestamp_s,1.9)

    def test_wrong_class_shape_or_elevation_cannot_recover(self):
        for impostor in (box(label='Pedestrian'),box(length=10.),box(z=4.),box(x=20.)):
            tracker,identifier = self.warmed()
            self.assertEqual(tracker.update(5.,[impostor]),[])
            result = tracker.update(5.1,[impostor])
            self.assertNotEqual(result[0].track_id,identifier)

    def test_ambiguous_dormant_targets_are_not_arbitrarily_claimed(self):
        tracker = LidarTracker()
        tracker.update(0.,[box(y=-.3),box(y=.3)])
        old = tracker.update(.1,[box(y=-.3),box(y=.3)])
        self.assertEqual(len(old),2)
        self.assertEqual(tracker.update(3.2,[box()]),[])
        self.assertGreater(tracker.diagnostics['ambiguous_recoveries'],0)

    def test_low_score_continues_but_does_not_spawn(self):
        tracker = LidarTracker(LidarTrackerConfig(high_confidence=.9,low_confidence=.5))
        self.assertEqual(tracker.update(0.,[box(confidence=.6)]),[])
        tracker.update(.1,[box()]); old=tracker.update(.2,[box()])[0]
        result=tracker.update(.3,[box(confidence=.6)])[0]
        self.assertEqual(result.track_id,old.track_id)
        self.assertFalse(result.predicted_only)

    def test_rejected_best_recovery_cannot_be_stolen_by_worse_track(self):
        tracker=LidarTracker()
        tracker.update(0.,[box(x=0.),box(x=8.)])
        tracker.update(.1,[box(x=0.),box(x=8.)])
        # At return, first identity has two almost equally good candidates.
        # The second old track must not steal either by elimination.
        output=tracker.update(3.2,[box(x=1.),box(x=-1.)])
        self.assertEqual(output,[])
        self.assertEqual(tracker.diagnostics['recovered'],0)

    def test_stop_and_restart_and_crossing_use_positions(self):
        tracker=LidarTracker(); initial=None
        for i in range(81):
            t=i*.1; x=min(2*t,4.) if t<4 else 4+2*(t-4)
            # Different shapes make the physical identities separable at crossing.
            ds=[box(x=x),box(x=10-2*t,y=.1,length=1.,width=.6,label='Pedestrian')]
            out=tracker.update(t,ds[::-1] if i%2 else ds)
            ids={o.label:o.track_id for o in out}
            if i==10: initial=ids
            if i>=10: self.assertEqual(ids,initial)

    def test_timestamp_rejection_does_not_mutate_tracker(self):
        tracker,_=self.warmed()
        for t in (1.9,1.,float('nan'),float('inf')):
            with self.assertRaises(ValueError): tracker.update(t,[box()])
        self.assertEqual(tracker.update(2.,[box()])[0].track_id,1)

    def test_long_run_has_bounded_history_and_track_capacity(self):
        tracker=LidarTracker(LidarTrackerConfig(maximum_tracks=2))
        for i in range(1000): tracker.update(i*.1,[box()])
        self.assertEqual(len(tracker._tracks[0].hits),0)
        tracker.update(100.,[box(),box(x=30.),box(x=60.)])
        self.assertEqual(len(tracker._tracks),2)
        self.assertEqual(tracker.diagnostics['capacity_drops'],1)


class GeometryAndEvaluationTests(unittest.TestCase):
    def test_iou_known_volumes_rotation_symmetry(self):
        a=box(); self.assertAlmostEqual(iou3d(a,a),1.)
        self.assertAlmostEqual(iou3d(a,box(x=2.)),1/3)
        self.assertEqual(iou3d(a,box(z=4.)),0.)
        b=box(x=.4,y=.7,yaw=.6)
        self.assertAlmostEqual(iou3d(a,b),iou3d(b,a))
        self.assertAlmostEqual(iou3d(a,box(yaw=math.pi)),1.)

    def test_camera_bottom_center_and_ego_motion(self):
        # Rectified camera x right, y down, z forward -> world x forward,y left,z up.
        transform=np.eye(4); transform[:3,:3]=[[0,0,1],[-1,0,0],[0,-1,0]]
        a=camera_box_to_world(2.,2.,4.,0.,1.,10.,0.,'Car',.9,transform)
        self.assertAlmostEqual(a.x,10.);self.assertAlmostEqual(a.z,0.)
        self.assertAlmostEqual(a.yaw,-math.pi/2)
        transform[0,3]=5.
        b=camera_box_to_world(2.,2.,4.,0.,1.,5.,0.,'Car',.9,transform)
        self.assertAlmostEqual(iou3d(a,b),1.)

    def test_independent_identity_metric_counts_switch(self):
        from types import SimpleNamespace
        truth=[TruthBox('physical',box())]; score=BoxScore()
        for identifier in (1,1,2,2):
            score.update(truth,[SimpleNamespace(**asdict(box()),track_id=identifier)],[],.25)
        result=score.result()
        self.assertEqual(result['id_changes'],1)
        self.assertEqual(result['idf1'],.5)
        self.assertEqual(result['f1'],1.)

    def test_ignore_boxes_do_not_remove_true_matches(self):
        from types import SimpleNamespace
        from dataclasses import asdict
        p=SimpleNamespace(**asdict(box()),track_id=1)
        score=BoxScore();score.update([TruthBox('one',box())],[p],[box()],.25)
        self.assertEqual(score.result()['tp'],1)
        self.assertEqual(score.result()['ignored_outputs'],0)
        score=BoxScore();score.update([],[p],[box()],.25)
        self.assertEqual(score.result()['fp'],0)

    def test_dropout_recovery_denominator_and_latency(self):
        rows=[dict(frame_index=i,observed_tracks=dict(matches={}),
                   birth_eligible_detections=dict(matched_gt_ids=[])) for i in range(40)]
        rows[4]['observed_tracks']['matches']={'a':1,'b':2,'c':3}
        rows[35]['observed_tracks']['matches']={'a':1,'b':9}
        rows[35]['birth_eligible_detections']['matched_gt_ids']=['a','b','c']
        result=dropout_recovery(rows,dict(start_frame=5,frames=30),.1)
        self.assertEqual(result['eligible'],3);self.assertEqual(result['same_id'],1)
        self.assertEqual(result['not_matched_after_return'],1)
        self.assertEqual(result['no_track_after_return'],1)


if __name__ == '__main__': unittest.main()
