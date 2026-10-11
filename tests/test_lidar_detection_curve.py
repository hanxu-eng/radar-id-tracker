import unittest
from dataclasses import replace
from radar_id_tracker.lidar_tracker import BoxDetection
from radar_id_tracker.lidar_detection_curve import DetectionCurve


class DetectionCurveTests(unittest.TestCase):
    def box(self,x,score=1.):return BoxDetection(x,0,0,4,2,1.5,0,score,'Pedestrian')

    def test_duplicate_boxes_and_score_order_cost_precision(self):
        score=DetectionCurve();score.update([self.box(10),self.box(20)],
            [self.box(40,.9),self.box(10,.8),self.box(10,.7),self.box(20,.6)])
        result=score.result();self.assertEqual(result['points'][-1]['tp'],2)
        self.assertEqual(result['points'][-1]['fp'],2)
        self.assertAlmostEqual(result['average_precision'],.5)

    def test_tied_scores_are_grouped_and_ignore_does_not_hide_true_match(self):
        score=DetectionCurve();score.update([self.box(10)],
            [self.box(10,.8),self.box(30,.8),self.box(50,.9)],[self.box(10),self.box(50)])
        result=score.result();self.assertEqual(result['predictions'],2)
        self.assertEqual(len(result['points']),1);self.assertAlmostEqual(result['average_precision'],.5)
        self.assertEqual(result['maximum_recall_at_precision_096'],0.)

    def test_empty_truth_and_missing_predictions(self):
        empty=DetectionCurve();self.assertIsNone(empty.result()['average_precision'])
        missing=DetectionCurve();missing.update([self.box(10)],[])
        self.assertEqual(missing.result()['average_precision'],0.)

    def test_monotonic_score_shift_does_not_improve_untruncated_ap(self):
        truth=[self.box(10),self.box(20)]
        proposals=[self.box(40,.09),self.box(10,.08),self.box(10,.07),self.box(20,.06)]
        before=DetectionCurve();after=DetectionCurve()
        before.update(truth,proposals)
        after.update(truth,[replace(b,confidence=.5+.4*b.confidence) for b in proposals])
        self.assertEqual(before.result()['average_precision'],after.result()['average_precision'])


if __name__=='__main__':unittest.main()
