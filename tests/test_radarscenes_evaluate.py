"""Known-answer checks for the offline RadarScenes scoring definitions."""

from __future__ import annotations

import unittest

import numpy as np

from radar_id_tracker.radarscenes_evaluate import (
    EvalFrame,
    Proposal,
    TruthObject,
    score_frames,
    truth_objects,
)


def frame(index: int, gt: tuple[TruthObject, ...],
          detections: tuple[Proposal, ...],
          tracks: tuple[Proposal, ...]) -> EvalFrame:
    return EvalFrame(1_000_000 + index * 100_000, index * 0.1,
                     gt, detections, tracks)


class RadarScenesEvaluationTests(unittest.TestCase):
    def test_perfect_detection_and_persistent_identity(self):
        frames = [
            frame(i, (TruthObject("A", float(i), 0.0, 2),),
                  (Proposal(float(i), 0.0),),
                  (Proposal(float(i), 0.0, track_id=1),))
            for i in range(2)
        ]
        report, details = score_frames(frames, 1.0, min_points=2)
        self.assertEqual(report["detections"]["f1"], 1.0)
        self.assertEqual(report["observed_tracks"]["idf1"], 1.0)
        self.assertEqual(report["observed_tracks"]["id_changes"], 0)
        self.assertEqual(report["gt_below_cluster_min_points"], 0)
        self.assertEqual(details[0]["observed_tracks"]["matches"][0]["gt_id"], "A")

    def test_switch_costs_identity_even_when_centers_are_perfect(self):
        frames = [
            frame(i, (TruthObject("A", float(i), 0.0, 2),),
                  (Proposal(float(i), 0.0),),
                  (Proposal(float(i), 0.0, track_id=i + 1),))
            for i in range(2)
        ]
        report, details = score_frames(frames, 1.0)
        self.assertEqual(report["observed_tracks"]["f1"], 1.0)
        self.assertEqual(report["observed_tracks"]["idf1"], 0.5)
        self.assertEqual(report["observed_tracks"]["id_changes"], 1)
        self.assertEqual(details[1]["observed_tracks"]["id_change_gt_ids"], ["A"])

    def test_miss_fragment_and_predicted_only_are_reported_separately(self):
        gt = (TruthObject("A", 10.0, 0.0, 1),)
        frames = [
            frame(0, gt, (Proposal(10.0, 0.0),), (Proposal(10.0, 0.0, 1),)),
            frame(1, gt, (), (Proposal(10.0, 0.0, 1, True),)),
            frame(2, gt, (Proposal(10.0, 0.0),), (Proposal(10.0, 0.0, 1),)),
        ]
        report, details = score_frames(frames, 1.0, min_points=2)
        self.assertEqual(report["gt_single_point_occurrences"], 3)
        self.assertEqual(report["gt_below_cluster_min_points"], 3)
        self.assertEqual(report["predicted_only_track_outputs"], 1)
        self.assertEqual(report["observed_tracks"]["fn"], 1)
        self.assertEqual(report["observed_tracks"]["fragmentations"], 1)
        self.assertEqual(report["observed_tracks"]["idf1"], 0.8)
        self.assertEqual(report["published_tracks_including_predictions"]["idf1"], 1.0)
        self.assertEqual(details[1]["observed_tracks"]["missed_gt_ids"], ["A"])

    def test_false_positive_and_one_to_one_matching(self):
        report, details = score_frames([
            frame(0, (TruthObject("A", 0.0, 0.0, 2),
                      TruthObject("B", 0.5, 0.0, 2)),
                  (Proposal(0.2, 0.0), Proposal(50.0, 0.0)),
                  (Proposal(0.2, 0.0, 1), Proposal(50.0, 0.0, 2)))
        ], 1.0)
        self.assertEqual(report["detections"]["tp"], 1)
        self.assertEqual(report["detections"]["fp"], 1)
        self.assertEqual(report["detections"]["fn"], 1)
        self.assertEqual(len(details[0]["observed_tracks"]["matches"]), 1)

    def test_matching_maximizes_cardinality_before_distance(self):
        # Nearest-only would match A to proposal 0 and miss B; matching both
        # requires the slightly longer A->1, B->0 assignment.
        report, details = score_frames([
            frame(0, (TruthObject("A", 0.0, 0.0, 2),
                      TruthObject("B", 0.9, 0.0, 2)),
                  (Proposal(0.0, 0.0), Proposal(-0.9, 0.0)), ())
        ], 1.0)
        self.assertEqual(report["detections"]["tp"], 2)
        self.assertEqual(report["detections"]["fn"], 0)
        self.assertEqual({item["proposal_id"] for item in details[0]["detections"]["matches"]},
                         {0, 1})

    def test_empty_truth_does_not_claim_perfect_score(self):
        report, _ = score_frames([frame(0, (), (), ())], 1.0)
        self.assertIsNone(report["detections"]["f1"])
        self.assertIsNone(report["observed_tracks"]["idf1"])
        with self.assertRaisesRegex(ValueError, "distance gate"):
            score_frames([frame(0, (), (), ())], 0.0)

    def test_truth_groups_dynamic_points_and_ignores_static(self):
        dtype = np.dtype([("x_seq", "f8"), ("y_seq", "f8"),
                          ("label_id", "i4"), ("track_id", "S16")])
        radar = np.array([
            (1.0, 1.0, 0, b"A"), (3.0, 1.0, 0, b"A"),
            (10.0, 2.0, 11, b""), (5.0, 3.0, 7, b""),
        ], dtype=dtype)
        objects, skipped = truth_objects(radar)
        self.assertEqual(len(objects), 1)
        self.assertEqual(objects[0], TruthObject("A", 2.0, 1.0, 2))
        self.assertEqual(skipped, 1)


if __name__ == "__main__":
    unittest.main()
