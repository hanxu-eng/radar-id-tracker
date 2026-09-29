import math
import unittest

from radar_id_tracker import Detection, RadarTracker, TrackerConfig


class TrackerTests(unittest.TestCase):
    def test_single_target_keeps_id_and_uses_real_timestamps(self):
        tracker = RadarTracker()
        ids = []
        for frame in range(30):
            time_s = frame * 0.1 + (0.012 if frame % 3 == 0 else 0.0)
            detection = Detection(x=10 + 2 * time_s, y=2, confidence=0.9, radial_velocity=2.0)
            outputs = tracker.update(time_s, [detection])
            if outputs:
                ids.append(outputs[0].track_id)
        self.assertEqual(set(ids), {1})
        self.assertEqual(len(ids), 29)

    def test_crossing_targets_keep_identity_with_doppler(self):
        tracker = RadarTracker()
        first_ids = None
        last = None
        sensor = (-10.0, 0.0)
        for frame in range(21):
            time_s = frame * 0.1
            left_to_right = Detection(x=-1 + 2 * time_s, y=0, confidence=0.9, radial_velocity=2.0)
            right_to_left = Detection(x=1 - 2 * time_s, y=0, confidence=0.9, radial_velocity=-2.0)
            ordered = [left_to_right, right_to_left] if frame % 2 == 0 else [right_to_left, left_to_right]
            outputs = tracker.update(time_s, ordered, sensor)
            if frame == 1:
                first_ids = {1: max(outputs, key=lambda item: item.vx).track_id,
                             -1: min(outputs, key=lambda item: item.vx).track_id}
            last = outputs
        self.assertEqual(len(last), 2)
        self.assertEqual(max(last, key=lambda item: item.vx).track_id, first_ids[1])
        self.assertEqual(min(last, key=lambda item: item.vx).track_id, first_ids[-1])

    def test_short_dropout_reuses_id_and_marks_prediction(self):
        tracker = RadarTracker()
        tracker.update(0.0, [Detection(10.0, 0.0, 0.9, 1.0)])
        original_id = tracker.update(0.1, [Detection(10.1, 0.0, 0.9, 1.0)])[0].track_id
        for time_s in (0.2, 0.3, 0.4):
            predicted = tracker.update(time_s, [])
            self.assertEqual(predicted[0].track_id, original_id)
            self.assertTrue(predicted[0].predicted_only)
        recovered = tracker.update(0.5, [Detection(10.5, 0.0, 0.8, 1.0)])
        self.assertEqual(recovered[0].track_id, original_id)
        self.assertFalse(recovered[0].predicted_only)

    def test_dormant_recovery_and_late_reappearance(self):
        tracker = RadarTracker()
        tracker.update(0.0, [Detection(10.0, 0.0, 0.9, 0.0)])
        old_id = tracker.update(0.1, [Detection(10.0, 0.0, 0.9, 0.0)])[0].track_id
        for frame in range(2, 9):
            output = tracker.update(frame * 0.1, [])
        self.assertEqual(output, [])
        restored = tracker.update(0.9, [Detection(10.0, 0.0, 0.9, 0.0)])
        self.assertEqual(restored[0].track_id, old_id)
        tracker.update(2.6, [])
        self.assertEqual(tracker.update(2.7, [Detection(10.0, 0.0, 0.9, 0.0)]), [])
        new_id = tracker.update(2.8, [Detection(10.0, 0.0, 0.9, 0.0)])[0].track_id
        self.assertGreater(new_id, old_id)

    def test_low_score_clutter_cannot_create_id(self):
        tracker = RadarTracker()
        for frame in range(10):
            self.assertEqual(
                tracker.update(frame * 0.1, [Detection(5.0, 2.0, 0.3)]), []
            )

    def test_accelerating_target_without_doppler_keeps_id(self):
        tracker = RadarTracker()
        observed_ids = []
        for frame in range(20):
            time_s = frame * 0.1
            x = 5.0 + 1.2 * time_s + 0.5 * 1.5 * time_s**2
            outputs = tracker.update(time_s, [Detection(x, 3.0, 0.9)])
            if outputs:
                observed_ids.append(outputs[0].track_id)
        self.assertEqual(set(observed_ids), {1})

    def test_time_validation_and_empty_frames(self):
        tracker = RadarTracker()
        self.assertEqual(tracker.update(1.0, []), [])
        with self.assertRaisesRegex(ValueError, "strictly increase"):
            tracker.update(1.0, [])
        with self.assertRaises(ValueError):
            tracker.update(math.nan, [])

    def test_detection_validation(self):
        with self.assertRaises(ValueError):
            Detection(x=1.0, y=2.0, confidence=1.2)
        with self.assertRaises(ValueError):
            Detection(x=1.0, y=2.0, confidence=0.5, radial_velocity=math.nan)

    def test_different_sensor_positions_with_compensated_doppler(self):
        tracker = RadarTracker(TrackerConfig(position_sigma_m=0.5))
        tracker.update(0.0, [Detection(x=10, y=0, confidence=0.9, radial_velocity=2)], (0, 0))
        outputs = tracker.update(0.1, [Detection(x=10.2, y=0, confidence=0.9, radial_velocity=2)], (1, 0))
        self.assertEqual(outputs[0].track_id, 1)

    def test_dormant_id_cannot_be_claimed_by_distant_target(self):
        tracker = RadarTracker()
        tracker.update(0.0, [Detection(10.0, 0.0, 0.9, 0.0)])
        old_id = tracker.update(0.1, [Detection(10.0, 0.0, 0.9, 0.0)])[0].track_id
        for frame in range(2, 9):
            tracker.update(frame * 0.1, [])
        self.assertEqual(tracker.update(0.9, [Detection(18.0, 0.0, 0.9, 0.0)]), [])
        new_id = tracker.update(1.0, [Detection(18.0, 0.0, 0.9, 0.0)])[0].track_id
        self.assertNotEqual(new_id, old_id)


if __name__ == "__main__":
    unittest.main()
