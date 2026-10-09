from dataclasses import asdict,replace
import json
from pathlib import Path
import tempfile
import unittest

from radar_id_tracker import ClassScorePolicy,LidarTracker,LidarTrackerConfig
from radar_id_tracker.lidar_cli import main
from test_lidar_tracker import box


def config(**changes):
    return replace(LidarTrackerConfig(high_confidence=.98,low_confidence=.5,
        score_policies=(ClassScorePolicy('Car',.5,.7,.98,.98),),maximum_tentative_tracks=4),**changes)


class ScorePolicyTests(unittest.TestCase):
    def test_three_medium_observations_confirm_without_future_input(self):
        tracker=LidarTracker(config());d=box(confidence=.8)
        self.assertEqual(tracker.update(0.,[d]),[])
        self.assertEqual(tracker.update(.1,[d]),[])
        out=tracker.update(.2,[d])
        self.assertEqual(len(out),1)
        self.assertEqual(out[0].confirmation_path,'medium')
        self.assertAlmostEqual(out[0].confirmation_delay_s,.2)
        self.assertEqual(tracker.tentative_outputs,[])

    def test_high_path_stays_fast_and_mixed_scores_do_not_count_as_two_high(self):
        tracker=LidarTracker(config())
        tracker.update(0.,[box()]);out=tracker.update(.1,[box()])
        self.assertEqual(out[0].confirmation_path,'strong')
        tracker=LidarTracker(config())
        tracker.update(0.,[box()]);self.assertEqual(tracker.update(.1,[box(confidence=.8)]),[])
        self.assertEqual(tracker.update(.2,[box(confidence=.8)])[0].confirmation_path,'medium')

    def test_strong_candidate_match_precedes_nearby_medium_duplicate(self):
        tracker=LidarTracker(config(duplicate_birth_iou=.85))
        tracker.update(0.,[box()])
        out=tracker.update(.1,[box(x=.2),box(confidence=.8)])
        self.assertEqual(out[0].confirmation_path,'strong')
        self.assertEqual(tracker.tentative_outputs,[])

    def test_single_spike_and_repeated_below_birth_detections_cannot_spawn(self):
        tracker=LidarTracker(config())
        tracker.update(0.,[box(confidence=.8)])
        for i in range(1,10):
            self.assertEqual(tracker.update(i*.1,[box(confidence=.6)]),[])
        self.assertEqual(tracker.tentative_outputs,[])

    def test_missed_update_resets_medium_confirmation(self):
        tracker=LidarTracker(config(medium_confirmation_window_s=.6))
        d=box(confidence=.8)
        tracker.update(0.,[d]);tracker.update(.1,[d]);tracker.update(.2,[])
        self.assertEqual(tracker.update(.3,[d]),[])
        self.assertEqual(tracker.update(.4,[d]),[])
        self.assertEqual(len(tracker.update(.5,[d])),1)

    def test_large_timestamp_gap_resets_medium_even_without_empty_update(self):
        tracker=LidarTracker(config(medium_confirmation_window_s=1.))
        d=box(confidence=.8)
        tracker.update(0.,[d]);tracker.update(.1,[d])
        self.assertEqual(tracker.update(.4,[d]),[])
        self.assertEqual(tracker.update(.5,[d]),[])
        self.assertEqual(len(tracker.update(.6,[d])),1)

    def test_expired_candidate_restarts_instead_of_using_old_evidence(self):
        tracker=LidarTracker(config());d=box(confidence=.8)
        tracker.update(0.,[d]);tracker.update(.1,[d])
        self.assertEqual(tracker.update(.6,[d]),[])
        self.assertEqual(tracker.tentative_outputs[0]['medium_hits'],1)
        self.assertEqual(tracker.diagnostics['tentative_expired'],1)

    def test_class_specific_rules_and_unknown_class_fallback(self):
        cfg=config(score_policies=(ClassScorePolicy('Car',.5,.7,.98,.98),
                                   ClassScorePolicy('Pedestrian',.5,.9,.98,.98)))
        tracker=LidarTracker(cfg)
        for i in range(3):out=tracker.update(i*.1,[box(confidence=.8),box(x=20.,confidence=.8,label='Pedestrian')])
        self.assertEqual([t.label for t in out],['Car'])
        for i in range(3,5):out=tracker.update(i*.1,[box(x=40.,label='Other')])
        self.assertIn('Other',[t.label for t in out])

    def test_confirmed_track_has_priority_over_medium_candidate(self):
        tracker=LidarTracker(config());tracker.update(0.,[box()]);old=tracker.update(.1,[box()])[0].track_id
        for i in range(2,10):
            out=tracker.update(i*.1,[box(confidence=.8)])
            self.assertEqual([t.track_id for t in out],[old])
            self.assertEqual(tracker.tentative_outputs,[])

    def test_medium_birth_does_not_lower_dormant_recovery_threshold(self):
        tracker=LidarTracker(config());tracker.update(0.,[box()]);old=tracker.update(.1,[box()])[0].track_id
        for i in range(2,32):tracker.update(i*.1,[])
        self.assertEqual(tracker.update(3.2,[box(confidence=.8)]),[])
        self.assertEqual(tracker.diagnostics['recovered'],0)
        # A strong return still reaches the old identity before pending candidates.
        out=tracker.update(3.3,[box()])
        self.assertEqual(out[0].track_id,old)
        self.assertEqual(tracker.diagnostics['recovered'],1)
        self.assertEqual(tracker.tentative_outputs[0]['medium_hits'],0)

    def test_shape_jump_does_not_pool_candidate_hits(self):
        tracker=LidarTracker(config());tracker.update(0.,[box(confidence=.8)])
        self.assertEqual(tracker.update(.1,[box(confidence=.8,length=12.)]),[])
        self.assertEqual(tracker.update(.2,[box(confidence=.8)]),[])
        self.assertEqual(tracker.update(.3,[box(confidence=.8)]),[])

    def test_candidate_limit_does_not_evict_confirmed_identity(self):
        tracker=LidarTracker(config(maximum_tentative_tracks=2,maximum_tracks=4))
        tracker.update(0.,[box()]);old=tracker.update(.1,[box()])[0].track_id
        ds=[box()]+[box(x=20*i,confidence=.8) for i in range(1,5)]
        out=tracker.update(.2,ds)
        self.assertEqual(out[0].track_id,old)
        self.assertEqual(len(tracker.tentative_outputs),2)
        self.assertEqual(tracker.diagnostics['capacity_drops'],2)

    def test_two_nearby_candidates_keep_distinct_ids(self):
        tracker=LidarTracker(config())
        for i in range(5):
            ds=[box(x=i*.1,y=-1.,confidence=.8),box(x=-i*.1,y=1.,confidence=.8)]
            out=tracker.update(i*.1,ds[::-1] if i%2 else ds)
            if i>=2:
                self.assertEqual(len(out),2)
                self.assertEqual({t.track_id for t in out},{1,2})

    def test_duplicate_boxes_do_not_create_duplicate_medium_ids(self):
        tracker=LidarTracker(config(duplicate_birth_iou=.85))
        for i in range(5):
            out=tracker.update(i*.1,[box(confidence=.8),box(confidence=.81)])
            if i>=2:self.assertEqual(len(out),1)
        self.assertEqual(len(tracker._tracks),1)

    def test_roundtrip_config_validation_and_legacy_medium_disabled(self):
        cfg=config();self.assertEqual(LidarTrackerConfig.from_dict(asdict(cfg)),cfg)
        tracker=LidarTracker(LidarTrackerConfig(high_confidence=.98,low_confidence=.5))
        for i in range(5):self.assertEqual(tracker.update(i*.1,[box(confidence=.8)]),[])
        for values in ((.9,.8,.98,.98),(.5,.7,float('nan'),.98)):
            with self.assertRaises(ValueError):ClassScorePolicy('Car',*values)
        with self.assertRaises(ValueError):config(score_policies=cfg.score_policies*2)
        with self.assertRaises(ValueError):config(medium_confirmation_hits=2)

    def test_disabled_policy_preserves_legacy_window_boundary(self):
        tracker=LidarTracker()
        tracker.update(.3,[box()])
        self.assertEqual(tracker.update(.7000000000000001,[box()]),[])
        self.assertEqual(len(tracker.update(.8,[box()])),1)

    def test_cli_config_and_unconfirmed_objects_are_not_formal_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);policy=root/'policy.json';policy.write_text(json.dumps(asdict(config())))
            source=root/'input.jsonl'
            source.write_text(''.join(json.dumps(dict(timestamp_s=i*.1,detections=[asdict(box(confidence=.8))]))+'\n' for i in range(3)))
            output=root/'tracks.jsonl'
            self.assertEqual(main([str(source),'--output',str(output),'--config',str(policy)]),0)
            rows=[json.loads(s) for s in output.read_text().splitlines()]
            self.assertFalse(rows[0]['tracks']);self.assertTrue(rows[0]['tentative'])
            self.assertEqual(rows[2]['tracks'][0]['confirmation_path'],'medium')
            with self.assertRaises(SystemExit):main([str(source),'--config',str(policy),'--high-confidence','.9'])

    def test_fixed_recovery_keeps_missing_pretrack_in_denominator(self):
        import sys
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'examples'))
        from benchmark_score_policy import fixed_recovery
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            rows=[dict(observed_tracks=dict(matches={})) for _ in range(40)]
            rows[4]['observed_tracks']['matches']={'a':1,'b':2}
            rows[35]['observed_tracks']['matches']={'a':1,'b':9,'c':2}
            (root/'evaluation_frames.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            results=fixed_recovery(root,'0000',5,[{'gt_id':g} for g in ('a','b','c')])
            self.assertEqual(len(results),3)
            self.assertEqual(sum(r['same_id'] for r in results),1)
            self.assertTrue(results[2]['missing_before'])
            self.assertTrue(results[2]['wrong_retained_identity'])


if __name__=='__main__':unittest.main()
