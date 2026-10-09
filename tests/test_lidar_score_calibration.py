import json
import hashlib
from pathlib import Path
import tempfile
import unittest
import numpy as np
from radar_id_tracker.lidar_score_calibration import (ScoreCalibrationModel,fit_sigmoid,
    load_calibration_plan,calibration_targets,calibrate_cache,sigmoid,logit)
from radar_id_tracker.lidar_tracker import BoxDetection,LidarTracker,LidarTrackerConfig,ClassScorePolicy
from radar_id_tracker.kitti_evaluate import TruthBox


class CalibrationTests(unittest.TestCase):
    def data(self):
        return dict(format_version=1,method='positive_slope_logit_sigmoid',front_only=True,
            evaluation_used_for_selection=False,models={label:dict(slope=2.,intercept=-1.,status='fitted') for label in ('Car','Pedestrian')},
            detector_training_sequences=['0000'],calibration_sequences=['0001'],evaluation_sequences=['0002'],
            head_model_sha256='head',checkpoint_sha256='base',inference_config={},recovery_raw_score_floor=.7)

    def test_fit_recovers_known_map_and_is_monotone(self):
        scores=np.linspace(.01,.99,500);prob=sigmoid(1.6*logit(scores)-.8)
        # Deterministic repeated Bernoulli counts avoid stochastic assertions.
        counts=np.round(prob*100).astype(int)
        x=np.repeat(scores,100);y=np.concatenate([np.r_[np.ones(n),np.zeros(100-n)] for n in counts])
        fit=fit_sigmoid(x,y,np.ones(len(x)),l2=1e-6)
        self.assertAlmostEqual(fit['slope'],1.6,delta=.03);self.assertAlmostEqual(fit['intercept'],-.8,delta=.03)
        self.assertTrue(np.isfinite(sigmoid([-1000,1000])).all())

    def test_missing_outcome_is_explicit_identity(self):
        fit=fit_sigmoid([.2,.8],[0,0],[1,1])
        self.assertEqual(fit['status'],'identity_missing_outcome')
        data=self.data();data['models']['Car']=fit
        np.testing.assert_array_equal(ScoreCalibrationModel(data).predict('Car',[0,.2,1]),[0,.2,1])
        with self.assertRaises(ValueError):fit_sigmoid([.3],[1],[0])
        with self.assertRaises(ValueError):logit([np.nan])

    def test_scope_geometry_and_recovery_floor_preserved(self):
        model=ScoreCalibrationModel(self.data())
        boxes=[BoxDetection(x,y,0,1,1,2,0,s,label) for x,y,s,label in
               [(10,0,.2,'Pedestrian'),(-10,0,.2,'Pedestrian'),(10,0,.2,'Truck'),(10,0,.8,'Car')]]
        changed=model.rescore(boxes,np.eye(4))
        self.assertEqual(changed[1:3],boxes[1:3]);self.assertEqual(changed[0].height,boxes[0].height)
        self.assertEqual(boxes[0].confidence,.2)
        floor=float(model.predict('Car',[.7])[0])
        self.assertLess(model.predict('Car',[.69])[0],floor);self.assertGreater(model.predict('Car',[.71])[0],floor)
        self.assertTrue(np.all(np.diff(model.predict('Car',np.linspace(0,1,100)))>=0))
        saturated=self.data();saturated['models']['Pedestrian'].update(slope=100,intercept=100)
        model=ScoreCalibrationModel(saturated)
        low=BoxDetection(10,0,0,1,1,2,0,.69,'Pedestrian')
        self.assertLess(model.rescore([low],np.eye(4))[0].confidence,model.predict('Pedestrian',[.7])[0])

    def test_role_leakage_domain_and_invalid_coefficients_rejected(self):
        data=self.data();data['evaluation_sequences']=['0001']
        with self.assertRaises(ValueError):ScoreCalibrationModel(data)
        data=self.data();data['models']['Car']['slope']=0
        with self.assertRaises(ValueError):ScoreCalibrationModel(data)
        model=ScoreCalibrationModel(self.data())
        with self.assertRaises(ValueError):model.validate_domain(dict(head_model_sha256='different'))
        data=json.loads(Path('configs/lidar_sequence_calibration.experimental.json').read_text());data['groups'][0].append('0020')
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'plan.json';path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):load_calibration_plan(path)

    def test_targets_are_one_to_one_and_ignore_only_unmatched(self):
        b=BoxDetection(10,0,0,1,1,2,0,.4,'Pedestrian')
        y=calibration_targets([TruthBox('0',b)],[b,b],[b])
        self.assertEqual(list(sorted(y)),[-1,1])
        self.assertEqual(list(sorted(calibration_targets([TruthBox('0',b)],[b,b],[]))),[0,1])
        self.assertEqual(calibration_targets([],[b],[b])[0],-1)

    def test_cache_inference_is_label_free_and_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cache=root/'raw';cache.mkdir();model_path=root/'model.json'
            model_path.write_text(json.dumps(self.data()))
            box=dict(x=10,y=0,z=0,length=1,width=1,height=2,yaw=0,confidence=.2,label='Pedestrian')
            row=dict(frame_index=0,timestamp_s=0,world_from_lidar=np.eye(4).tolist(),detections=[box],raw_scan_available=True)
            raw=json.dumps(row)+'\n';(cache/'detections.jsonl').write_text(raw)
            meta=dict(sequence='0002',frames=1,period_s=.1,head_model_sha256='head',checkpoint_sha256='base',
                inference_config={},ground_truth_used_as_input=False,output_sha256=hashlib.sha256(raw.encode()).hexdigest())
            # Write bytes so newline conversion cannot differ from the checksum.
            (cache/'detections.jsonl').write_bytes(raw.encode())
            (cache/'inference_summary.json').write_text(json.dumps(meta))
            result=calibrate_cache(cache,root/'calibrated',model_path)
            changed=json.loads((root/'calibrated/detections.jsonl').read_text())
            self.assertNotEqual(changed['detections'][0]['confidence'],.2)
            self.assertEqual(result['calibration_inference_labels_used'],False)
            self.assertEqual((cache/'detections.jsonl').read_bytes(),raw.encode())
            with self.assertRaises(ValueError):calibrate_cache(cache,root/'calibrated',model_path)
            (cache/'detections.jsonl').write_bytes(b'corrupt')
            with self.assertRaises(ValueError):calibrate_cache(cache,root/'other',model_path)

    def test_saturated_calibration_cannot_resurrect_below_raw_recovery_floor(self):
        data=self.data();data['models']['Pedestrian'].update(slope=100,intercept=100)
        model=ScoreCalibrationModel(data)
        policy=ClassScorePolicy('Pedestrian',.1,.5,.7,1.)
        tracker=LidarTracker(LidarTrackerConfig(score_policies=(policy,)))
        make=lambda score: model.rescore([BoxDetection(10,0,0,1,1,2,0,score,'Pedestrian')],np.eye(4))
        tracker.update(0,make(.9));before=tracker.update(.1,make(.9))[0].track_id
        tracker.update(1.,[]);returned=tracker.update(3.1,make(.69))
        self.assertFalse(any(t.track_id==before and not t.predicted_only for t in returned))
        self.assertEqual(tracker.diagnostics['recovered'],0)


if __name__=='__main__':unittest.main()
