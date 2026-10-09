"""Optional monotone score calibration; NumPy-only inference, no GT or IDs.

Probabilities refer to the frozen front-ROI, IoU and equal-sequence development
mixture. Calibration cannot change detector ranking or repair box geometry.
"""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import numpy as np
from .lidar_inference_config import RawLidarConfig
from .lidar_scoped_evaluate import in_scope

LABELS = ('Car', 'Pedestrian')


def sigmoid(values):
    x = np.asarray(values, dtype=np.float64)
    return np.exp(-np.logaddexp(0., -x))


def logit(scores):
    x = np.asarray(scores, dtype=np.float64)
    if not np.isfinite(x).all() or ((x < 0) | (x > 1)).any():
        raise ValueError('scores must be finite probabilities')
    x = np.clip(x, 1e-6, 1-1e-6)
    return np.log(x)-np.log1p(-x)


def load_calibration_plan(path):
    raw = Path(path).read_bytes()
    if len(raw) > 65536: raise ValueError('calibration plan too large')
    data = json.loads(raw)
    if data['format_version'] != 1 or data['evaluation_used_for_selection'] is not False:
        raise ValueError('evaluation cannot select calibration or policies')
    groups = data['groups']
    if not isinstance(groups, list) or len(groups) != 3 or any(not g for g in groups):
        raise ValueError('three nonempty whole-sequence groups required')
    all_ids = [s for g in groups for s in g]+data['reserved_sequences']+data['excluded_viewed_sequences']
    if any(not isinstance(s, str) or len(s) != 4 or not s.isascii() or not s.isdecimal() or not 0 <= int(s) <= 20 for s in all_ids):
        raise ValueError('invalid KITTI training sequence')
    if len(set(all_ids)) != len(all_ids): raise ValueError('sequence roles overlap')
    if sorted(data['roles']) != [[0,1,2],[1,2,0],[2,0,1]]:
        raise ValueError('each group must occupy each role exactly once')
    RawLidarConfig.from_dict(data['inference_config'])
    c = data['calibration']
    if (c['method'] != 'positive_slope_logit_sigmoid' or c['sequence_equal_weight'] is not True or
            not np.isfinite([c['l2'], c['target_iou']]).all() or c['l2'] <= 0 or not 0 < c['target_iou'] <= 1 or
            type(c['iterations']) is not int or not 1 <= c['iterations'] <= 1000):
        raise ValueError('invalid calibration parameters')
    from .lidar_tracker import LidarTrackerConfig, ClassScorePolicy
    LidarTrackerConfig.from_dict(data['tracker_config'])
    p = data['medium_policy']
    ClassScorePolicy('Pedestrian',p['continuation_confidence'],p['birth_confidence'],p['strong_confidence'],data['recovery_raw_score_floor'])
    if p['medium_confirmation_hits'] < 3: raise ValueError('medium confirmation requires at least three hits')
    for key in ('acceptance_precision','acceptance_recall','recovery_raw_score_floor'):
        if not np.isfinite(data[key]) or not 0 < data[key] <= 1: raise ValueError('invalid acceptance/floor')
    return data, hashlib.sha256(raw).hexdigest()


def fit_sigmoid(scores, targets, weights, l2=.01, iterations=100):
    """Offline weighted logistic fit with line search and positive slope.

    A class lacking either outcome explicitly falls back to the raw score.
    No fitted class balance weights: retain the supplied sequence-mixture prior.
    """
    x = logit(scores); y = np.asarray(targets, dtype=np.float64); w = np.asarray(weights, dtype=np.float64)
    if x.ndim != 1 or y.shape != x.shape or w.shape != x.shape or not len(x):
        raise ValueError('nonempty aligned score/target/weight vectors required')
    if not np.isfinite(y).all() or not np.isfinite(w).all() or not np.isin(y,[0,1]).all() or (w <= 0).any():
        raise ValueError('binary targets and positive finite weights required')
    if not np.isfinite(l2) or l2 <= 0 or type(iterations) is not int or not 1 <= iterations <= 1000:
        raise ValueError('invalid calibration optimization')
    if len(np.unique(y)) < 2:
        return dict(slope=1.,intercept=0.,status='identity_missing_outcome')
    w = w/w.sum(); design = np.column_stack([x,np.ones(len(x))]); theta = np.array([1.,0.])
    # The regularizer is .5*l2*((slope-1)^2 + intercept^2).
    def objective(t):
        z=design@t
        return float(w@(np.logaddexp(0,z)-y*z)+.5*l2*((t[0]-1)**2+t[1]**2))
    for _ in range(iterations):
        p=sigmoid(design@theta)
        gradient=design.T@(w*(p-y))+l2*(theta-[1.,0.])
        hessian=(design.T*(w*p*(1-p)))@design+l2*np.eye(2)
        step=np.linalg.solve(hessian,gradient); old=objective(theta)
        for scale in 2.**-np.arange(30):
            proposal=theta-scale*step;proposal[0]=max(1e-6,proposal[0])
            if objective(proposal) <= old+1e-14: break
        else: break
        change=np.max(np.abs(proposal-theta));theta=proposal
        if change < 1e-9: break
    return dict(slope=float(theta[0]),intercept=float(theta[1]),status='fitted')


class ScoreCalibrationModel:
    def __init__(self,data):
        if data.get('format_version') != 1 or data.get('method') != 'positive_slope_logit_sigmoid':
            raise ValueError('unsupported score calibration')
        self.data=data
        if set(data['models']) != set(LABELS): raise ValueError('Car/Pedestrian calibrators required')
        for m in data['models'].values():
            if not np.isfinite([m['slope'],m['intercept']]).all() or not 0 < m['slope'] <= 100 or abs(m['intercept']) > 100:
                raise ValueError('invalid calibration coefficients')
            if m.get('status') not in ('fitted','identity_missing_outcome'):
                raise ValueError('unknown calibration fit status')
        roles=[data[k] for k in ('detector_training_sequences','calibration_sequences','evaluation_sequences')]
        if any(not r or len(set(r)) != len(r) for r in roles) or any(set(roles[i])&set(roles[j]) for i in range(3) for j in range(i)):
            raise ValueError('detector training, calibration and evaluation must be disjoint')
        if data.get('front_only') is not True or data.get('evaluation_used_for_selection') is not False:
            raise ValueError('calibration scope or isolation differs')
        RawLidarConfig.from_dict(data['inference_config'])
        if not np.isfinite(data['recovery_raw_score_floor']) or not 0 < data['recovery_raw_score_floor'] < 1:
            raise ValueError('invalid raw recovery floor')

    @classmethod
    def load(cls,path):
        if Path(path).stat().st_size > 1024*1024: raise ValueError('calibration model too large')
        return cls(json.loads(Path(path).read_text()))

    def validate_domain(self,metadata):
        if metadata.get('head_model_sha256') != self.data['head_model_sha256'] or metadata.get('checkpoint_sha256') != self.data['checkpoint_sha256'] or metadata.get('inference_config') != self.data['inference_config']:
            raise ValueError('score calibration detector domain differs')
        if metadata.get('ground_truth_used_as_input') is not False:
            raise ValueError('label-free detector cache required')

    def predict(self,label,scores):
        if label not in LABELS: raise ValueError('unsupported calibration class')
        m=self.data['models'][label]
        x=logit(scores)
        if m['status'] == 'identity_missing_outcome': return np.asarray(scores,dtype=np.float64).copy()
        return sigmoid(m['slope']*x+m['intercept'])

    def rescore(self,boxes,lidar_from_world):
        result=[]
        for b in boxes:
            if not in_scope(b,lidar_from_world):result.append(b);continue
            score=float(self.predict(b.label,[b.confidence])[0])
            floor=float(self.predict(b.label,[self.data['recovery_raw_score_floor']])[0])
            # Floating-point sigmoid saturation must not allow below-floor raw
            # inputs to pass an independently mapped recovery threshold.
            if b.confidence < self.data['recovery_raw_score_floor']:
                score=min(score,float(np.nextafter(floor,0.)))
            result.append(replace(b,confidence=score))
        return result


def calibration_targets(truth,boxes,ignored,threshold=.25):
    """Offline one-to-one labels; unmatched ignore overlaps do not train negatives."""
    from .kitti_evaluate import match_boxes
    from .box_geometry import iou3d
    pairs,_=match_boxes(truth,boxes,threshold)
    matched={j for _,j in pairs};targets=np.zeros(len(boxes),np.int8)
    for j,b in enumerate(boxes):
        if j in matched:targets[j]=1
        elif any(b.label==g.label and iou3d(b,g)>=threshold for g in ignored):targets[j]=-1
    return targets


def calibrate_cache(cache_dir,output_dir,model_path):
    """Apply a frozen, domain-bound calibration without opening scans or labels."""
    from dataclasses import asdict
    from .lidar_tracker import BoxDetection
    cache_dir,output_dir,model_path=map(Path,(cache_dir,output_dir,model_path))
    model=ScoreCalibrationModel.load(model_path)
    metadata=json.loads((cache_dir/'inference_summary.json').read_text());model.validate_domain(metadata)
    if metadata['sequence'] not in model.data['calibration_sequences']+model.data['evaluation_sequences']:
        raise ValueError('calibration cache cannot include detector training sequences')
    source=cache_dir/'detections.jsonl'
    if hashlib.sha256(source.read_bytes()).hexdigest()!=metadata['output_sha256']:
        raise ValueError('raw cache checksum differs')
    if output_dir.resolve()==cache_dir.resolve():raise ValueError('never overwrite raw detector output')
    output_dir.mkdir(parents=True,exist_ok=True);path=output_dir/'detections.jsonl'
    if path.exists():raise ValueError('calibrated cache exists; choose a new directory')
    rows=0;preserved=0
    with source.open() as src,path.with_suffix('.part').open('w') as dest:
        for line in src:
            row=json.loads(line)
            if row['frame_index']!=rows or abs(row['timestamp_s']-rows*metadata['period_s'])>1e-8:
                raise ValueError('cache index or timestamp differs')
            boxes=[BoxDetection(**d) for d in row['detections']]
            inverse=np.linalg.inv(np.asarray(row['world_from_lidar']))
            calibrated=model.rescore(boxes,inverse)
            preserved+=sum(a==b for a,b in zip(boxes,calibrated) if not in_scope(a,inverse))
            row['detections']=[asdict(d) for d in calibrated]
            dest.write(json.dumps(row,allow_nan=False)+'\n');rows+=1
    if rows!=metadata['frames']:raise ValueError('cache frame count differs')
    path.with_suffix('.part').replace(path)
    result={**metadata,'output_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'score_calibration_sha256':hashlib.sha256(model_path.read_bytes()).hexdigest(),
        'score_calibration_applied':True,'confidence_is_calibrated':False,
        'calibration_status':'candidate; held-out reliability must be evaluated',
        'calibration_inference_labels_used':False,'raw_detector_output_sha256':metadata['output_sha256'],
        'outside_scope_preserved_instances':preserved}
    (output_dir/'inference_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
