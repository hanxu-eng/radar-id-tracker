"""Train small class-specific quality trees with sequence-held-out predictions.

scikit-learn is training-only; exported JSON inference uses NumPy only. Fixed
hyperparameters and tracking settings are frozen before new-sequence evaluation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
os.environ.setdefault('LOKY_MAX_CPU_COUNT','4')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from radar_id_tracker.lidar_box_features import FEATURE_NAMES,FEATURE_VERSION
from radar_id_tracker.lidar_box_quality import BoxQualityModel

PARAMETERS=dict(max_iter=60,max_leaf_nodes=7,max_depth=3,min_samples_leaf=40,
                learning_rate=.1,l2_regularization=1.,early_stopping=False,
                categorical_features=None,random_state=17)


def export(model):
    trees=[]
    for predictors in model._predictors:
        if len(predictors)!=1:raise ValueError('binary model required')
        nodes=predictors[0].nodes
        if nodes['is_categorical'].any():raise ValueError('categorical tree unsupported')
        trees.append([[int(n['feature_idx']),float(n['num_threshold']),int(n['left']),
                       int(n['right']),int(n['is_leaf']),float(n['value'])] for n in nodes])
    return dict(intercept=float(model._baseline_prediction[0,0]),trees=trees)


def train(datasets,sequences):
    result=dict(format_version=1,feature_version=FEATURE_VERSION,features=list(FEATURE_NAMES),models={},
                training_sequences=sequences,sklearn_version=sklearn.__version__,training_parameters=PARAMETERS,
                inference_config=dict(sweeps=5,period_s=.1,intensity_scale=64,score_threshold=.1,voxel_backend='numba'),
                checkpoint_sha256='191a38226d2ce990f26cc3f78c454310cc562d81d29e86d8ca8fa39c0a44215d',
                score_is_calibrated=False,training_domain='KITTI front Car/Pedestrian; rear and RoboSense unvalidated')
    estimators={}
    for label in ('Car','Pedestrian'):
        xs=[];ys=[];ws=[]
        for sequence in sequences:
            data=datasets[sequence];mask=(data['labels']==label)&(data['y']>=0)
            x,y=data['x'][mask],data['y'][mask]
            if len(x):
                xs.append(x);ys.append(y);ws.append(np.full(len(x),1/len(x)))
        x=np.concatenate(xs);y=np.concatenate(ys);w=np.concatenate(ws);w*=len(w)/w.sum()
        if len(np.unique(y))!=2:raise ValueError('each class needs positives and negatives')
        model=HistGradientBoostingClassifier(**PARAMETERS)
        with threadpool_limits(2):model.fit(x,y,sample_weight=w)
        result['models'][label]=export(model);estimators[label]=model
    return result,estimators


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features-root',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--training-sequences',nargs='+',default=['0000','0001','0002','0005','0006'])
    p.add_argument('--validation-sequences',nargs='+',default=['0003','0004','0008'])
    args=p.parse_args();out=args.output_dir
    if (out/'protocol.json').exists():raise ValueError('frozen protocol exists; choose a new training directory')
    out.mkdir(parents=True,exist_ok=True)
    if set(args.training_sequences)&set(args.validation_sequences):p.error('training and validation must be disjoint')
    datasets={};provenance={}
    for sequence in args.training_sequences:
        root=args.features_root/sequence
        meta=json.loads((root/'manifest.json').read_text());targets=json.loads((root/'target_manifest.json').read_text())
        raw=json.loads((Path(meta['cache'])/'inference_summary.json').read_text())
        expected=dict(sweeps=5,period_s=.1,intensity_scale=64,score_threshold=.1)
        if (raw.get('ground_truth_used_as_input') is not False or
                raw.get('checkpoint_sha256')!='191a38226d2ce990f26cc3f78c454310cc562d81d29e86d8ca8fa39c0a44215d' or
                any(raw.get(k if k!='score_threshold' else 'detector_score_threshold')!=v for k,v in expected.items())):
            raise ValueError('training features require the fixed raw detector input domain')
        if meta['features_sha256']!=hashlib.sha256((root/'features.npz').read_bytes()).hexdigest() or targets['targets_sha256']!=hashlib.sha256((root/'targets.npz').read_bytes()).hexdigest():
            raise ValueError('training artifact checksum differs')
        if meta['source_sha256']!=hashlib.sha256(Path('src/radar_id_tracker/lidar_box_features.py').read_bytes()).hexdigest():
            raise ValueError('training feature implementation differs')
        arrays=np.load(root/'features.npz',allow_pickle=False)
        datasets[sequence]={**{key:arrays[key] for key in arrays.files},'y':np.load(root/'targets.npz',allow_pickle=False)['y']}
        provenance[sequence]=dict(features_sha256=meta['features_sha256'],targets_sha256=targets['targets_sha256'],label_sha256=targets['label_sha256'])
    audits=[]
    for held_out in args.training_sequences+[None]:
        training=[s for s in args.training_sequences if s!=held_out]
        data,estimators=train(datasets,training);data['training_provenance']={s:provenance[s] for s in training}
        candidate=BoxQualityModel(data)
        directory=out/('full' if held_out is None else f'fold_{held_out}');directory.mkdir(exist_ok=True)
        model_path=directory/'model.json';model_path.write_text(json.dumps(data,allow_nan=False)+'\n')
        audit=dict(held_out_sequence=held_out,training_sequences=training,model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),parity={})
        for label,estimator in estimators.items():
            x=np.concatenate([d['x'][d['labels']==label] for d in datasets.values()])
            # Audit trained and unseen held-out features against the source estimator.
            with threadpool_limits(2):reference=estimator.predict_proba(x)[:,1]
            actual=candidate.predict(label,x)
            error=float(np.max(np.abs(reference-actual),initial=0))
            np.testing.assert_allclose(reference,actual,rtol=0,atol=1e-12)
            audit['parity'][label]=dict(samples=len(x),maximum_absolute_difference=error)
        audits.append(audit);print(json.dumps(audit),flush=True)
    protocol=dict(training_sequences=args.training_sequences,validation_sequences=args.validation_sequences,
                  feature_names=list(FEATURE_NAMES),features_current_scan_only=True,identifiers_used_as_features=False,
                  training_parameters=PARAMETERS,sklearn_version=sklearn.__version__,
                  sequence_equal_weight=True,frame_random_split=False,threshold_tuned=False,
                  frozen_inference_config=data['inference_config'],
                  frozen_config=dict(high_confidence=.7,low_confidence=.1,maximum_tentative_tracks=128,duplicate_birth_iou=.85),
                  model_sha256=audits[-1]['model_sha256'],folds=audits,acceptance_precision=.96,acceptance_recall=.85,
                  source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                    [Path(__file__),Path('src/radar_id_tracker/lidar_box_features.py'),Path('src/radar_id_tracker/lidar_box_quality.py')]})
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')


if __name__=='__main__':main()
