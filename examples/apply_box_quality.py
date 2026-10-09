"""Apply a frozen JSON verifier to label-free feature/cache artifacts."""
import argparse
from dataclasses import asdict,replace
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from radar_id_tracker.lidar_box_features import FEATURE_NAMES,FEATURE_VERSION
from radar_id_tracker.lidar_box_quality import BoxQualityModel
from radar_id_tracker.lidar_tracker import BoxDetection


def apply(cache,features,model_path,out):
    if out.resolve() in (cache.resolve(),features.resolve()):raise ValueError('output must be separate')
    meta=json.loads((cache/'inference_summary.json').read_text())
    manifest=json.loads((features/'manifest.json').read_text())
    digest=hashlib.sha256((cache/'detections.jsonl').read_bytes()).hexdigest()
    if digest!=meta['output_sha256'] or digest!=manifest['cache_sha256'] or manifest['ground_truth_used_as_features'] is not False:
        raise ValueError('valid label-free matching cache/features required')
    if meta['ground_truth_used_as_input'] is not False:raise ValueError('label-free inference required')
    feature_source=Path(__file__).resolve().parents[1]/'src/radar_id_tracker/lidar_box_features.py'
    if (manifest['feature_version']!=FEATURE_VERSION or tuple(manifest['features'])!=FEATURE_NAMES or
            manifest['source_sha256']!=hashlib.sha256(feature_source.read_bytes()).hexdigest()):
        raise ValueError('cached feature implementation differs')
    if hashlib.sha256((features/'features.npz').read_bytes()).hexdigest()!=manifest['features_sha256']:
        raise ValueError('feature checksum mismatch')
    model=BoxQualityModel.load(model_path)
    if model.data['checkpoint_sha256']!=meta['checkpoint_sha256'] or any(
            meta.get(key if key!='score_threshold' else 'detector_score_threshold')!=value
            for key,value in model.data['inference_config'].items() if key!='voxel_backend'):
        raise ValueError('quality model detector input domain differs')
    arrays=np.load(features/'features.npz',allow_pickle=False);start=perf_counter()
    scores=np.zeros(len(arrays['x']))
    for label in ('Car','Pedestrian'):
        selected=arrays['labels']==label;scores[selected]=model.predict(label,arrays['x'][selected])
    prediction_ms=(perf_counter()-start)*1000
    frames=[json.loads(line) for line in (cache/'detections.jsonl').read_text().splitlines()]
    for index,((i,j),score) in enumerate(zip(arrays['indices'],scores)):
        original=BoxDetection(**frames[i]['detections'][j])
        if original.label!=arrays['labels'][index]:raise ValueError('feature label mismatch')
        frames[i]['detections'][j]=asdict(replace(original,confidence=float(score)))
        frames[i].setdefault('box_quality_scores',[]).append(dict(detection_index=int(j),detector_confidence=original.confidence,quality_score=float(score)))
    out.mkdir(parents=True,exist_ok=True)
    target=out/'detections.jsonl'
    target.write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in frames))
    result={**meta,'output_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
            'raw_cache_sha256':digest,'quality_features_sha256':manifest['features_sha256'],
            'box_quality_model_sha256':hashlib.sha256(model_path.read_bytes()).hexdigest(),
            'box_quality_training_sequences':model.data['training_sequences'],
            'quality_inference_ground_truth_used':False,'box_quality_score_is_calibrated':False,
            'quality_score_prediction_ms':prediction_ms,
            'quality_features_median_ms':manifest['median_ms'],'quality_features_p99_ms':manifest['p99_ms'],
            'latency_note':'source detector timing excludes cached box quality extraction/prediction; use integrated latency probe'}
    (out/'inference_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(sequence=meta['sequence'],model=str(model_path),quality_scores=len(scores))),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache-dir',type=Path,required=True)
    p.add_argument('--features-dir',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args();apply(args.cache_dir,args.features_dir,args.model,args.output_dir)


if __name__=='__main__':main()
