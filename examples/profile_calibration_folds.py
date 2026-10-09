"""Fixed 100-frame isolated latency probe for fold-zero calibrated medium policy."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from evaluate_calibration_folds import tracking_configs
from radar_id_tracker.lidar_score_calibration import load_calibration_plan,ScoreCalibrationModel


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset-root','checkpoint','plan','run-root'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();plan,digest=load_calibration_plan(args.plan);root=args.run_root
    model=root/'fold_0/calibration/model.json'
    config=tracking_configs(plan,ScoreCalibrationModel.load(model))['calibrated_medium']
    path=root/'latency_protocol.json'
    data=dict(plan_sha256=digest,frozen_config=asdict(config),frozen_inference_config=plan['inference_config'])
    if path.exists() and json.loads(path.read_text())!=data:raise ValueError('latency protocol differs')
    path.write_text(json.dumps(data,indent=2)+'\n')
    sys.argv=['profile_surround.py','--dataset-root',str(args.dataset_root),'--checkpoint',str(args.checkpoint),
        '--config',str(path),'--sequence','0009','--start','100','--frames','100',
        '--head-model',str(root/'fold_0/model/model.json'),'--score-calibration-model',str(model),'--output',str(root/'latency.json')]
    from profile_surround import main as profile
    profile()


if __name__=='__main__':main()
