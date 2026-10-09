"""Freeze matched geometry controls before additional optimization/evaluation."""
import argparse
import json
from pathlib import Path
from radar_id_tracker.lidar_score_calibration import load_calibration_plan
from radar_id_tracker.lidar_head_targets import load_head_manifest
from fit_calibration_folds import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--parent-run',type=Path,required=True)
    p.add_argument('--run-root',type=Path,required=True);args=p.parse_args()
    data=json.loads(args.config.read_text());parent_protocol=json.loads((args.parent_run/'protocol.json').read_text())
    if data['parent_calibration_plan_sha256']!=parent_protocol['plan_sha256'] or data['evaluation_used_for_selection'] is not False or data['reserved_sequence_opened'] is not False:
        raise ValueError('geometry experiment parent/freeze differs')
    if data['variants']!=['continued_l1','weighted_axis_huber']:raise ValueError('matched geometry variants required')
    for i in range(3):
        path=args.parent_run/f'fold_{i}'/'model/model.json';load_head_manifest(path)
        if sha(path)!=data['parent_head_sha256'][str(i)]:raise ValueError('geometry parent head checksum differs')
    frozen=dict(**data,config_sha256=sha(args.config),parent_plan=parent_protocol['plan'],
        parent_evaluation_protocol_sha256=sha(args.parent_run/'evaluation_protocol.json'),
        geometry_diagnostics_sha256=sha(args.parent_run/'geometry_diagnostics.json'),
        split_role_audit_sha256=sha(args.parent_run/'split_role_audit.json'),
        source_sha256={n:sha(n) for n in ('examples/train_pedestrian_geometry.py','src/radar_id_tracker/lidar_geometry_loss.py')},
        final_deployment_model_trained=False,development_folds_previously_reviewed=True)
    args.run_root.mkdir(parents=True,exist_ok=True);target=args.run_root/'protocol.json'
    if target.exists() and json.loads(target.read_text())!=frozen:raise ValueError('frozen geometry experiment exists and differs')
    target.write_text(json.dumps(frozen,indent=2)+'\n');print(json.dumps(dict(protocol_sha256=sha(target),parent_heads=data['parent_head_sha256'])))


if __name__=='__main__':main()
