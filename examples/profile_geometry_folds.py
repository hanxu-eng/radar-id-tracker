"""Isolated geometry candidate probe on the same fixed fold/sequence/frame range."""
import argparse
import json
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset-root','checkpoint','run-root'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();root=args.run_root;protocol=json.loads((root/'protocol.json').read_text());plan=protocol['parent_plan']
    config=dict(frozen_config=plan['tracker_config'],frozen_inference_config=plan['inference_config'])
    target=root/'latency_protocol.json'
    if target.exists() and json.loads(target.read_text())!=config:raise ValueError('geometry latency protocol differs')
    target.write_text(json.dumps(config,indent=2)+'\n')
    sys.argv=['profile_surround.py','--dataset-root',str(args.dataset_root),'--checkpoint',str(args.checkpoint),
        '--config',str(target),'--sequence','0009','--start','100','--frames','100',
        '--head-model',str(root/'fold_0/weighted_axis_huber/model/model.json'),'--output',str(root/'latency.json')]
    from profile_surround import main as profile
    profile()


if __name__=='__main__':main()
