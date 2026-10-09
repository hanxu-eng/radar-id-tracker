"""Additional frozen-model IoU=.5 audit; leaves original IoU=.25 files intact."""
import argparse
import hashlib
import json
from pathlib import Path
from radar_id_tracker.lidar_scoped_evaluate import evaluate_scoped

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root',type=Path,required=True)
    p.add_argument('--run-root',type=Path,required=True)
    args=p.parse_args();rows=[]
    protocol=json.loads((args.run_root/'evaluation_protocol.json').read_text())
    for sequence in protocol['validation_sequences']:
        for profile in ('baseline','candidate'):
            run=args.run_root/sequence/'natural'/profile
            digest=hashlib.sha256((run/'evaluation.json').read_bytes()).hexdigest()
            result=evaluate_scoped(args.dataset_root,run,threshold=.5,write=False)
            assert digest==hashlib.sha256((run/'evaluation.json').read_bytes()).hexdigest()
            rows.append(dict(sequence=sequence,profile=profile,role='validation',evaluation=result))
            print(json.dumps(dict(sequence=sequence,profile=profile,**result['observed_tracks'])),flush=True)
    (args.run_root/'strict_iou_results.json').write_text(json.dumps(rows,indent=2)+'\n')

if __name__=='__main__':main()
