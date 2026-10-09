"""Attribute existing quality-model failures across already-viewed sequences."""
import argparse
import json
from pathlib import Path
from radar_id_tracker.lidar_proposal_diagnostics import diagnostic

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root',type=Path,required=True)
    p.add_argument('--quality-run',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args();rows=[]
    protocol=json.loads((args.quality_run/'models/protocol.json').read_text())
    for sequence in protocol['training_sequences']+protocol['validation_sequences']:
        feature=json.loads((args.quality_run/'features'/sequence/'manifest.json').read_text())
        rows.append(diagnostic(args.dataset_root,Path(feature['cache']),
            args.quality_run/sequence/'natural/candidate',args.output_dir/sequence))
        (args.output_dir/'diagnostics.json').write_text(json.dumps(rows,indent=2)+'\n')

if __name__=='__main__':main()
