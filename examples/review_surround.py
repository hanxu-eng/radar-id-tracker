"""Attach original pointcloud assets to the scoped detector comparison."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from radar_id_tracker.pointcloud_review import prepare_clouds
from radar_id_tracker.lidar_visualization import build_review
from benchmark_surround import label_review


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root',type=Path,required=True)
    p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--reuse-assets',type=Path,help='existing pointclouds directory, source hashes checked')
    p.add_argument('--comparison',choices=('detector','input','quality'),default='detector')
    args=p.parse_args()
    rows=json.loads((args.run_root/'benchmark.json').read_text())
    sequences=sorted({r['sequence'] for r in rows})
    if args.reuse_assets:
        source_manifest=json.loads((args.reuse_assets/'manifest.json').read_text())
        manifest=dict(source_manifest)
        manifest['frames']={s:source_manifest['frames'][s] for s in sequences}
        manifest['missing_frames']=[f for f in source_manifest['missing_frames'] if f['sequence'] in sequences]
        manifest['original_points']=sum(f['original_points'] for seq in manifest['frames'].values() for f in seq.values())
        manifest['display_points']=sum(f['display_points'] for seq in manifest['frames'].values() for f in seq.values())
        manifest['dataset_root']=str(args.dataset_root.resolve())
        target=args.run_root/'pointclouds';target.mkdir(parents=True,exist_ok=True)
        for sequence,frames in manifest['frames'].items():
            (target/sequence).mkdir(exist_ok=True)
            for index,metadata in frames.items():
                name=f'{int(index):06d}'
                scan=args.dataset_root/'training/velodyne'/sequence/(name+'.bin')
                if hashlib.sha256(scan.read_bytes()).hexdigest()!=metadata['source_sha256']:
                    raise ValueError('source cloud mismatch: '+str(scan))
                destination=target/sequence/(name+'.js')
                source=args.reuse_assets/sequence/(name+'.js')
                if not destination.exists():os.link(source,destination)
        (target/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    else:prepare_clouds(args.dataset_root,args.run_root,sequences)
    build_review(args.run_root,rows);label_review(args.run_root/'review.html',args.comparison)
    print(args.run_root/'review.html')


if __name__=='__main__':main()
