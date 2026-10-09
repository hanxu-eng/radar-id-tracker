"""Download only pinned published predictions; keep the external KITTI dataset read-only."""
import argparse
import json
from pathlib import Path

from download_kitti_tracking import AB3DMOT_COMMIT, BASE, download


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,default=Path('datasets/kitti_tracking/detections'))
    p.add_argument('--sequences',nargs='+',required=True)
    args=p.parse_args()
    if any(len(s)!=4 or not s.isdecimal() or int(s)>20 for s in args.sequences):
        p.error('sequences must be training IDs 0000..0020')
    records=[]
    for sequence in args.sequences:
        for label in ('Car','Pedestrian','Cyclist'):
            source=f'{BASE}/data/KITTI/detection/pointrcnn_{label}_val/{sequence}.txt'
            records.append(download(source,args.output_dir/label/f'{sequence}.txt'))
        print(f'published predictions downloaded: {sequence}',flush=True)
    path=args.output_dir/'prediction_manifest.json'
    previous=json.loads(path.read_text()) if path.exists() else {}
    sources={r['url']:r for r in previous.get('sources',[])}
    sources.update({r['url']:r for r in records})
    path.write_text(json.dumps(dict(ab3dmot_commit=AB3DMOT_COMMIT,
                                    detector='published PointRCNN / AB3DMOT',
                                    sources=list(sources.values())),indent=2)+'\n')


if __name__=='__main__':main()
