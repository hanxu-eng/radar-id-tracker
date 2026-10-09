"""Fetch a small KITTI tracking slice and published PointRCNN detections.

No model code is downloaded or executed. KITTI is CC BY-NC-SA 3.0.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

AB3DMOT_COMMIT = '61f3bd72574093e367916c757b4747ca445f978c'
BASE = f'https://raw.githubusercontent.com/xinshuoweng/AB3DMOT/{AB3DMOT_COMMIT}'
S3 = 'https://s3.eu-central-1.amazonaws.com/avg-kitti'


def download(url: str, path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        part = path.with_suffix(path.suffix + '.part')
        request = urllib.request.Request(url, headers={'User-Agent': 'radar-id-tracker-dataset-test'})
        with urllib.request.urlopen(request, timeout=90) as response, part.open('wb') as target:
            total = 0
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > 20 * 1024 * 1024:
                    raise ValueError(f'download exceeded 20 MiB limit: {url}')
                target.write(chunk)
        part.replace(path)
    return dict(url=url, path=str(path), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('datasets/kitti_tracking'))
    parser.add_argument('--sequences', nargs='+', default=['0000', '0001', '0005'])
    args = parser.parse_args()
    if any(len(s) != 4 or not s.isdecimal() or int(s) > 20 for s in args.sequences):
        parser.error('sequences must be training IDs 0000..0020')
    root = args.output_dir
    records = []
    for name, directory in [('data_tracking_label_2.zip', 'label_02'),
                            ('data_tracking_oxts.zip', 'oxts')]:
        archive = root / 'archives' / name
        records.append(download(f'{S3}/{name}', archive))
        with zipfile.ZipFile(archive) as source:
            for sequence in args.sequences:
                member = f'training/{directory}/{sequence}.txt'
                info = source.getinfo(member)
                if info.file_size > 20 * 1024 * 1024:
                    raise ValueError(f'archive member too large: {member}')
                target = root / member
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read(member))
    for sequence in args.sequences:
        path = f'data/KITTI/mini/training/calib/{sequence}.txt'
        records.append(download(f'{BASE}/{path}', root / 'training/calib' / f'{sequence}.txt'))
        for label in ('Car', 'Pedestrian', 'Cyclist'):
            path = f'data/KITTI/detection/pointrcnn_{label}_val/{sequence}.txt'
            records.append(download(f'{BASE}/{path}', root / 'detections' / label / f'{sequence}.txt'))
        print(f'downloaded sequence {sequence}', flush=True)
    manifest = dict(dataset='KITTI tracking training slice', sequences=args.sequences,
                    detector='published PointRCNN / AB3DMOT', ab3dmot_commit=AB3DMOT_COMMIT,
                    sources=records, raw_lidar_points_downloaded=(root/'pointcloud_manifest.json').exists(),
                    dataset_license='CC BY-NC-SA 3.0',
                    dataset_home='https://www.cvlibs.net/datasets/kitti/eval_tracking.php')
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(root / 'manifest.json')


if __name__ == '__main__':
    main()
