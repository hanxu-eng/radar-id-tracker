"""Pin upstream sources and download official KITTI checkpoints as data only."""
import argparse
import json
from pathlib import Path
import subprocess

from radar_id_tracker.openpcdet_adapter import MODEL_SPECS, UPSTREAM_COMMIT, UPSTREAM_URL, sha256


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path('outputs/vendor/OpenPCDet'))
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/models/openpcdet'))
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--models', nargs='+', choices=MODEL_SPECS, default=list(MODEL_SPECS))
    args = parser.parse_args(argv)
    repo = args.repo.resolve()
    if not repo.exists():
        subprocess.run(['git', 'clone', '--depth', '1', UPSTREAM_URL + '.git', str(repo)], check=True)
        # Pin a newly created checkout; never reset an existing checkout.
        subprocess.run(['git', '-C', str(repo), 'fetch', '--depth', '1', 'origin', UPSTREAM_COMMIT], check=True)
        subprocess.run(['git', '-C', str(repo), 'checkout', '--detach', UPSTREAM_COMMIT], check=True)
    commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != UPSTREAM_COMMIT:
        raise ValueError('checkout must be pinned to ' + UPSTREAM_COMMIT + '; existing sources will not be reset automatically')
    subprocess.run(['git', '-C', str(repo), 'diff', '--exit-code', 'HEAD', '--', 'pcdet', 'tools/cfgs'], check=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / 'assets.json'
    data = json.loads(path.read_text()) if path.exists() else {}
    source = {str(p.relative_to(repo)).replace('\\', '/'): sha256(p)
              for directory, pattern in (('pcdet', '*.py'), ('pcdet', '*.cpp'),
                                         ('pcdet', '*.cu'), ('pcdet', '*.h'), ('tools/cfgs', '*.yaml'))
              for p in sorted((repo / directory).rglob(pattern)) if p.name != 'version.py'}
    if data and (data.get('upstream_commit') != commit or data.get('source_sha256') != source):
        raise ValueError('existing asset manifest differs from source checkout')
    data.update(format_version=1, upstream_url=UPSTREAM_URL, upstream_commit=commit,
                source_sha256=source, model_zoo_url=UPSTREAM_URL + '#model-zoo',
                checkpoint_hash_origin='locally measured after first official-link download; not an upstream signed hash',
                pretrained_scene_overlap_status='not audited', models=data.get('models', {}))
    failed = False
    for name, spec in MODEL_SPECS.items():
        record = data['models'].get(name, dict(spec, status='not_downloaded'))
        target = args.output_dir / spec['filename']
        if record.get('status') == 'downloaded':
            if not target.is_file() or sha256(target) != record['sha256']:
                raise ValueError('existing checkpoint changed: ' + name)
        elif args.download and name in args.models:
            try:
                import gdown
                partial = target.with_suffix('.download')
                result = gdown.download(id=spec['drive_id'], output=str(partial), quiet=False, use_cookies=False)
                if result is None or not partial.exists():
                    raise ValueError('official download did not return a file')
                with partial.open('rb') as source_file:
                    header = source_file.read(512).lstrip().lower()
                if partial.stat().st_size < 1024 * 1024 or b'<html' in header or b'<!doctype html' in header:
                    raise ValueError('download is not a plausible checkpoint (small file or HTML response)')
                partial.replace(target)
                record = dict(spec, status='downloaded', bytes=target.stat().st_size,
                              sha256=sha256(target), downloaded_via='official OpenPCDet model-zoo Google Drive link')
            except Exception as error:
                failed = True
                record = dict(spec, status='download_failed', error=str(error))
        data['models'][name] = record
        path.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(dict(model=name, **record)), flush=True)
    return 2 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
