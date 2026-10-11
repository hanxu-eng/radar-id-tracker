import json
from pathlib import Path
import tempfile
import unittest

from examples.check_openpcdet_runtime import validate_platform
from examples.benchmark_openpcdet import portable_path, baseline_caches
from radar_id_tracker.openpcdet_adapter import sha256


class Ubuntu2404ContractTests(unittest.TestCase):
    def test_platform_check_rejects_system_python_and_other_architectures(self):
        info = {'ID': 'ubuntu', 'VERSION_ID': '24.04'}
        validate_platform(info, 'x86_64', (3, 11, 12))
        for os_info, arch, python in ((info, 'aarch64', (3, 11)),
                                       (info, 'x86_64', (3, 12)),
                                       ({'ID': 'ubuntu', 'VERSION_ID': '22.04'}, 'x86_64', (3, 11))):
            with self.assertRaises(RuntimeError): validate_platform(os_info, arch, python)

    def test_relative_manifest_resolves_from_manifest_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); cache = root / 'raw_cache/0000'; cache.mkdir(parents=True)
            (cache / 'detections.jsonl').write_text('{}\n')
            digest = sha256(cache / 'detections.jsonl')
            (cache / 'inference_summary.json').write_text(json.dumps(dict(
                sequence='0000', frames=1, period_s=.1, ground_truth_used_as_input=False,
                raw_pointcloud_detector_tested=True, output_sha256=digest)))
            manifest = root / 'cache_manifest.json'
            manifest.write_text(json.dumps({'records': {'0/0000/baseline': dict(
                role='evaluation', cache='raw_cache/0000', output_sha256=digest)}}))
            resolved = baseline_caches(manifest, ['0000'])
            self.assertEqual(Path(resolved['0000']['cache']), cache)

    def test_legacy_relocation_is_explicit_and_cannot_escape_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(portable_path(r'C:\old\repo\outputs\baseline\0000', cache_root=root),
                             root / 'baseline/0000')
            with self.assertRaisesRegex(ValueError, 'escapes'):
                portable_path(r'C:\old\repo\outputs\..\outside', cache_root=root)


if __name__ == '__main__':
    unittest.main()
