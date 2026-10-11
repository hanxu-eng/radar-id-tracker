"""Fail-closed Ubuntu 24.04/CUDA smoke check, separate from accuracy evaluation."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

from radar_id_tracker.openpcdet_adapter import OpenPCDetDetector, predictions_to_world


def validate_platform(os_info, architecture, python_version):
    if os_info.get('ID') != 'ubuntu' or os_info.get('VERSION_ID') != '24.04':
        raise RuntimeError('This runtime recipe requires native Ubuntu 24.04')
    if architecture != 'x86_64':
        raise RuntimeError('This recipe requires x86_64; Jetson/ARM needs a separate dependency recipe')
    if tuple(python_version[:2]) != (3, 11):
        raise RuntimeError('Activate the isolated Python 3.11 environment')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path('outputs/vendor/OpenPCDet'))
    parser.add_argument('--assets', type=Path, default=Path('outputs/models/openpcdet/assets.json'))
    parser.add_argument('--output', type=Path, default=Path('outputs/ubuntu2404/runtime_check.json'))
    parser.add_argument('--scan', type=Path, help='optional one real Nx4 scan for both model forward passes')
    args = parser.parse_args(argv)
    report = dict(passed=False, accuracy_evaluated=False, gpu_forward_tested=False,
                  platform=platform.platform(), python=sys.version, checks=[])
    try:
        validate_platform(platform.freedesktop_os_release(), platform.machine(), sys.version_info)
        report['checks'].append('Ubuntu 24.04 x86_64 / Python 3.11')
        import numpy as np
        import torch
        if torch.__version__.split('+')[0] != '2.7.1' or torch.version.cuda != '12.6':
            raise RuntimeError('Expected torch 2.7.1 with CUDA 12.6 wheel')
        if not torch.cuda.is_available():
            raise RuntimeError('NVIDIA GPU is unavailable to torch')
        nvcc = subprocess.check_output(['nvcc', '--version'], text=True)
        if 'release 12.6' not in nvcc:
            raise RuntimeError('Expected CUDA Toolkit/nvcc 12.6 for custom operators')
        if importlib.metadata.version('spconv-cu120') != '2.3.6':
            raise RuntimeError('Expected spconv-cu120 2.3.6')
        if np.__version__ != '1.26.4':
            raise RuntimeError('Expected NumPy 1.26.4')
        report.update(torch=torch.__version__, torch_cuda=torch.version.cuda,
                      gpu=torch.cuda.get_device_name(), nvcc=nvcc,
                      cuda_capability=list(torch.cuda.get_device_capability()))
        # Build/load the real selected networks, with strict tensor coverage.
        # This also checks all upstream registry imports, including av2.
        for name in ('second_iou', 'pv_rcnn'):
            detector = OpenPCDetDetector(args.repo, args.assets, name)
            report.setdefault('model_tensors_loaded', {})[name] = detector.loaded_tensors
            if args.scan:
                from radar_id_tracker.pointcloud_review import read_scan
                boxes, scores, labels = detector.predict(read_scan(args.scan))
                predictions_to_world(boxes, scores, labels, np.eye(4))
                detector.synchronize()
                report.setdefault('scan_detection_counts', {})[name] = len(boxes)
            del detector
            torch.cuda.empty_cache()
        report['checks'].append('both official networks initialized with complete model tensors')
        from pcdet.ops.iou3d_nms import iou3d_nms_utils
        from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
        boxes = torch.tensor([[0., 0., 0., 2., 2., 2., 0.],
                              [0., 0., 0., 2., 2., 2., 0.]], device='cuda')
        overlaps = iou3d_nms_utils.boxes_iou3d_gpu(boxes, boxes)
        if not torch.allclose(overlaps, torch.ones_like(overlaps), atol=1e-5):
            raise RuntimeError('CUDA IoU known-answer check failed')
        keep, _ = iou3d_nms_utils.nms_gpu(boxes, torch.tensor([.9, .8], device='cuda'), .1)
        if keep.numel() != 1 or keep.item() != 0:
            raise RuntimeError('CUDA rotated NMS known-answer check failed')
        points = torch.tensor([[[0., 0., 0.], [10., 0., 0.]]], device='cuda')
        inside = roiaware_pool3d_utils.points_in_boxes_gpu(points, boxes[:1].unsqueeze(0))
        if inside.cpu().tolist() != [[0, -1]]:
            raise RuntimeError('CUDA point-in-box known-answer check failed')
        import spconv.pytorch as spconv
        coords = torch.tensor([[0, 1, 1, 1], [0, 1, 1, 2]], dtype=torch.int32, device='cuda')
        sparse = spconv.SparseConvTensor(torch.ones((2, 4), device='cuda'), coords, [4, 4, 4], 1)
        convolution = spconv.SubMConv3d(4, 4, 3, padding=1, bias=False).cuda().eval()
        with torch.inference_mode():
            features = convolution(sparse).features
        torch.cuda.synchronize()
        if features.shape != (2, 4) or not torch.isfinite(features).all():
            raise RuntimeError('spconv CUDA convolution smoke check failed')
        report['checks'].append('CUDA IoU, rotated NMS, point-in-box and sparse convolution executed')
        report['gpu_forward_tested'] = args.scan is not None
        report['passed'] = True
    except Exception as error:
        report['error'] = str(error)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    return 0 if report['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
