"""Official OpenPCDet inference bridge; no learned architecture is ported here.

KITTI boxes are gravity centers in x-forward/y-left/z-up LiDAR coordinates.
The upstream demo input recipe uses one Nx4 scan, unchanged reflectance and
the official voxel/range/NMS settings. It does not apply image-FOV cropping.
Annotations are read only by separate evaluation code.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np

from .kitti_adapter import calibration_transforms, load_poses
from .lidar_tracker import BoxDetection
from .pointcloud_review import read_scan

UPSTREAM_URL = 'https://github.com/open-mmlab/OpenPCDet'
UPSTREAM_COMMIT = '233f849829b6ac19afb8af8837a0246890908755'
CLASS_NAMES = ('Car', 'Pedestrian', 'Cyclist')
MODEL_SPECS = {
    'second_iou': {
        'config': 'tools/cfgs/kitti_models/second_iou.yaml',
        'filename': 'second_iou7909.pth',
        'drive_id': '1AQkeNs4bxhvhDQ-5sEo_yvQUlfo73lsW',
    },
    'parta2_free': {
        'config': 'tools/cfgs/kitti_models/PartA2_free.yaml',
        'filename': 'PartA2_free.pth',
        'drive_id': '1lcUUxF8mJgZ_e-tZhP1XNQtTBuC-R0zr',
    },
    'pv_rcnn': {
        'config': 'tools/cfgs/kitti_models/pv_rcnn.yaml',
        'filename': 'pv_rcnn_8369.pth',
        'drive_id': '1lIOq4Hxr0W3qsX83ilQv0nk1Cls6KAr-',
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_transform(transform):
    matrix = np.asarray(transform, dtype=np.float64)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1])
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(matrix[:3, :3]), 1., atol=1e-6)):
        raise ValueError('expected a proper rigid LiDAR-to-world transform')
    return matrix


def predictions_to_world(boxes, scores, labels, world_from_lidar):
    """Preserve upstream center, dimensions, yaw and probability scores.

    Reject malformed predictions instead of clipping scores or silently dropping
    failed outputs. OpenPCDet class indices are one-based, not CenterPoint IDs.
    """
    matrix = validate_transform(world_from_lidar)
    boxes = np.asarray(boxes, dtype=float)
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    if (boxes.ndim != 2 or boxes.shape[1] != 7
            or scores.shape != (len(boxes),) or labels.shape != scores.shape
            or not np.isfinite(boxes).all() or not np.isfinite(scores).all()
            or not np.isfinite(labels).all()):
        raise ValueError('expected finite Nx7 boxes and N scores/class indices')
    if (np.any(boxes[:, 3:6] <= 0) or np.any((scores < 0) | (scores > 1))
            or np.any(labels != np.floor(labels)) or np.any((labels < 1) | (labels > 3))):
        raise ValueError('invalid box dimensions, probability score or one-based class')
    result = []
    for box, score, label in zip(boxes, scores, labels):
        center = matrix @ np.r_[box[:3], 1.]
        heading = matrix[:3, :3] @ [np.cos(box[6]), np.sin(box[6]), 0.]
        result.append(BoxDetection(*map(float, center[:3]), *map(float, box[3:6]),
                                  float(np.arctan2(heading[1], heading[0])),
                                  float(score), CLASS_NAMES[int(label) - 1]))
    return result


def validate_assets(repo: Path, manifest: Path, model_name: str):
    if model_name not in MODEL_SPECS:
        raise ValueError('unsupported OpenPCDet model')
    repo = repo.resolve()
    commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    data = json.loads(manifest.read_text(encoding='utf-8'))
    if commit != UPSTREAM_COMMIT or data.get('upstream_commit') != commit:
        raise ValueError('OpenPCDet source commit differs from pinned manifest')
    if data.get('upstream_url') != UPSTREAM_URL:
        raise ValueError('unexpected upstream source')
    for relative, expected in data['source_sha256'].items():
        path = (repo / relative).resolve()
        if not path.is_relative_to(repo) or sha256(path) != expected:
            raise ValueError('OpenPCDet source checksum mismatch: ' + relative)
    record = data['models'][model_name]
    if record.get('status') != 'downloaded':
        raise ValueError('official checkpoint is not downloaded: ' + model_name)
    checkpoint = (manifest.parent / record['filename']).resolve()
    if not checkpoint.is_relative_to(manifest.parent.resolve()):
        raise ValueError('checkpoint escapes asset directory')
    if sha256(checkpoint) != record['sha256']:
        raise ValueError('official checkpoint checksum mismatch')
    if record.get('drive_id') != MODEL_SPECS[model_name]['drive_id']:
        raise ValueError('checkpoint source differs from official model-zoo link')
    return checkpoint, data


@contextmanager
def _working_directory(directory):
    previous = Path.cwd()
    os.chdir(directory)
    try:
        yield
    finally:
        os.chdir(previous)


class OpenPCDetDetector:
    """Lazy CUDA runtime: tracker-only installations need neither torch nor pcdet."""
    def __init__(self, repo: Path, manifest: Path, model_name: str):
        checkpoint, self.assets = validate_assets(repo, manifest, model_name)
        repo = repo.resolve()
        if sys.platform != 'linux':
            raise RuntimeError('Use a prepared Linux CUDA environment for official OpenPCDet operators; Windows runtime has not been validated')
        sys.path.insert(0, str(repo))
        import torch
        import pcdet
        from easydict import EasyDict
        from pcdet.config import cfg_from_yaml_file
        from pcdet.datasets import DatasetTemplate
        from pcdet.models import build_network, load_data_to_gpu
        if not Path(pcdet.__file__).resolve().is_relative_to(repo):
            raise ValueError('imported pcdet does not belong to the pinned checkout')
        if not torch.cuda.is_available():
            raise RuntimeError('OpenPCDet baseline requires CUDA')
        self.torch = torch
        torch.manual_seed(0)
        np.random.seed(0)
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark = False
        self.model_name = model_name
        self.config = EasyDict()
        with _working_directory(repo / 'tools'):
            cfg_from_yaml_file(str(repo / MODEL_SPECS[model_name]['config']), self.config)
        if tuple(self.config.CLASS_NAMES) != CLASS_NAMES:
            raise ValueError('official three-class KITTI configuration required')
        logger = logging.getLogger('radar_id_tracker.openpcdet')
        self.dataset = DatasetTemplate(dataset_cfg=self.config.DATA_CONFIG,
                                       class_names=list(CLASS_NAMES), training=False,
                                       root_path=repo, logger=logger)
        self.model = build_network(model_cfg=self.config.MODEL, num_class=3, dataset=self.dataset)
        # Trusted official asset only, checked before deserialization. Do not use
        # upstream's permissive loader: missing model tensors must fail closed.
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        _, loaded = self.model._load_state_dict(state['model_state'], strict=True)
        self.loaded_tensors = len(loaded)
        self.model.cuda().eval()
        self.load_data_to_gpu = load_data_to_gpu

    def synchronize(self):
        self.torch.cuda.synchronize()

    def predict(self, points):
        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 4 or not np.isfinite(points).all():
            raise ValueError('official KITTI model expects a finite single Nx4 scan')
        prepared = self.dataset.prepare_data({'points': points.copy(), 'frame_id': 0})
        batch = self.dataset.collate_batch([prepared])
        self.load_data_to_gpu(batch)
        with self.torch.inference_mode():
            predictions, _ = self.model(batch)
        record = predictions[0]
        return tuple(record[key].detach().cpu().numpy() for key in
                     ('pred_boxes', 'pred_scores', 'pred_labels'))

    def provenance(self):
        processors = self.config.DATA_CONFIG.DATA_PROCESSOR
        return dict(model_name=self.model_name, upstream_url=UPSTREAM_URL,
                    upstream_commit=UPSTREAM_COMMIT,
                    checkpoint_sha256=self.assets['models'][self.model_name]['sha256'],
                    checkpoint_tensors_loaded_strictly=self.loaded_tensors,
                    source_sha256=self.assets['source_sha256'],
                    point_cloud_range=list(self.config.DATA_CONFIG.POINT_CLOUD_RANGE),
                    data_processors=processors,
                    detector_score_threshold=float(self.config.MODEL.POST_PROCESSING.SCORE_THRESH),
                    post_processing=dict(self.config.MODEL.POST_PROCESSING),
                    input_recipe='official DemoDataset prepare_data, single raw Nx4 scan; no image-FOV crop',
                    confidence_is_calibrated=False, torch_version=self.torch.__version__,
                    gpu=self.torch.cuda.get_device_name())


def infer_sequence(root: Path, sequence: str, output_dir: Path, detector, period_s=.1):
    """Produce the existing label-free replay cache without importing annotations."""
    if len(sequence) != 4 or not sequence.isascii() or not sequence.isdecimal():
        raise ValueError('sequence must be a four-digit ID')
    if not np.isfinite(period_s) or period_s <= 0:
        raise ValueError('period must be finite and positive')
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError('output directory must be new or empty')
    poses = load_poses(root / 'training/oxts' / f'{sequence}.txt')
    _, lidar_to_imu = calibration_transforms(root / 'training/calib' / f'{sequence}.txt')
    provenance = detector.provenance()
    # Warm-up an actual current scan; no prior/future scans are inference inputs.
    warm_path = next((root / 'training/velodyne' / sequence / f'{i:06d}.bin'
                      for i in range(len(poses))
                      if (root / 'training/velodyne' / sequence / f'{i:06d}.bin').is_file()), None)
    if warm_path is None:
        raise ValueError('sequence contains no raw scans')
    detector.predict(read_scan(warm_path))
    detector.synchronize()
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / 'detections.jsonl'
    partial = output_dir / 'detections.jsonl.part'
    timing, inputs, missing = [], [], []
    with partial.open('w', encoding='utf-8') as destination:
        for index, pose in enumerate(poses):
            start = perf_counter()
            transform = pose @ lidar_to_imu
            path = root / 'training/velodyne' / sequence / f'{index:06d}.bin'
            available = path.is_file()
            detections, inference_ms = [], None
            if available:
                inputs.append(dict(frame_index=index, bytes=path.stat().st_size, sha256=sha256(path)))
                points = read_scan(path)
                detector.synchronize()
                model_start = perf_counter()
                predictions = detector.predict(points)
                detector.synchronize()
                inference_ms = (perf_counter() - model_start) * 1000
                detections = predictions_to_world(*predictions, transform)
            else:
                missing.append(index)
            elapsed = (perf_counter() - start) * 1000
            if available:
                timing.append(elapsed)
            row = dict(frame_index=index, timestamp_s=index * period_s,
                       coordinate_frame='sequence_enu_z_up', sensor_xyz=transform[:3, 3].tolist(),
                       world_from_lidar=transform.tolist(), raw_scan_available=available,
                       synthetic_detector_dropout=False, detections=[asdict(d) for d in detections],
                       sweeps_used=[index] if available else [],
                       end_to_end_ms=elapsed, inference_ms=inference_ms)
            destination.write(json.dumps(row, allow_nan=False) + '\n')
            if index % 50 == 0 or index == len(poses) - 1:
                print(json.dumps(dict(sequence=sequence, frame=index, total=len(poses),
                                      detections=len(detections))), flush=True)
    partial.replace(target)
    summary = dict(sequence=sequence, frames=len(poses), period_s=period_s,
                   dataset_root=str(root.resolve()), ground_truth_used_as_input=False,
                   raw_pointcloud_detector_tested=True, detector='official OpenPCDet ' + provenance['model_name'],
                   **provenance, input_scans=inputs, missing_scan_frames=missing,
                   causal_past_only=True, sweeps=1, intensity_scale=1., point_deskewed=False,
                   ego_motion_compensated=False, world_pose_applied_to_output=True,
                   rear_validated=False, domain_transfer='KITTI Object pretrained -> KITTI Tracking; RoboSense unvalidated',
                   pretrained_scene_overlap_status='not audited; development results only',
                   time_source='nominal_frame_period_not_measured_timestamps', warmup_excluded=True,
                   median_detector_pipeline_ms=float(np.median(timing)),
                   p99_detector_pipeline_ms=float(np.quantile(timing, .99)),
                   latency_includes='scan read + SHA256 + official preprocessing + GPU inference + NMS + world transform',
                   output_sha256=sha256(target),
                   implementation_sha256=sha256(Path(__file__)))
    (output_dir / 'inference_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return summary
