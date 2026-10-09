"""Label-free whole-sequence split validation for experimental detector training."""
import hashlib
import json
from pathlib import Path
from .lidar_inference_config import RawLidarConfig
from .lidar_tracker import LidarTrackerConfig


def load_training_plan(path: Path):
    if path.stat().st_size > 65536:
        raise ValueError('training plan too large')
    raw = path.read_bytes(); data = json.loads(raw)
    if data['format_version'] != 1 or data['validation_used_for_selection'] is not False:
        raise ValueError('unsupported or unfrozen training plan')
    keys = ('training_sequences', 'validation_sequences', 'reserved_sequences',
            'previously_viewed_sequences', 'diagnostic_replay_sequences')
    for key in keys:
        values = data[key]
        if (not isinstance(values, list) or len(values) > 21 or
                any(not isinstance(s, str) or len(s) != 4 or not s.isascii() or
                    not s.isdecimal() or not 0 <= int(s) <= 20 for s in values) or
                len(set(values)) != len(values)):
            raise ValueError('split must contain unique KITTI training sequence IDs')
    training, validation, reserved = (set(data[k]) for k in keys[:3])
    if not training or not validation or training & validation or training & reserved or validation & reserved:
        raise ValueError('training, validation and reserved splits must be disjoint')
    viewed = set(data['previously_viewed_sequences'])
    if not viewed <= training or validation & viewed or reserved & viewed:
        raise ValueError('previously viewed sequences belong in development training')
    if not set(data['diagnostic_replay_sequences']) <= training:
        raise ValueError('diagnostic training replay must belong to training')
    RawLidarConfig.from_dict(data['inference_config'])
    LidarTrackerConfig.from_dict(data['tracker_config'])
    for key in ('acceptance_precision', 'acceptance_recall'):
        value = data[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value <= 1:
            raise ValueError('invalid frozen acceptance guard')
    return data, hashlib.sha256(raw).hexdigest()
