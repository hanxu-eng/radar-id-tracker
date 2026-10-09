"""Explicit, reproducible raw-scan settings, independent of PyTorch and labels."""
from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True, slots=True)
class RawLidarConfig:
    sweeps: int = 5
    period_s: float = .1
    intensity_scale: float = 255.
    score_threshold: float = .1
    voxel_backend: str = 'numba'

    def __post_init__(self):
        if type(self.sweeps) is not int or not 1 <= self.sweeps <= 10:
            raise ValueError('sweeps must be an integer between 1 and 10')
        for name in ('period_s', 'intensity_scale', 'score_threshold'):
            value = getattr(self, name)
            if type(value) not in (float, int) or not isfinite(value):
                raise ValueError(f'{name} must be finite and numeric')
        if min(self.period_s, self.intensity_scale) <= 0:
            raise ValueError('period and intensity scale must be positive')
        if not 0 < self.score_threshold < 1:
            raise ValueError('score threshold must be within (0, 1)')
        if self.voxel_backend not in ('numpy', 'numba'):
            raise ValueError('voxel backend must be numpy or numba')

    @classmethod
    def from_dict(cls, values):
        if not isinstance(values, dict):
            raise ValueError('inference config must be a JSON object')
        return cls(**values)
