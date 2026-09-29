"""Public input and output types. Coordinates use a fixed metric frame."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True, slots=True)
class Detection:
    """One radar detection at a frame timestamp.

    ``radial_velocity`` is the target's ego-motion-compensated radial speed
    in m/s, positive away from the radar. Missing Doppler is allowed.
    """

    x: float
    y: float
    confidence: float
    radial_velocity: float | None = None
    z: float = 0.0
    length: float | None = None
    width: float | None = None
    height: float | None = None
    yaw: float | None = None

    def __post_init__(self) -> None:
        for name in ("x", "y", "z", "confidence"):
            if not isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be within [0, 1]")
        for name in ("radial_velocity", "yaw"):
            value = getattr(self, name)
            if value is not None and not isfinite(value):
                raise ValueError(f"{name} must be finite")
        for name in ("length", "width", "height"):
            value = getattr(self, name)
            if value is not None and (not isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be positive and finite")


@dataclass(frozen=True, slots=True)
class TrackOutput:
    track_id: int
    timestamp_s: float
    x: float
    y: float
    vx: float
    vy: float
    z: float
    confidence: float
    predicted_only: bool
    missed_frames: int
    length: float | None
    width: float | None
    height: float | None
    yaw: float | None
