"""Causal 3D box tracking in a fixed, z-up metric coordinate frame.

No annotation IDs, Doppler or future frames enter this module. Box yaw is an
axis angle (modulo pi); it is not an appearance signature or a velocity heading.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import exp, isfinite, log, pi

import numpy as np

from .assignment import assign


def axis_difference(a: float, b: float) -> float:
    return (a - b + pi / 2) % pi - pi / 2


@dataclass(frozen=True, slots=True)
class BoxDetection:
    x: float
    y: float
    z: float
    length: float
    width: float
    height: float
    yaw: float
    confidence: float
    label: str

    def __post_init__(self) -> None:
        for name in ('x', 'y', 'z', 'length', 'width', 'height', 'yaw', 'confidence'):
            if not isfinite(getattr(self, name)):
                raise ValueError(f'{name} must be finite')
        if min(self.length, self.width, self.height) <= 0:
            raise ValueError('box dimensions must be positive')
        if not 0 <= self.confidence <= 1:
            raise ValueError('confidence must be within [0, 1]')
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError('label must be a nonempty string')


@dataclass(frozen=True, slots=True)
class BoxTrack:
    track_id: int
    timestamp_s: float
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    length: float
    width: float
    height: float
    yaw: float
    label: str
    confidence: float
    predicted_only: bool
    missed_frames: int
    last_observed_timestamp_s: float
    position_covariance: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class LidarTrackerConfig:
    high_confidence: float = 0.6
    low_confidence: float = 0.2
    confirmation_hits: int = 2
    confirmation_window_s: float = 0.4
    prediction_seconds: float = 0.5
    retention_seconds: float = 4.0  # total time since last measurement, not extra
    position_sigma_m: float = 0.45
    vertical_sigma_m: float = 0.3
    acceleration_sigma_mps2: float = 2.0
    position_gate_squared: float = 16.3
    active_residual_m: float = 4.0
    recovery_residual_m: float = 9.0
    maximum_height_residual_m: float = 1.5
    maximum_size_ratio: float = 2.0
    recovery_size_ratio: float = 1.6
    recovery_yaw_gate_rad: float = pi / 3
    recovery_ambiguity_margin: float = 0.08
    confidence_decay_s: float = 0.6
    maximum_tracks: int = 256

    def __post_init__(self) -> None:
        if not 0 <= self.low_confidence <= self.high_confidence <= 1:
            raise ValueError('confidence thresholds must satisfy 0 <= low <= high <= 1')
        if type(self.confirmation_hits) is not int or self.confirmation_hits < 2:
            raise ValueError('confirmation_hits must be an integer >= 2')
        if type(self.maximum_tracks) is not int or self.maximum_tracks < 1:
            raise ValueError('maximum_tracks must be a positive integer')
        for name in self.__dataclass_fields__:
            if name in ('high_confidence', 'low_confidence', 'confirmation_hits', 'maximum_tracks'):
                continue
            if not isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be positive and finite')
        if self.retention_seconds < self.prediction_seconds:
            raise ValueError('retention_seconds must cover prediction_seconds')
        if self.maximum_size_ratio <= 1 or self.recovery_size_ratio <= 1:
            raise ValueError('size ratios must exceed 1')


@dataclass(slots=True)
class _BoxState:
    mean: np.ndarray
    covariance: np.ndarray
    shape: BoxDetection
    first_seen_s: float
    last_seen_s: float
    hits: deque[float]
    track_id: int | None = None
    missed_frames: int = 0


class LidarTracker:
    """3D CV Kalman filter, class/shape gating and staged global assignment.

    Dormant identities are retained but never published. A dormant association
    must be unambiguous in both row and column; increasing retention alone does
    not relax the maximum recovery distance or the shape consistency checks.
    """

    def __init__(self, config: LidarTrackerConfig | None = None) -> None:
        self.config = config or LidarTrackerConfig()
        self._tracks: list[_BoxState] = []
        self._next_id = 1
        self._time_s: float | None = None
        self.diagnostics: dict[str, int] = {}

    def _noise(self, detection: BoxDetection) -> np.ndarray:
        scale = 1.0 + (1.0 - detection.confidence)
        return np.diag([self.config.position_sigma_m**2,
                        self.config.position_sigma_m**2,
                        self.config.vertical_sigma_m**2]) * scale**2

    def _predict(self, track: _BoxState, dt: float) -> None:
        f = np.eye(6)
        f[:3, 3:] = np.eye(3) * dt
        # Continuous white acceleration noise remains consistent when frames
        # are missing rather than depending on the number of update calls.
        q = np.zeros((6, 6))
        variance = self.config.acceleration_sigma_mps2**2
        for p in range(3):
            q[p, p] = variance * dt**3 / 3
            q[p, p+3] = q[p+3, p] = variance * dt**2 / 2
            q[p+3, p+3] = variance * dt
        track.mean = f @ track.mean
        track.covariance = f @ track.covariance @ f.T + q

    def _record(self, track: _BoxState, detection: BoxDetection, time_s: float) -> None:
        r = self._noise(detection)
        innovation = track.covariance[:3, :3] + r
        gain = np.linalg.solve(innovation, track.covariance[:3, :]).T
        residual = np.array([detection.x, detection.y, detection.z]) - track.mean[:3]
        track.mean += gain @ residual
        h = np.zeros((3, 6)); h[:, :3] = np.eye(3)
        a = np.eye(6) - gain @ h
        track.covariance = a @ track.covariance @ a.T + gain @ r @ gain.T
        track.covariance = (track.covariance + track.covariance.T) / 2
        # Use measured shape for association. Partial occlusion can change the
        # detected box; an EMA should not indefinitely mix distinct objects.
        track.shape = detection
        track.last_seen_s = time_s
        track.missed_frames = 0
        if track.track_id is None:
            track.hits.append(time_s)
            while track.hits and time_s - track.hits[0] > self.config.confirmation_window_s:
                track.hits.popleft()
            if len(track.hits) >= self.config.confirmation_hits:
                track.track_id = self._next_id
                self._next_id += 1
                track.hits.clear()

    def _cost(self, track: _BoxState, detection: BoxDetection, dormant: bool) -> float | None:
        old = track.shape
        if detection.label != old.label:
            return None
        delta = np.array([detection.x, detection.y, detection.z]) - track.mean[:3]
        gate = self.config.recovery_residual_m if dormant else self.config.active_residual_m
        if np.linalg.norm(delta[:2]) > gate or abs(delta[2]) > self.config.maximum_height_residual_m:
            return None
        distance = float(delta @ np.linalg.solve(track.covariance[:3, :3] + self._noise(detection), delta))
        if distance > self.config.position_gate_squared:
            return None
        size_errors = [abs(log(a/b)) for a,b in zip(
            (detection.length, detection.width, detection.height),
            (old.length, old.width, old.height))]
        ratio = self.config.recovery_size_ratio if dormant else self.config.maximum_size_ratio
        if max(size_errors) > log(ratio):
            return None
        angle = abs(axis_difference(detection.yaw, old.yaw))
        # Pedestrian box axes are unreliable; enforce orientation for vehicles.
        if dormant and old.label in ('Car', 'Cyclist', 'Vehicle') and angle > self.config.recovery_yaw_gate_rad:
            return None
        return (.45 * distance / self.config.position_gate_squared
                + .25 * float(np.linalg.norm(delta[:2])) / gate
                + .2 * sum(size_errors) / (3 * log(ratio))
                + .1 * angle / (pi/2))

    def update(self, timestamp_s: float, detections: list[BoxDetection]) -> list[BoxTrack]:
        if not isfinite(timestamp_s):
            raise ValueError('timestamp_s must be finite')
        if self._time_s is not None and timestamp_s <= self._time_s:
            raise ValueError('frame timestamps must strictly increase')
        if any(not isinstance(d, BoxDetection) for d in detections):
            raise TypeError('detections must be BoxDetection instances')
        cfg = self.config
        self.diagnostics = dict(recovered=0, ambiguous_recoveries=0, capacity_drops=0, expired=0)
        retained = []
        for track in self._tracks:
            age = timestamp_s - track.last_seen_s
            expired = age > cfg.retention_seconds + 1e-9 or (
                track.track_id is None and timestamp_s-track.first_seen_s > cfg.confirmation_window_s)
            if expired:
                self.diagnostics['expired'] += 1
            else:
                retained.append(track)
        self._tracks = retained
        if self._time_s is not None:
            for track in self._tracks:
                self._predict(track, timestamp_s - self._time_s)
        self._time_s = timestamp_s
        high = {i for i,d in enumerate(detections) if d.confidence >= cfg.high_confidence}
        low = {i for i,d in enumerate(detections) if cfg.low_confidence <= d.confidence < cfg.high_confidence}
        active = [i for i,t in enumerate(self._tracks) if t.track_id is not None
                  and timestamp_s-t.last_seen_s <= cfg.prediction_seconds + 1e-9]
        dormant = [i for i,t in enumerate(self._tracks) if t.track_id is not None
                   and timestamp_s-t.last_seen_s > cfg.prediction_seconds + 1e-9]
        tentative = [i for i,t in enumerate(self._tracks) if t.track_id is None]
        matched: set[int] = set()

        def stage(indices: list[int], candidates: set[int], recover: bool = False) -> None:
            ordered = sorted(candidates)
            if not indices or not ordered:
                return
            costs = [[self._cost(self._tracks[i], detections[j], recover) for j in ordered] for i in indices]
            if recover:
                # Determine best edges BEFORE removing ambiguity. Otherwise a
                # rejected best track can leave its detection to a worse track,
                # accidentally resurrecting somebody else's identity.
                row_best = {}; column_best = {}; ambiguous_rows = set(); ambiguous_columns = set()
                margin = cfg.recovery_ambiguity_margin
                for row, values in enumerate(costs):
                    valid = sorted((v,col) for col,v in enumerate(values) if v is not None)
                    if valid:
                        row_best[row] = valid[0][1]
                    if len(valid) > 1 and valid[1][0]-valid[0][0] < margin:
                        ambiguous_rows.add(row)
                for col in range(len(ordered)):
                    valid = sorted((values[col],row) for row,values in enumerate(costs) if values[col] is not None)
                    if valid:
                        column_best[col] = valid[0][1]
                    if len(valid) > 1 and valid[1][0]-valid[0][0] < margin:
                        ambiguous_columns.add(col)
                affected = ambiguous_rows | {row for row,values in enumerate(costs)
                                             if any(values[col] is not None for col in ambiguous_columns)}
                self.diagnostics['ambiguous_recoveries'] += len(affected)
                for row,values in enumerate(costs):
                    for col in range(len(ordered)):
                        if (row in ambiguous_rows or col in ambiguous_columns or
                                row_best.get(row) != col or column_best.get(col) != row):
                            values[col] = None
            for row,col in assign(costs, len(ordered)):
                i,j = indices[row],ordered[col]
                self._record(self._tracks[i], detections[j], timestamp_s)
                matched.add(i); candidates.remove(j)
                if recover:
                    self.diagnostics['recovered'] += 1

        stage(active, high)
        stage([i for i in active if i not in matched], low)
        stage(dormant, high, True)
        stage(tentative, high)
        for i,track in enumerate(self._tracks):
            if i not in matched:
                track.missed_frames += 1
        for j in sorted(high):
            if len(self._tracks) >= cfg.maximum_tracks:
                self.diagnostics['capacity_drops'] += 1
                continue
            d = detections[j]
            covariance = np.zeros((6,6))
            covariance[:3,:3] = self._noise(d)
            covariance[3:,3:] = np.eye(3)*25.
            self._tracks.append(_BoxState(
                np.array([d.x,d.y,d.z,0.,0.,0.]), covariance, d,
                timestamp_s,timestamp_s,deque([timestamp_s],maxlen=cfg.confirmation_hits)))
        output = []
        for track in self._tracks:
            age = timestamp_s-track.last_seen_s
            if track.track_id is None or age > cfg.prediction_seconds + 1e-9:
                continue
            d = track.shape; m = track.mean
            output.append(BoxTrack(
                track.track_id,timestamp_s,*map(float,m[:3]),*map(float,m[3:]),
                d.length,d.width,d.height,d.yaw,d.label,
                d.confidence*exp(-age/cfg.confidence_decay_s), age > 0,
                track.missed_frames,track.last_seen_s,
                tuple(map(float,track.covariance[:3,:3].ravel()))))
        return sorted(output,key=lambda t:t.track_id)
