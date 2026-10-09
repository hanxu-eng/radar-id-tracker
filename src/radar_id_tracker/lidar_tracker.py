"""Causal 3D box tracking in a fixed, z-up metric coordinate frame.

No annotation IDs, Doppler or future frames enter this module. Box yaw is an
axis angle (modulo pi); it is not an appearance signature or a velocity heading.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from math import exp, isfinite, log, pi

import numpy as np

from .assignment import assign
from .box_geometry import iou3d


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
    confirmation_path: str = 'strong'
    confirmation_delay_s: float = 0.


@dataclass(frozen=True, slots=True)
class ClassScorePolicy:
    label: str
    continuation_confidence: float
    birth_confidence: float
    strong_confidence: float
    recovery_confidence: float

    def __post_init__(self) -> None:
        if not isinstance(self.label,str) or not self.label.strip():
            raise ValueError('policy label must be nonempty')
        scores=(self.continuation_confidence,self.birth_confidence,self.strong_confidence)
        if not all(isfinite(v) for v in (*scores,self.recovery_confidence)) or not (
                0 <= scores[0] <= scores[1] <= scores[2] <= 1 and scores[0] <= self.recovery_confidence <= 1):
            raise ValueError('policy requires 0 <= continuation <= birth <= strong <= 1 and continuation <= recovery <= 1')


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
    score_policies: tuple[ClassScorePolicy, ...] = ()
    medium_confirmation_hits: int = 3
    medium_confirmation_window_s: float = 0.4
    maximum_confirmation_gap_s: float = 0.2
    maximum_tentative_tracks: int | None = None
    duplicate_birth_iou: float | None = None

    def score_policy(self,label: str) -> ClassScorePolicy:
        return next((p for p in self.score_policies if p.label == label),
                    ClassScorePolicy(label,self.low_confidence,self.high_confidence,
                                     self.high_confidence,self.high_confidence))

    @classmethod
    def from_dict(cls,values: dict) -> LidarTrackerConfig:
        values=dict(values)
        values['score_policies']=tuple(ClassScorePolicy(**p) for p in values.get('score_policies',()))
        return cls(**values)

    def __post_init__(self) -> None:
        if not isinstance(self.score_policies,tuple) or any(not isinstance(p,ClassScorePolicy) for p in self.score_policies):
            raise ValueError('score_policies must be a tuple of ClassScorePolicy')
        if len({p.label for p in self.score_policies}) != len(self.score_policies):
            raise ValueError('duplicate class score policy')
        if type(self.medium_confirmation_hits) is not int or self.medium_confirmation_hits < 3:
            raise ValueError('medium_confirmation_hits must be an integer >= 3')
        if self.maximum_tentative_tracks is not None and (
                type(self.maximum_tentative_tracks) is not int or self.maximum_tentative_tracks < 1):
            raise ValueError('maximum_tentative_tracks must be a positive integer or None')
        if self.duplicate_birth_iou is not None and (not isfinite(self.duplicate_birth_iou) or not 0 < self.duplicate_birth_iou <= 1):
            raise ValueError('duplicate_birth_iou must be in (0,1] or None')
        if not 0 <= self.low_confidence <= self.high_confidence <= 1:
            raise ValueError('confidence thresholds must satisfy 0 <= low <= high <= 1')
        if type(self.confirmation_hits) is not int or self.confirmation_hits < 2:
            raise ValueError('confirmation_hits must be an integer >= 2')
        if type(self.maximum_tracks) is not int or self.maximum_tracks < 1:
            raise ValueError('maximum_tracks must be a positive integer')
        for name in self.__dataclass_fields__:
            if name in ('high_confidence', 'low_confidence', 'confirmation_hits', 'maximum_tracks',
                        'score_policies','medium_confirmation_hits','maximum_tentative_tracks','duplicate_birth_iou'):
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
    medium_hits: deque[float] = field(default_factory=deque)
    confirmation_path: str = 'strong'
    confirmation_delay_s: float = 0.
    candidate_id: int = 0


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
        self.class_diagnostics: dict[str,dict[str,int]] = {}
        self.tentative_outputs: list[dict] = []
        self._next_candidate_id = 1

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
        gap=time_s-track.last_seen_s
        track.last_seen_s = time_s
        track.missed_frames = 0
        if track.track_id is None:
            policy=self.config.score_policy(detection.label)
            if detection.confidence >= policy.strong_confidence:
                track.hits.append(time_s)
            while track.hits and time_s - track.hits[0] > self.config.confirmation_window_s:
                track.hits.popleft()
            if gap > self.config.maximum_confirmation_gap_s+1e-9:
                track.medium_hits.clear()
            track.medium_hits.append(time_s)
            while track.medium_hits and time_s-track.medium_hits[0] > self.config.medium_confirmation_window_s+1e-9:
                track.medium_hits.popleft()
            strong=len(track.hits) >= self.config.confirmation_hits
            medium=(policy.birth_confidence < policy.strong_confidence and
                    len(track.medium_hits) >= self.config.medium_confirmation_hits)
            if strong or medium:
                track.track_id = self._next_id
                self._next_id += 1
                track.confirmation_path='strong' if strong else 'medium'
                track.confirmation_delay_s=time_s-track.first_seen_s
                self.diagnostics[f'confirmed_{track.confirmation_path}'] += 1
                self._count(detection.label,f'confirmed_{track.confirmation_path}')
                track.hits.clear()
                track.medium_hits.clear()

    def _count(self,label: str,key: str) -> None:
        counts=self.class_diagnostics.setdefault(label,{})
        counts[key]=counts.get(key,0)+1

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
        self.diagnostics = dict(recovered=0, ambiguous_recoveries=0, capacity_drops=0, expired=0,
                                confirmed_strong=0,confirmed_medium=0,tentative_created=0,tentative_expired=0)
        self.class_diagnostics={}
        policies=[cfg.score_policy(d.label) for d in detections]
        for d,p in zip(detections,policies):
            self._count(d.label,'input_detections')
            if d.confidence < p.continuation_confidence:self._count(d.label,'below_continuation')
        retained = []
        for track in self._tracks:
            age = timestamp_s - track.last_seen_s
            policy=cfg.score_policy(track.shape.label)
            window=(max(cfg.confirmation_window_s,cfg.medium_confirmation_window_s)
                    if policy.birth_confidence < policy.strong_confidence else cfg.confirmation_window_s)
            window_tolerance=1e-9 if policy.birth_confidence < policy.strong_confidence else 0.
            expired = age > cfg.retention_seconds + 1e-9 or (
                track.track_id is None and timestamp_s-track.first_seen_s > window+window_tolerance)
            if expired:
                self.diagnostics['expired'] += 1
                if track.track_id is None:
                    self.diagnostics['tentative_expired'] += 1
                    self._count(track.shape.label,'tentative_expired')
            else:
                retained.append(track)
        self._tracks = retained
        if self._time_s is not None:
            for track in self._tracks:
                self._predict(track, timestamp_s - self._time_s)
        self._time_s = timestamp_s
        high = {i for i,(d,p) in enumerate(zip(detections,policies)) if d.confidence >= p.strong_confidence}
        low = {i for i,(d,p) in enumerate(zip(detections,policies)) if p.continuation_confidence <= d.confidence < p.strong_confidence}
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
                high.discard(j);low.discard(j)
                self._count(detections[j].label,'recovered' if recover else
                            'continued' if i in active else 'tentative_matched')
                if recover:
                    self.diagnostics['recovered'] += 1

        stage(active, high)
        stage([i for i in active if i not in matched], low)
        recovery={i for i in high|low if detections[i].confidence >= policies[i].recovery_confidence}
        stage(dormant, recovery, True)
        births={i for i in high|low if detections[i].confidence >= policies[i].birth_confidence}
        stage(tentative, births & high)
        births.intersection_update(high|low)
        stage([i for i in tentative if i not in matched], births)
        for i,track in enumerate(self._tracks):
            if i not in matched:
                track.missed_frames += 1
                if track.track_id is None:track.medium_hits.clear()
        tentative_count=sum(t.track_id is None for t in self._tracks)
        order=(sorted(births,key=lambda j:(-detections[j].confidence,j)) if cfg.score_policies else sorted(births))
        for j in order:
            if cfg.duplicate_birth_iou is not None and any(
                    t.last_seen_s==timestamp_s and t.shape.label==detections[j].label and
                    iou3d(t.shape,detections[j]) >= cfg.duplicate_birth_iou for t in self._tracks):
                self._count(detections[j].label,'duplicate_birth_suppressed')
                continue
            if len(self._tracks) >= cfg.maximum_tracks or (
                    cfg.maximum_tentative_tracks is not None and tentative_count >= cfg.maximum_tentative_tracks):
                self.diagnostics['capacity_drops'] += 1
                self._count(detections[j].label,'capacity_dropped')
                continue
            d = detections[j]
            covariance = np.zeros((6,6))
            covariance[:3,:3] = self._noise(d)
            covariance[3:,3:] = np.eye(3)*25.
            self._tracks.append(_BoxState(
                np.array([d.x,d.y,d.z,0.,0.,0.]), covariance, d,
                timestamp_s,timestamp_s,
                deque([timestamp_s] if d.confidence >= policies[j].strong_confidence else [],maxlen=cfg.confirmation_hits),
                medium_hits=deque([timestamp_s],maxlen=cfg.medium_confirmation_hits),candidate_id=self._next_candidate_id))
            self._next_candidate_id += 1;tentative_count += 1
            self.diagnostics['tentative_created'] += 1
            self._count(d.label,'tentative_created')
        for j in (high|low)-births:self._count(detections[j].label,'below_birth_unmatched')
        self.tentative_outputs=[dict(asdict(t.shape),x=float(t.mean[0]),y=float(t.mean[1]),z=float(t.mean[2]),
                                     candidate_id=t.candidate_id,strong_hits=len(t.hits),medium_hits=len(t.medium_hits),
                                     age_s=timestamp_s-t.first_seen_s,observed=t.last_seen_s==timestamp_s)
                                for t in self._tracks if t.track_id is None]
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
                tuple(map(float,track.covariance[:3,:3].ravel())),
                track.confirmation_path,track.confirmation_delay_s))
        return sorted(output,key=lambda t:t.track_id)
