"""Causal, time-based radar tracking and stable public Track IDs."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import isfinite, log
from typing import Sequence

from .assignment import assign
from .imm import IMMFilter
from .types import Detection, TrackOutput


@dataclass(frozen=True, slots=True)
class TrackerConfig:
    high_confidence: float = 0.5
    low_confidence: float = 0.2
    confirmation_hits: int = 2
    confirmation_window_s: float = 0.3
    lost_seconds: float = 0.5
    dormant_seconds: float = 1.0
    position_sigma_m: float = 0.6
    doppler_sigma_mps: float = 0.8
    position_gate_squared: float = 9.21
    maximum_position_residual_m: float = 6.0
    doppler_gate_mps: float = 3.0
    maximum_tracks: int = 256

    def __post_init__(self) -> None:
        if not 0.0 <= self.low_confidence <= self.high_confidence <= 1.0:
            raise ValueError("confidence thresholds must satisfy 0 <= low <= high <= 1")
        if self.confirmation_hits < 2 or self.maximum_tracks < 1:
            raise ValueError("confirmation_hits must be >= 2 and maximum_tracks >= 1")
        for name in (
            "confirmation_window_s", "lost_seconds", "dormant_seconds",
            "position_sigma_m", "doppler_sigma_mps", "position_gate_squared",
            "maximum_position_residual_m", "doppler_gate_mps",
        ):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")


@dataclass(slots=True)
class _Track:
    filter: IMMFilter
    first_seen_s: float
    last_seen_s: float
    observation: Detection
    hit_times: deque[float]
    track_id: int | None = None
    missed_frames: int = 0

    def record(self, detection: Detection, time_s: float, sensor_xy: tuple[float, float]) -> None:
        self.filter.update(detection, sensor_xy)
        self.observation = detection
        self.last_seen_s = time_s
        self.missed_frames = 0
        self.hit_times.append(time_s)


class RadarTracker:
    """Online tracker accepting detections in a fixed world/odometry frame.

    A Track ID is assigned after confirmation and is never reused by this
    tracker instance. Construct one instance per independent radar stream.
    """

    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.config = config or TrackerConfig()
        self._tracks: list[_Track] = []
        self._next_id = 1
        self._last_timestamp_s: float | None = None

    def update(
        self,
        timestamp_s: float,
        detections: Sequence[Detection],
        sensor_xy: tuple[float, float] = (0.0, 0.0),
    ) -> list[TrackOutput]:
        """Process one radar frame, including empty frames.

        ``sensor_xy`` is the radar position in the same fixed frame as the
        detections. Doppler must already be compensated for ego motion.
        """
        if not isfinite(timestamp_s):
            raise ValueError("timestamp_s must be finite")
        if len(sensor_xy) != 2 or not all(isfinite(value) for value in sensor_xy):
            raise ValueError("sensor_xy must contain two finite coordinates")
        if self._last_timestamp_s is not None and timestamp_s <= self._last_timestamp_s:
            raise ValueError("frame timestamps must strictly increase")
        if any(not isinstance(detection, Detection) for detection in detections):
            raise TypeError("detections must be Detection instances")

        if self._last_timestamp_s is not None:
            dt = timestamp_s - self._last_timestamp_s
            if dt > 10.0:
                self._tracks.clear()
            else:
                for track in self._tracks:
                    track.filter.predict(dt)
        self._last_timestamp_s = timestamp_s

        expiration = self.config.lost_seconds + self.config.dormant_seconds
        self._tracks = [
            track for track in self._tracks
            if timestamp_s - track.last_seen_s <= expiration
            and (track.track_id is not None
                 or timestamp_s - track.first_seen_s <= self.config.confirmation_window_s)
        ]

        high = [i for i, det in enumerate(detections) if det.confidence >= self.config.high_confidence]
        low = [
            i for i, det in enumerate(detections)
            if self.config.low_confidence <= det.confidence < self.config.high_confidence
        ]
        available_high = set(high)
        available_low = set(low)
        matched_tracks: set[int] = set()

        active = [
            i for i, track in enumerate(self._tracks)
            if track.track_id is not None
            and timestamp_s - track.last_seen_s <= self.config.lost_seconds
        ]
        dormant = [
            i for i, track in enumerate(self._tracks)
            if track.track_id is not None
            and timestamp_s - track.last_seen_s > self.config.lost_seconds
        ]
        tentative = [i for i, track in enumerate(self._tracks) if track.track_id is None]

        def match_stage(track_indices: list[int], candidate_indices: set[int], strict: bool = False) -> None:
            if not track_indices or not candidate_indices:
                return
            ordered_candidates = sorted(candidate_indices)
            costs = [
                [
                    self._association_cost(self._tracks[track_i], detections[det_i], sensor_xy, strict)
                    for det_i in ordered_candidates
                ]
                for track_i in track_indices
            ]
            for row, col in assign(costs, len(ordered_candidates)):
                track_i = track_indices[row]
                det_i = ordered_candidates[col]
                track = self._tracks[track_i]
                track.record(detections[det_i], timestamp_s, sensor_xy)
                matched_tracks.add(track_i)
                candidate_indices.remove(det_i)
                if track.track_id is None:
                    while track.hit_times and timestamp_s - track.hit_times[0] > self.config.confirmation_window_s:
                        track.hit_times.popleft()
                    if len(track.hit_times) >= self.config.confirmation_hits:
                        track.track_id = self._next_id
                        self._next_id += 1

        match_stage(active, available_high)
        match_stage([i for i in active if i not in matched_tracks], available_low)
        match_stage(dormant, available_high, strict=True)
        match_stage(tentative, available_high)

        for index, track in enumerate(self._tracks):
            if index not in matched_tracks:
                track.missed_frames += 1

        for det_i in sorted(available_high):
            if len(self._tracks) >= self.config.maximum_tracks:
                break
            detection = detections[det_i]
            self._tracks.append(
                _Track(
                    filter=IMMFilter(
                        detection, sensor_xy,
                        self.config.position_sigma_m,
                        self.config.doppler_sigma_mps,
                    ),
                    first_seen_s=timestamp_s,
                    last_seen_s=timestamp_s,
                    observation=detection,
                    hit_times=deque([timestamp_s]),
                )
            )

        outputs: list[TrackOutput] = []
        for track in self._tracks:
            if track.track_id is None or timestamp_s - track.last_seen_s > self.config.lost_seconds:
                continue
            mean, _ = track.filter.mean_covariance()
            observed = track.observation
            outputs.append(
                TrackOutput(
                    track_id=track.track_id,
                    timestamp_s=timestamp_s,
                    x=float(mean[0]), y=float(mean[1]),
                    vx=float(mean[2]), vy=float(mean[3]),
                    z=observed.z,
                    confidence=observed.confidence,
                    predicted_only=track.last_seen_s != timestamp_s,
                    missed_frames=track.missed_frames,
                    length=observed.length, width=observed.width,
                    height=observed.height, yaw=observed.yaw,
                )
            )
        return sorted(outputs, key=lambda output: output.track_id)

    def _association_cost(
        self,
        track: _Track,
        detection: Detection,
        sensor_xy: tuple[float, float],
        strict: bool,
    ) -> float | None:
        mean, _ = track.filter.mean_covariance()
        position_residual = ((detection.x - mean[0]) ** 2 + (detection.y - mean[1]) ** 2) ** 0.5
        residual_gate = self.config.maximum_position_residual_m * (0.5 if strict else 1.0)
        if position_residual > residual_gate:
            return None
        position_squared = track.filter.position_distance_squared(detection)
        position_gate = self.config.position_gate_squared * (0.5 if strict else 1.0)
        if position_squared > position_gate:
            return None
        terms = [(0.5, position_squared / position_gate)]

        radial_residual = track.filter.radial_residual(detection, sensor_xy)
        if radial_residual is not None:
            radial_gate = self.config.doppler_gate_mps * (0.65 if strict else 1.0)
            if radial_residual > radial_gate:
                return None
            terms.append((0.35, radial_residual / radial_gate))

        old = track.observation
        if all(value is not None for value in (old.length, old.width, detection.length, detection.width)):
            assert old.length is not None and old.width is not None
            assert detection.length is not None and detection.width is not None
            size_error = (
                abs(log(detection.length / old.length))
                + abs(log(detection.width / old.width))
            ) / (2 * log(2.0))
            if size_error > (0.7 if strict else 1.0):
                return None
            terms.append((0.15, size_error))

        weight_sum = sum(weight for weight, _ in terms)
        return sum(weight * value for weight, value in terms) / weight_sum
