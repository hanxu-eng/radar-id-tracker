"""Small synthetic tracker-only CPU latency probe."""

from __future__ import annotations

import statistics
import time

from radar_id_tracker import Detection, RadarTracker


def run(target_count: int = 32, frame_count: int = 100) -> None:
    tracker = RadarTracker()
    timings_ms = []
    for frame in range(frame_count):
        timestamp = frame * 0.1
        detections = [
            Detection(
                x=10.0 + 8.0 * (index % 8) + 0.7 * timestamp,
                y=5.0 + 8.0 * (index // 8),
                confidence=0.9,
                radial_velocity=0.5,
            )
            for index in range(target_count)
        ]
        start = time.perf_counter()
        tracker.update(timestamp, detections)
        timings_ms.append((time.perf_counter() - start) * 1000.0)
    ordered = sorted(timings_ms[10:])
    p99_index = max(0, int(0.99 * (len(ordered) - 1)))
    print(f"targets={target_count} frames={frame_count} tracker_only_cpu=true")
    print(f"median_ms={statistics.median(ordered):.3f} p99_ms={ordered[p99_index]:.3f}")


if __name__ == "__main__":
    run()
