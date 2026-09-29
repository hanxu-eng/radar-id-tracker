# Radar ID Tracker

An online multi-object tracker for a **10 Hz radar**. It assigns stable,
monotonically increasing IDs to radar targets using a two-model interacting
multiple model (IMM) filter, position and Doppler gating, deterministic
Hungarian association, and time-based track management.

The package consumes **detections**. These can come from a radar's built-in
target list, a clustering stage, or a trained 3D detector. No detector weights
or real radar data are included. The tracker itself runs on the CPU; an RTX
4080 may be used by an upstream detector, but this repository does not claim
an end-to-end GPU latency or accuracy result.

## Install and try

Python 3.10 or newer and NumPy are required.

```bash
python -m pip install .
radar-id-track examples/detections.jsonl
python -m unittest discover -s tests -v
```

The first frame creates a tentative internal track. An ID is published after
two high-confidence observations within 0.3 seconds. The example emits ID 1
on its second frame, keeps it through an empty frame, and reports
`predicted_only: true` for that prediction.

## Input contract

Each JSONL line is one complete frame, including empty frames:

```json
{"timestamp_s":0.1,"sensor_xy":[0.0,0.0],"detections":[{"x":10.3,"y":2.0,"confidence":0.88,"radial_velocity":3.0,"length":4.2,"width":1.8}]}
```

- `timestamp_s` must increase strictly; pass actual timestamps, not frame
  numbers. Nominal 10 Hz means about 0.1 seconds between frames.
- `x`, `y`, and `sensor_xy` must be in the **same fixed metric frame** (for
  example an odometry frame). If the vehicle moves, transform radar detections
  into that frame before calling the tracker. `z` and dimensions are optional.
- `radial_velocity` is optional, in m/s, positive **away from the radar**.
  It must be compensated for sensor motion. With line-of-sight unit vector
  `u` and sensor velocity `v_sensor`, convert raw relative Doppler by
  `v_compensated = v_raw + dot(u, v_sensor)`. Check the sign convention of your
  radar driver before applying this equation.
- `confidence` is a detector score in `[0, 1]`. High detections (default
  `>= 0.5`) can create tracks. Low detections (`0.2` to `0.5`) may continue
  confirmed tracks but cannot create IDs.
- If detections come from a neural model, use its predicted box center and
  measured/aggregated radar Doppler. Do not substitute a Cartesian speed
  magnitude for radial velocity.

Python integration:

```python
from radar_id_tracker import Detection, RadarTracker

tracker = RadarTracker()
tracks = tracker.update(
    timestamp_s=42.1,
    detections=[Detection(x=12.0, y=1.8, confidence=0.91,
                          radial_velocity=2.4)],
    sensor_xy=(0.0, 0.0),
)
for track in tracks:
    print(track.track_id, track.x, track.y, track.predicted_only)
```

Use one `RadarTracker` instance per independent stream. Each instance starts
IDs at 1 and never reuses them while running. For ID uniqueness across process
restarts, persist a stream/epoch identifier alongside the numeric ID.

## Algorithm

1. Predict every existing track to the frame timestamp. The IMM has a
   low-acceleration constant-velocity model and a constant-acceleration model.
2. Reject impossible detection-track pairs with a position innovation gate,
   absolute position residual cap, Doppler residual gate, and optional size
   change gate. Doppler prediction is the track velocity projected onto the
   current radar line of sight.
3. Run four deterministic association passes: confirmed tracks with high
   detections, unmatched confirmed tracks with low detections, dormant tracks
   with high detections under stricter gates, then tentative tracks with
   remaining high detections.
4. Start tentative tracks from remaining high detections. Publish an ID after
   two hits within 0.3 seconds. Publish prediction-only outputs for at most
   0.5 seconds after a missed detection. Keep a further 1.0-second dormant
   recovery window without output, then delete the track.

These thresholds are starting values, not calibrated performance claims. Set
them with held-out sequences from the target radar, speeds, range, and false
alarm conditions. `TrackerConfig` exposes the thresholds.

## Validation and deployment

The test suite checks deterministic assignment, crossing objects with
opposite Doppler, variable timestamps, short dropouts, dormant recovery,
low-score clutter, a distant new target, and malformed inputs. Run
`python examples/benchmark.py` for a **tracker-only** CPU latency probe on
synthetic detections. A production gate should additionally measure IDF1,
HOTA association accuracy, ID switches, false tracks, recovery after missed
frames, and P50/P95/P99 end-to-end latency on real labeled recordings.

For an RTX 4080 deployment, connect an upstream detector through the
`Detection` interface. Keep detection and tracking asynchronous, timestamp
their outputs, and drop stale queued frames. A trained CenterPoint-Pillar
detector, TensorRT engine, radar-specific preprocessing, and calibration data
must be supplied and validated for the actual sensor. The code here does not
include those assets.

Limitations: 2D ground-plane motion is tracked; `z` and box dimensions are
carried from the latest observation rather than filtered. An unresolved merge
of two physical objects can still cause an ID switch, especially when Doppler
is absent or nearly equal. The Python implementation is a reference and
should be profiled against the target object count before production use.

## License

MIT. See [LICENSE](LICENSE).
