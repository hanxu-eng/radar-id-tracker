"""Dependency-free BEV replay for one RadarScenes radar stream.

Coordinates stay in the sequence frame. The viewport follows the radar position,
but does not rotate with the vehicle; the same world direction always points up.
"""

from __future__ import annotations

import json
from collections import deque
from html import escape
from math import ceil, floor, isfinite
from pathlib import Path
from shutil import copy2

import numpy as np

from .types import Detection, TrackOutput


_PALETTE = (
    "#2563eb", "#16a34a", "#9333ea", "#db2777", "#0891b2",
    "#b45309", "#4f46e5", "#15803d", "#be123c", "#0d9488",
)
_SIZE = 1000
_CENTER = 500
_PLOT_HALF = 420


def _pixel(x: float, y: float, sensor_xy: tuple[float, float], range_m: float) -> tuple[float, float]:
    scale = _PLOT_HALF / range_m
    return (_CENTER + (x - sensor_xy[0]) * scale,
            _CENTER - (y - sensor_xy[1]) * scale)


def _visible(x: float, y: float, sensor_xy: tuple[float, float], range_m: float) -> bool:
    return (isfinite(x) and isfinite(y)
            and abs(x - sensor_xy[0]) <= range_m
            and abs(y - sensor_xy[1]) <= range_m)


def render_bev_svg(
    radar_data: np.ndarray,
    detections: list[Detection],
    tracks: list[TrackOutput],
    histories: dict[int, deque[tuple[float, float, float, bool]]],
    sensor_xy: tuple[float, float],
    timestamp_s: float,
    sequence_name: str,
    sensor_id: int,
    range_m: float,
) -> str:
    """Render one scan; annotations are visual only and never feed the tracker."""
    if not isfinite(range_m) or range_m <= 0:
        raise ValueError("visualization range must be positive and finite")
    names = set(radar_data.dtype.names or ())
    if not {"x_seq", "y_seq"} <= names:
        raise ValueError("radar_data missing x_seq or y_seq for visualization")

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="1000" '
        'viewBox="0 0 1000 1000" role="img" aria-label="Radar BEV tracking">',
        '<rect width="1000" height="1000" fill="#f8fafc"/>',
        '<rect x="80" y="80" width="840" height="840" fill="#ffffff" '
        'stroke="#94a3b8" stroke-width="2"/>',
    ]
    # World-aligned grid; the radar remains at the center while driving.
    grid_step = 10 ** floor(np.log10(range_m / 4))
    if range_m / grid_step > 12:
        grid_step *= 2
    if range_m / grid_step > 12:
        grid_step *= 2.5
    for axis in (0, 1):
        low = sensor_xy[axis] - range_m
        high = sensor_xy[axis] + range_m
        for multiple in range(ceil(low / grid_step), floor(high / grid_step) + 1):
            value = multiple * grid_step
            coordinate = _pixel(value, sensor_xy[1], sensor_xy, range_m)[0] if axis == 0 else \
                _pixel(sensor_xy[0], value, sensor_xy, range_m)[1]
            if axis == 0:
                parts.append(f'<line x1="{coordinate:.2f}" y1="80" x2="{coordinate:.2f}" '
                             'y2="920" stroke="#e2e8f0" stroke-width="1"/>')
            else:
                parts.append(f'<line x1="80" y1="{coordinate:.2f}" x2="920" '
                             f'y2="{coordinate:.2f}" stroke="#e2e8f0" stroke-width="1"/>')

    # Raw points are deliberately neutral: neither labels nor truth IDs are used.
    for point in radar_data:
        x, y = float(point["x_seq"]), float(point["y_seq"])
        if _visible(x, y, sensor_xy, range_m):
            px, py = _pixel(x, y, sensor_xy, range_m)
            parts.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="2" fill="#94a3b8"/>')

    for detection in detections:
        if _visible(detection.x, detection.y, sensor_xy, range_m):
            px, py = _pixel(detection.x, detection.y, sensor_xy, range_m)
            parts.append(f'<rect x="{px - 4:.2f}" y="{py - 4:.2f}" width="8" height="8" '
                         'fill="#f97316" stroke="#7c2d12" stroke-width="1"/>')

    for track in tracks:
        color = _PALETTE[(track.track_id - 1) % len(_PALETTE)]
        history = histories.get(track.track_id, ())
        for previous, current in zip(history, list(history)[1:]):
            _, x0, y0, predicted0 = previous
            _, x1, y1, predicted1 = current
            if not (_visible(x0, y0, sensor_xy, range_m)
                    and _visible(x1, y1, sensor_xy, range_m)):
                continue
            px0, py0 = _pixel(x0, y0, sensor_xy, range_m)
            px1, py1 = _pixel(x1, y1, sensor_xy, range_m)
            dash = ' stroke-dasharray="5 4"' if predicted0 or predicted1 else ""
            parts.append(f'<line x1="{px0:.2f}" y1="{py0:.2f}" x2="{px1:.2f}" '
                         f'y2="{py1:.2f}" stroke="{color}" stroke-width="3"{dash}/>')
        if not _visible(track.x, track.y, sensor_xy, range_m):
            continue
        px, py = _pixel(track.x, track.y, sensor_xy, range_m)
        predicted = "true" if track.predicted_only else "false"
        fill = "#ffffff" if track.predicted_only else color
        dash = ' stroke-dasharray="4 3"' if track.predicted_only else ""
        parts.append(f'<g data-track-id="{track.track_id}" data-predicted-only="{predicted}">')
        parts.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="9" fill="{fill}" '
                     f'stroke="{color}" stroke-width="3"{dash}/>')
        label = f'ID {track.track_id}' + (' (P)' if track.predicted_only else '')
        parts.append(f'<text x="{px + 12:.2f}" y="{py - 12:.2f}" font-size="18" '
                     f'font-family="sans-serif" font-weight="bold" fill="{color}" '
                     f'stroke="#ffffff" stroke-width="4" paint-order="stroke">{label}</text>')
        parts.append('</g>')

    parts.extend([
        '<circle cx="500" cy="500" r="8" fill="#111827" stroke="#ffffff" stroke-width="2"/>',
        '<line x1="486" y1="500" x2="514" y2="500" stroke="#111827" stroke-width="2"/>',
        '<line x1="500" y1="486" x2="500" y2="514" stroke="#111827" stroke-width="2"/>',
        f'<text x="80" y="42" font-size="23" font-family="sans-serif" fill="#0f172a">'
        f'{escape(sequence_name)} · radar {sensor_id} · t={timestamp_s:.3f} s</text>',
        '<text x="80" y="960" font-size="16" font-family="sans-serif" fill="#475569">'
        'gray: raw points | orange: clusters | color: observed ID | hollow/dashed (P): prediction</text>',
        f'<text x="700" y="42" font-size="16" font-family="sans-serif" fill="#475569">'
        f'view: ±{range_m:g} m · +x right · +y up</text>',
        '</svg>',
    ])
    return "\n".join(parts) + "\n"


class RadarScenesVisualizer:
    """Stream SVG frames and write a local player without extra dependencies."""

    def __init__(self, output_dir: Path, sequence_name: str, sensor_id: int,
                 range_m: float = 60.0, trail_seconds: float = 2.0,
                 with_camera: bool = False) -> None:
        if not isfinite(range_m) or range_m <= 0:
            raise ValueError("visualization range must be positive and finite")
        if not isfinite(trail_seconds) or trail_seconds <= 0:
            raise ValueError("trail seconds must be positive and finite")
        self.output_dir = output_dir
        self.frames_dir = output_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.with_camera = with_camera
        self.camera_dir = output_dir / "camera"
        if with_camera:
            self.camera_dir.mkdir(parents=True, exist_ok=True)
        self.camera_files: dict[Path, str] = {}
        self.sequence_name = sequence_name
        self.sensor_id = sensor_id
        self.range_m = range_m
        self.trail_seconds = trail_seconds
        self.histories: dict[int, deque[tuple[float, float, float, bool]]] = {}
        self.frames: list[tuple[float, str, str | None, float | None]] = []

    def add_frame(self, radar_data: np.ndarray, detections: list[Detection],
                  tracks: list[TrackOutput], sensor_xy: tuple[float, float],
                  timestamp_s: float, camera_path: Path | None = None,
                  camera_offset_ms: float | None = None) -> None:
        if self.with_camera != (camera_path is not None):
            raise ValueError("camera_path must be supplied exactly when camera view is enabled")
        cutoff = timestamp_s - self.trail_seconds
        for track_id, history in list(self.histories.items()):
            while history and history[0][0] < cutoff:
                history.popleft()
            if not history:
                del self.histories[track_id]
        for track in tracks:
            self.histories.setdefault(track.track_id, deque()).append(
                (timestamp_s, track.x, track.y, track.predicted_only))
        name = f"frame_{len(self.frames):06d}.svg"
        svg = render_bev_svg(radar_data, detections, tracks, self.histories,
                             sensor_xy, timestamp_s, self.sequence_name,
                             self.sensor_id, self.range_m)
        (self.frames_dir / name).write_text(svg, encoding="utf-8")
        camera_relative: str | None = None
        if camera_path is not None:
            source = camera_path.resolve()
            camera_relative = self.camera_files.get(source)
            if camera_relative is None:
                camera_name = f"camera_{len(self.camera_files):06d}{source.suffix.lower()}"
                copy2(source, self.camera_dir / camera_name)
                camera_relative = f"camera/{camera_name}"
                self.camera_files[source] = camera_relative
        self.frames.append((timestamp_s, f"frames/{name}",
                            camera_relative, camera_offset_ms))

    def finish(self) -> Path:
        manifest = json.dumps(self.frames, ensure_ascii=False)
        title = escape(f"{self.sequence_name} · radar {self.sensor_id}")
        camera_panel = """
<div class="pane">
  <h2>Nearest camera view</h2>
  <img id="camera" alt="Documentary camera image nearest to this radar scan">
  <p id="camera-status"></p>
</div>""" if self.with_camera else ""
        html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Radar BEV replay — {title}</title>
<style>
body {{ margin: 0; background: #e2e8f0; color: #0f172a; font: 16px system-ui, sans-serif; }}
main {{ max-width: 1850px; margin: 0 auto; padding: 16px; }}
h1 {{ font-size: 22px; margin: 0 0 12px; }}
h2 {{ font-size: 18px; margin: 0 0 8px; }}
.controls {{ display: flex; align-items: center; gap: 12px; flex-wrap: wrap; margin-bottom: 12px; }}
button, select {{ font: inherit; padding: 6px 10px; }}
input[type=range] {{ flex: 1; min-width: 180px; }}
.viewer {{ display: flex; gap: 16px; align-items: flex-start; flex-wrap: wrap; }}
.pane {{ flex: 1 1 500px; min-width: 0; max-width: 900px; }}
.pane img {{ display: block; width: 100%; max-height: 900px; object-fit: contain; background: white; border: 1px solid #94a3b8; }}
p {{ color: #475569; }}
</style>
</head>
<body><main>
<h1>Radar BEV replay — {title}</h1>
<div class="controls">
  <button id="play" type="button">Play</button>
  <button id="previous" type="button">Previous</button>
  <button id="next" type="button">Next</button>
  <input id="seek" type="range" min="0" max="{max(0, len(self.frames) - 1)}" value="0" aria-label="Frame">
  <select id="speed" aria-label="Playback speed">
    <option value="0.25">0.25×</option><option value="0.5">0.5×</option>
    <option value="1" selected>1×</option><option value="2">2×</option>
    <option value="4">4×</option>
  </select>
  <span id="status"></span>
</div>
<div class="viewer">
  <div class="pane"><h2>Radar BEV</h2><img id="frame" alt="Bird's-eye radar scan with numbered tracks"></div>
  {camera_panel}
</div>
<p>Playback follows the original radar scan timestamp gaps at 1× (subject to browser timer scheduling). The viewport follows the radar without rotating. The camera is the nearest recorded image, not a calibrated radar projection. IDs are tracker outputs, not ground truth; this visualization does not evaluate tracking accuracy.</p>
<script>
const frames = {manifest};
const image = document.getElementById('frame');
const camera = document.getElementById('camera');
const cameraStatus = document.getElementById('camera-status');
const seek = document.getElementById('seek');
const status = document.getElementById('status');
const playButton = document.getElementById('play');
let index = 0;
let timer = null;
function show(i) {{
  index = Math.max(0, Math.min(frames.length - 1, i));
  image.src = frames[index][1];
  if (camera !== null) {{
    camera.src = frames[index][2];
    const offset = frames[index][3];
    cameraStatus.textContent = offset === null
      ? 'Nearest camera image; timestamp offset unavailable'
      : `Camera − radar timestamp: ${{offset >= 0 ? '+' : ''}}${{offset.toFixed(1)}} ms`;
  }}
  seek.value = String(index);
  status.textContent = `Frame ${{index + 1}}/${{frames.length}} · t=${{frames[index][0].toFixed(3)}} s`;
}}
function stop() {{ clearTimeout(timer); timer = null; playButton.textContent = 'Play'; }}
function schedule() {{
  if (index >= frames.length - 1) {{ stop(); return; }}
  const deltaMs = (frames[index + 1][0] - frames[index][0]) * 1000;
  const speed = Number(document.getElementById('speed').value);
  timer = setTimeout(() => {{ show(index + 1); schedule(); }}, Math.max(1, deltaMs / speed));
}}
playButton.onclick = () => {{
  if (timer !== null) {{ stop(); return; }}
  if (index === frames.length - 1) show(0);
  playButton.textContent = 'Pause';
  schedule();
}};
document.getElementById('previous').onclick = () => {{ stop(); show(index - 1); }};
document.getElementById('next').onclick = () => {{ stop(); show(index + 1); }};
seek.oninput = () => {{ stop(); show(Number(seek.value)); }};
document.getElementById('speed').onchange = () => {{ if (timer !== null) {{ clearTimeout(timer); schedule(); }} }};
if (frames.length) show(0);
</script>
</main></body></html>
"""
        path = self.output_dir / "visualization.html"
        path.write_text(html, encoding="utf-8")
        return path
