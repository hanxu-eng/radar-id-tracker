"""Offline, label-isolated RadarScenes center/identity diagnostics.

Run *after* radarscenes_adapter. Ground-truth fields are read only in this
module; they never enter clustering, tracking, or the visualization pipeline.
These are custom center-distance metrics, not an official RadarScenes benchmark.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from math import hypot, isfinite
from pathlib import Path
from statistics import median

import numpy as np

from .assignment import assign


@dataclass(frozen=True, slots=True)
class TruthObject:
    object_id: str
    x: float
    y: float
    point_count: int


@dataclass(frozen=True, slots=True)
class Proposal:
    x: float
    y: float
    track_id: int | None = None
    predicted_only: bool = False


@dataclass(frozen=True, slots=True)
class EvalFrame:
    timestamp_us: int
    timestamp_s: float
    truth: tuple[TruthObject, ...]
    detections: tuple[Proposal, ...]
    tracks: tuple[Proposal, ...]
    skipped_dynamic_points: int = 0


def _truth_id(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").rstrip("\x00").strip()
    return str(value).rstrip("\x00").strip()


def truth_objects(radar_data: np.ndarray) -> tuple[tuple[TruthObject, ...], int]:
    """Group labeled moving points by official ID within exactly one scan."""
    required = {"x_seq", "y_seq", "label_id", "track_id"}
    names = set(radar_data.dtype.names or ())
    if not required <= names:
        raise ValueError(f"radar_data missing truth fields: {', '.join(sorted(required - names))}")
    groups: dict[str, list[tuple[float, float]]] = defaultdict(list)
    skipped = 0
    for point in radar_data:
        label_id = int(point["label_id"])
        if label_id == 11:  # RadarScenes STATIC; dynamic classes are 0..10.
            continue
        if not 0 <= label_id <= 10:
            raise ValueError(f"unknown RadarScenes label_id: {label_id}")
        object_id = _truth_id(point["track_id"])
        x, y = float(point["x_seq"]), float(point["y_seq"])
        if not object_id or not (isfinite(x) and isfinite(y)):
            skipped += 1
            continue
        groups[object_id].append((x, y))
    objects = tuple(
        TruthObject(object_id, float(np.median([xy[0] for xy in points])),
                    float(np.median([xy[1] for xy in points])), len(points))
        for object_id, points in sorted(groups.items())
    )
    return objects, skipped


def _as_proposals(items: object, *, tracks: bool) -> tuple[Proposal, ...]:
    if not isinstance(items, list):
        raise ValueError("detections/tracks must be JSON arrays")
    proposals: list[Proposal] = []
    seen_ids: set[int] = set()
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("each detection/track must be a JSON object")
        try:
            x, y = float(item["x"]), float(item["y"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("detection/track has invalid x or y") from error
        if not (isfinite(x) and isfinite(y)):
            raise ValueError("detection/track coordinates must be finite")
        track_id: int | None = None
        predicted_only = False
        if tracks:
            track_id = item.get("track_id")
            if type(track_id) is not int or track_id < 1 or track_id in seen_ids:
                raise ValueError("track IDs must be unique positive integers within a frame")
            seen_ids.add(track_id)
            predicted_only = item.get("predicted_only")
            if type(predicted_only) is not bool:
                raise ValueError("predicted_only must be a boolean")
        proposals.append(Proposal(x, y, track_id, predicted_only))
    return tuple(proposals)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(f"replay output missing: {path}")
    result: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON in {path} line {line_number}") from error
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object in {path} line {line_number}")
            result.append(value)
    return result


def load_frames(scenes_path: Path, run_dir: Path) -> tuple[str, int, list[EvalFrame], int | None]:
    """Align saved predictions to the same sensor scans by exact timestamps."""
    try:
        from radar_scenes.sequence import Sequence
    except ImportError as error:
        raise RuntimeError(
            "RadarScenes reader missing; install h5py and radar_scenes as in docs/radarscenes.md"
        ) from error
    if not scenes_path.is_file():
        raise FileNotFoundError(f"scenes.json not found: {scenes_path}")
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"replay summary missing: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise ValueError("summary.json must contain an object")
    sensor_id = summary.get("sensor_id")
    count = summary.get("frames")
    if type(sensor_id) is not int or sensor_id not in (1, 2, 3, 4):
        raise ValueError("summary.json has invalid sensor_id")
    if type(count) is not int or count < 1:
        raise ValueError("summary.json has invalid frame count")
    if summary.get("ground_truth_used_as_input") is not False:
        raise ValueError("cannot verify that replay was label-free")
    sequence = Sequence.from_json(str(scenes_path))
    if summary.get("sequence") != sequence.sequence_name:
        raise ValueError("scenes.json and summary.json refer to different sequences")
    detections = _read_jsonl(run_dir / "detections.jsonl")
    tracks = _read_jsonl(run_dir / "tracks.jsonl")
    if len(detections) != count or len(tracks) != count:
        raise ValueError("JSONL line counts do not match summary.json frames")
    selected = [timestamp for timestamp in sorted(int(t) for t in sequence.timestamps)
                if sequence.get_scene(timestamp).sensor_id == sensor_id]
    if len(selected) < count:
        raise ValueError("sequence has fewer selected-sensor scans than the saved replay")
    selected = selected[:count]
    first_us = selected[0]
    frames: list[EvalFrame] = []
    radar_point_count = 0
    for index, timestamp_us in enumerate(selected):
        timestamp_s = (timestamp_us - first_us) / 1_000_000.0
        detection_row, track_row = detections[index], tracks[index]
        for row in (detection_row, track_row):
            try:
                saved_time = float(row["timestamp_s"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"frame {index} missing valid timestamp_s") from error
            if not isfinite(saved_time) or abs(saved_time - timestamp_s) > 1e-6:
                raise ValueError(f"frame {index} timestamp mismatch with scenes.json")
        scene = sequence.get_scene(timestamp_us)
        assert scene is not None
        radar_point_count += len(scene.radar_data)
        truth, skipped = truth_objects(scene.radar_data)
        frames.append(EvalFrame(
            timestamp_us, timestamp_s, truth,
            _as_proposals(detection_row.get("detections"), tracks=False),
            _as_proposals(track_row.get("tracks"), tracks=True),
            skipped,
        ))
    if summary.get("radar_points") != radar_point_count:
        raise ValueError("radar point count does not match saved replay")
    config = summary.get("config", {})
    min_points = config.get("min_points") if isinstance(config, dict) else None
    if type(min_points) is not int or min_points < 1:
        min_points = None
    return sequence.sequence_name, sensor_id, frames, min_points


def _match(truth: tuple[TruthObject, ...], proposals: tuple[Proposal, ...],
           gate_m: float) -> list[tuple[int, int, float]]:
    costs: list[list[float | None]] = []
    for gt in truth:
        row = []
        for proposal in proposals:
            distance = hypot(gt.x - proposal.x, gt.y - proposal.y)
            row.append(distance / gate_m if distance <= gate_m else None)
        costs.append(row)
    # An unmatched row must cost more than the maximum possible increase in
    # total normalized distance, so cardinality wins before proximity.
    unmatched_cost = float(len(truth) + 1)
    return [(gi, pi, hypot(truth[gi].x - proposals[pi].x,
                           truth[gi].y - proposals[pi].y))
            for gi, pi in assign(costs, len(proposals), unmatched_cost=unmatched_cost)]


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


class _Accumulator:
    def __init__(self, identity: bool) -> None:
        self.identity = identity
        self.tp = self.fp = self.fn = 0
        self.distances: list[float] = []
        self.gt_counts: Counter[str] = Counter()
        self.pred_counts: Counter[int] = Counter()
        self.potential: Counter[tuple[str, int]] = Counter()
        self.last_id: dict[str, int] = {}
        self.last_observed_match: dict[str, bool] = {}
        self.id_changes = self.fragmentations = 0

    def update(self, truth: tuple[TruthObject, ...], proposals: tuple[Proposal, ...],
               gate_m: float) -> dict[str, object]:
        matches = _match(truth, proposals, gate_m)
        self.tp += len(matches)
        self.fp += len(proposals) - len(matches)
        self.fn += len(truth) - len(matches)
        self.distances.extend(distance for _, _, distance in matches)
        matched_gt = {gi: (pi, distance) for gi, pi, distance in matches}
        matched_pred = {pi for _, pi, _ in matches}
        changes: list[str] = []
        fragments: list[str] = []
        if self.identity:
            # Identity potential matches are counted for every gated ID pair,
            # independently of the one-to-one per-frame localization match.
            for gt in truth:
                self.gt_counts[gt.object_id] += 1
                for proposal in proposals:
                    assert proposal.track_id is not None
                    if hypot(gt.x - proposal.x, gt.y - proposal.y) <= gate_m:
                        self.potential[gt.object_id, proposal.track_id] += 1
            for proposal in proposals:
                assert proposal.track_id is not None
                self.pred_counts[proposal.track_id] += 1
            for gi, gt in enumerate(truth):
                if gi not in matched_gt:
                    if gt.object_id in self.last_id:
                        self.last_observed_match[gt.object_id] = False
                    continue
                proposal = proposals[matched_gt[gi][0]]
                assert proposal.track_id is not None
                previous_id = self.last_id.get(gt.object_id)
                if previous_id is not None and previous_id != proposal.track_id:
                    self.id_changes += 1
                    changes.append(gt.object_id)
                if previous_id is not None and not self.last_observed_match[gt.object_id]:
                    self.fragmentations += 1
                    fragments.append(gt.object_id)
                self.last_id[gt.object_id] = proposal.track_id
                self.last_observed_match[gt.object_id] = True
        return {
            "tp": len(matches), "fp": len(proposals) - len(matches),
            "fn": len(truth) - len(matches),
            "matches": [
                {"gt_id": truth[gi].object_id,
                 "proposal_id": proposals[pi].track_id if self.identity else pi,
                 "distance_m": round(distance, 4)}
                for gi, pi, distance in matches
            ],
            "missed_gt_ids": [gt.object_id for gi, gt in enumerate(truth)
                              if gi not in matched_gt],
            "unmatched_proposal_ids": [
                proposal.track_id if self.identity else pi
                for pi, proposal in enumerate(proposals) if pi not in matched_pred
            ],
            "id_change_gt_ids": changes,
            "fragmented_gt_ids": fragments,
        }

    def result(self) -> dict[str, object]:
        precision = _ratio(self.tp, self.tp + self.fp)
        recall = _ratio(self.tp, self.tp + self.fn)
        result: dict[str, object] = {
            "tp": self.tp, "fp": self.fp, "fn": self.fn,
            "precision": precision, "recall": recall,
            "f1": _ratio(2 * self.tp, 2 * self.tp + self.fp + self.fn),
            "median_matched_distance_m": median(self.distances) if self.distances else None,
        }
        if self.identity:
            gt_ids = sorted(self.gt_counts)
            pred_ids = sorted(self.pred_counts)
            costs = [
                [-float(self.potential[gt_id, pred_id])
                 if self.potential[gt_id, pred_id] else None
                 for pred_id in pred_ids]
                for gt_id in gt_ids
            ]
            id_pairs = assign(costs, len(pred_ids), unmatched_cost=0.0)
            idtp = sum(self.potential[gt_ids[gi], pred_ids[pi]] for gi, pi in id_pairs)
            idfn = sum(self.gt_counts.values()) - idtp
            idfp = sum(self.pred_counts.values()) - idtp
            result.update({
                "idtp": idtp, "idfp": idfp, "idfn": idfn,
                "id_precision": _ratio(idtp, idtp + idfp),
                "id_recall": _ratio(idtp, idtp + idfn),
                "idf1": _ratio(2 * idtp, 2 * idtp + idfp + idfn),
                "id_changes": self.id_changes,
                "fragmentations": self.fragmentations,
                "unique_predicted_ids": len(pred_ids),
                "unique_gt_ids": len(gt_ids),
            })
        return result


def score_frames(frames: list[EvalFrame], gate_m: float,
                 min_points: int | None = None) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Score detections, observed tracks, and all published tracks separately."""
    if not isfinite(gate_m) or gate_m <= 0:
        raise ValueError("distance gate must be positive and finite")
    if not frames:
        raise ValueError("cannot evaluate an empty sequence")
    detection_score = _Accumulator(identity=False)
    observed_score = _Accumulator(identity=True)
    published_score = _Accumulator(identity=True)
    details: list[dict[str, object]] = []
    gt_occurrences = gt_single_point = skipped_points = predicted_outputs = 0
    below_min_points = 0
    for index, frame in enumerate(frames):
        gt_occurrences += len(frame.truth)
        gt_single_point += sum(gt.point_count == 1 for gt in frame.truth)
        if min_points is not None:
            below_min_points += sum(gt.point_count < min_points for gt in frame.truth)
        skipped_points += frame.skipped_dynamic_points
        predicted_outputs += sum(proposal.predicted_only for proposal in frame.tracks)
        observed = tuple(proposal for proposal in frame.tracks if not proposal.predicted_only)
        details.append({
            "frame_index": index,
            "timestamp_us": frame.timestamp_us,
            "timestamp_s": frame.timestamp_s,
            "gt_objects": [{"gt_id": gt.object_id, "x": gt.x, "y": gt.y,
                            "point_count": gt.point_count} for gt in frame.truth],
            "detections": detection_score.update(frame.truth, frame.detections, gate_m),
            "observed_tracks": observed_score.update(frame.truth, observed, gate_m),
            "published_tracks": published_score.update(frame.truth, frame.tracks, gate_m),
            "predicted_only_outputs": len(frame.tracks) - len(observed),
        })
    worst = sorted(details, key=lambda row: (
        row["observed_tracks"]["fp"] + row["observed_tracks"]["fn"],
        row["detections"]["fp"] + row["detections"]["fn"],
        -row["frame_index"],
    ), reverse=True)[:10]
    summary: dict[str, object] = {
        "method": "single_scan_gt_point_median_center_euclidean_one_to_one",
        "assignment_priority": "maximum_valid_matches_then_minimum_total_distance",
        "distance_gate_m": gate_m,
        "official_radar_scenes_benchmark": False,
        "gt_used_for_detection_or_tracking": False,
        "frames": len(frames),
        "gt_object_occurrences": gt_occurrences,
        "gt_single_point_occurrences": gt_single_point,
        "gt_below_cluster_min_points": below_min_points if min_points is not None else None,
        "cluster_min_points": min_points,
        "skipped_dynamic_points_without_usable_id_or_coordinates": skipped_points,
        "predicted_only_track_outputs": predicted_outputs,
        "detections": detection_score.result(),
        "observed_tracks": observed_score.result(),
        "published_tracks_including_predictions": published_score.result(),
        "worst_frame_indices_by_observed_fp_plus_fn": [row["frame_index"] for row in worst],
        "notes": [
            "Identity scores use TrackEval-style global ID assignment under this custom center-distance gate; they are not official RadarScenes/HOTA scores.",
            "Observed-track scores exclude predicted_only; published-track scores include them and may penalize valid short occlusion predictions when no radar truth points exist.",
            "RadarScenes labels only moving measured objects and may assign a new truth ID after >500 ms occlusion or stop.",
        ],
    }
    return summary, details


def evaluate(scenes_path: Path, run_dir: Path, gate_m: float = 2.0) -> dict[str, object]:
    name, sensor_id, frames, min_points = load_frames(scenes_path, run_dir)
    report, details = score_frames(frames, gate_m, min_points)
    report["sequence"] = name
    report["sensor_id"] = sensor_id
    report["scenes_file"] = str(scenes_path)
    report["run_dir"] = str(run_dir)
    frames_path = run_dir / "evaluation_frames.jsonl"
    report_path = run_dir / "evaluation.json"
    with frames_path.open("w", encoding="utf-8") as target:
        for row in details:
            target.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    report["per_frame_file"] = str(frames_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
                           encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", required=True, type=Path, help="original sequence scenes.json")
    parser.add_argument("--run-dir", required=True, type=Path, help="directory from radarscenes_adapter")
    parser.add_argument("--distance-m", type=float, default=2.0,
                        help="fixed maximum GT/prediction center distance in meters")
    args = parser.parse_args(argv)
    try:
        report = evaluate(args.scenes, args.run_dir, args.distance_m)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        parser.exit(2, f"RadarScenes evaluation error: {error}\n")
    print(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
