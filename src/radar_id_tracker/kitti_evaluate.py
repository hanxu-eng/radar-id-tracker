"""Independent, local upright-3D-IoU diagnostics, NOT official KITTI scores."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
from math import isfinite
from pathlib import Path

from .assignment import assign
from .box_geometry import iou3d
from .kitti_adapter import calibration_transforms, camera_box_to_world, load_poses
from .lidar_tracker import BoxDetection, BoxTrack


@dataclass(frozen=True)
class TruthBox:
    object_id: str
    box: BoxDetection


def load_truth(root: Path, sequence: str) -> tuple[list[list[TruthBox]], list[list[BoxDetection]]]:
    poses = load_poses(root/'training/oxts'/f'{sequence}.txt')
    camera_to_imu, _ = calibration_transforms(root/'training/calib'/f'{sequence}.txt')
    truth = [[] for _ in poses]; ignored = [[] for _ in poses]
    for line in (root/'training/label_02'/f'{sequence}.txt').read_text().splitlines():
        row = line.split()
        if len(row) not in (17, 18):
            raise ValueError('expected KITTI tracking label fields')
        frame, object_id = int(row[0]), int(row[1]); label = row[2]
        if not 0 <= frame < len(poses):
            raise ValueError('label frame outside pose sequence')
        if label not in ('Car','Pedestrian','Cyclist','Van','Person_sitting'):
            continue
        box = camera_box_to_world(*map(float,row[10:17]),
                                  {'Van':'Car','Person_sitting':'Pedestrian'}.get(label,label),
                                  1., poses[frame] @ camera_to_imu)
        if label in ('Van','Person_sitting'):
            ignored[frame].append(box)
        elif object_id >= 0:
            truth[frame].append(TruthBox(str(object_id),box))
    return truth, ignored


def overlaps(truth, proposals):
    return [[iou3d(gt.box,p) if gt.box.label == p.label else 0. for p in proposals] for gt in truth]


def match_boxes(truth, proposals, threshold):
    values = overlaps(truth,proposals)
    pairs = assign([[1-v if v >= threshold else None for v in row] for row in values],
                   len(proposals), unmatched_cost=float(len(truth)+1))
    return pairs, values


class BoxScore:
    def __init__(self, identity=True):
        self.identity = identity
        self.tp = self.fp = self.fn = self.id_changes = self.fragmentations = 0
        self.gt_count = Counter(); self.pred_count = Counter(); self.potential = Counter()
        self.previous = {}; self.missing = set(); self.ignored_outputs = 0

    def update(self, truth, proposals, ignored, threshold):
        pairs, values = match_boxes(truth,proposals,threshold)
        matched = {j for _,j in pairs}
        # Ignore only unmatched predictions overlapping valid Van/sitting boxes.
        # KITTI 2D DontCare handling is deliberately not approximated in 3D.
        ignored_indices = {j for j,p in enumerate(proposals) if j not in matched and
                           any(p.label == b.label and iou3d(p,b) >= threshold for b in ignored)}
        self.ignored_outputs += len(ignored_indices)
        self.tp += len(pairs); self.fp += len(proposals)-len(pairs)-len(ignored_indices)
        self.fn += len(truth)-len(pairs)
        changes = []
        mapping = {truth[i].object_id: proposals[j].track_id for i,j in pairs} if self.identity else {}
        if self.identity:
            for i,gt in enumerate(truth):
                gid = gt.object_id; self.gt_count[gid] += 1
                for j,p in enumerate(proposals):
                    if j not in ignored_indices and values[i][j] >= threshold:
                        self.potential[gid,p.track_id] += 1
                if gid in mapping:
                    if gid in self.previous:
                        if self.previous[gid] != mapping[gid]:
                            self.id_changes += 1; changes.append(gid)
                        if gid in self.missing:
                            self.fragmentations += 1
                    self.previous[gid] = mapping[gid]; self.missing.discard(gid)
                elif gid in self.previous:
                    self.missing.add(gid)
            for j,p in enumerate(proposals):
                if j not in ignored_indices:
                    self.pred_count[p.track_id] += 1
        return dict(matches=mapping,matched_gt_ids=[truth[i].object_id for i,j in pairs],
                    id_change_gt_ids=changes,tp=len(pairs),
                    fp=len(proposals)-len(pairs)-len(ignored_indices),fn=len(truth)-len(pairs))

    def result(self):
        ratio = lambda a,b: a/b if b else None
        result = dict(tp=self.tp,fp=self.fp,fn=self.fn,
                      precision=ratio(self.tp,self.tp+self.fp),recall=ratio(self.tp,self.tp+self.fn),
                      f1=ratio(2*self.tp,2*self.tp+self.fp+self.fn),ignored_outputs=self.ignored_outputs)
        if self.identity:
            gids = sorted(self.gt_count); pids = sorted(self.pred_count)
            costs = [[-float(self.potential[g,p]) if self.potential[g,p] else None for p in pids] for g in gids]
            pairs = assign(costs,len(pids),unmatched_cost=0.)
            idtp = sum(self.potential[gids[i],pids[j]] for i,j in pairs)
            idfn = sum(self.gt_count.values())-idtp; idfp = sum(self.pred_count.values())-idtp
            result.update(idtp=idtp,idfn=idfn,idfp=idfp,idf1=ratio(2*idtp,2*idtp+idfp+idfn),
                          id_changes=self.id_changes,fragmentations=self.fragmentations,
                          unique_predicted_ids=len(pids),unique_gt_ids=len(gids))
        return result


def dropout_recovery(details, interval, period):
    if interval is None:
        return None
    start = interval['start_frame']; end = start+interval['frames']
    # Establish eligibility from pre-dropout observed tracks and post-return
    # strong INPUT detections. A tracker failing to publish a post-return match
    # must count as failure rather than vanish from its own denominator.
    window = max(1,int(round(.5/period)))
    before = {}; after = {}; detectable_after = set()
    for row in details[max(0,start-window):start]:
        before.update(row['observed_tracks']['matches'])
    for row in details[end:end+window]:
        detectable_after.update(row['birth_eligible_detections']['matched_gt_ids'])
        for gid,pid in row['observed_tracks']['matches'].items():
            after.setdefault(gid,(pid,row['frame_index']))
    eligible = sorted(before.keys() & detectable_after)
    previous_owner = {pid:gid for gid,pid in before.items()}
    records = [dict(gt_id=g,before_id=before[g],after_id=after[g][0] if g in after else None,
                    same_id=g in after and before[g] == after[g][0],
                    return_latency_s=(after[g][1]-end)*period if g in after else None)
               for g in eligible]
    for record in records:
        record['previous_owner_gt_id'] = previous_owner.get(record['after_id'])
        record['wrong_retained_identity'] = (not record['same_id'] and
                                             record['previous_owner_gt_id'] is not None)
    return dict(kind=f'{interval["frames"]}-frame all-detection dropout, not natural occlusion',
                eligibility='pre observed GT match; post strong input detection GT match within 0.5 s',
                eligible=len(eligible),same_id=sum(r['same_id'] for r in records),
                changed_id=sum(not r['same_id'] and r['after_id'] is not None for r in records),
                no_track_after_return=sum(r['after_id'] is None for r in records),
                wrong_retained_identity=sum(r['wrong_retained_identity'] for r in records),
                pre_matched_targets=len(before),not_matched_after_return=len(before.keys()-after.keys()),
                not_detectable_after_return=len(before.keys()-detectable_after),
                targets=records)


def evaluate(root: Path, run_dir: Path, threshold=.25, write=True):
    if not isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError('IoU threshold must be within (0, 1]')
    summary = json.loads((run_dir/'summary.json').read_text())
    if summary.get('ground_truth_used_as_input') is not False:
        raise ValueError('replay must be label-free')
    truth, ignored = load_truth(root,summary['sequence'])
    ds = [json.loads(line) for line in (run_dir/'detections.jsonl').read_text().splitlines()]
    ts = [json.loads(line) for line in (run_dir/'tracks.jsonl').read_text().splitlines()]
    if len(ds) != len(ts) or len(ts) != len(truth) or len(ts) != summary['frames']:
        raise ValueError('frame counts do not align')
    scores = dict(observed_tracks=BoxScore(),published_tracks=BoxScore(),all_input_detections=BoxScore(False),
                  birth_eligible_detections=BoxScore(False))
    classes = {label:BoxScore() for label in ('Car','Pedestrian','Cyclist')}
    details = []
    for i,(d,t,gt,ig) in enumerate(zip(ds,ts,truth,ignored)):
        expected = i*summary['period_s']
        if any(row.get('frame_index') != i or abs(row['timestamp_s']-expected)>1e-8
               or not isfinite(row['timestamp_s']) for row in (d,t)):
            raise ValueError('frame index/timestamp mismatch')
        detections = [BoxDetection(**p) for p in d['detections']]
        tracks = [BoxTrack(**p) for p in t['tracks']]
        if len({p.track_id for p in tracks}) != len(tracks):
            raise ValueError('duplicate output ID in a frame')
        observed = [p for p in tracks if not p.predicted_only]
        proposals = dict(observed_tracks=observed,published_tracks=tracks,all_input_detections=detections,
                         birth_eligible_detections=[p for p in detections if p.confidence >= summary['tracker_config']['high_confidence']])
        row = dict(frame_index=i,timestamp_s=expected,truth=[dict(gt_id=g.object_id,**asdict(g.box)) for g in gt])
        for name,score in scores.items():
            row[name] = score.update(gt,proposals[name],ig,threshold)
        for label,score in classes.items():
            score.update([g for g in gt if g.box.label == label],
                         [p for p in observed if p.label == label],
                         [b for b in ig if b.label == label],threshold)
        details.append(row)
    result = dict(sequence=summary['sequence'],frames=len(ts),iou3d_threshold=threshold,
                  method='local class-aware upright 3D IoU, max-cardinality frame assignment and global IDF1',
                  official_kitti_score=False,dontcare_2d_regions_handled=False,
                  all_valid_3d_truth_included=True,**{k:v.result() for k,v in scores.items()},
                  per_class_observed={k:v.result() for k,v in classes.items()},
                  dropout_recovery=dropout_recovery(details,summary['hidden_interval'],summary['period_s']))
    if write:
        (run_dir/'evaluation.json').write_text(json.dumps(result,indent=2)+'\n')
        (run_dir/'evaluation_frames.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in details))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root',type=Path,required=True)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--iou3d',type=float,default=.25)
    args = parser.parse_args(argv)
    print(json.dumps(evaluate(args.dataset_root,args.run_dir,args.iou3d),indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
