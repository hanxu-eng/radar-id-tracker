"""Freeze 0000-only score selection, then compare regression/holdout and fixed dropout targets."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from radar_id_tracker.kitti_adapter import load_sequence,replay
from radar_id_tracker.kitti_evaluate import evaluate
from radar_id_tracker.lidar_tracker import ClassScorePolicy,LidarTrackerConfig
from radar_id_tracker.lidar_visualization import build_review


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def fixed_recovery(run_dir,sequence,start,targets,period=.1):
    """Never remove a baseline opportunity when the new policy loses its pre-track."""
    rows=[json.loads(s) for s in (run_dir/'evaluation_frames.jsonl').read_text().splitlines()]
    end=start+30;window=round(.5/period);before={};after={}
    for r in rows[max(0,start-window):start]:before.update(r['observed_tracks']['matches'])
    for r in rows[end:end+window]:
        for gid,pid in r['observed_tracks']['matches'].items():after.setdefault(gid,pid)
    owner={pid:gid for gid,pid in before.items()};records=[]
    for target in targets:
        gid=target['gt_id'];old=before.get(gid);new=after.get(gid)
        records.append(dict(sequence=sequence,start_frame=start,gt_id=gid,
                            before_id=old,after_id=new,same_id=old is not None and new==old,
                            missing_before=old is None,no_track_after_return=new is None,
                            wrong_retained_identity=new in owner and owner[new]!=gid))
    return records


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root',type=Path,required=True)
    p.add_argument('--detections-root',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,default=Path('outputs/kitti_score_policy'))
    p.add_argument('--protocol',type=Path,default=Path('configs/kitti_score_validation.json'))
    p.add_argument('--stress-reference',type=Path,default=Path('outputs/kitti_final/stress_grid.json'))
    p.add_argument('--stress-grid',action='store_true')
    args=p.parse_args();protocol=json.loads(args.protocol.read_text());root=args.output_dir
    baseline=LidarTrackerConfig(high_confidence=.98,low_confidence=.8)
    original_sequences=protocol['tuning_sequences']+protocol['regression_sequences']
    sequences=original_sequences+protocol['holdout_sequences']
    if len(set(sequences))!=len(sequences):raise ValueError('tuning/regression/holdout sequences must be disjoint')
    write(root/'protocol.json',protocol)
    baseline_scores={};rows=[]

    def run(seq,profile,cfg,scenario='natural',hide_start=None):
        out=root/seq/scenario/profile
        summary=replay(args.dataset_root,seq,out,cfg,hide_start=hide_start,detections_root=args.detections_root)
        score=evaluate(args.dataset_root,out)
        row=dict(sequence=seq,scenario=scenario,profile=profile,run=summary,evaluation=score)
        return row

    # Nothing from regression or holdout scoring is opened before the policy is frozen.
    for seq in protocol['tuning_sequences']:
        row=run(seq,'baseline',baseline);rows.append(row);baseline_scores[seq]=row['evaluation']
    if len(protocol['tuning_sequences'])!=1:raise ValueError('selection currently requires one tuning sequence')
    seq=protocol['tuning_sequences'][0];selected=[];sweep=[]
    for label,thresholds in protocol['birth_candidates'].items():
        original=baseline_scores[seq]['per_class_observed'][label];valid=[]
        for birth in thresholds:
            policy=ClassScorePolicy(label,min(protocol['continuation_confidence'],birth),birth,
                                    protocol['strong_confidence'],protocol['recovery_confidence'])
            cfg=LidarTrackerConfig(high_confidence=.98,low_confidence=.8,score_policies=(policy,),
                medium_confirmation_hits=protocol['medium_confirmation_hits'],
                medium_confirmation_window_s=protocol['medium_confirmation_window_s'],
                maximum_confirmation_gap_s=protocol['maximum_confirmation_gap_s'],
                maximum_tentative_tracks=protocol['maximum_tentative_tracks'],
                duplicate_birth_iou=protocol['duplicate_birth_iou'])
            row=run(seq,f'{label}_{birth}',cfg,scenario='tuning')
            score=row['evaluation']['per_class_observed'][label]
            eligible=(score['recall'] is not None and score['precision'] is not None and
                      score['precision'] >= (original['precision'] or 0)-.02 and
                      score['id_changes'] <= original['id_changes'])
            sweep.append(dict(label=label,birth_confidence=birth,score=score,eligible=eligible))
            if eligible:valid.append((score['recall'],birth,policy))
            print(f'tune {label} birth={birth}: precision={score["precision"]} recall={score["recall"]} eligible={eligible}',flush=True)
        chosen=max(valid,key=lambda v:(v[0],v[1]))[2] if valid else baseline.score_policy(label)
        selected.append(chosen)
    candidate=LidarTrackerConfig(high_confidence=.98,low_confidence=.8,score_policies=tuple(selected),
        medium_confirmation_hits=protocol['medium_confirmation_hits'],
        medium_confirmation_window_s=protocol['medium_confirmation_window_s'],
        maximum_confirmation_gap_s=protocol['maximum_confirmation_gap_s'],
        maximum_tentative_tracks=protocol['maximum_tentative_tracks'],duplicate_birth_iou=protocol['duplicate_birth_iou'])
    write(root/'tuning.json',sweep);write(root/'frozen_config.json',asdict(candidate))
    config_sha=hashlib.sha256((root/'frozen_config.json').read_bytes()).hexdigest()
    print('FROZEN POLICY '+json.dumps(asdict(candidate.score_policies[0]))+' sha256='+config_sha,flush=True)
    # Regression/holdout data are now scored with the frozen policy. No retuning.
    for seq in sequences:
        if seq not in baseline_scores:
            row=run(seq,'baseline',baseline);rows.append(row);baseline_scores[seq]=row['evaluation']
        row=run(seq,'candidate',candidate);rows.append(row)
        print(json.dumps(dict(sequence=seq,split='tuning' if seq in protocol['tuning_sequences'] else
                              'regression' if seq in protocol['regression_sequences'] else 'holdout',
                              score=row['evaluation']['observed_tracks'])),flush=True)
    write(root/'benchmark.json',rows)
    scores={(r['sequence'],r['profile']):r['evaluation']['observed_tracks'] for r in rows}
    b=scores['0005','baseline'];c=scores['0005','candidate'];gates=protocol['acceptance']
    checks=dict(recall_gain_0005=c['recall']-b['recall']>=gates['0005_minimum_recall_gain'],
                precision_0005=c['precision']>=gates['0005_minimum_precision'],
                regression_precision=all(scores[s,'candidate']['precision']>=scores[s,'baseline']['precision']-
                                         gates['maximum_regression_precision_loss'] for s in original_sequences),
                natural_id_changes=all(scores[s,'candidate']['id_changes']<=scores[s,'baseline']['id_changes'] for s in sequences))
    fixed=[];stress=[]
    if args.stress_grid:
        reference=json.loads(args.stress_reference.read_text())
        for seq in original_sequences:
            frames,_=load_sequence(args.dataset_root,seq,detections_root=args.detections_root)
            for start in range(20,len(frames)-35,30):
                targets=[r for r in reference['paired_targets'] if r['sequence']==seq and r['start_frame']==start]
                for name,cfg in (('baseline',baseline),('candidate',candidate)):
                    scenario=f'stress_start_{start}';row=run(seq,name,cfg,scenario,start);stress.append(row)
                    out=root/seq/scenario/name
                    fixed.extend(dict(profile=name,**r) for r in fixed_recovery(out,seq,start,targets))
                print(f'fixed stress {seq} start={start}: {len(targets)} baseline opportunities',flush=True)
        totals={name:dict(eligible=sum(r['profile']==name for r in fixed),
                         same_id=sum(r['profile']==name and r['same_id'] for r in fixed),
                         wrong_retained_identity=sum(r['profile']==name and r['wrong_retained_identity'] for r in fixed),
                         missing_before=sum(r['profile']==name and r['missing_before'] for r in fixed),
                         no_track_after_return=sum(r['profile']==name and r['no_track_after_return'] for r in fixed))
                for name in ('baseline','candidate')}
        checks['fixed_recovery']=totals['candidate']['eligible']==gates['fixed_recovery_opportunities'] and totals['candidate']['same_id']>=gates['minimum_fixed_recovery_successes']
        # Reject new wrong restorations even if an old wrong case disappears elsewhere.
        previous={(r['sequence'],r['start_frame'],r['gt_id']):r for r in fixed if r['profile']=='baseline'}
        checks['no_new_wrong_restorations']=not any(r['wrong_retained_identity'] and not previous[r['sequence'],r['start_frame'],r['gt_id']]['wrong_retained_identity'] for r in fixed if r['profile']=='candidate')
        write(root/'fixed_stress.json',dict(reference_sha256=hashlib.sha256(args.stress_reference.read_bytes()).hexdigest(),totals=totals,targets=fixed,
                                           new_eligible_targets={name:sum(r['evaluation']['dropout_recovery']['eligible'] for r in stress if r['profile']==name) for name in totals}))
        failed={(r['sequence'],r['start_frame']) for r in fixed if r['profile']=='candidate' and not r['same_id']}
        rows.extend(r for r in stress if (r['sequence'],int(r['scenario'].removeprefix('stress_start_'))) in failed)
    else:
        checks['fixed_recovery']=None;checks['no_new_wrong_restorations']=None
    write(root/'review_cases.json',rows);build_review(root,rows)
    report=dict(config_sha256=config_sha,checks=checks,recommended=all(v is True for v in checks.values()),
                baseline_retained=True,thresholds=[asdict(p) for p in candidate.score_policies])
    write(root/'acceptance.json',report);print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
