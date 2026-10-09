"""Evaluate cached raw surround detections and the old detector on one common ROI.

Threshold selection uses sequence 0000 only; other sequences are held out.
Raw GPU inference is a separate command so tuning never repeats neural inference.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from radar_id_tracker.lidar_raw import replay_cache
from radar_id_tracker.lidar_tracker import LidarTrackerConfig
from radar_id_tracker.kitti_adapter import replay
from radar_id_tracker.lidar_scoped_evaluate import evaluate_scoped,SCOPE,fixed_recovery
from radar_id_tracker.lidar_visualization import build_review


def label_review(path,comparison='detector'):
    html=path.read_text(encoding='utf-8')
    if comparison=='input':
        html=html.replace('原分数策略','CenterPoint 原输入')
        html=html.replace('候选确认策略（实验）','CenterPoint 输入适配（实验）')
        html=html.replace('左侧为CenterPoint 原输入，右侧为按类别候选确认策略；两侧身份均保留 4 秒',
                          '两侧使用相同 CenterPoint 权重和跟踪门限，比较原输入与开发集冻结后的输入适配；身份保留 4 秒')
        html=html.replace('点云仅加入可视化，跟踪仍使用公开 PointRCNN 检测框。',
                          '两侧都从真实点云全周检测；后方无 KITTI 真值。指标只统计前方 ±40°、50m 内 Car/Pedestrian，未达到部署验收。')
        path.write_text(html,encoding='utf-8')
        return
    html=html.replace('原分数策略','公开 PointRCNN 检测')
    html=html.replace('候选确认策略（实验）','原始点云 CenterPoint（实验）')
    html=html.replace('左侧为公开 PointRCNN 检测，右侧为按类别候选确认策略；两侧身份均保留 4 秒',
                      '左侧为公开 PointRCNN 检测，右侧直接使用原始点云进行全周 CenterPoint 推理；两侧身份均保留 4 秒')
    html=html.replace('点云仅加入可视化，跟踪仍使用公开 PointRCNN 检测框。',
                      '右侧点云参与神经检测；后方框仅供人工检查，KITTI 无后方真值。指标只统计前方 ±40°、50m 内 Car/Pedestrian。')
    path.write_text(html,encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root',type=Path,required=True)
    p.add_argument('--detections-root',type=Path,required=True)
    p.add_argument('--cache-root',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--sequences',nargs='+',default=['0000','0001','0005'])
    args=p.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if '0000' not in args.sequences:p.error('0000 is required for explicit tuning protocol')
    tuning=[]
    for threshold in (.2,.3,.4,.5,.6,.7,.8,.9,.95):
        config=LidarTrackerConfig(high_confidence=threshold,low_confidence=.1,
                                  maximum_tentative_tracks=128,duplicate_birth_iou=.85)
        path=args.output_dir/'tuning'/str(threshold)
        replay_cache(args.cache_root/'0000',path,config)
        result=evaluate_scoped(args.dataset_root,path)
        score=result['observed_tracks']
        tuning.append(dict(threshold=threshold,observed_tracks=score,config=asdict(config)))
        print(json.dumps(dict(tuning='0000',threshold=threshold,**score)),flush=True)
    # Selection must prioritize precision; if none meets 96%, retain explicit failure.
    eligible=[row for row in tuning if (row['observed_tracks']['precision'] or 0)>=.96]
    best=max(eligible,key=lambda r:(r['observed_tracks']['recall'] or 0,r['threshold'])) if eligible else max(tuning,key=lambda r:(r['observed_tracks']['f1'] or 0,r['threshold']))
    config=LidarTrackerConfig.from_dict(best['config'])
    (args.output_dir/'protocol.json').write_text(json.dumps(dict(tuning_sequence='0000',
        held_out_sequences=[s for s in args.sequences if s!='0000'],scope=SCOPE,
        acceptance_precision=.96,tuning_precision_passed=bool(eligible),
        fallback='highest tuning F1 if precision guard fails; experimental only',
        tuning=tuning,frozen_config=asdict(config)),indent=2)+'\n')
    rows=[]
    for sequence in args.sequences:
        for scenario,start in (('natural',None),('dropout_3s',50)):
            for profile in ('baseline','candidate'):
                path=args.output_dir/sequence/scenario/profile
                if profile=='baseline':
                    run=replay(args.dataset_root,sequence,path,LidarTrackerConfig(high_confidence=.98,low_confidence=.8),
                               hide_start=start,detections_root=args.detections_root)
                else:
                    run=replay_cache(args.cache_root/sequence,path,config,hide_start=start)
                result=evaluate_scoped(args.dataset_root,path)
                rows.append(dict(sequence=sequence,scenario=scenario,profile=profile,run=run,evaluation=result))
                print(json.dumps(dict(sequence=sequence,scenario=scenario,profile=profile,
                                     **result['observed_tracks'])),flush=True)
    (args.output_dir/'benchmark.json').write_text(json.dumps(rows,indent=2)+'\n')
    fixed={sequence:fixed_recovery(args.output_dir/sequence/'dropout_3s/baseline',
                                  args.output_dir/sequence/'dropout_3s/candidate') for sequence in args.sequences}
    (args.output_dir/'fixed_recovery.json').write_text(json.dumps(fixed,indent=2)+'\n')
    build_review(args.output_dir,rows)
    label_review(args.output_dir/'review.html')


if __name__=='__main__':main()
