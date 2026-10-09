"""Pointcloud review with explicit development-fold roles, including failures."""
import argparse
import json
from pathlib import Path
import re
from radar_id_tracker.lidar_score_calibration import load_calibration_plan
from radar_id_tracker.lidar_visualization import build_review
from radar_id_tracker.pointcloud_review import prepare_clouds


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--run-root',type=Path,required=True);p.add_argument('--prepare-clouds',action='store_true')
    args=p.parse_args();root=args.run_root;plan,digest=load_calibration_plan(args.plan)
    sequences=[s for g in plan['groups'] for s in g]
    if args.prepare_clouds:prepare_clouds(args.dataset_root,root,sequences)
    rows=json.loads((root/'benchmark.json').read_text());audit={}
    note=('所有序列属于已查看开发数据的整序列外层评价；每折检测训练、校准和评价相互隔离。'
          '0011/0012 同组，跨角色无扫描重复及 20m 内自车位置。不是新数据最终测试，0020 未打开。'
          '点云实际进入检测，分数校准仅使用检测分数；GT 仅用于离线评分/叠图。'
          '前向 ±40°、50m 内 Car/Pedestrian，3D IoU≥0.25，后方无真值；不是 KITTI 官方评分。')
    for left,name,title in (('head_raw','review_heads.html','相同分折检测头：原分数 vs 校准＋中分确认'),('baseline','review.html','原检测头 vs 分折检测头＋校准＋中分确认')):
        selected=[]
        for row in rows:
            if row['profile'] not in (left,'calibrated_medium'):continue
            selected.append({**row,'profile':'baseline' if row['profile']==left else 'candidate'})
        path=build_review(root,selected);html=path.read_text(encoding='utf-8')
        html=re.sub(r'<p class="note">.*?</p>',lambda m:'<p class="note">'+note+'</p>',html,count=1,flags=re.S)
        html=html.replace('KITTI：原始点云与稳定 ID 标注','KITTI：开发分折校准与稳定 ID 标注')
        html=html.replace('原分数策略</h2>',('分折检测头原分数' if left=='head_raw' else '原检测头')+'</h2>')
        html=html.replace('候选确认策略（实验）</h2>','分折检测头＋校准＋中分确认（实验）</h2>')
        html=html.replace('STRESS_NOTE','')
        destination=root/name;destination.write_text(html,encoding='utf-8')
        pattern=r'<script id="data" type="application/json">(.*?)</script>'
        payload=json.loads(re.search(pattern,html,re.S).group(1))
        clouds=json.loads(re.search(r'<script id="cloudMeta" type="application/json">(.*?)</script>',html,re.S).group(1))
        assets=sum(len(v) for v in clouds['frames'].values())
        missing_assets=[f'{s}/{i}' for s,frames in clouds['frames'].items() for i in frames if not (root/'pointclouds'/s/f'{int(i):06d}.js').exists()]
        if missing_assets:raise ValueError('cloud assets missing')
        audit[name]=dict(cases=len(payload),sequence_count=len({c['sequence'] for c in payload}),
            profile_frames=sum(len(p['frames']) for c in payload for p in c['profiles'].values()),cloud_assets=assets,
            missing_assets=missing_assets,missing_raw_scans=clouds['missing_frames'],
            static_payload_audit=True,browser_render_verified=False,title=title)
        if name!='review.html':path.unlink()
    (root/'review_audit.json').write_text(json.dumps(dict(plan_sha256=digest,pages=audit),indent=2)+'\n')
    print(json.dumps(audit),flush=True)


if __name__=='__main__':main()
