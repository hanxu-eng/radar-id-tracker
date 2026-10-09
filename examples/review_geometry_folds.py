"""Per-sequence pointcloud review pages to bound browser payload size."""
import argparse
import html as markup
import json
import os
from pathlib import Path
import re
from radar_id_tracker.lidar_visualization import build_review
from fit_calibration_folds import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('parent-run','run-root'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();root=args.run_root
    protocol=json.loads((root/'protocol.json').read_text());plan=protocol['parent_plan']
    sequences=sorted(s for group in plan['groups'] for s in group)
    clouds=json.loads((args.parent_run/'pointclouds/manifest.json').read_text());assets=0
    for sequence in sequences:
        directory=root/'pointclouds'/sequence;directory.mkdir(parents=True,exist_ok=True)
        for frame in clouds['frames'][sequence]:
            source=args.parent_run/'pointclouds'/sequence/f'{int(frame):06d}.js';target=directory/source.name
            if not source.exists():raise ValueError('parent pointcloud asset missing')
            if target.exists():
                if sha(target)!=sha(source):raise ValueError('shared pointcloud differs')
            else:os.link(source,target)
            assets+=1
    (root/'pointclouds/manifest.json').write_text(json.dumps(clouds)+'\n')
    rows=json.loads((root/'benchmark.json').read_text());pages=[]
    if len(rows)!=136:raise ValueError('review requires all geometry evaluations')
    # Both natural and fixed dropout stay visible; no failure/sample selection.
    for sequence in sequences:
        for scenario in ('natural','dropout_3s'):
            selected=[{**r,'profile':'baseline' if r['profile']=='continued_l1' else 'candidate'} for r in rows if r['sequence']==sequence and r['scenario']==scenario and r['profile'] in ('continued_l1','weighted_axis_huber')]
            if len(selected)!=2:raise ValueError('review matched pair missing')
            path=build_review(root,selected);html=path.read_text(encoding='utf-8')
            note=('已查看开发数据上的匹配训练预算对照：左侧行人原 L1 额外训练 6 轮，右侧加权 SmoothL1／轴向角损失 6 轮。'
                  '每折只用训练组优化；分类、车辆、速度和 BN 统计冻结，不应用上一轮校准。GT 只作离线评分和叠图。'
                  '前向 ±40°、50m、Car/Pedestrian、3D IoU≥0.25；后方无真值，0020 未打开，不是实车或官方 KITTI 验收。')
            html=re.sub(r'<p class="note">.*?</p>',lambda m:'<p class="note">'+note+'</p>',html,count=1,flags=re.S)
            html=html.replace('KITTI：原始点云与稳定 ID 标注','KITTI：行人框几何与稳定 ID 对照')
            html=html.replace('原分数策略</h2>','匹配预算：原 L1</h2>').replace('候选确认策略（实验）</h2>','匹配预算：加权／轴向损失</h2>')
            html=html.replace('左侧为原分数策略，右侧为按类别候选确认策略；两侧身份均保留 4 秒','两侧均使用原 0.7/0.1 跟踪及 4 秒身份保留；仅行人几何训练损失不同')
            payload=json.loads(re.search(r'<script id="data" type="application/json">(.*?)</script>',html,re.S).group(1))
            frames=sum(len(p['frames']) for c in payload for p in c['profiles'].values())
            name=f'review_{sequence}_{scenario}.html';(root/name).write_text(html,encoding='utf-8');path.unlink()
            pages.append(dict(sequence=sequence,scenario=scenario,page=name,profile_frames=frames,bytes=(root/name).stat().st_size))
    links=''.join('<li><a href="'+markup.escape(r['page'])+'">'+r['sequence']+' '+r['scenario']+'</a></li>' for r in pages)
    index='<!doctype html><html lang="zh"><meta charset="utf-8"><title>行人框几何对照</title><body><h1>行人框几何：原始点云与 ID</h1><p>左：匹配预算原 L1；右：加权轴向损失。开发数据，后方未评分。按序列加载，避免一次解析所有完整序列。</p><ul>'+links+'</ul></body></html>'
    (root/'review.html').write_text(index,encoding='utf-8')
    audit=dict(cloud_assets=assets,missing_raw_scans=clouds['missing_frames'],pages=pages,static_payload_audit=True,browser_render_verified=False,
        total_profile_frames=sum(r['profile_frames'] for r in pages),all_sequences_and_scenarios_included=True)
    (root/'review_audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit))


if __name__=='__main__':main()
